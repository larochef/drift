package drift.backend.download

import java.net.URI
import java.net.http.*
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.security.MessageDigest
import java.util.concurrent.{ConcurrentHashMap, Semaphore}
import java.util.concurrent.atomic.AtomicLong
import scala.jdk.CollectionConverters.*
import scala.jdk.OptionConverters.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import com.typesafe.scalalogging.Logger
import ox.flow.Flow

enum DownloadOutcome derives CanEqual {
  case Completed(path: Path, bytes: Long)

  /** The partial file is kept for a later resume. */
  case Cancelled

  case Failed(reason: String)
}

/** Which pieces of a chunked transfer are already on disk — the `<part>.chunks`
  * sidecar that makes an out-of-order partial resumable. Without it a partial
  * is only usable as a plain prefix.
  */
private case class ChunkState(
    totalBytes: Long,
    chunkBytes: Long,
    completed: List[Int] = List.empty
)
private object ChunkState {
  given JsonValueCodec[ChunkState] = JsonCodecMaker.make
}

/** Streams one URL to disk, hashing as it writes.
  *
  * The rules come from `specs/05-model-cache-and-downloads.md`:
  *   - never trust a partial file: write to `partFile`, atomically rename to
  *     `finalFile` only after the checksum passes;
  *   - resume, priming the digest from what is already on disk — which also
  *     catches a partial corrupted between runs;
  *   - a checksum mismatch deletes the partial and fails with both hashes; it
  *     is not retried automatically, because a repeatable mismatch means the
  *     source changed, not that the network hiccuped.
  *
  * Large files on range-capable hosts download as **parallel chunks**
  * ([[Downloader.WorkersPerDownload]] connections filling a work queue of
  * [[Downloader.ChunkBytes]] pieces): a single CDN stream rarely fills the
  * pipe, several usually do. Chunk completion is tracked in a `.chunks` sidecar
  * so cancel/crash resumes at chunk granularity, and a prefix-style partial
  * from the sequential path is adopted as its fully-covered chunks. The
  * checksum then runs over the assembled file, catching any assembly or
  * adoption mistake loudly. Everything else — no range support, unknown length,
  * small files — takes the sequential path unchanged.
  *
  * Whole transfers, chunked or not, also honour a cap of
  * [[Downloader.MaxTransfersPerHost]] concurrent downloads per host, shared by
  * every caller (models, LoRAs, runtimes) — saturating a host with five
  * transfers of four connections each helps nobody.
  */
final class Downloader(client: HttpClient) {
  private val logger = Logger[Downloader]
  private val bufferSize = 256 * 1024
  private val progressEveryBytes = 4L * 1024 * 1024

  def fetch(
      url: String,
      partFile: Path,
      finalFile: Path,
      expectedSha256: Option[String],
      headers: Map[String, String] = Map.empty,
      isCancelled: () => Boolean = () => false,
      onProgress: (Long, Option[Long]) => Unit = (_, _) => ()
  ): DownloadOutcome = {
    val permit = Downloader.hostPermit(url)
    permit.acquire()
    try
      fetchWithPermit(
        url,
        partFile,
        finalFile,
        expectedSha256,
        headers,
        isCancelled,
        onProgress
      )
    finally permit.release()
  }

  private def fetchWithPermit(
      url: String,
      partFile: Path,
      finalFile: Path,
      expectedSha256: Option[String],
      headers: Map[String, String],
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    try {
      Files.createDirectories(finalFile.getParent)
      Files.createDirectories(partFile.getParent)

      probeRangeSupport(url, headers) match {
        case Some(totalBytes) if totalBytes >= Downloader.ChunkedThreshold =>
          chunkedFetch(
            url,
            partFile,
            finalFile,
            totalBytes,
            expectedSha256,
            headers,
            isCancelled,
            onProgress
          )
        case _ =>
          // A chunked partial is sparse and only meaningful with its sidecar;
          // the sequential path can only prefix-resume, so a leftover chunk
          // state means starting clean.
          if (Files.exists(chunkStateFile(partFile))) {
            Files.deleteIfExists(chunkStateFile(partFile))
            Files.deleteIfExists(partFile)
          }
          sequentialFetch(
            url,
            partFile,
            finalFile,
            expectedSha256,
            headers,
            isCancelled,
            onProgress
          )
      }
    } catch {
      case NonFatal(err) =>
        logger.warn(s"Download failed: $url", err)
        DownloadOutcome.Failed(Option(err.getMessage).getOrElse(err.toString))
    }

  /** Whether the server honours ranges for this URL, and the full length — both
    * read from a one-byte range request. A server that answers 200 sent the
    * whole file; the stream is closed immediately and the sequential path takes
    * over.
    */
  private def probeRangeSupport(
      url: String,
      headers: Map[String, String]
  ): Option[Long] =
    try {
      val response = client.send(
        request(url, headers).header("Range", "bytes=0-0").build(),
        HttpResponse.BodyHandlers.ofInputStream()
      )
      try
        if (response.statusCode == 206)
          response
            .headers()
            .firstValue("Content-Range")
            .toScala
            .map(_.split("/").last.trim)
            .filter(_ != "*")
            .flatMap(_.toLongOption)
        else None
      finally response.body().close()
    } catch { case NonFatal(_) => None }

  // ----------------------------------------------------------------- chunked

  private def chunkedFetch(
      url: String,
      partFile: Path,
      finalFile: Path,
      totalBytes: Long,
      expectedSha256: Option[String],
      headers: Map[String, String],
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome = {
    val chunkBytes = Downloader.ChunkBytes
    val chunkCount = ((totalBytes + chunkBytes - 1) / chunkBytes).toInt
    val state = loadOrAdoptChunkState(partFile, totalBytes, chunkBytes)
    val completed = ConcurrentHashMap.newKeySet[Int]()
    state.completed.foreach(completed.add)

    def completedBytes: Long =
      completed.asScala.iterator
        .map(chunkLength(_, chunkBytes, totalBytes))
        .sum

    val chunksToFetch =
      (0 until chunkCount).filterNot(completed.contains).toList

    val downloadedTotal = AtomicLong(completedBytes)
    val lastReported = AtomicLong(downloadedTotal.get)
    onProgress(downloadedTotal.get, Some(totalBytes))

    val channel = FileChannel.open(
      partFile,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )
    // Cancellation travels as InterruptedException on purpose: `NonFatal`
    // excludes it, so the per-chunk retry below cannot swallow it, and the
    // flow's structured scope interrupts the sibling chunks.
    try {
      Flow
        .fromIterable(chunksToFetch)
        .mapParUnordered(Downloader.WorkersPerDownload) { chunk =>
          var attempt = 0
          var done = false
          while (!done) {
            if (isCancelled()) throw InterruptedException("cancelled")
            attempt += 1
            try {
              downloadChunk(
                url,
                headers,
                channel,
                chunk,
                chunkBytes,
                totalBytes,
                isCancelled,
                read => {
                  val total = downloadedTotal.addAndGet(read)
                  val last = lastReported.get
                  if (
                    total - last >= progressEveryBytes &&
                    lastReported.compareAndSet(last, total)
                  ) onProgress(total, Some(totalBytes))
                }
              )
              done = true
            } catch {
              case NonFatal(err) if attempt < Downloader.ChunkAttempts =>
                logger.info(
                  s"Chunk $chunk attempt $attempt failed (${err.getMessage}); retrying"
                )
                // The failed attempt's partial bytes were counted; recount
                // from the completed set to keep progress honest.
                downloadedTotal.set(completedBytes)
            }
          }
          chunk
        }
        // Completions land here on one thread: record each, so a cancel or
        // crash resumes at chunk granularity.
        .runForeach { chunk =>
          completed.add(chunk)
          saveChunkState(
            partFile,
            ChunkState(totalBytes, chunkBytes, completed.asScala.toList.sorted)
          )
        }
    } catch {
      case _ if isCancelled() =>
        logger.info(
          s"Chunked download cancelled at ${downloadedTotal.get}: $url"
        )
        return DownloadOutcome.Cancelled
      case NonFatal(err) =>
        return DownloadOutcome.Failed(
          Option(err.getMessage).getOrElse(err.toString)
        )
    } finally channel.close()

    if (completed.size != chunkCount)
      return DownloadOutcome.Failed(
        s"only ${completed.size} of $chunkCount chunks arrived"
      )
    onProgress(totalBytes, Some(totalBytes))

    // The chunks landed out of order, so the hash runs over the assembled
    // file — this is also what catches a wrong adoption of a sparse partial.
    expectedSha256.map(_.toLowerCase) match {
      case Some(expected) =>
        val actual = hashFile(partFile)
        if (actual != expected) {
          Files.deleteIfExists(partFile)
          Files.deleteIfExists(chunkStateFile(partFile))
          return DownloadOutcome.Failed(
            s"checksum mismatch: expected $expected, got $actual"
          )
        }
      case None => ()
    }
    Files.deleteIfExists(chunkStateFile(partFile))
    moveIntoPlace(partFile, finalFile)
    DownloadOutcome.Completed(finalFile, totalBytes)
  }

  private def downloadChunk(
      url: String,
      headers: Map[String, String],
      channel: FileChannel,
      chunk: Int,
      chunkBytes: Long,
      totalBytes: Long,
      isCancelled: () => Boolean,
      onRead: Long => Unit
  ): Unit = {
    val start = chunk.toLong * chunkBytes
    val endInclusive = math.min(start + chunkBytes, totalBytes) - 1
    val response = client.send(
      request(url, headers)
        .header("Range", s"bytes=$start-$endInclusive")
        .build(),
      HttpResponse.BodyHandlers.ofInputStream()
    )
    if (response.statusCode != 206) {
      response.body().close()
      throw new java.io.IOException(
        s"HTTP ${response.statusCode} for range $start-$endInclusive"
      )
    }
    val in = response.body()
    try {
      val buffer = new Array[Byte](bufferSize)
      var position = start
      var read = in.read(buffer)
      while (read > 0) {
        // Thrown, not returned: a half-written chunk must never look like a
        // success to the caller recording completions.
        if (isCancelled()) throw InterruptedException("cancelled")
        var written = 0
        while (written < read)
          written += channel.write(
            ByteBuffer.wrap(buffer, written, read - written),
            position + written
          )
        position += read
        onRead(read)
        read = in.read(buffer)
      }
      if (position != endInclusive + 1)
        throw new java.io.IOException(
          s"range $start-$endInclusive ended early at $position"
        )
    } finally
      try in.close()
      catch { case NonFatal(_) => () }
  }

  /** The saved chunk map, or — for a partial left by the sequential path — the
    * chunks its prefix fully covers, so nothing already fetched is refetched. A
    * mismatched total means the source changed: start clean.
    */
  private def loadOrAdoptChunkState(
      partFile: Path,
      totalBytes: Long,
      chunkBytes: Long
  ): ChunkState = {
    val sidecar = chunkStateFile(partFile)
    val saved =
      if (!Files.isRegularFile(sidecar)) None
      else
        try Some(readFromString[ChunkState](Files.readString(sidecar)))
        catch { case NonFatal(_) => None }
    saved match {
      case Some(state)
          if state.totalBytes == totalBytes &&
            state.chunkBytes == chunkBytes =>
        state
      case Some(_) =>
        Files.deleteIfExists(partFile)
        Files.deleteIfExists(sidecar)
        ChunkState(totalBytes, chunkBytes)
      case None if Files.isRegularFile(partFile) =>
        // A prefix-style partial: every chunk wholly below its length is
        // done. The boundary chunk is refetched.
        val prefix =
          try Files.size(partFile)
          catch { case NonFatal(_) => 0L }
        val covered =
          (0 until ((totalBytes + chunkBytes - 1) / chunkBytes).toInt)
            .filter(index => (index.toLong + 1) * chunkBytes <= prefix)
            .toList
        logger.info(
          s"Adopting prefix partial of $prefix bytes as ${covered.size} chunks: $partFile"
        )
        ChunkState(totalBytes, chunkBytes, covered)
      case None => ChunkState(totalBytes, chunkBytes)
    }
  }

  private def saveChunkState(partFile: Path, state: ChunkState): Unit =
    Files.write(
      chunkStateFile(partFile),
      writeToString(state).getBytes(StandardCharsets.UTF_8)
    )

  private def chunkStateFile(partFile: Path): Path =
    partFile.resolveSibling(partFile.getFileName.toString + ".chunks")

  private def chunkLength(
      index: Int,
      chunkBytes: Long,
      totalBytes: Long
  ): Long =
    math.min((index.toLong + 1) * chunkBytes, totalBytes) - index * chunkBytes

  private def hashFile(file: Path): String = {
    val digest = MessageDigest.getInstance("SHA-256")
    val in = Files.newInputStream(file)
    try {
      val buffer = new Array[Byte](1024 * 1024)
      var read = in.read(buffer)
      while (read > 0) {
        digest.update(buffer, 0, read)
        read = in.read(buffer)
      }
    } finally in.close()
    digest.digest().map("%02x".format(_)).mkString
  }

  // -------------------------------------------------------------- sequential

  private def sequentialFetch(
      url: String,
      partFile: Path,
      finalFile: Path,
      expectedSha256: Option[String],
      headers: Map[String, String],
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome = {
    val digest = MessageDigest.getInstance("SHA-256")
    var alreadyDownloaded = primeDigest(digest, partFile)

    var builder = request(url, headers)
    if (alreadyDownloaded > 0)
      builder = builder.header("Range", s"bytes=$alreadyDownloaded-")

    val response =
      client.send(builder.build(), HttpResponse.BodyHandlers.ofInputStream())
    val status = response.statusCode()

    val appending = status match {
      case 206 => true
      case 200 =>
        // The server ignored the range: start over, including the digest.
        if (alreadyDownloaded > 0) {
          digest.reset()
          alreadyDownloaded = 0L
        }
        false
      case other =>
        response.body().close()
        return DownloadOutcome.Failed(s"HTTP $other from $url")
    }

    val totalBytes = contentTotal(response, status, alreadyDownloaded)

    val out = Files.newOutputStream(
      partFile,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE,
      if (appending) StandardOpenOption.APPEND
      else StandardOpenOption.TRUNCATE_EXISTING
    )
    val in = response.body()
    var downloaded = alreadyDownloaded
    var lastReported = 0L
    try {
      onProgress(downloaded, totalBytes)
      val buffer = new Array[Byte](bufferSize)
      var read = in.read(buffer)
      while (read > 0) {
        if (isCancelled()) {
          logger.info(s"Download cancelled at $downloaded bytes: $url")
          return DownloadOutcome.Cancelled
        }
        out.write(buffer, 0, read)
        digest.update(buffer, 0, read)
        downloaded += read
        if (downloaded - lastReported >= progressEveryBytes) {
          lastReported = downloaded
          onProgress(downloaded, totalBytes)
        }
        read = in.read(buffer)
      }
    } finally {
      try in.close()
      catch { case NonFatal(_) => () }
      out.close()
    }
    onProgress(downloaded, totalBytes)

    val actualSha256 = digest.digest().map("%02x".format(_)).mkString
    expectedSha256.map(_.toLowerCase) match {
      case Some(expected) if expected != actualSha256 =>
        Files.deleteIfExists(partFile)
        DownloadOutcome.Failed(
          s"checksum mismatch: expected $expected, got $actualSha256"
        )
      case _ =>
        moveIntoPlace(partFile, finalFile)
        DownloadOutcome.Completed(finalFile, downloaded)
    }
  }

  // ----------------------------------------------------------------- helpers

  private def request(
      url: String,
      headers: Map[String, String]
  ): HttpRequest.Builder = {
    var builder = HttpRequest
      .newBuilder(URI.create(url))
      .header("User-Agent", "drift/0.1.0")
    headers.foreach { case (name, value) =>
      builder = builder.header(name, value)
    }
    builder
  }

  /** Feeds an existing partial file through the digest so a resumed download
    * still produces the hash of the whole content.
    */
  private def primeDigest(digest: MessageDigest, partFile: Path): Long = {
    if (!Files.isRegularFile(partFile)) return 0L
    var size = 0L
    val in = Files.newInputStream(partFile)
    try {
      val buffer = new Array[Byte](1024 * 1024)
      var read = in.read(buffer)
      while (read > 0) {
        digest.update(buffer, 0, read)
        size += read
        read = in.read(buffer)
      }
    } finally in.close()
    size
  }

  private def contentTotal(
      response: HttpResponse[?],
      status: Int,
      offset: Long
  ): Option[Long] =
    if (status == 206)
      // Content-Range: bytes <from>-<to>/<total>
      response
        .headers()
        .firstValue("Content-Range")
        .toScala
        .map(_.split("/").last.trim)
        .filter(_ != "*")
        .flatMap(_.toLongOption)
    else {
      val length = response.headers().firstValueAsLong("Content-Length")
      if (length.isPresent && length.getAsLong > 0) Some(length.getAsLong)
      else None
    }

  private def moveIntoPlace(partFile: Path, finalFile: Path): Unit =
    try
      Files.move(
        partFile,
        finalFile,
        StandardCopyOption.REPLACE_EXISTING,
        StandardCopyOption.ATOMIC_MOVE
      )
    catch {
      case _: AtomicMoveNotSupportedException =>
        Files.move(partFile, finalFile, StandardCopyOption.REPLACE_EXISTING)
    }
}

object Downloader {

  /** Concurrent range connections per chunked download. */
  val WorkersPerDownload = 4

  /** How often one chunk may fail before the download does. */
  val ChunkAttempts = 3

  /** One piece of the work queue — also the resume granularity. */
  val ChunkBytes: Long = 64L * 1024 * 1024

  /** Below this, one stream is fine and chunking is overhead. */
  val ChunkedThreshold: Long = 32L * 1024 * 1024

  /** Concurrent *transfers* per host across every caller; each chunked one
    * multiplies into [[WorkersPerDownload]] connections.
    */
  val MaxTransfersPerHost = 2

  private val hostPermits = ConcurrentHashMap[String, Semaphore]()

  private def hostPermit(url: String): Semaphore = {
    val host =
      try Option(URI.create(url).getHost).getOrElse("unknown").toLowerCase
      catch { case NonFatal(_) => "unknown" }
    hostPermits.computeIfAbsent(host, _ => Semaphore(MaxTransfersPerHost, true))
  }
}
