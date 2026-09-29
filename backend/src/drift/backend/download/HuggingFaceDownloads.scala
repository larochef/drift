package drift.backend.download

import drift.backend.cache.{ModelCache, SafetensorsIndex}
import drift.shared.*

import java.net.URI
import java.net.http.*
import java.nio.channels.FileChannel
import java.nio.file.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.typesafe.scalalogging.Logger

/** Downloads a HuggingFace file into the *shared* HuggingFace cache, in
  * HuggingFace's own layout, so one copy serves both drift and
  * `huggingface_hub`. The contract is documented in
  * `specs/05-model-cache-and-downloads.md`, "Writing into the HuggingFace
  * cache", and was verified against a live cache:
  *
  * {{{
  * <root>/.locks/models--{org}--{repo}/<sha256>.lock   held during the download
  * <root>/models--{org}--{repo}/
  *   refs/main              = <commit sha>
  *   blobs/<sha256>         = the content (etag == sha256 for LFS files)
  *   snapshots/<sha>/<path> -> relative symlink into blobs/
  * }}}
  *
  * Only ever adds — blobs, snapshot directories, symlinks. Never rewrites or
  * deletes anything it did not create.
  */
final class HuggingFaceDownloads(
    huggingFaceRoot: Path,
    driftRoot: Path,
    downloader: Downloader,
    client: HttpClient,
    /** Resolved per request, so a token saved in Settings applies without a
      * restart.
      */
    token: () => Option[String]
) {
  private val logger = Logger[HuggingFaceDownloads]

  def download(
      source: HuggingFace,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    fetchDetail(source.repo) match {
      case Left(reason) => DownloadOutcome.Failed(reason)
      case Right(detail) if ShardedSafetensors.isIndex(source.filename) =>
        downloadSharded(source, detail, isCancelled, onProgress)
      case Right(detail) =>
        val sibling = detail.siblings.find(_.rfilename == source.filename)
        val commit = detail.sha
        (sibling.flatMap(blobNameOf), commit) match {
          case (Some(blobName), Some(revision)) =>
            // an LFS file is verified by its sha256; a small file outside LFS
            // (a tokenizer.json) is named by its git blob id, unverified
            downloadIntoSharedCache(
              source,
              blobName,
              sibling.flatMap(_.sha256),
              revision,
              isCancelled,
              onProgress
            )
          case _ if sibling.isEmpty =>
            DownloadOutcome.Failed(
              s"'${source.filename}' is not in ${source.repo}"
            )
          case _ =>
            // No blob name or no commit: the shared cache layout is keyed by
            // both, so fall back to drift's own tree, unverified — the "trust
            // the file" case of the spec.
            downloadIntoDriftTree(source, isCancelled, onProgress)
        }
    }

  /** The file's entry in its repository listing — size and LFS hash — for a
    * caller that stores the file itself: a LoRA lives in the LoRA store, where
    * sd-cpp's `--lora-model-dir` finds it, not in the shared cache
    * (`specs/33-lora-sources.md`).
    */
  def lookup(source: HuggingFace): Either[String, HuggingFaceFileInfo] =
    fetchDetail(source.repo).flatMap(detail =>
      detail.siblings
        .find(_.rfilename == source.filename)
        .toRight(s"'${source.filename}' is not in ${source.repo}")
    )

  /** Fetches the file to `target` through a `.part` file (resumable), with the
    * saved token, verified when a hash is given.
    */
  def downloadTo(
      source: HuggingFace,
      target: Path,
      expectedSha256: Option[String],
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    downloader.fetch(
      url = resolveUrl(source, "main"),
      partFile = target.resolveSibling(target.getFileName.toString + ".part"),
      finalFile = target,
      expectedSha256 = expectedSha256,
      headers = authHeaders,
      isCancelled = isCancelled,
      onProgress = onProgress
    )

  /** A split safetensors model (`specs/34-sharded-safetensors.md`): the index,
    * then every shard it lists, all linked into the same snapshot so sd-cpp
    * finds the shards beside the index. Shards already in the cache are only
    * linked; progress runs over the shards' total.
    */
  private def downloadSharded(
      source: HuggingFace,
      detail: HuggingFaceModelDetail,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome = {
    val folder = source.filename.lastIndexOf('/') match {
      case -1    => ""
      case slash => source.filename.take(slash + 1)
    }
    (detail.siblings.find(_.rfilename == source.filename), detail.sha) match {
      case (None, _) =>
        DownloadOutcome.Failed(s"'${source.filename}' is not in ${source.repo}")
      case (_, None) =>
        DownloadOutcome.Failed(
          s"HuggingFace named no commit for ${source.repo}"
        )
      case (Some(indexInfo), Some(revision)) =>
        blobNameOf(indexInfo) match {
          case None =>
            DownloadOutcome.Failed(
              s"HuggingFace gave no content id for '${source.filename}'"
            )
          case Some(indexBlob) =>
            downloadIntoSharedCache(
              source,
              indexBlob,
              indexInfo.sha256,
              revision,
              isCancelled,
              (_, _) => ()
            ) match {
              case DownloadOutcome.Completed(blob, _) =>
                SafetensorsIndex.shardNames(blob) match {
                  case Left(reason) =>
                    DownloadOutcome.Failed(
                      s"'${source.filename}' cannot be read as a safetensors index: $reason"
                    )
                  case Right(names) =>
                    val listed = names.map(name =>
                      (folder + name) ->
                        detail.siblings.find(_.rfilename == folder + name)
                    )
                    listed.collectFirst { case (path, None) => path } match {
                      case Some(missing) =>
                        DownloadOutcome.Failed(
                          s"'$missing', listed by '${source.filename}', is not in ${source.repo}"
                        )
                      case None =>
                        val shards = listed.collect { case (path, Some(info)) =>
                          (path, info)
                        }
                        val total = shards
                          .flatMap((_, info) =>
                            info.lfs.flatMap(_.size).orElse(info.size)
                          )
                          .sum
                        onProgress(0L, Some(total))
                        val snapshotIndex = huggingFaceRoot
                          .resolve(ModelCache.repoDirectoryName(source.repo))
                          .resolve("snapshots")
                          .resolve(revision)
                          .resolve(source.filename)
                        fetchShards(
                          source.repo,
                          shards,
                          revision,
                          0L,
                          total,
                          snapshotIndex,
                          isCancelled,
                          onProgress
                        )
                    }
                }
              case other => other
            }
        }
    }
  }

  /** The shards one after the other, stopping at the first that does not
    * complete; `before` is what the earlier ones weighed.
    */
  private def fetchShards(
      repo: String,
      shards: List[(String, HuggingFaceFileInfo)],
      revision: String,
      before: Long,
      total: Long,
      snapshotIndex: Path,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    shards match {
      case Nil => DownloadOutcome.Completed(snapshotIndex, before)
      case _ if isCancelled()   => DownloadOutcome.Cancelled
      case (path, info) :: rest =>
        blobNameOf(info) match {
          case None =>
            DownloadOutcome.Failed(
              s"HuggingFace gave no content id for '$path'"
            )
          case Some(blobName) =>
            downloadIntoSharedCache(
              HuggingFace(repo, path),
              blobName,
              info.sha256,
              revision,
              isCancelled,
              (bytes, _) => onProgress(before + bytes, Some(total))
            ) match {
              case DownloadOutcome.Completed(_, bytes) =>
                fetchShards(
                  repo,
                  rest,
                  revision,
                  before + bytes,
                  total,
                  snapshotIndex,
                  isCancelled,
                  onProgress
                )
              case other => other
            }
        }
    }

  /** The name `huggingface_hub` gives a file's blob: its ETag, which is the LFS
    * sha256, or the git blob id for a small file outside LFS.
    */
  private def blobNameOf(info: HuggingFaceFileInfo): Option[String] =
    info.sha256.orElse(info.blobId)

  /** One file into the shared cache as `blobs/<blobName>`, linked from the
    * revision's snapshot; only a sha256 is verified.
    */
  private def downloadIntoSharedCache(
      source: HuggingFace,
      blobName: String,
      expectedSha256: Option[String],
      revision: String,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome = {
    val repoDirectory =
      huggingFaceRoot.resolve(ModelCache.repoDirectoryName(source.repo))
    val blob = repoDirectory.resolve("blobs").resolve(blobName)

    withLock(source.repo, blobName) {
      val outcome =
        if (Files.isRegularFile(blob))
          // Someone — possibly `huggingface_hub` — already has the content.
          DownloadOutcome.Completed(blob, Files.size(blob))
        else
          downloader.fetch(
            url = resolveUrl(source, revision),
            // `.incomplete` rather than `.part`: the shared cache's own
            // convention for in-flight blobs.
            partFile =
              repoDirectory.resolve("blobs").resolve(s"$blobName.incomplete"),
            finalFile = blob,
            expectedSha256 = expectedSha256,
            headers = authHeaders,
            isCancelled = isCancelled,
            onProgress = onProgress
          )

      outcome match {
        case done: DownloadOutcome.Completed =>
          linkSnapshot(repoDirectory, revision, source.filename, blob)
          writeRef(repoDirectory, revision)
          done
        case other => other
      }
    }
  }

  private def downloadIntoDriftTree(
      source: HuggingFace,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome = {
    val target = driftRoot
      .resolve("models")
      .resolve("huggingface")
      .resolve(ModelCache.repoDirectoryName(source.repo))
      .resolve(source.filename)
    downloadTo(source, target, None, isCancelled, onProgress)
  }

  private def resolveUrl(source: HuggingFace, revision: String): String =
    s"https://huggingface.co/${source.repo}/resolve/$revision/${source.filename}"

  private def fetchDetail(
      repo: String
  ): Either[String, HuggingFaceModelDetail] =
    try {
      var builder = HttpRequest
        .newBuilder(
          URI.create(s"https://huggingface.co/api/models/$repo?blobs=true")
        )
        .header("User-Agent", "drift/0.1.0")
      token().foreach(t =>
        builder = builder.header("Authorization", s"Bearer $t")
      )
      val response =
        client.send(builder.build(), HttpResponse.BodyHandlers.ofString())
      if (response.statusCode() == 200)
        Right(readFromString[HuggingFaceModelDetail](response.body()))
      else
        Left(
          s"HuggingFace API answered HTTP ${response.statusCode()} for $repo"
        )
    } catch {
      case NonFatal(err) =>
        Left(s"HuggingFace API request failed: ${err.getMessage}")
    }

  /** The snapshot entry is a *relative* symlink into `blobs/`, exactly as
    * `huggingface_hub` writes it. `relativize` computes the `../..` depth, so
    * nested repo paths need no counting.
    */
  private def linkSnapshot(
      repoDirectory: Path,
      revision: String,
      filename: String,
      blob: Path
  ): Unit = {
    val link =
      repoDirectory.resolve("snapshots").resolve(revision).resolve(filename)
    Files.createDirectories(link.getParent)
    if (!Files.exists(link, java.nio.file.LinkOption.NOFOLLOW_LINKS))
      Files.createSymbolicLink(link, link.getParent.relativize(blob))
  }

  private def writeRef(repoDirectory: Path, revision: String): Unit =
    try {
      val refs = repoDirectory.resolve("refs")
      Files.createDirectories(refs)
      Files.writeString(refs.resolve("main"), revision)
    } catch {
      case NonFatal(err) =>
        logger.warn(s"Could not write refs/main: ${err.getMessage}")
    }

  /** `huggingface_hub` holds a lock per blob while downloading; taking the same
    * lock keeps a concurrent `hf download` of the same file from racing us.
    */
  private def withLock[A](repo: String, sha256: String)(body: => A): A = {
    val lockDirectory =
      huggingFaceRoot
        .resolve(".locks")
        .resolve(ModelCache.repoDirectoryName(repo))
    Files.createDirectories(lockDirectory)
    val channel = FileChannel.open(
      lockDirectory.resolve(s"$sha256.lock"),
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )
    val lock = channel.lock()
    try body
    finally {
      lock.release()
      channel.close()
    }
  }

  private def authHeaders: Map[String, String] =
    token().map(t => "Authorization" -> s"Bearer $t").toMap
}
