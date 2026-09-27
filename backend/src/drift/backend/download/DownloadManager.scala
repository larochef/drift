package drift.backend.download

import drift.backend.cache.{CacheEntry, ModelCache}
import drift.backend.routes.CivitaiClient
import drift.backend.storage.StorageService
import drift.shared.*

import java.net.URI
import java.net.http.*
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.time.Instant
import java.util.concurrent.{ConcurrentHashMap, Executors}
import java.util.concurrent.atomic.AtomicBoolean
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.{writeToString, WriterConfig}
import com.typesafe.scalalogging.Logger

/** Owns the download queue: at most `concurrency` transfers at once (saturating
  * the link with several 15 GB fetches helps nobody), one job per model, and a
  * registry the UI polls.
  */
final class DownloadManager(
    storage: StorageService,
    cache: ModelCache,
    civitaiClient: CivitaiClient,
    huggingFace: HuggingFaceDownloads,
    modelScope: drift.backend.modelscope.ModelScopeDownloads,
    downloader: Downloader,
    client: HttpClient,
    /** Resolved per request, so a token saved in Settings applies without a
      * restart.
      */
    civitaiToken: () => Option[String],
    concurrency: Int = 2
) {
  private val logger = Logger[DownloadManager]

  final private class Entry(
      @volatile var job: DownloadJob,
      val cancelled: AtomicBoolean = AtomicBoolean(false)
  )
  private val entries = ConcurrentHashMap[String, Entry]()

  private val executor = Executors.newFixedThreadPool(
    concurrency,
    runnable => {
      val thread = Thread(runnable, "drift-download")
      thread.setDaemon(true)
      thread
    }
  )

  def list: List[DownloadJob] =
    entries.values.asScala.map(_.job).toList.sortBy(_.modelId)

  def cancel(modelId: String): DownloadJob =
    Option(entries.get(modelId)) match {
      case None =>
        DownloadJob(
          modelId,
          DownloadState.Failed,
          error = Some("no download for this model")
        )
      case Some(entry) =>
        entry.cancelled.set(true)
        if (entry.job.state == DownloadState.Queued)
          entry.job = entry.job.copy(
            state = DownloadState.Cancelled,
            completedAt = Some(System.currentTimeMillis())
          )
        entry.job
    }

  def start(modelId: String): DownloadJob = synchronized {
    Option(entries.get(modelId)).map(_.job).filter(_.state.isActive) match {
      case Some(active) => active
      case None         =>
        val job = buildJob(modelId)
        job
    }
  }

  private def buildJob(modelId: String): DownloadJob =
    storage.get[Model]("models", modelId) match {
      case None =>
        terminal(modelId, DownloadState.Failed, Some("model is not registered"))
      case Some(model) =>
        cache.resolve(model.source) match {
          case CacheEntry.Present(path, bytes) =>
            terminal(
              modelId,
              DownloadState.Completed,
              None,
              downloadedBytes = bytes,
              totalBytes = Some(bytes)
            )
          case CacheEntry.BrokenLocal(path) =>
            terminal(
              modelId,
              DownloadState.Failed,
              Some(s"local path '$path' does not exist; nothing to download")
            )
          case CacheEntry.Absent =>
            model.source match {
              case Local(path) =>
                terminal(
                  modelId,
                  DownloadState.Failed,
                  Some("local source; nothing to download")
                )
              case source =>
                val entry = Entry(DownloadJob(modelId, DownloadState.Queued))
                entries.put(modelId, entry)
                executor.submit(runnable(model, source, entry))
                entry.job
            }
        }
    }

  private def runnable(model: Model, source: ModelSource, entry: Entry) =
    new Runnable {
      def run(): Unit =
        try {
          if (entry.cancelled.get()) return
          entry.job = entry.job.copy(
            state = DownloadState.Downloading,
            startedAt = Some(System.currentTimeMillis())
          )
          val onProgress: (Long, Option[Long]) => Unit = (done, total) =>
            entry.job = entry.job.copy(
              downloadedBytes = done,
              totalBytes = total.orElse(entry.job.totalBytes)
            )
          val outcome = source match {
            case huggingFaceSource: HuggingFace =>
              huggingFace.download(
                huggingFaceSource,
                () => entry.cancelled.get(),
                onProgress
              )
            case modelScopeSource: ModelScope =>
              modelScope.download(
                modelScopeSource,
                () => entry.cancelled.get(),
                onProgress
              )
            case civitaiSource: Civitai =>
              downloadCivitai(model, civitaiSource, entry, onProgress)
            case Local(_) => DownloadOutcome.Failed("local source")
          }
          val now = Some(System.currentTimeMillis())
          entry.job = outcome match {
            case DownloadOutcome.Completed(_, bytes) =>
              entry.job.copy(
                state = DownloadState.Completed,
                downloadedBytes = bytes,
                totalBytes = entry.job.totalBytes.orElse(Some(bytes)),
                completedAt = now
              )
            case DownloadOutcome.Cancelled =>
              entry.job
                .copy(state = DownloadState.Cancelled, completedAt = now)
            case DownloadOutcome.Failed(reason) =>
              logger.warn(s"Download of ${model.id} failed: $reason")
              entry.job.copy(
                state = DownloadState.Failed,
                error = Some(reason),
                completedAt = now
              )
          }
        } catch {
          case NonFatal(err) =>
            logger.warn(s"Download of ${model.id} blew up", err)
            entry.job = entry.job.copy(
              state = DownloadState.Failed,
              error = Some(Option(err.getMessage).getOrElse(err.toString)),
              completedAt = Some(System.currentTimeMillis())
            )
        }
    }

  private def downloadCivitai(
      model: Model,
      source: Civitai,
      entry: Entry,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    source.modelId.toIntOption
      .flatMap(civitaiClient.getModelDetail)
      .fold[DownloadOutcome](
        DownloadOutcome.Failed(
          s"Civitai has no model '${source.modelId}' (or the lookup failed)"
        )
      ) { detail =>
        val version = detail.modelVersions
          .find(_.id.toString == source.versionId)
          .orElse(
            detail.modelVersions
              .find(_.files.exists(_.id.toString == source.fileId))
          )
        val file = version.flatMap(v =>
          v.files
            .find(_.id.toString == source.fileId)
            .orElse(v.files.find(_.name == source.filename))
        )
        (version, file) match {
          case (Some(foundVersion), Some(foundFile)) =>
            val url = foundFile.downloadUrl.getOrElse(
              s"https://civitai.com/api/download/models/${foundVersion.id}?fileId=${foundFile.id}"
            )
            val target = cache.civitaiTarget(
              source,
              model.familyId,
              detail.name,
              foundVersion.name
            )
            val outcome = downloader.fetch(
              url = url,
              partFile =
                target.resolveSibling(target.getFileName.toString + ".part"),
              finalFile = target,
              expectedSha256 = foundFile.sha256,
              headers = civitaiToken()
                .map(t => "Authorization" -> s"Bearer $t")
                .toMap,
              isCancelled = () => entry.cancelled.get(),
              onProgress = onProgress
            )
            outcome match {
              case done: DownloadOutcome.Completed =>
                writeSidecar(target, detail, foundVersion, foundFile, source)
                done
              case other => other
            }
          case _ =>
            DownloadOutcome.Failed(
              s"version ${source.versionId} / file ${source.fileId} not found on Civitai model ${source.modelId}"
            )
        }
      }

  /** The metadata sidecar plus up to two small preview images, best effort — a
    * failed preview must not fail a verified multi-gigabyte download.
    */
  private def writeSidecar(
      target: Path,
      detail: CivitaiModelDetail,
      version: CivitaiModelVersion,
      file: CivitaiModelFile,
      source: Civitai
  ): Unit =
    try {
      val previews = version.images
        .filter(_.`type` == "image")
        .take(2)
        .zipWithIndex
        .flatMap { case (image, index) =>
          fetchPreview(image.url, target.getParent, index + 1)
        }
      val metadata = ModelMetadata(
        source = "civitai",
        modelId = source.modelId,
        versionId = source.versionId,
        fileId = source.fileId,
        name = detail.name,
        versionName = version.name,
        creator = detail.creator.flatMap(_.username),
        description = detail.description,
        tags = detail.tags,
        baseModel = version.baseModel,
        trainedWords = version.trainedWords,
        publishedAt = version.publishedAt,
        nsfwLevel = detail.nsfwLevel,
        downloadCount = detail.stats.map(_.downloadCount.toLong),
        thumbsUpCount = detail.stats.map(_.thumbsUpCount.toLong),
        sha256 = file.sha256,
        sizeBytes = file.sizeKB.map(kb => (kb * 1024).toLong),
        previews = previews,
        fetchedAt = Instant.now().toString
      )
      Files.write(
        target.getParent.resolve("drift-metadata.json"),
        writeToString(metadata, WriterConfig.withIndentionStep(2))
          .getBytes(StandardCharsets.UTF_8)
      )
    } catch {
      case NonFatal(err) =>
        logger.warn(s"Sidecar for ${source.modelId} not written", err)
    }

  private def fetchPreview(
      url: String,
      directory: Path,
      index: Int
  ): Option[String] =
    try {
      // The width segment is the same transform `CivitaiBrowser` uses for its
      // own thumbnails; recognition is the goal, not a gallery.
      val small = url.replace("/original=true/", "/width=450/original=true/")
      val extension =
        url.split('.').lastOption.filter(_.length <= 4).getOrElse("jpg")
      val name = s"preview-$index.$extension"
      val request = HttpRequest
        .newBuilder(URI.create(small))
        .header("User-Agent", "drift/0.1.0")
        .build()
      val response =
        client.send(request, HttpResponse.BodyHandlers.ofByteArray())
      if (response.statusCode() == 200) {
        Files.write(directory.resolve(name), response.body())
        Some(name)
      } else None
    } catch {
      case NonFatal(err) =>
        logger.debug(s"Preview fetch failed: ${err.getMessage}")
        None
    }

  private def terminal(
      modelId: String,
      state: DownloadState,
      error: Option[String],
      downloadedBytes: Long = 0L,
      totalBytes: Option[Long] = None
  ): DownloadJob = {
    val job = DownloadJob(
      modelId,
      state,
      downloadedBytes = downloadedBytes,
      totalBytes = totalBytes,
      error = error,
      completedAt = Some(System.currentTimeMillis())
    )
    entries.put(modelId, Entry(job))
    job
  }
}
