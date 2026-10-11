package drift.backend.models

import drift.backend.Background
import drift.backend.cache.{CacheEntry, ModelCache}
import drift.backend.civitai.CivitaiClient
import drift.backend.download.*
import drift.backend.huggingface.HuggingFaceDownloads
import drift.backend.storage.{Collections, StorageService}
import drift.shared.*

import java.net.URI
import java.net.http.*
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.time.Instant
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.{writeToString, WriterConfig}
import com.typesafe.scalalogging.Logger

/** The downloads of registered models: what each one fetches and where to — its
  * site, its cache folder, Civitai's metadata beside the file — on a queue of
  * its own (`DownloadQueue`).
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
    /** Where its downloads run. */
    background: Background,
    concurrency: Int = 2
) {
  private val logger = Logger[DownloadManager]

  /** One transfer per model, by its id (`DownloadQueue`). */
  private val downloads =
    DownloadQueue[String](background, "drift-download", concurrency)

  private def jobOf(modelId: String, transfer: Transfer): DownloadJob =
    DownloadJob(
      modelId,
      transfer.state,
      downloadedBytes = transfer.downloadedBytes,
      totalBytes = transfer.totalBytes,
      error = transfer.error,
      startedAt = transfer.startedAt,
      completedAt = transfer.completedAt
    )

  def list: List[DownloadJob] =
    downloads.all.toList.map(jobOf).sortBy(_.modelId)

  def cancel(modelId: String): DownloadJob =
    downloads
      .cancel(modelId)
      .fold(
        DownloadJob(
          modelId,
          DownloadState.Failed,
          error = Some("no download for this model")
        )
      )(jobOf(modelId, _))

  /** Queues the model's download, unless one is under way or there is nothing
    * to fetch: the job as it stands.
    */
  def start(modelId: String): DownloadJob =
    jobOf(
      modelId,
      downloads.get(modelId).filter(_.state.isActive).getOrElse(queued(modelId))
    )

  private def queued(modelId: String): Transfer = {
    def nothing(reason: String) =
      downloads.settled(modelId, DownloadState.Failed, Some(reason))
    storage.get(Collections.models, modelId) match {
      case None        => nothing("model is not registered")
      case Some(model) =>
        cache.resolve(model.source) match {
          case CacheEntry.Present(_, bytes) =>
            downloads
              .settled(modelId, DownloadState.Completed, bytes = Some(bytes))
          case CacheEntry.BrokenLocal(path) =>
            nothing(s"local path '$path' does not exist; nothing to download")
          case CacheEntry.Absent =>
            model.source match {
              case Local(_) => nothing("local source; nothing to download")
              case source   =>
                downloads.queue(modelId)((isCancelled, onProgress) =>
                  source match {
                    case huggingFaceSource: HuggingFace =>
                      huggingFace
                        .download(huggingFaceSource, isCancelled, onProgress)
                    case modelScopeSource: ModelScope =>
                      modelScope
                        .download(modelScopeSource, isCancelled, onProgress)
                    case civitaiSource: Civitai =>
                      downloadCivitai(
                        model,
                        civitaiSource,
                        isCancelled,
                        onProgress
                      )
                    case Local(_) => DownloadOutcome.Failed("local source")
                  }
                )
            }
        }
    }
  }

  private def downloadCivitai(
      model: Model,
      source: Civitai,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    source.modelId.toIntOption
      .toRight(s"'${source.modelId}' is not a Civitai model's number")
      .flatMap(civitaiClient.getModelDetail)
      .fold[DownloadOutcome](
        DownloadOutcome.Failed(_),
        { detail =>
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
                civitaiClient.downloadAddress(
                  foundVersion.id.toString,
                  foundFile.id.toString
                )
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
                headers = civitaiClient.downloadHeaders,
                isCancelled = isCancelled,
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
      )

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
}
