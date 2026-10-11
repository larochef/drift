package drift.backend.upscale

import drift.backend.Background
import drift.backend.civitai.CivitaiClient
import drift.backend.download.*
import drift.backend.storage.{Collections, StorageService}
import drift.shared.*

import java.nio.file.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger
import ox.discard

/** Owns the upscaler store (`specs/10-generation-time-upscaling.md`): a flat
  * `~/.cache/drift/upscale/` directory of ESRGAN weight files, read by the
  * post-hoc upscale (`specs/15-post-hoc-resize.md`) and handed to no launch.
  * Unlike models and LoRAs there is no grouping structure, just weight files
  * whose stems are the upscaler names.
  *
  * Transfers share [[Downloader]] (resume, sha256, progress) with the other
  * stores. `resumeInterrupted` queues every installed entity at startup: files
  * on disk become immediately `Completed` jobs — so the job list is also the
  * "what is actually fetched" answer — and missing ones resume off their
  * `.part`.
  */
final class UpscalerManager(
    storage: StorageService,
    civitaiClient: CivitaiClient,
    downloader: Downloader,
    /** Resolved per request, so a token saved in Settings applies without a
      * restart.
      */
    val upscaleRoot: Path,
    /** Where its downloads run. */
    background: Background
) {
  private val logger = Logger[UpscalerManager]

  /** One transfer per upscaler, by its id (`DownloadQueue`). */
  private val downloads =
    DownloadQueue[String](background, "drift-upscaler-download")

  private def jobOf(upscalerId: String, transfer: Transfer) =
    UpscalerDownloadJob(
      upscalerId,
      transfer.state,
      downloadedBytes = transfer.downloadedBytes,
      totalBytes = transfer.totalBytes,
      error = transfer.error
    )

  def list: List[Upscaler] =
    storage.list(Collections.upscalers).sortBy(_.label)

  /** Stops a download from the downloads panel: a queued one at once, a running
    * one at its next read, its `.part` kept for a resume. The upscaler stays
    * registered, so a restart resumes it (`resumeInterrupted`).
    */
  def cancelDownload(upscalerId: String): Option[UpscalerDownloadJob] =
    downloads.cancel(upscalerId).map(jobOf(upscalerId, _))

  def listJobs: List[UpscalerDownloadJob] =
    downloads.all.toList.map(jobOf).sortBy(_.upscalerId)

  // ----------------------------------------------------------------- install

  def installFromUrl(
      request: InstallUpscalerRequest
  ): InstallUpscalerResponse = {
    val url = request.url.trim
    if (url.isEmpty)
      return InstallUpscalerResponse(error = Some("no URL given"))
    val fileName = request.fileName
      .map(_.trim)
      .filter(_.nonEmpty)
      .orElse(UpscalerManager.fileNameOf(url))
      .getOrElse("")
    UpscalerManager.stemOf(fileName) match {
      case None =>
        InstallUpscalerResponse(error =
          Some(
            s"'$fileName' is not a weight file (expected " +
              s"${UpscalerManager.WeightExtensions.mkString(", ")})"
          )
        )
      case Some(stem) =>
        install(
          Upscaler(
            id = stem,
            label =
              request.label.map(_.trim).filter(_.nonEmpty).getOrElse(stem),
            fileName = fileName,
            downloadUrl = url,
            sha256 = request.sha256.map(_.trim.toLowerCase).filter(_.nonEmpty),
            sizeBytes = None,
            civitaiModelId = None,
            createdAt = System.currentTimeMillis()
          )
        ) match {
          case Left(reason)    => InstallUpscalerResponse(error = Some(reason))
          case Right(upscaler) =>
            InstallUpscalerResponse(upscalers = List(upscaler))
        }
    }
  }

  /** The browser flow: install the version's weight files, Civitai naming and
    * hashes included. On-disk names carry the `<fileId>-` prefix like the other
    * Civitai stores, so two versions publishing the same filename cannot
    * collide.
    */
  def installFromCivitai(
      request: InstallUpscalerFromCivitaiRequest
  ): InstallUpscalerResponse =
    request.civitaiModelId.toIntOption
      .toRight(s"'${request.civitaiModelId}' is not a Civitai model's number")
      .flatMap(civitaiClient.getModelDetail) match {
      case Left(reason)  => InstallUpscalerResponse(error = Some(reason))
      case Right(detail) =>
        detail.modelVersions.find(_.id.toString == request.versionId) match {
          case None =>
            InstallUpscalerResponse(error =
              Some(
                s"model '${detail.name}' has no version '${request.versionId}'"
              )
            )
          case Some(version) =>
            val allWeightFiles = version.files.filter(file =>
              UpscalerManager.stemOf(file.name).isDefined
            )
            val weightFiles =
              if (request.fileIds.isEmpty) allWeightFiles
              else
                allWeightFiles.filter(file =>
                  request.fileIds.contains(file.id.toString)
                )
            if (weightFiles.isEmpty)
              InstallUpscalerResponse(error =
                Some(
                  s"version '${version.name}' has no weight files to install"
                )
              )
            else {
              val results = weightFiles.map { file =>
                val fileName = s"${file.id}-${file.name}"
                install(
                  Upscaler(
                    id = UpscalerManager.stemOf(fileName).get,
                    label =
                      if (weightFiles.sizeIs == 1) detail.name
                      else s"${detail.name} — ${file.name}",
                    fileName = fileName,
                    downloadUrl = file.downloadUrl.getOrElse(
                      civitaiClient
                        .downloadAddress(version.id.toString, file.id.toString)
                    ),
                    sha256 = file.sha256,
                    sizeBytes = file.sizeKB.map(kb => (kb * 1024).toLong),
                    civitaiModelId = Some(detail.id.toString),
                    createdAt = System.currentTimeMillis()
                  )
                )
              }
              val installed = results.collect { case Right(upscaler) =>
                upscaler
              }
              val refused = results.collect { case Left(reason) => reason }
              InstallUpscalerResponse(
                upscalers = installed,
                error =
                  if (refused.isEmpty) None else Some(refused.mkString("; "))
              )
            }
        }
    }

  /** A re-install of something already on disk is refused rather than silently
    * re-queued — deleting first is the explicit way to refetch.
    */
  private def install(upscaler: Upscaler): Either[String, Upscaler] = {
    val existing = storage.get(Collections.upscalers, upscaler.id)
    if (
      existing.isDefined &&
      Files.isRegularFile(upscaleRoot.resolve(existing.get.fileName))
    )
      Left(s"upscaler '${upscaler.id}' is already installed")
    else {
      storage.save(Collections.upscalers, upscaler.id, upscaler)
      queueDownload(upscaler)
      Right(upscaler)
    }
  }

  /** Queues every installed upscaler. Files on disk get their `Completed` job
    * immediately; missing ones resume off their `.part`. Called once at
    * startup.
    */
  def resumeInterrupted(): Unit =
    list.foreach { upscaler =>
      if (!Files.isRegularFile(upscaleRoot.resolve(upscaler.fileName)))
        logger.info(s"Resuming interrupted upscaler download ${upscaler.id}")
      queueDownload(upscaler)
    }

  private def queueDownload(upscaler: Upscaler): Unit = {
    val target = upscaleRoot.resolve(upscaler.fileName)
    if (Files.isRegularFile(target))
      downloads
        .settled(
          upscaler.id,
          DownloadState.Completed,
          bytes = Some(Files.size(target))
        )
        .discard
    else
      downloads
        .queue(upscaler.id) { (isCancelled, onProgress) =>
          Files.createDirectories(upscaleRoot)
          downloader.fetch(
            url = upscaler.downloadUrl,
            partFile =
              target.resolveSibling(target.getFileName.toString + ".part"),
            finalFile = target,
            expectedSha256 = upscaler.sha256,
            headers = civitaiClient.headersFor(upscaler.downloadUrl),
            isCancelled = isCancelled,
            onProgress = onProgress
          )
        }
        .discard
  }

  // ------------------------------------------------------------------ delete

  /** Deletes the entity, the weight file and any `.part` leftover. */
  def delete(id: String): Boolean = {
    storage.get(Collections.upscalers, id).foreach { upscaler =>
      val target = upscaleRoot.resolve(upscaler.fileName)
      try {
        Files.deleteIfExists(
          target.resolveSibling(target.getFileName.toString + ".part")
        )
        Files.deleteIfExists(target)
      } catch {
        case NonFatal(err) =>
          logger.warn(s"Deleting '$target' failed: ${err.getMessage}")
      }
    }
    downloads.forget(id)
    storage.delete(Collections.upscalers, id)
  }
}

object UpscalerManager {

  /** What sd-server can load as an ESRGAN upscaler — `.pth` included: the
    * classic RealESRGAN releases ship as torch checkpoints.
    */
  val WeightExtensions: Set[String] =
    Set(".pth", ".pt", ".safetensors", ".ckpt", ".gguf")

  /** The last path segment of a URL, query and fragment stripped. */
  def fileNameOf(url: String): Option[String] =
    url
      .takeWhile(c => c != '?' && c != '#')
      .split('/')
      .lastOption
      .map(_.trim)
      .filter(_.nonEmpty)

  /** The stem sd-server will name the upscaler by, or None for anything that is
    * not a weight file.
    */
  def stemOf(fileName: String): Option[String] =
    WeightExtensions
      .find(extension => fileName.toLowerCase.endsWith(extension))
      .map(extension => fileName.dropRight(extension.length))
      .filter(_.nonEmpty)
}
