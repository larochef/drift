package drift.backend.upscale

import drift.backend.download.{DownloadOutcome, Downloader}
import drift.backend.routes.CivitaiClient
import drift.backend.storage.StorageService
import drift.shared.*

import java.net.URI
import java.nio.file.*
import java.util.concurrent.{ConcurrentHashMap, Executors}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Owns the upscaler store (`specs/10-generation-time-upscaling.md`): a flat
  * `~/.cache/drift/upscale/` directory that every session receives as
  * `--hires-upscalers-dir`. Flat because sd-server scans only the top level —
  * subdirectories would hide the files — so unlike models and LoRAs there is no
  * grouping structure, just weight files whose stems become the upscaler names.
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
    civitaiToken: () => Option[String],
    val upscaleRoot: Path
) {
  private val logger = Logger[UpscalerManager]

  final private class Entry(@volatile var job: UpscalerDownloadJob)
  private val jobs = ConcurrentHashMap[String, Entry]()

  private val executor = Executors.newFixedThreadPool(
    2,
    runnable => {
      val thread = Thread(runnable, "drift-upscaler-download")
      thread.setDaemon(true)
      thread
    }
  )

  def list: List[Upscaler] =
    storage.list[Upscaler]("upscalers").sortBy(_.label)

  def listJobs: List[UpscalerDownloadJob] =
    jobs.values.asScala.map(_.job).toList.sortBy(_.upscalerId)

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
    request.civitaiModelId.toIntOption.flatMap(
      civitaiClient.getModelDetail
    ) match {
      case None =>
        InstallUpscalerResponse(error =
          Some(
            s"Civitai has no model '${request.civitaiModelId}' (or the lookup failed)"
          )
        )
      case Some(detail) =>
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
                      s"https://civitai.com/api/download/models/${version.id}?fileId=${file.id}"
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
    val existing = storage.get[Upscaler]("upscalers", upscaler.id)
    if (
      existing.isDefined &&
      Files.isRegularFile(upscaleRoot.resolve(existing.get.fileName))
    )
      Left(s"upscaler '${upscaler.id}' is already installed")
    else {
      storage.save("upscalers", upscaler.id, upscaler)
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
    if (Files.isRegularFile(target)) {
      jobs.put(
        upscaler.id,
        Entry(
          UpscalerDownloadJob(
            upscaler.id,
            DownloadState.Completed,
            downloadedBytes = Files.size(target),
            totalBytes = Some(Files.size(target))
          )
        )
      )
      return
    }
    Option(jobs.get(upscaler.id)).map(_.job).filter(_.state.isActive) match {
      case Some(_) => ()
      case None    =>
        val entry =
          Entry(UpscalerDownloadJob(upscaler.id, DownloadState.Queued))
        jobs.put(upscaler.id, entry)
        executor.submit(new Runnable {
          def run(): Unit =
            try {
              entry.job = entry.job.copy(state = DownloadState.Downloading)
              Files.createDirectories(upscaleRoot)
              val outcome = downloader.fetch(
                url = upscaler.downloadUrl,
                partFile =
                  target.resolveSibling(target.getFileName.toString + ".part"),
                finalFile = target,
                expectedSha256 = upscaler.sha256,
                headers = authHeaders(upscaler.downloadUrl),
                isCancelled = () => false,
                onProgress = (done, total) =>
                  entry.job = entry.job.copy(
                    downloadedBytes = done,
                    totalBytes = total.orElse(entry.job.totalBytes)
                  )
              )
              entry.job = outcome match {
                case DownloadOutcome.Completed(_, bytes) =>
                  entry.job.copy(
                    state = DownloadState.Completed,
                    downloadedBytes = bytes,
                    totalBytes = entry.job.totalBytes.orElse(Some(bytes))
                  )
                case DownloadOutcome.Cancelled =>
                  entry.job.copy(state = DownloadState.Cancelled)
                case DownloadOutcome.Failed(reason) =>
                  entry.job
                    .copy(state = DownloadState.Failed, error = Some(reason))
              }
            } catch {
              case NonFatal(err) =>
                logger.warn(
                  s"Upscaler download ${upscaler.id} blew up",
                  err
                )
                entry.job = entry.job.copy(
                  state = DownloadState.Failed,
                  error = Some(Option(err.getMessage).getOrElse(err.toString))
                )
            }
        })
    }
  }

  /** The Civitai token authorizes Civitai downloads only — sending it to
    * github.com or huggingface.co would leak it.
    */
  private def authHeaders(url: String): Map[String, String] = {
    val isCivitai =
      try {
        val host = Option(URI.create(url).getHost).getOrElse("")
        host == "civitai.com" || host.endsWith(".civitai.com")
      } catch { case NonFatal(_) => false }
    if (isCivitai)
      civitaiToken().map(t => "Authorization" -> s"Bearer $t").toMap
    else Map.empty
  }

  // ------------------------------------------------------------------ delete

  /** Deletes the entity, the weight file and any `.part` leftover. */
  def delete(id: String): Boolean = {
    storage.get[Upscaler]("upscalers", id).foreach { upscaler =>
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
    jobs.remove(id)
    storage.delete("upscalers", id)
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
