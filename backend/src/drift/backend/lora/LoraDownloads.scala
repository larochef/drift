package drift.backend.lora

import drift.backend.download.*
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.*
import java.util.concurrent.{ConcurrentHashMap, Executors}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** Getting a LoRA's weights onto the disk (`specs/09-lora-management.md`,
  * `specs/33-lora-sources.md`): the queue, its two threads, and what each file
  * costs to fetch from the site it came from — Civitai, HuggingFace, ModelScope
  * — or to copy from the host.
  *
  * The entity is written before its files arrive, so a LoRA is always the thing
  * that says what it is made of; this is only the fetching, and a file already
  * on disk is simply reported done.
  */
final private[lora] class LoraDownloads(
    storage: StorageService,
    huggingFace: HuggingFaceDownloads,
    modelScope: drift.backend.modelscope.ModelScopeDownloads,
    downloader: Downloader,
    civitaiToken: () => Option[String],
    lorasRoot: Path
) {

  private val logger = Logger[LoraDownloads]

  final private class Entry(@volatile var job: LoraDownloadJob)
  private val jobs = ConcurrentHashMap[String, Entry]()

  private val executor = Executors.newFixedThreadPool(
    2,
    runnable => {
      val thread = Thread(runnable, "drift-lora-download")
      thread.setDaemon(true)
      thread
    }
  )

  def listJobs: List[LoraDownloadJob] =
    jobs.values.asScala.map(_.job).toList.sortBy(j => (j.loraId, j.fileName))

  // ----------------------------------------------------------------- install

  /** Re-queues every installed LoRA file that is not on disk. Called once at
    * startup, so a drift restart does not strand a half-fetched wan pair — the
    * `.part` file makes the retry a resume, not a fresh transfer.
    */
  def resumeInterrupted(): Unit =
    storage.list[Lora]("loras").foreach { lora =>
      lora.files
        .filterNot(file =>
          Files.isRegularFile(lorasRoot.resolve(lora.storagePathOf(file)))
        )
        .foreach { file =>
          logger.info(
            s"Resuming interrupted LoRA download ${lora.id}/${file.fileName}"
          )
          queueDownload(lora, file)
        }
    }

  def queueDownload(lora: Lora, file: LoraFile): Unit = {
    val key = s"${lora.id}/${file.fileName}"
    val target = lorasRoot.resolve(lora.storagePathOf(file))
    if (Files.isRegularFile(target)) {
      jobs.put(
        key,
        Entry(
          LoraDownloadJob(
            lora.id,
            file.fileName,
            DownloadState.Completed,
            downloadedBytes = Files.size(target),
            totalBytes = Some(Files.size(target))
          )
        )
      )
      return
    }
    Option(jobs.get(key)).map(_.job).filter(_.state.isActive) match {
      case Some(_) => ()
      case None    =>
        val entry = Entry(
          LoraDownloadJob(
            lora.id,
            file.fileName,
            DownloadState.Queued,
            totalBytes = file.sizeBytes
          )
        )
        jobs.put(key, entry)
        executor.submit(new Runnable {
          def run(): Unit =
            try {
              entry.job = entry.job.copy(state = DownloadState.Downloading)
              Files.createDirectories(target.getParent)
              val outcome = fetch(
                file,
                target,
                (done, total) =>
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
                logger.warn(s"LoRA download $key blew up", err)
                entry.job = entry.job.copy(
                  state = DownloadState.Failed,
                  error = Some(Option(err.getMessage).getOrElse(err.toString))
                )
            }
        })
    }
  }

  /** One file from its source into the LoRA store. */
  private def fetch(
      file: LoraFile,
      target: Path,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    file.source match {
      case Civitai(_, versionId, fileId, _) =>
        downloader.fetch(
          url =
            s"https://civitai.com/api/download/models/$versionId?fileId=$fileId",
          partFile =
            target.resolveSibling(target.getFileName.toString + ".part"),
          finalFile = target,
          expectedSha256 = file.sha256,
          headers =
            civitaiToken().map(t => "Authorization" -> s"Bearer $t").toMap,
          isCancelled = () => false,
          onProgress = onProgress
        )
      case source: HuggingFace =>
        huggingFace.downloadTo(
          source,
          target,
          file.sha256,
          () => false,
          onProgress
        )
      case source: ModelScope =>
        modelScope.downloadTo(
          source,
          target,
          file.sha256,
          () => false,
          onProgress
        )
      case Local(path) => copyFromDisk(Paths.get(path), target, onProgress)
    }

  /** Through a `.part` file, so a LoRA folder never holds a half-written weight
    * file a launching session could pick up.
    */
  private def copyFromDisk(
      source: Path,
      target: Path,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    if (!Files.isRegularFile(source))
      DownloadOutcome.Failed(s"'$source' is not there to copy any more")
    else {
      val size = Files.size(source)
      val part = target.resolveSibling(target.getFileName.toString + ".part")
      onProgress(0L, Some(size))
      Files.copy(source, part, StandardCopyOption.REPLACE_EXISTING)
      Files.move(part, target, StandardCopyOption.ATOMIC_MOVE)
      onProgress(size, Some(size))
      DownloadOutcome.Completed(target, size)
    }

  // ------------------------------------------------------------------- adopt
}
