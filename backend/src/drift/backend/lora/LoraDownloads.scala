package drift.backend.lora

import drift.backend.Background
import drift.backend.civitai.CivitaiClient
import drift.backend.download.*
import drift.backend.huggingface.HuggingFaceDownloads
import drift.backend.storage.{Collections, StorageService}
import drift.shared.*

import java.nio.file.*

import com.typesafe.scalalogging.Logger
import ox.discard

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
    civitaiClient: CivitaiClient,
    lorasRoot: Path,
    background: Background
) {

  private val logger = Logger[LoraDownloads]

  /** One transfer per file of a LoRA, by the LoRA's id and the file's name
    * (`DownloadQueue`).
    */
  private val downloads =
    DownloadQueue[(String, String)](background, "drift-lora-download")

  private def jobOf(key: (String, String), transfer: Transfer) =
    LoraDownloadJob(
      key._1,
      key._2,
      transfer.state,
      downloadedBytes = transfer.downloadedBytes,
      totalBytes = transfer.totalBytes,
      error = transfer.error
    )

  def listJobs: List[LoraDownloadJob] =
    downloads.all.toList.map(jobOf).sortBy(j => (j.loraId, j.fileName))

  /** Stops one file's transfer from the downloads panel: a queued one at once,
    * a running one at its next read, its `.part` kept for a resume. The LoRA
    * stays installed without that file, so a restart resumes it
    * (`resumeInterrupted`) — deleting the LoRA is what drops it for good.
    */
  def cancel(loraId: String, fileName: String): Option[LoraDownloadJob] =
    downloads.cancel((loraId, fileName)).map(jobOf((loraId, fileName), _))

  // ----------------------------------------------------------------- install

  /** Re-queues every installed LoRA file that is not on disk. Called once at
    * startup, so a drift restart does not strand a half-fetched wan pair — the
    * `.part` file makes the retry a resume, not a fresh transfer.
    */
  def resumeInterrupted(): Unit =
    storage.list(Collections.loras).foreach { lora =>
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
    val key = (lora.id, file.fileName)
    val target = lorasRoot.resolve(lora.storagePathOf(file))
    if (Files.isRegularFile(target))
      downloads
        .settled(key, DownloadState.Completed, bytes = Some(Files.size(target)))
        .discard
    else
      downloads
        .queue(key, file.sizeBytes) { (isCancelled, onProgress) =>
          Files.createDirectories(target.getParent)
          fetch(file, target, isCancelled, onProgress)
        }
        .discard
  }

  /** One file from its source into the LoRA store. */
  private def fetch(
      file: LoraFile,
      target: Path,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    file.source match {
      case Civitai(_, versionId, fileId, _) =>
        downloader.fetch(
          url = civitaiClient.downloadAddress(versionId, fileId),
          partFile =
            target.resolveSibling(target.getFileName.toString + ".part"),
          finalFile = target,
          expectedSha256 = file.sha256,
          headers = civitaiClient.downloadHeaders,
          isCancelled = isCancelled,
          onProgress = onProgress
        )
      case source: HuggingFace =>
        huggingFace.downloadTo(
          source,
          target,
          file.sha256,
          isCancelled,
          onProgress
        )
      case source: ModelScope =>
        modelScope.downloadTo(
          source,
          target,
          file.sha256,
          isCancelled,
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
