package drift.frontend.services

import drift.shared.*

import com.raquo.laminar.api.L.*

/** One row of the global downloads panel, whatever kind of transfer it is. */
case class ActiveDownload(
    /** "model", "lora", "upscaler", "runtime" or "conversion" — shown as a
      * small kind marker.
      */
    kind: String,
    label: String,
    /** What is happening beyond plain downloading: a runtime's step
      * ("unpacking…"), a queued wait, a LoRA's file id for pairs.
      */
    detail: String,
    downloadedBytes: Long,
    /** Absent means the bar is indeterminate (queued, unpacking…). */
    totalBytes: Option[Long],
    /** Waiting for a transfer slot. The panel counts these in one summary line
      * instead of listing them, so a bulk install cannot grow it beyond the
      * viewport.
      */
    queued: Boolean = false,
    /** What the two counts count when they are not bytes — "tensors" for a
      * conversion, whose bar is sd-cli's tensor count.
      */
    unit: Option[String] = None
)

/** Feeds the downloads panel in the shell: every current and pending transfer —
  * model weights, LoRA files, runtime installs, conversions — wherever it was
  * started.
  *
  * Deliberately its own service, mounted once by `AppShell`: the per-page
  * download services listen only while their page is open, and mounting their
  * effects a second time would run every command twice (the bugs/15 class).
  * This one lives entirely off the status socket instead, because a global view
  * has to *discover* work started elsewhere — the on-connect snapshot and the
  * pushes are exactly that.
  */
class GlobalDownloadsService(statusSocket: StatusSocketService) {

  private val modelJobs = Var(List.empty[DownloadJob])
  private val loraJobs = Var(List.empty[LoraDownloadJob])
  private val upscalerJobs = Var(List.empty[UpscalerDownloadJob])
  private val runtimeJobs = Var(List.empty[RuntimeInstallJob])
  private val conversionJobs = Var(List.empty[ConversionJob])

  /** Everything currently queued or transferring, steadiest first. */
  val activeDownloads: Signal[List[ActiveDownload]] = Signal
    .combine(
      modelJobs.signal,
      loraJobs.signal,
      upscalerJobs.signal,
      runtimeJobs.signal,
      conversionJobs.signal
    )
    .map { (models, loras, upscalers, runtimes, conversions) =>
      val fromModels = models.filter(_.state.isActive).map { job =>
        ActiveDownload(
          kind = "model",
          label = job.modelId,
          detail = if (job.state == DownloadState.Queued) "queued" else "",
          downloadedBytes = job.downloadedBytes,
          totalBytes = job.totalBytes,
          queued = job.state == DownloadState.Queued
        )
      }
      val fromLoras = loras.filter(_.state.isActive).map { job =>
        ActiveDownload(
          kind = "lora",
          label = job.loraId,
          detail =
            if (job.state == DownloadState.Queued) "queued"
            else job.fileName,
          downloadedBytes = job.downloadedBytes,
          totalBytes = job.totalBytes,
          queued = job.state == DownloadState.Queued
        )
      }
      val fromUpscalers = upscalers.filter(_.state.isActive).map { job =>
        ActiveDownload(
          kind = "upscaler",
          label = job.upscalerId,
          detail = if (job.state == DownloadState.Queued) "queued" else "",
          downloadedBytes = job.downloadedBytes,
          totalBytes = job.totalBytes,
          queued = job.state == DownloadState.Queued
        )
      }
      val fromRuntimes = runtimes.filter(_.state.isActive).map { job =>
        val stateText = job.state match {
          case RuntimeInstallState.Queued     => "queued"
          case RuntimeInstallState.Unpacking  => "unpacking…"
          case RuntimeInstallState.Validating => "validating…"
          case _                              => job.step
        }
        ActiveDownload(
          kind = "runtime",
          label = job.runtimeId,
          detail = stateText,
          downloadedBytes = job.downloadedBytes,
          totalBytes = job.totalBytes
            // Bytes only mean a bar while actually downloading.
            .filter(_ => job.state == RuntimeInstallState.Downloading),
          queued = job.state == RuntimeInstallState.Queued
        )
      }
      // A conversion's bar counts tensors written, as sd-cli reports them.
      val fromConversions = conversions.filter(_.state.isActive).map { job =>
        ActiveDownload(
          kind = "conversion",
          label = job.outputPath.split('/').last,
          detail = job.detail,
          downloadedBytes = job.progress.map(_.completed.toLong).getOrElse(0L),
          totalBytes = job.progress
            .map(_.total.toLong)
            .filter(_ =>
              job.state == ConversionState.Converting ||
                job.state == ConversionState.Dequantizing
            ),
          queued = job.state == ConversionState.Queued,
          unit = Some("tensors")
        )
      }
      fromModels ++ fromLoras ++ fromUpscalers ++ fromRuntimes ++ fromConversions
    }

  val effects: Modifier[HtmlElement] = Seq(
    statusSocket.downloads --> modelJobs.writer,
    statusSocket.loraDownloads --> loraJobs.writer,
    statusSocket.upscalerDownloads --> upscalerJobs.writer,
    statusSocket.runtimeInstalls --> runtimeJobs.writer,
    statusSocket.conversions --> conversionJobs.writer
  )
}
