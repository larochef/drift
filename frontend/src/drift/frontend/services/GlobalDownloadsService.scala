package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

/** What a downloads-panel row stops when cancelled — each kind has its own
  * endpoint.
  */
enum DownloadTarget {
  case Model(modelId: String)
  case Lora(loraId: String, fileName: String)
  case Upscaler(upscalerId: String)
  case Runtime(runtimeId: String)
  case Conversion(conversionId: String)
}

/** One row of the global downloads panel, whatever kind of transfer it is. */
case class ActiveDownload(
    /** What cancelling the row stops. */
    target: DownloadTarget,
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
    unit: Option[String] = None,
    /** A cancel was sent and the transfer has not stopped yet: the row shows it
      * instead of offering the cancel again.
      */
    cancelling: Boolean = false
)

/** Feeds the downloads panel in the shell: every current and pending transfer —
  * model weights, LoRA files, runtime installs, conversions — wherever it was
  * started.
  *
  * Deliberately its own service, mounted once by `AppShell`: the per-page
  * download services listen only while their page is open, and mounting their
  * effects a second time would run every command twice. This one lives entirely
  * off the status socket instead, because a global view has to *discover* work
  * started elsewhere — the on-connect snapshot and the pushes are exactly that.
  */
class GlobalDownloadsService(statusSocket: StatusSocketService) {

  private val modelJobs = Var(List.empty[DownloadJob])
  private val loraJobs = Var(List.empty[LoraDownloadJob])
  private val upscalerJobs = Var(List.empty[UpscalerDownloadJob])
  private val runtimeJobs = Var(List.empty[RuntimeInstallJob])
  private val conversionJobs = Var(List.empty[ConversionJob])

  private val cancelling = Var(Set.empty[DownloadTarget])

  /** Everything currently queued or transferring, steadiest first. */
  val activeDownloads: Signal[List[ActiveDownload]] = listed
    .combineWith(cancelling.signal)
    .map((items, cancelling) =>
      items.map(item =>
        if (cancelling.contains(item.target)) item.copy(cancelling = true)
        else item
      )
    )

  private lazy val listed: Signal[List[ActiveDownload]] = Signal
    .combine(
      modelJobs.signal,
      loraJobs.signal,
      upscalerJobs.signal,
      runtimeJobs.signal,
      conversionJobs.signal
    )
    .map { (models, loras, upscalers, runtimes, conversions) =>
      val fromModels =
        transfers(models, "model")(
          job => DownloadTarget.Model(job.modelId),
          _.modelId,
          _ => ""
        )
      // A pair's two files are told apart by their names.
      val fromLoras =
        transfers(loras, "lora")(
          job => DownloadTarget.Lora(job.loraId, job.fileName),
          _.loraId,
          _.fileName
        )
      val fromUpscalers =
        transfers(upscalers, "upscaler")(
          job => DownloadTarget.Upscaler(job.upscalerId),
          _.upscalerId,
          _ => ""
        )
      val fromRuntimes = runtimes.filter(_.state.isActive).map { job =>
        val stateText = job.state match {
          case RuntimeInstallState.Queued     => "queued"
          case RuntimeInstallState.Unpacking  => "unpacking…"
          case RuntimeInstallState.Validating => "validating…"
          case _                              => job.step
        }
        ActiveDownload(
          target = DownloadTarget.Runtime(job.runtimeId),
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
          target = DownloadTarget.Conversion(job.id),
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

  /** The rows of the transfers under way of one kind: each job's key says what
    * a cancel stops and what the row is called, `transferring` what it shows
    * beside its bar while it is not waiting its turn.
    */
  private def transfers[J <: TransferJob](jobs: List[J], kind: String)(
      target: J => DownloadTarget,
      label: J => String,
      transferring: J => String
  ): List[ActiveDownload] =
    jobs.filter(_.state.isActive).map { job =>
      val queued = job.state == DownloadState.Queued
      ActiveDownload(
        target = target(job),
        kind = kind,
        label = label(job),
        detail = if (queued) "queued" else transferring(job),
        downloadedBytes = job.downloadedBytes,
        totalBytes = job.totalBytes,
        queued = queued
      )
    }

  private val cancelModelFn = ApiClient.stream(drift.shared.cancelDownload)
  private val cancelLoraFn = ApiClient.stream(drift.shared.cancelLoraDownload)
  private val cancelUpscalerFn =
    ApiClient.stream(drift.shared.cancelUpscalerDownload)
  private val cancelRuntimeFn =
    ApiClient.stream(drift.shared.cancelRuntimeInstall)
  private val cancelConversionFn =
    ApiClient.stream(drift.shared.cancelConversion)

  private val cancelBus = new EventBus[DownloadTarget]

  /** Stops a transfer; the socket brings the job's new state. */
  def cancel(target: DownloadTarget): Unit = {
    cancelling.update(_ + target)
    cancelBus.writer.onNext(target)
  }

  val effects: Modifier[HtmlElement] = Seq(
    cancelBus.events.flatMapMerge { target =>
      (target match {
        case DownloadTarget.Model(id)      => cancelModelFn(id).mapTo(())
        case DownloadTarget.Lora(id, file) =>
          cancelLoraFn((id, file)).mapTo(())
        case DownloadTarget.Upscaler(id)   => cancelUpscalerFn(id).mapTo(())
        case DownloadTarget.Runtime(id)    => cancelRuntimeFn(id).mapTo(())
        case DownloadTarget.Conversion(id) => cancelConversionFn(id).mapTo(())
      }).recoverToTry
    } --> Observer[Try[Unit]] {
      case Failure(err) =>
        org.scalajs.dom.console
          .warn(s"Cancelling a download: ${err.getMessage}")
      case Success(_) => ()
    },
    // A transfer that has stopped leaves the list; so does its cancelling mark.
    listed --> Observer[List[ActiveDownload]] { items =>
      val active = items.map(_.target).toSet
      cancelling.update(_.intersect(active))
    },
    statusSocket.downloads --> modelJobs.writer,
    statusSocket.loraDownloads --> loraJobs.writer,
    statusSocket.upscalerDownloads --> upscalerJobs.writer,
    statusSocket.runtimeInstalls --> runtimeJobs.writer,
    statusSocket.conversions --> conversionJobs.writer
  )
}
