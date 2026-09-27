package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object UpscalerService {
  enum Command {
    case Load

    /** A direct-URL install — the curated RealESRGAN list and the free-form URL
      * field both land here.
      */
    case InstallUrl(
        url: String,
        label: Option[String],
        fileName: Option[String],
        sha256: Option[String]
    )

    /** The Civitai browser flow; empty `fileIds` means every weight file of the
      * version.
      */
    case InstallCivitai(
        civitaiModelId: String,
        versionId: String,
        fileIds: List[String]
    )
    case Delete(upscalerId: String)
    case LoadJobs
  }
}

/** The upscaler store and its downloads
  * (`specs/10-generation-time-upscaling.md`). The job list doubles as the
  * on-disk answer: a file already fetched shows as an immediately `Completed`
  * job.
  */
class UpscalerService(statusSocket: StatusSocketService) extends ServiceErrors {
  import UpscalerService.Command

  private val listFn = ApiClient.stream(drift.shared.listUpscalers)
  private val installUrlFn = ApiClient.stream(installUpscaler)
  private val installCivitaiFn = ApiClient.stream(installUpscalerFromCivitai)
  private val deleteFn = ApiClient.stream(deleteUpscaler)
  private val jobsFn = ApiClient.stream(listUpscalerDownloads)

  private val _upscalers = Var(List.empty[Upscaler])
  val upscalers: Signal[List[Upscaler]] = _upscalers.signal

  private val _jobs = Var(List.empty[UpscalerDownloadJob])
  val jobs: Signal[List[UpscalerDownloadJob]] = _jobs.signal

  private val cmdBus = new EventBus[Command]

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private def upsert(installed: List[Upscaler]): Unit = {
    val ids = installed.map(_.id).toSet
    _upscalers.update(all =>
      (all.filterNot(u => ids.contains(u.id)) ++ installed).sortBy(_.label)
    )
  }

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listFn(()).recoverToTry)
      --> Observer[Try[List[Upscaler]]] {
        case Success(listing) =>
          clearError()
          _upscalers.set(listing.sortBy(_.label))
        case Failure(err) => reportFailure("Listing upscalers", err)
      },
    cmdBus.events
      .collect { case Command.InstallUrl(url, label, fileName, sha256) =>
        InstallUpscalerRequest(url, label, fileName, sha256)
      }
      .flatMapMerge(request => installUrlFn(request).recoverToTry)
      --> Observer[Try[InstallUpscalerResponse]] {
        case Success(response) =>
          response.error match {
            case Some(reason) =>
              reportFailure("Installing the upscaler", reason)
            case None =>
              clearError()
              upsert(response.upscalers)
              push(Command.LoadJobs)
          }
        case Failure(err) => reportFailure("Installing the upscaler", err)
      },
    cmdBus.events
      .collect { case Command.InstallCivitai(modelId, versionId, fileIds) =>
        InstallUpscalerFromCivitaiRequest(modelId, versionId, fileIds)
      }
      .flatMapMerge(request => installCivitaiFn(request).recoverToTry)
      --> Observer[Try[InstallUpscalerResponse]] {
        case Success(response) =>
          response.error match {
            case Some(reason) =>
              reportFailure("Installing the upscaler", reason)
            case None =>
              clearError()
              upsert(response.upscalers)
              push(Command.LoadJobs)
          }
        case Failure(err) => reportFailure("Installing the upscaler", err)
      },
    cmdBus.events
      .collect { case Command.Delete(upscalerId) => upscalerId }
      .flatMapMerge(id => deleteFn(id).map(ok => (id, ok)).recoverToTry)
      --> Observer[Try[(String, Boolean)]] {
        case Success((id, _)) =>
          clearError()
          _upscalers.update(_.filterNot(_.id == id))
          push(Command.LoadJobs)
        case Failure(err) => reportFailure("Deleting the upscaler", err)
      },
    cmdBus.events
      .collect { case Command.LoadJobs => () }
      .flatMapSwitch(_ => jobsFn(()).recoverToTry)
      --> Observer[Try[List[UpscalerDownloadJob]]] {
        case Success(listing) => _jobs.set(listing)
        // Job polling is decoration; a failed poll reports nothing.
        case Failure(_) => ()
      },
    // The socket pushes the job list whenever it changes, so progress and
    // completion arrive on their own.
    statusSocket.upscalerDownloads
      --> Observer[List[UpscalerDownloadJob]](_jobs.set)
  )
}
