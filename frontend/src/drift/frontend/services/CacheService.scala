package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object CacheService {
  enum Command {
    case Load
    case LoadFiles
    case DeleteFile(path: String)
  }
}

/** Which registered models have their weights on disk.
  *
  * Read-only for now: [`05b`] adds downloading. Until then this is what lets
  * the UI say what is present, and what stops a run configuration launching
  * without its weights.
  */
class CacheService extends ServiceErrors {
  import CacheService.Command

  private val statusFn = ApiClient.stream(drift.shared.cacheStatus)
  private val filesFn = ApiClient.stream(drift.shared.listCachedFiles)
  private val deleteFileFn = ApiClient.stream(drift.shared.deleteCachedFile)

  private val _statuses = Var(List.empty[ModelCacheStatus])
  private val _files = Var(List.empty[CachedFileEntry])

  /** What the cache holds on disk, including orphans. */
  val files: Signal[List[CachedFileEntry]] = _files.signal

  /** Keyed by model id, which is how `CommandLine.blockers` wants it. */
  val statuses: Signal[Map[String, ModelCacheStatus]] =
    _statuses.signal.map(_.map(s => s.modelId -> s).toMap)

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private val cmdBus = new EventBus[Command]

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => statusFn(()).recoverToTry)
      --> Observer[Try[List[ModelCacheStatus]]] {
        case Success(rows) =>
          clearError()
          _statuses.set(rows)
        case Failure(err) => reportFailure("Reading the model cache", err)
      },
    cmdBus.events
      .collect { case Command.LoadFiles => () }
      .flatMapSwitch(_ => filesFn(()).recoverToTry)
      --> Observer[Try[List[CachedFileEntry]]] {
        case Success(rows) =>
          clearError()
          _files.set(rows)
        case Failure(err) => reportFailure("Listing cached files", err)
      },
    cmdBus.events
      .collect { case Command.DeleteFile(path) => path }
      .flatMapMerge(path =>
        deleteFileFn(path).map(deleted => (path, deleted)).recoverToTry
      ) --> Observer[Try[(String, Boolean)]] {
      case Success((_, true)) =>
        clearError()
        // Both views change: the file is gone and models may have lost it.
        push(Command.LoadFiles)
        push(Command.Load)
      case Success((path, false)) =>
        reportFailure("Deleting the cached file", s"'$path' was not deleted.")
      case Failure(err) => reportFailure("Deleting the cached file", err)
    }
  )
}
