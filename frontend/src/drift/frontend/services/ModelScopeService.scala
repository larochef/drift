package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object ModelScopeService {
  enum Command {

    /** A new search, from its first page. */
    case Search(
        query: String,
        sort: String,
        baseModels: List[String],
        lorasOnly: Boolean
    )

    /** The next page of the current search, appended. */
    case LoadMore

    /** The files and examples of `owner/name`. */
    case LoadDetail(repo: String)

    /** The README of `owner/name`. */
    case LoadCard(repo: String)
  }
}

/** The ModelScope browser's state (`specs/37-modelscope.md`): its own, apart
  * from the HuggingFace service, since the two sites page, sort and answer
  * differently.
  */
class ModelScopeService
    extends SearchService[ModelScopeModelInfo, ModelScopeService.Command]
    with RepositoryCards[ModelScopeModelCard]
    with RepositoryDetails[ModelScopeModelDetail] {
  import ModelScopeService.Command

  private val searchFn = ApiClient.streamWithFailureReason(searchModelScope)
  private val detailFn =
    ApiClient.streamWithFailureReason(getModelScopeModelDetail)
  private val cardFn = ApiClient.streamWithFailureReason(getModelScopeModelCard)

  /** ModelScope answers how many a search has in all; a page short of that is a
    * page before the last.
    */
  private val _total = Var(0)

  val hasMore: Signal[Boolean] =
    _results.signal
      .combineWith(_total.signal)
      .map((shown, total) => shown.size < total)

  protected def idOf(result: ModelScopeModelInfo): String = result.id
  protected def resetPaging(): Unit = _total.set(0)

  def loadMore(): Unit = push(Command.LoadMore)
  protected def loadCard(repo: String): Unit = push(Command.LoadCard(repo))

  protected def loadDetail(repo: String): Unit = push(Command.LoadDetail(repo))

  // The search in effect and the page it reached, so `LoadMore` continues it.
  private val current = Var(Option.empty[Command.Search])
  private val page = Var(1)

  private def split(repo: String): (String, String) =
    repo.split("/", 2) match {
      case Array(owner, name) => (owner, name)
      case _                  => (repo, "")
    }

  // Every request recovers into a Try: an unrecovered failure ends the stream
  // for good and strands `_searching` at true.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case search: Command.Search => search }
      .flatMapSwitch { search =>
        current.set(Some(search))
        _searching.set(true)
        searchFn(
          (
            search.query,
            1,
            Some(search.sort),
            search.baseModels,
            search.lorasOnly
          )
        ).recoverToTry
      } --> Observer[Try[ModelScopeSearchPage]] {
      case Success(found) =>
        clearError()
        _results.set(found.models)
        _total.set(found.total)
        page.set(1)
        _searching.set(false)
      case Failure(err) =>
        _results.set(Nil)
        _total.set(0)
        _searching.set(false)
        reportFailure("Searching ModelScope", err)
    },
    cmdBus.events
      .collect { case Command.LoadMore => () }
      .flatMapSwitch { _ =>
        current.now() match {
          case None         => EventStream.empty
          case Some(search) =>
            val next = page.now() + 1
            _searching.set(true)
            searchFn(
              (
                search.query,
                next,
                Some(search.sort),
                search.baseModels,
                search.lorasOnly
              )
            ).recoverToTry.map(result => (next, result))
        }
      } --> Observer[(Int, Try[ModelScopeSearchPage])] {
      case (next, Success(found)) =>
        clearError()
        appendPage(found.models)
        _total.set(found.total)
        // Advanced only on success, so a failed page is asked for again.
        page.set(next)
        _searching.set(false)
      case (_, Failure(err)) =>
        _searching.set(false)
        reportFailure("Loading more results", err)
    },
    cmdBus.events
      .collect { case Command.LoadDetail(repo) => repo }
      .flatMapSwitch { repo =>
        detailRequested(repo)
        detailFn(split(repo)).recoverToTry
      } --> Observer[Try[Option[ModelScopeModelDetail]]] {
      case Success(found) =>
        clearError()
        detailArrived(found)
      case Failure(err) =>
        detailFailed(reasonOf(err))
        reportFailure("Loading the repository", err)
    },
    cmdBus.events
      .collect { case Command.LoadCard(repo) => repo }
      .flatMapSwitch { repo =>
        cardRequested(repo)
        cardFn(split(repo)).recoverToTry
      } --> Observer[Try[ModelScopeModelCard]] {
      case Success(found) => cardArrived(found)
      case Failure(err)   => cardFailed(reasonOf(err))
    }
  )
}
