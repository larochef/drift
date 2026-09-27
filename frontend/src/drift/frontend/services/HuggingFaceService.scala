package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object HuggingFaceService {
  val PageSize = 25

  enum Command {
    case Search(
        query: String,
        sort: Option[String] = None,
        direction: Option[Int] = None,
        /** Only adapters of this repository (`specs/33-lora-sources.md`). */
        baseModel: Option[String] = None
    )

    /** Fetch the next page of the current search and append it. */
    case LoadMore

    /** Load the file list of `owner/repo`. */
    case LoadFiles(modelId: String)

    /** Load the model card of `owner/repo` (`specs/24`). */
    case LoadCard(modelId: String)
  }
}

class HuggingFaceService
    extends SearchService[HuggingFaceModelInfo, HuggingFaceService.Command]
    with RepositoryCards[HuggingFaceModelCard]
    with RepositoryDetails[HuggingFaceModelDetail] {
  import HuggingFaceService.{Command, PageSize}

  private val searchFn = ApiClient.stream(drift.shared.searchHuggingFaceModels)
  private val detailFn =
    ApiClient.stream(drift.shared.getHuggingFaceModelDetail)
  private val cardFn =
    ApiClient.streamWithFailureReason(drift.shared.getHuggingFaceModelCard)

  private val _hasMore = Var(false)
  val hasMore: Signal[Boolean] = _hasMore.signal

  /** The opened repository: its files and what its model makes
    * (`specs/36-huggingface-examples.md`). A signal like ModelScope's, so both
    * repository browsers read their detail the same way.
    */

  protected def idOf(result: HuggingFaceModelInfo): String = result.id
  protected def resetPaging(): Unit = _hasMore.set(false)

  def loadMore(): Unit = push(Command.LoadMore)
  protected def loadCard(repo: String): Unit = push(Command.LoadCard(repo))

  protected def loadDetail(repo: String): Unit = push(Command.LoadFiles(repo))

  // The parameters of the search in effect, so `LoadMore` can continue it.
  private val query = Var("")
  private val sort = Var(Option.empty[String])
  private val direction = Var(Option.empty[Int])
  private val baseModel = Var(Option.empty[String])
  private val offset = Var(0)

  // Every request recovers into a Try: an unrecovered failure ends the stream
  // for good and strands `_searching` at true, which disables the search UI
  // until the page or modal is remounted.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Search(q, s, d, b) => (q, s, d, b) }
      .flatMapSwitch { (q, s, d, b) =>
        query.set(q)
        sort.set(s)
        direction.set(d)
        baseModel.set(b)
        _searching.set(true)
        searchFn((q, Some(PageSize), Some(0), s, d, b)).recoverToTry
      } --> Observer[Try[List[HuggingFaceModelInfo]]] {
      case Success(r) =>
        clearError()
        _results.set(r)
        offset.set(PageSize)
        _hasMore.set(r.size >= PageSize)
        _searching.set(false)
      case Failure(err) =>
        _results.set(Nil)
        _hasMore.set(false)
        _searching.set(false)
        reportFailure("Searching HuggingFace", err)
    },
    cmdBus.events
      .collect { case Command.LoadMore => () }
      .flatMapSwitch { _ =>
        val from = offset.now()
        _searching.set(true)
        searchFn(
          (
            query.now(),
            Some(PageSize),
            Some(from),
            sort.now(),
            direction.now(),
            baseModel.now()
          )
        ).recoverToTry.map(t => (from, t))
      } --> Observer[(Int, Try[List[HuggingFaceModelInfo]])] {
      case (from, Success(r)) =>
        clearError()
        appendPage(r)
        // Advance only on success, so a failed page is retried rather than skipped.
        offset.set(from + PageSize)
        _hasMore.set(r.size >= PageSize)
        _searching.set(false)
      case (_, Failure(err)) =>
        _searching.set(false)
        reportFailure("Loading more results", err)
    },
    cmdBus.events
      .collect { case Command.LoadFiles(id) => id }
      .flatMapSwitch { id =>
        detailRequested(id)
        id.split("/", 2) match {
          case Array(owner, repo) => detailFn((owner, repo)).recoverToTry
          case _                  =>
            EventStream.fromValue(
              Success(Option.empty[HuggingFaceModelDetail])
            )
        }
      } --> Observer[Try[Option[HuggingFaceModelDetail]]] {
      case Success(found) =>
        clearError()
        detailArrived(found)
      case Failure(err) =>
        detailFailed(reasonOf(err))
        reportFailure("Loading the model files", err)
    },
    cmdBus.events
      .collect { case Command.LoadCard(id) => id }
      .flatMapSwitch { id =>
        cardRequested(id)
        id.split("/", 2) match {
          case Array(owner, repo) => cardFn((owner, repo)).recoverToTry
          case _                  =>
            EventStream.fromValue(Success(HuggingFaceModelCard()))
        }
      } --> Observer[Try[HuggingFaceModelCard]] {
      case Success(found) => cardArrived(found)
      case Failure(err)   => cardFailed(reasonOf(err))
    }
  )
}
