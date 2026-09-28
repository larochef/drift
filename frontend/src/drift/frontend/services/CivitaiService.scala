package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object CivitaiService {
  val PageSize = 25

  enum Command {
    case Search(
        query: String,
        sort: Option[String] = None,
        modelType: Option[String] = None,
        baseModels: List[String] = List.empty,
        includeNsfw: Boolean = false
    )

    /** Fetch the next page of the current search and append it. */
    case LoadMore

    /** Load the version/file listing of a Civitai model. */
    case LoadFiles(modelId: Int)

    /** The images people posted with a model (`specs/24`), replacing the
      * gallery: `versionId` narrows them to one version, `sort` is one of
      * Civitai's image orders.
      */
    case LoadImages(
        modelId: Int,
        versionId: Option[Int],
        sort: String,
        includeNsfw: Boolean
    )

    /** The next page of the current gallery, appended. */
    case LoadMoreImages
  }
  enum Event {
    case SearchResults(results: List[CivitaiModelListInfo])
    case FilesLoaded(versions: List[CivitaiModelVersion])
  }

  /** Why the gallery's last page did not load, and the command that asks for it
    * again — the first page or the next one (`specs/24`).
    */
  case class ImageLoadFailure(reason: String, retry: Command)
}

class CivitaiService
    extends SearchService[CivitaiModelListInfo, CivitaiService.Command] {
  import CivitaiService.{Command, Event, ImageLoadFailure, PageSize}

  private val searchFn = ApiClient.stream(drift.shared.searchCivitaiModels)
  private val detailFn = ApiClient.stream(drift.shared.getCivitaiModelDetail)
  private val cachedFn = ApiClient.stream(drift.shared.listCivitaiCachedFileIds)
  private val imagesFn =
    ApiClient.streamWithFailureReason(drift.shared.listCivitaiImages)

  /** The file ids of the currently opened model that are already in the cache —
    * what the file list's "downloaded" markers show. Refreshed alongside
    * `LoadFiles`.
    */
  private val _cachedFileIds = Var(Set.empty[String])
  val cachedFileIds: Signal[Set[String]] = _cachedFileIds.signal

  /** The opened model's detail: its cleaned description, tags and version notes
    * (`specs/24`). `None` while it loads, or when the load failed.
    */
  private val _detail = Var(Option.empty[CivitaiModelDetail])
  val detail: Signal[Option[CivitaiModelDetail]] = _detail.signal

  private val _hasMore = Var(false)
  val hasMore: Signal[Boolean] = _hasMore.signal

  protected def idOf(result: CivitaiModelListInfo): String = result.id.toString
  protected def resetPaging(): Unit = _hasMore.set(false)

  def loadMore(): Unit = push(Command.LoadMore)

  // The parameters of the search in effect, so `LoadMore` can continue it.
  // Civitai pages with an opaque cursor, not an offset — `nextCursor` from one
  // page is the only way to the next.
  private val query = Var(Option.empty[String])
  private val sort = Var(Option.empty[String])
  private val modelType = Var(Option.empty[String])
  private val baseModels = Var(List.empty[String])
  private val includeNsfw = Var(false)
  private val nextCursor = Var(Option.empty[String])

  /** The opened model's gallery (`specs/24`) and the query it answers — kept
    * across tab switches, so returning to the Images tab reloads nothing — with
    * the cursor to its next page.
    */
  private val _images = Var(List.empty[CivitaiPostedImage])
  val images: Signal[List[CivitaiPostedImage]] = _images.signal
  private val _loadingImages = Var(false)
  val loadingImages: Signal[Boolean] = _loadingImages.signal
  private val _hasMoreImages = Var(false)
  val hasMoreImages: Signal[Boolean] = _hasMoreImages.signal
  private val imageQuery = Var(Option.empty[Command.LoadImages])
  private val imageCursor = Var(Option.empty[String])

  /** The gallery's failed load, shown in the gallery beside a Retry rather than
    * in the banner: Civitai's usual failure is an overload, and asking again is
    * the whole remedy.
    */
  private val _imageFailure = Var(Option.empty[ImageLoadFailure])
  val imageFailure: Signal[Option[String]] =
    _imageFailure.signal.map(_.map(_.reason))

  def currentImageQuery: Option[Command.LoadImages] = imageQuery.now()

  /** Asks again for exactly what failed. */
  def retryImages(): Unit =
    _imageFailure.now().foreach(failure => push(failure.retry))

  private def resetImages(): Unit = {
    _images.set(Nil)
    _loadingImages.set(false)
    _hasMoreImages.set(false)
    _imageFailure.set(None)
    imageQuery.set(None)
    imageCursor.set(None)
  }

  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  // See HuggingFaceService.effects for why every request recovers into a Try.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Search(q, sortValue, typeValue, bases, nsfw) =>
        (q, sortValue, typeValue, bases, nsfw)
      }
      .flatMapSwitch { (q, sortValue, typeValue, bases, nsfw) =>
        val queryValue = if (q.isEmpty) None else Some(q)
        query.set(queryValue)
        sort.set(sortValue)
        modelType.set(typeValue)
        baseModels.set(bases)
        includeNsfw.set(nsfw)
        _searching.set(true)
        searchFn(
          (
            queryValue,
            Some(PageSize),
            None,
            sortValue,
            typeValue,
            bases,
            Some(nsfw)
          )
        ).recoverToTry
      } --> Observer[Try[CivitaiSearchResult]] {
      case Success(result) =>
        clearError()
        _results.set(result.items)
        nextCursor.set(result.nextCursor)
        _hasMore.set(result.nextCursor.nonEmpty)
        _searching.set(false)
        evtBus.writer.onNext(Event.SearchResults(result.items))
      case Failure(err) =>
        _results.set(Nil)
        _hasMore.set(false)
        _searching.set(false)
        reportFailure("Searching Civitai", err)
    },
    cmdBus.events
      .collect { case Command.LoadMore => () }
      .flatMapSwitch { _ =>
        _searching.set(true)
        searchFn(
          (
            query.now(),
            Some(PageSize),
            nextCursor.now(),
            sort.now(),
            modelType.now(),
            baseModels.now(),
            Some(includeNsfw.now())
          )
        ).recoverToTry
      } --> Observer[Try[CivitaiSearchResult]] {
      case Success(result) =>
        clearError()
        appendPage(result.items)
        // Advance only on success, so a failed page is retried rather than
        // skipped.
        nextCursor.set(result.nextCursor)
        _hasMore.set(result.nextCursor.nonEmpty)
        _searching.set(false)
      case Failure(err) =>
        _searching.set(false)
        reportFailure("Loading more results", err)
    },
    cmdBus.events
      .collect { case Command.LoadFiles(id) => id }
      .flatMapSwitch { id =>
        _searching.set(true)
        _detail.set(None)
        // A new model: the previous one's gallery must not show under it.
        resetImages()
        detailFn(id).recoverToTry
      } --> Observer[Try[Option[CivitaiModelDetail]]] {
      case Success(detail) =>
        clearError()
        _searching.set(false)
        _detail.set(detail)
        val versions = detail.map(_.modelVersions).getOrElse(Nil)
        evtBus.writer.onNext(Event.FilesLoaded(versions))
      case Failure(err) =>
        _searching.set(false)
        evtBus.writer.onNext(Event.FilesLoaded(Nil))
        reportFailure("Loading the model files", err)
    },
    cmdBus.events
      .collect { case Command.LoadFiles(id) => id }
      .flatMapSwitch { id =>
        _cachedFileIds.set(Set.empty)
        cachedFn(id.toString).recoverToTry
      } --> Observer[Try[List[String]]] {
      // A failed lookup only costs the markers, so it reports no error.
      case Success(ids) => _cachedFileIds.set(ids.toSet)
      case Failure(_)   => ()
    },
    // Each answer carries the query it was for: a page that arrives after the
    // gallery moved on (another model, version or order) is dropped, whichever
    // of the two streams it came from.
    cmdBus.events
      .collect { case load: Command.LoadImages => load }
      .flatMapSwitch { load =>
        resetImages()
        imageQuery.set(Some(load))
        _loadingImages.set(true)
        imagesFn(
          (
            load.modelId,
            load.versionId,
            Some(load.sort),
            None,
            Some(load.includeNsfw)
          )
        ).recoverToTry.map(result => (load, result))
      } --> Observer[(Command.LoadImages, Try[CivitaiImagePage])] {
      case (load, Success(page)) if imageQuery.now().contains(load) =>
        clearError()
        _images.set(page.items)
        imageCursor.set(page.nextCursor)
        _hasMoreImages.set(page.nextCursor.nonEmpty)
        _loadingImages.set(false)
      case (load, Failure(err)) if imageQuery.now().contains(load) =>
        _loadingImages.set(false)
        _imageFailure.set(Some(ImageLoadFailure(reasonOf(err), load)))
      case _ => ()
    },
    cmdBus.events
      .collect { case Command.LoadMoreImages => imageQuery.now() }
      .collect { case Some(load) => load }
      .flatMapSwitch { load =>
        _imageFailure.set(None)
        _loadingImages.set(true)
        imagesFn(
          (
            load.modelId,
            load.versionId,
            Some(load.sort),
            imageCursor.now(),
            Some(load.includeNsfw)
          )
        ).recoverToTry.map(result => (load, result))
      } --> Observer[(Command.LoadImages, Try[CivitaiImagePage])] {
      case (load, Success(page)) if imageQuery.now().contains(load) =>
        clearError()
        // Dedupe: a post can repeat across a page boundary.
        _images.update { current =>
          val seen = current.map(_.id).toSet
          current ++ page.items.filterNot(image => seen.contains(image.id))
        }
        imageCursor.set(page.nextCursor)
        _hasMoreImages.set(page.nextCursor.nonEmpty)
        _loadingImages.set(false)
      case (load, Failure(err)) if imageQuery.now().contains(load) =>
        // The cursor only advances on success, so the retry asks for this
        // same page.
        _loadingImages.set(false)
        _imageFailure.set(
          Some(ImageLoadFailure(reasonOf(err), Command.LoadMoreImages))
        )
      case _ => ()
    }
  )
}
