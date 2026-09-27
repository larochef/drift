package drift.frontend.services

import com.raquo.laminar.api.L.*

/** What the three site services are underneath (`specs/24`): the results of the
  * search in effect, whether one is in flight, whether the site has another
  * page, and the command bus a browser pushes to. Each service keeps its own
  * client, its own `Command` enum and its own way of paging — an offset, a page
  * number, an opaque cursor — and adds whatever its site has besides: a model's
  * detail, a card, a gallery of posts.
  */
trait SearchService[Result, Command] extends ServiceErrors {

  protected val _results = Var(List.empty[Result])
  val results: Signal[List[Result]] = _results.signal

  protected val _searching = Var(false)
  val searching: Signal[Boolean] = _searching.signal

  /** The site has another page for the search in effect. */
  def hasMore: Signal[Boolean]

  /** The next page, appended: the service's own `LoadMore` command. */
  def loadMore(): Unit

  /** The service's subscriptions, mounted by the browser it serves. */
  val effects: Modifier[HtmlElement]

  /** What tells two results apart, for the paging dedupe. */
  protected def idOf(result: Result): String

  /** What a new context forgets besides the results: how far the paging of the
    * search in effect had gone.
    */
  protected def resetPaging(): Unit

  /** What the results were searched for: the context a browser was opened in
    * (what is browsed, for which base models, from which starting search). A
    * browser opened in another context clears them before its first search,
    * rather than showing the previous context's results until it answers; the
    * same context keeps them, so coming back to a search costs nothing.
    */
  private var context = Option.empty[String]

  def enterContext(key: String): Unit =
    if (!context.contains(key)) {
      context = Some(key)
      _results.set(Nil)
      resetPaging()
    }

  /** A page after the first, appended minus what is already shown — every one
    * of the three sites can repeat an entry across a page boundary.
    */
  protected def appendPage(page: List[Result]): Unit =
    _results.update { shown =>
      val seen = shown.map(idOf).toSet
      shown ++ page.filterNot(result => seen.contains(idOf(result)))
    }

  protected val cmdBus = new EventBus[Command]

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  protected def reasonOf(err: Throwable): String =
    Option(err.getMessage).getOrElse(err.toString)
}
