package drift.frontend.components

import drift.frontend.services.SearchService

import com.raquo.laminar.api.L.*

/** The two-phase browser the three site browsers are (`specs/03`, `24`, `37`):
  * a search form over a grid of result cards, and one opened result's detail
  * view with Back. What a site does its own way is passed in -- its filters,
  * its cards, its detail view -- and what none of them do differently lives
  * here once: the modal and its width, the error banner, the grid, Load More,
  * Back and Cancel, and the scroll the list is left at (François, 2026-09-17).
  */
class ResultBrowser(
    browsing: BrowserView,
    /** The modal's title over the list, and over an opened result. */
    searchTitle: Signal[String],
    detailTitle: Signal[String],
    /** The site's results, its paging and its failures: the shell mounts its
      * subscriptions, shows its banner, and asks it for the next page.
      */
    service: SearchService[?, ?],
    /** What the modal must carry besides: the opening search, the browser's own
      * event handling.
      */
    modalMods: Seq[Mod[HtmlElement]],
    /** Where else the same thing could be looked for: the row of sources in the
      * modal's head, when the browser was opened by `SourceBrowser`.
      */
    sourceSwitch: Mod[HtmlElement] = emptyNode,
    /** What is searched for, and what asks for it again. */
    query: Var[String],
    onSearch: () => Unit,
    searchHint: String = "Search models...",
    /** Searches worth one click, as chips under the field: a component's family
      * and the architecture it comes packaged with.
      */
    suggestions: List[String] = Nil,
    /** The site's own controls, under the search field. */
    filters: Seq[Mod[HtmlElement]] = Nil,
    /** Said when a search came back with nothing. */
    emptyText: String,
    /** Gates that line: an untouched browser must not accuse itself of having
      * found nothing.
      */
    searched: Signal[Boolean] = Val(true),
    cards: Signal[Seq[HtmlElement]],
    /** Built afresh every time a result is opened. */
    detail: () => HtmlElement,
    onCancel: () => Unit,
    /** How many cards a row holds while the modal is wide enough for them to
      * keep their width: five leaves each one room to be looked at. A narrower
      * window fits as many 280px-wide ones as it can instead — narrower than
      * that, a tile's chips take more room than its picture.
      */
    cardsPerRow: Int = 5
) extends Component {

  lazy val element: HtmlElement = BrowserModal(
    title = browsing.showingDetail
      .combineWith(searchTitle, detailTitle)
      .map((opened, list, one) => if (opened) one else list),
    onCancel = onCancel,
    // The head carries what the browser as a whole is set to: where it looks
    // (`specs/03`) and whether videos play — one switch for the three
    // browsers and their galleries, not one per site's filter row.
    headerAction = div(
      cls := "browser-head-actions",
      sourceSwitch,
      BrowserFilters.checkbox(
        "Play videos",
        "Play the videos among the results; off, a tile shows a frame of " +
          "one and loads nothing more",
        BrowserMedia.playVideos
      )
    ),
    modalMods =
      Seq[Mod[HtmlElement]](cls := "browser-modal", service.effects) ++
        modalMods,
    body = Seq(
      browsing.binder,
      ErrorBanner(service),
      // The list is one element, taken out and put back; a detail view is a
      // new one each time, built over whatever the site browser has just
      // opened.
      child <-- browsing.showingDetail.map(if (_) detail() else searchView)
    ),
    footerLeft = child <-- browsing.showingDetail.map {
      case true  => BrowserModal.backButton(() => browsing.back())
      case false => emptyNode
    },
    footerRight = BrowserModal.cancelButton(onCancel)
  ).element

  private lazy val searchView: HtmlElement = div(
    SearchField(query, service.searching, onSearch, hint = searchHint),
    if (suggestions.isEmpty) emptyNode
    else
      div(
        cls := "buttons are-small mb-2",
        span(cls := "text-secondary is-size-7 mr-2", "Search for"),
        suggestions.map(suggestion =>
          button(
            cls := "button is-rounded",
            cls("is-info") <-- query.signal.map(_.trim == suggestion),
            suggestion,
            onClick --> { _ =>
              query.set(suggestion)
              onSearch()
            }
          )
        )
      ),
    filters,
    ResultsPlaceholder(
      busy = service.searching,
      isEmpty = cards.map(_.isEmpty),
      busyText = "Searching...",
      emptyText = emptyText,
      show = searched
    ),
    div(
      cls := "civitai-results-grid",
      styleAttr := "grid-template-columns: repeat(auto-fill, minmax(" +
        s"max(280px, calc((100% - ${cardsPerRow - 1} * 0.75rem - 1px)" +
        s" / $cardsPerRow)), 1fr));",
      children <-- cards
    ),
    child <-- service.hasMore.combineWith(service.searching).map {
      (more, working) =>
        if (more && !working)
          div(
            cls := "has-text-centered mt-3",
            button(
              cls := "button is-outlined is-fullwidth",
              "Load More",
              onClick --> (_ => service.loadMore())
            )
          )
        else emptyNode
    }
  )
}
