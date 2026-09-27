package drift.frontend.components

import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** A result of a repository site, in the shape its tile needs
  * (`specs/24-model-details-in-browsers.md`): the two sites count different
  * things and call them differently, but a tile says the same.
  */
case class RepositoryResult(
    /** `owner/name`. */
    id: String,
    previews: List[ModelExample] = Nil,
    exampleCount: Int = 0,
    downloads: Option[Long] = None,
    /** Whatever the site counts as approval. */
    likes: Option[Long] = None,
    /** When the site names one; otherwise the owner half of the id serves. */
    author: Option[String] = None,
    /** Lines under the id in the tile's tooltip. */
    notes: List[String] = Nil
)

/** What the HuggingFace and ModelScope browsers do the same way — which is
  * nearly everything they do (François, 2026-09-18: a behaviour asked for once
  * is wanted in both). The two stay separate classes, each with its own client
  * and service, and put here only what they do identically: the shell and its
  * title, the opening search, the tiles, opening a repository, and the Examples
  * · Files · Model card view over it.
  *
  * A site says the rest: how it sorts and filters, what its search command
  * takes, how its results and its card map onto the shapes above.
  */
abstract class RepositoryBrowser(
    initialQuery: String,
    /** Registering a model: the repository and the file picked in it. */
    onSelect: (String, String) => Unit,
    onCancel: () => Unit,
    /** For a LoRA: what the architecture's LoRAs are trained on, as this site
      * names it (`specs/33-lora-sources.md`).
      */
    baseModels: List[String],
    /** Browsing for a LoRA: LoRAs and results with examples only at first, and
      * a repository opens on its Examples — a LoRA is chosen by what it makes.
      */
    forLoras: Boolean,
    /** The architecture's installed LoRAs, marked on the repositories and files
      * they came from.
      */
    installed: Signal[Installed],
    /** Install mode: the files ticked in the opened repository, and where they
      * land. Args: (repo, paths, grouping).
      */
    onInstall: Option[(String, List[String], LoraGrouping) => Unit],
    /** The row of sources `SourceBrowser` puts in the modal's head. */
    sourceSwitch: Mod[HtmlElement]
) extends Component {

  // ------------------------------------------------- what each site must say

  protected def siteName: String
  protected def siteHost: String

  /** The site's service: results, paging and failures for the shell, and the
    * card state both sites hold the same way (`RepositoryCards`).
    */
  protected def service
      : SearchService[?, ?] & RepositoryCards[?] & RepositoryDetails[?]

  /** Sends this site's search command, with whatever its filters say. */
  protected def search(): Unit

  /** Sends the commands that open a repository: its files and its card. */
  protected def load(repo: String): Unit

  /** The results as tiles show them. Neither site can filter on examples, so
    * this is every result it sent and the box below does the rest.
    */
  protected def allResults: Signal[List[RepositoryResult]]

  /** The rows of controls under the search field. */
  protected def filters: Seq[Mod[HtmlElement]]

  /** Said when a search came back with nothing. */
  protected def emptyText: String

  /** Whether that line may show yet: a browser that has searched for nothing
    * must not accuse itself of having found nothing.
    */
  protected def searched: Signal[Boolean] = Val(true)

  // ------------------------------------------------- the opened repository

  protected def files: Signal[List[BrowserFile]]
  protected def examples: Signal[List[ModelExample]]
  protected def loadingDetail: Signal[Boolean]

  /** How the site gates the opened repository, in the site's own word — none
    * where the site has no such notion (ModelScope says nothing until a 403).
    */
  protected def gated: Signal[Option[String]] = Val(None)

  /** The site's card, as the tab shows it. The rest of the card's state —
    * loading, failure, retry — is the service's, the same on both sites.
    */
  protected def card: Signal[ModelCardContent]

  /** What drift already installed from this repository, and from one of its
    * files: the same lookup, by site.
    */
  protected def installedFrom(
      known: Installed,
      repo: String,
      file: Option[String] = None
  ): List[InstalledItem]

  /** What the site calls its approval count, for a tile's tooltip. */
  protected def likesLabel: String = "likes"

  /** The repository's page on its site. */
  protected def pageUrl(repo: String): String = s"https://$siteHost/$repo"

  /** Anything else the modal must carry while it is open. */
  protected def extraModalMods: Seq[Mod[HtmlElement]] = Nil

  /** Why a result with no example is usually not worth showing. */
  protected def examplesOnlyReason: String

  // ------------------------------------------------------ what they share

  protected val browsing = BrowserView()
  protected val searchQuery = Var(initialQuery)
  protected val opened = Var("")

  /** Kept from one repository to the next: reading cards in a row costs no
    * click per repository.
    */
  protected val tab =
    Var(if (forLoras) RepositoryTab.Examples else RepositoryTab.Files)

  /** Neither site can ask for results with examples, so a page is filtered here
    * and can show fewer than it fetched. Browsing for a LoRA it starts ticked:
    * a LoRA is chosen by what it makes.
    */
  protected val examplesOnly = Var(forLoras)

  /** The box for it, which both sites put in their first filter row. */
  protected def examplesOnlyBox: HtmlElement =
    BrowserFilters.checkbox(
      "With examples only",
      examplesOnlyReason,
      examplesOnly
    )

  private val results: Signal[List[RepositoryResult]] = allResults
    .combineWith(examplesOnly.signal)
    .map((all, only) => if (only) all.filter(_.exampleCount > 0) else all)

  lazy val element: HtmlElement = {
    // Before anything renders: stale results of another context never show.
    service.enterContext(
      s"lora=$forLoras|${baseModels.mkString(",")}|$initialQuery"
    )
    browser.element
  }

  private lazy val browser = ResultBrowser(
    browsing = browsing,
    searchTitle = Val(
      if (forLoras) s"Browse $siteName LoRAs" else s"Browse $siteName Models"
    ),
    detailTitle = opened.signal.map(repo => s"Select a file in $repo"),
    service = service,
    // A site that has nothing to search for yet searches for nothing: its own
    // `search` is what decides.
    modalMods = Seq(onMountCallback(_ => search())) ++ extraModalMods,
    sourceSwitch = sourceSwitch,
    query = searchQuery,
    onSearch = () => search(),
    filters = filters,
    emptyText = emptyText,
    searched = searched,
    cards = results.split(_.id)((_, result, _) => tile(result)),
    detail = () => repositoryView,
    onCancel = onCancel
  )

  private def tile(result: RepositoryResult): HtmlElement = {
    val (owner, name) = BrowserCard.ownerAndName(result.id)
    BrowserCard(
      examples = result.previews,
      exampleCount = result.exampleCount,
      name = name,
      onOpen = () => open(result),
      installed = installed,
      installedAs = known => installedFrom(known, result.id),
      author = result.author.orElse(owner),
      downloads = result.downloads,
      likes = result.likes,
      likesLabel = likesLabel,
      // The tile says the same things on every site; what only one site knows
      // goes in the tooltip.
      tooltip = Some((result.id :: result.notes).mkString("\n"))
    ).element
  }

  private def open(result: RepositoryResult): Unit = {
    opened.set(result.id)
    RepositoryDetail.chooseTab(tab, result.exampleCount)
    load(result.id)
    browsing.open()
  }

  private def repositoryView: HtmlElement = RepositoryDetail(
    tab = tab,
    siteName = siteName,
    siteHost = siteHost,
    pageUrl = pageUrl(opened.now()),
    examples = examples,
    files = files,
    loading = loadingDetail,
    gated = gated,
    onSelect = path => onSelect(opened.now(), path),
    tagsOf = path =>
      Seq(
        Installed.mark(
          installed,
          known => installedFrom(known, opened.now(), Some(path)),
          cls := "ml-2"
        )
      ),
    onInstall = onInstall.map(install =>
      (paths, grouping) => install(opened.now(), paths, grouping)
    ),
    candidates = installed
      .combineWith(opened.signal)
      .map((known, repo) => installedFrom(known, repo)),
    card = card,
    loadingCard = service.loadingCard,
    cardFailure = service.cardFailure,
    onRetryCard = () => service.retryCard(),
    detailFailure = service.detailFailure,
    onRetryDetail = () => service.retryDetail()
  ).element
}
