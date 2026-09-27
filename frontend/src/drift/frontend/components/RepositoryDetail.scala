package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** Which of an opened repository's three views shows. The browser holds it, so
  * the tab chosen stays chosen for the next repository opened (`specs/24`).
  */
enum RepositoryTab { case Examples, Files, Card }

/** An opened repository, HuggingFace or ModelScope (`specs/24`, `36`, `37`):
  * what its model makes, the files it holds, and its README. The two sites
  * differ in the service behind these, not in the view — so the view is one.
  */
class RepositoryDetail(
    tab: Var[RepositoryTab],
    /** The site's name and host, and the repository's page there. */
    siteName: String,
    siteHost: String,
    pageUrl: String,
    examples: Signal[List[ModelExample]],
    files: Signal[List[BrowserFile]],
    /** The repository's own request is in flight. */
    loading: Signal[Boolean],
    /** How the site gates this repository, when it does (`specs/24`). */
    gated: Signal[Option[String]],
    onSelect: String => Unit,
    /** The chips one file carries, by path: the ✓ installed mark. */
    tagsOf: String => Seq[Mod[HtmlElement]],
    /** Install mode: the ticked files, and where they land. */
    onInstall: Option[(List[String], LoraGrouping) => Unit] = None,
    /** What they could join: the LoRAs this repository already made. */
    candidates: Signal[List[InstalledItem]] = Val(Nil),
    card: Signal[ModelCardContent],
    loadingCard: Signal[Boolean],
    cardFailure: Signal[Option[String]],
    onRetryCard: () => Unit,
    /** Why the repository itself did not load — its files and its pictures are
      * one request, and ModelScope fails it often enough to need a Retry
      * (`specs/37-modelscope.md`).
      */
    detailFailure: Signal[Option[String]],
    onRetryDetail: () => Unit
) extends Component {

  private lazy val examplesTab = ModelExampleGallery(
    pageUrl,
    s"Open on $siteHost ↗",
    examples,
    loading
  )

  private lazy val fileList = RepositoryFiles(
    files,
    loading,
    onSelect,
    gated,
    siteName,
    tagsOf,
    onInstall,
    candidates
  )

  private lazy val cardTab = ModelCardTab(
    siteName,
    siteHost,
    pageUrl,
    card,
    loadingCard,
    cardFailure,
    onRetryCard
  )

  lazy val element: HtmlElement = div(
    // Above the tabs: whichever one is open shows nothing at all when the
    // repository did not load, which reads as an empty repository.
    child.maybe <-- detailFailure.map(
      _.map(reason =>
        RetryNotice(
          s"This repository did not load: $reason",
          onRetryDetail
        ).element
      )
    ),
    BrowserTabs(
      tab,
      RepositoryTab.Examples -> "Examples",
      RepositoryTab.Files -> "Files",
      RepositoryTab.Card -> "Model card"
    ),
    child <-- tab.signal.map {
      case RepositoryTab.Examples => examplesTab.element
      case RepositoryTab.Files    => fileList.element
      case RepositoryTab.Card     => cardTab.element
    }
  )
}

object RepositoryDetail {

  /** The tab a repository opens on: the one chosen last, unless this repository
    * has no example to show there.
    */
  def chooseTab(tab: Var[RepositoryTab], exampleCount: Int): Unit =
    if (tab.now() == RepositoryTab.Examples && exampleCount == 0)
      tab.set(RepositoryTab.Files)
}
