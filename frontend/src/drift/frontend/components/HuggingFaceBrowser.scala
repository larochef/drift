package drift.frontend.components

import drift.frontend.services.HuggingFaceService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Browses HuggingFace (`specs/03-model-discovery.md`, `specs/24`,
  * `specs/36-huggingface-examples.md`). What it shares with the ModelScope
  * browser is `RepositoryBrowser`'s; here is what HuggingFace does its own way:
  * it sorts on four fields with a direction, filters adapters by one base model
  * at a time — it ANDs repeated filters — and its card can be gated.
  */
class HuggingFaceBrowser(
    huggingFace: HuggingFaceService,
    initialQuery: String,
    onSelect: (String, String) => Unit,
    onCancel: () => Unit,
    /** For a LoRA: the repositories its architecture is trained on. The first
      * one's adapters show on opening, without typing; the choice also offers
      * the others and a search of every repository
      * (`specs/33-lora-sources.md`).
      */
    baseModels: List[String] = Nil,
    /** Browsing for a LoRA (`specs/36-huggingface-examples.md`): results
      * without examples are hidden at first, and a repository opens on its
      * Examples — a LoRA is chosen by what it makes.
      */
    forLoras: Boolean = false,
    // The three below mean the same on both sites and are documented on
    // `RepositoryBrowser`, which is what uses them.
    installed: Signal[Installed] = Val(Installed.none),
    onInstall: Option[(String, List[String], LoraGrouping) => Unit] = None,
    sourceSwitch: Mod[HtmlElement] = emptyNode,
    suggestions: List[String] = Nil
) extends RepositoryBrowser(
      initialQuery,
      onSelect,
      onCancel,
      baseModels,
      forLoras,
      installed,
      onInstall,
      sourceSwitch,
      suggestions
    ) {

  private val sortOptions = List(
    "downloads" -> "Most downloaded",
    "likes" -> "Most likes",
    "lastModified" -> "Recently updated",
    "createdAt" -> "Recently created"
  )

  private val sortBy = Var[String]("downloads")
  private val sortDir = Var[Int](-1)
  private val baseModel = Var(baseModels.headOption)
  private val hasSearched = Var(false)

  protected val siteName = "HuggingFace"
  protected val siteHost = "huggingface.co"
  protected def service = huggingFace

  protected def examplesOnlyReason: String =
    "Hide repositories with no example image or video — most test uploads " +
      "have none"

  /** A base model is a search in itself: its adapters, most downloaded first.
    * With neither it nor a query there is nothing to ask for.
    */
  protected def search(): Unit = {
    val query = searchQuery.now().trim
    if (query.nonEmpty || baseModel.now().isDefined) {
      hasSearched.set(true)
      huggingFace.push(
        HuggingFaceService.Command.Search(
          query,
          Some(sortBy.now()),
          Some(sortDir.now()),
          baseModel.now()
        )
      )
    }
  }

  protected def load(repo: String): Unit = {
    huggingFace.push(HuggingFaceService.Command.LoadFiles(repo))
    huggingFace.push(HuggingFaceService.Command.LoadCard(repo))
  }

  protected def allResults: Signal[List[RepositoryResult]] =
    huggingFace.results.map(
      _.map(model =>
        RepositoryResult(
          id = model.id,
          previews = model.previews,
          exampleCount = model.exampleCount,
          downloads = model.downloads,
          likes = model.likes,
          author = model.author
        )
      )
    )

  protected def filters: Seq[Mod[HtmlElement]] =
    Seq(sortAndFilters, baseModelChoice)

  protected def emptyText: String =
    (if (baseModels.isEmpty) "No models found."
     else
       "No models found — few LoRAs may name this base model yet; " +
         "try another one or Any repository.") +
      (if (forLoras) " Unticking With examples only shows the rest." else "")

  override protected def searched: Signal[Boolean] = hasSearched.signal

  override protected def gated: Signal[Option[String]] =
    huggingFace.detail.map(_.flatMap(_.gated.kind)).distinct

  protected def files: Signal[List[BrowserFile]] = huggingFace.detail.map(
    _.map(_.siblings.map(file => BrowserFile(file.rfilename, file.size)))
      .getOrElse(Nil)
  )

  protected def examples: Signal[List[ModelExample]] =
    huggingFace.detail.map(_.map(_.examples).getOrElse(Nil))

  protected def loadingDetail: Signal[Boolean] = huggingFace.loadingDetail

  protected def card: Signal[ModelCardContent] = huggingFace.card.map {
    case Some(HuggingFaceModelCard(Some(html), _)) =>
      ModelCardContent.Rendered(html)
    case Some(HuggingFaceModelCard(None, true)) => ModelCardContent.Gated
    case Some(_)                                => ModelCardContent.Absent
    case None                                   => ModelCardContent.Pending
  }

  protected def installedFrom(
      known: Installed,
      repo: String,
      file: Option[String]
  ): List[InstalledItem] = known.fromHuggingFace(repo, file)

  private def sortAndFilters: HtmlElement =
    BrowserFilters.row(
      BrowserFilters.choice(
        sortOptions,
        sortBy.now(),
        "sort order",
        picked => {
          sortBy.set(picked)
          search()
        }
      ),
      examplesOnlyBox
    )

  /** The repositories this architecture's LoRAs are trained on, one at a time —
    * HuggingFace ANDs repeated filters — or any repository at all.
    */
  private def baseModelChoice: Mod[HtmlElement] =
    if (baseModels.isEmpty) emptyNode
    else
      BrowserFilters.row(
        BrowserFilters.choice(
          baseModels.map(repo => repo -> s"LoRAs of $repo") :+
            ("" -> "Any repository"),
          baseModel.now().getOrElse(""),
          "trained on",
          picked => {
            baseModel.set(Option(picked).filter(_.nonEmpty))
            search()
          }
        )
      )
}
