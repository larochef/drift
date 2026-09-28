package drift.frontend.components

import drift.frontend.services.ModelScopeService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Browses ModelScope (`specs/37-modelscope.md`). What it shares with the
  * HuggingFace browser is `RepositoryBrowser`'s; here is what ModelScope does
  * its own way: it lists every model without a query, files some of them as
  * LoRAs, searches all of an architecture's base models at once — it ORs the
  * values of one criterion — and serves repositories under `/models/`.
  */
class ModelScopeBrowser(
    modelScope: ModelScopeService,
    initialQuery: String,
    onSelect: (String, String) => Unit,
    onCancel: () => Unit,
    /** For a LoRA: the base model names its architecture's LoRAs declare on
      * ModelScope. All of them are searched at once by default, or one, or
      * none.
      */
    baseModels: List[String] = Nil,
    /** Browsing for a LoRA: LoRAs and results with examples only at first, and
      * a repository opens on its Examples.
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

  private val sort = Var("DownloadsCount")

  /** "all", "any", or the index of one base model. */
  private val baseChoice = Var(if (baseModels.isEmpty) "any" else "all")
  private val lorasOnly = Var(forLoras)

  protected val siteName = "ModelScope"
  protected val siteHost = "modelscope.cn"
  protected def service = modelScope

  protected def examplesOnlyReason: String =
    "Hide repositories with no cover image and no image or video file"

  override protected def likesLabel: String = "stars"

  override protected def pageUrl(repo: String): String =
    s"https://$siteHost/models/$repo"

  /** ModelScope lists every model without a query: something shows at once. */
  protected def search(): Unit =
    modelScope.push(
      ModelScopeService.Command.Search(
        searchQuery.now().trim,
        sort.now(),
        chosenBaseModels,
        lorasOnly.now()
      )
    )

  protected def load(repo: String): Unit = {
    modelScope.push(ModelScopeService.Command.LoadDetail(repo))
    modelScope.push(ModelScopeService.Command.LoadCard(repo))
  }

  protected def allResults: Signal[List[RepositoryResult]] =
    modelScope.results.map(
      _.map(model =>
        RepositoryResult(
          id = model.id,
          previews = model.previews,
          exampleCount = model.exampleCount,
          downloads = Some(model.downloads),
          likes = Some(model.stars),
          notes = List(
            Option.when(model.lora)("ModelScope files it as a LoRA"),
            Option.when(model.baseModels.nonEmpty)(
              s"trained on ${model.baseModels.mkString(", ")}"
            )
          ).flatten
        )
      )
    )

  protected def filters: Seq[Mod[HtmlElement]] =
    Seq(sortAndFilters, baseModelChoice)

  protected def emptyText: String = "No repositories found." +
    (if (forLoras) " Unticking LoRAs only or With examples only shows more."
     else "")

  protected def files: Signal[List[BrowserFile]] = modelScope.detail.map(
    _.map(_.files.map(file => BrowserFile(file.path, file.size)))
      .getOrElse(Nil)
  )

  protected def examples: Signal[List[ModelExample]] =
    modelScope.detail.map(_.map(_.examples).getOrElse(Nil))

  protected def loadingDetail: Signal[Boolean] = modelScope.loadingDetail

  protected def card: Signal[ModelCardContent] = modelScope.card.map {
    case Some(ModelScopeModelCard(Some(html))) =>
      ModelCardContent.Rendered(html)
    case Some(_) => ModelCardContent.Absent
    case None    => ModelCardContent.Pending
  }

  protected def installedFrom(
      known: Installed,
      repo: String,
      file: Option[String]
  ): List[InstalledItem] = known.fromModelScope(repo, file)

  private def chosenBaseModels: List[String] =
    baseChoice.now() match {
      case "all"   => baseModels
      case "any"   => Nil
      case indexed => indexed.toIntOption.flatMap(baseModels.lift).toList
    }

  private def sortAndFilters: HtmlElement =
    BrowserFilters.row(
      BrowserFilters.choice(
        ModelScopeSorts,
        sort.now(),
        "sort order",
        picked => {
          sort.set(picked)
          search()
        }
      ),
      BrowserFilters.checkbox(
        "LoRAs only",
        "Only the repositories ModelScope files as LoRAs",
        lorasOnly,
        () => search()
      ),
      examplesOnlyBox
    )

  /** All the base models this architecture's LoRAs name on ModelScope at once —
    * ModelScope ORs the values of one criterion — or one of them, or any.
    */
  private def baseChoices: Seq[(String, String)] =
    Option
      .when(baseModels.size > 1)(
        "all" -> s"Any of its ${baseModels.size} base models"
      )
      .toList ++
      baseModels.zipWithIndex.map((name, index) =>
        (if (baseModels.size == 1) "all" else index.toString) -> name
      ) :+ ("any" -> "Any base model")

  private def baseModelChoice: Mod[HtmlElement] =
    if (baseModels.isEmpty) emptyNode
    else
      BrowserFilters.row(
        BrowserFilters.choice(
          baseChoices,
          baseChoice.now(),
          "trained on",
          picked => {
            baseChoice.set(picked)
            search()
          }
        )
      )
}
