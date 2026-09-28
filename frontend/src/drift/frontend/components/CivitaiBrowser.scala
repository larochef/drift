package drift.frontend.components

import drift.frontend.services.{AuthTokenService, CivitaiService}
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Browses Civitai (`specs/03-model-discovery.md`,
  * `specs/24-model-details-in-browsers.md`): a search grid of model cards, and
  * an opened model's files, description and posted images. The modal, the grid,
  * Load More, Back and the remembered scroll are `ResultBrowser`'s; its tiles
  * are `CivitaiMedia`, its chips `CivitaiTags`, its About tab
  * `CivitaiModelAbout`.
  */
class CivitaiBrowser(
    service: CivitaiService,
    authTokens: AuthTokenService,
    initialQuery: String,
    initialModelType: Option[String] = Some("CHECKPOINT"),
    civitaiBaseModels: List[String] = List.empty,
    // (modelId, versionId, fileId, filename, label)
    onSelect: (String, String, String, String, String) => Unit,
    onCancel: () => Unit,
    /** Install mode (`specs/33-lora-sources.md`): when set, the file rows are
      * ticked rather than picked, and the bar over them installs the lot. Args:
      * (civitaiModelId, files, grouping).
      */
    onInstall: Option[
      (String, List[CivitaiFileRef], LoraGrouping) => Unit
    ] = None,
    /** LoRA browsing: the architecture's installed LoRAs, marked on the models
      * and versions they came from (`specs/33-lora-sources.md`).
      */
    installed: Signal[Installed] = Val(Installed.none),
    /** The row of sources `SourceBrowser` puts in the modal's head. */
    sourceSwitch: Mod[HtmlElement] = emptyNode
) extends Component {

  /** The opened model's tabs (`specs/24`). Kept from one model to the next, so
    * reading descriptions or galleries in a row costs no click per model.
    */
  private enum Tab { case Files, About, Images }

  private val sortOptions = List(
    "Most Downloaded" -> "Most downloaded",
    "Most Liked" -> "Most likes",
    "Newest" -> "Recently published",
    "Oldest" -> "Oldest"
  )

  private val browsing = BrowserView()
  private val tab = Var(Tab.Files)
  private val searchQuery = Var(initialQuery)
  private val selectedModel = Var("")
  private val modelId = Var("")
  private val modelLabel = Var("")
  // Seeded from the search payload so files show instantly, then refined by
  // the detail response the service fetches.
  private val modelFiles = Var(List.empty[CivitaiModelVersion])
  private val searched = Var(false)

  private val results = service.results
  private val searching = service.searching
  private val sortFilter = Var(sortOptions.head._1)
  private val baseModelsVar = Var(civitaiBaseModels)
  // Sent explicitly either way — Civitai's absent-parameter default hides
  // models the site shows, some of them SFW-flagged (bugs/19). Off by
  // default, per François.
  private val includeNsfw = Var(false)

  /** Civitai answers every download 401 without a token, so while none is set
    * (saved and active, or `CIVITAI_API_TOKEN`) the browser asks for one first;
    * saving it makes it active, which swaps the browser in.
    */
  lazy val element: HtmlElement = div(
    authTokens.effects,
    onMountCallback(_ => authTokens.push(AuthTokenService.Command.Load)),
    child <-- authTokens.hasToken(AuthProvider.Civitai).map {
      case None        => emptyNode
      case Some(false) => tokenRequest
      case Some(true)  =>
        // Before anything renders: stale results of another context never
        // show.
        service.enterContext(
          s"${initialModelType.getOrElse("")}|" +
            s"${civitaiBaseModels.mkString(",")}|$initialQuery"
        )
        browser.element
    }
  )

  private lazy val tokenForm =
    AuthTokenForm(authTokens, Some(AuthProvider.Civitai), "Civitai")

  private lazy val tokenRequest: HtmlElement = BrowserModal(
    title = Val("Civitai needs an API token"),
    onCancel = onCancel,
    headerAction = div(cls := "browser-head-actions", sourceSwitch),
    modalMods = Seq(cls := "browser-modal"),
    body = Seq(
      ErrorBanner(authTokens),
      p(
        cls := "text-secondary mb-4",
        "Civitai refuses downloads without an API key. Create one under ",
        a(
          href := "https://civitai.com/user/account",
          target := "_blank",
          rel := "noopener noreferrer",
          "API Keys in your Civitai account settings"
        ),
        " and paste it here; it is saved in Settings → Authentication and ",
        "sent to civitai.com only."
      ),
      tokenForm.fields
    ),
    footerRight = div(
      cls := "buttons",
      tokenForm.saveButton(() => ()),
      BrowserModal.cancelButton(onCancel)
    )
  ).element

  private lazy val browser = ResultBrowser(
    browsing = browsing,
    searchTitle = Val("Browse Civitai Models"),
    detailTitle = selectedModel.signal.map(name => s"Select a file for $name"),
    service = service,
    modalMods = Seq(
      styleAttr := "overflow: hidden;",
      onMountCallback { _ => doSearch() },
      // `.changes` and not `.signal`: a signal replays its current value on
      // subscribe, which fired a duplicate search on every open alongside the
      // mount callback above (bugs/13).
      baseModelsVar.signal.changes --> Observer { _ => doSearch() },
      includeNsfw.signal.changes --> Observer { _ => doSearch() },
      service.events --> Observer {
        case CivitaiService.Event.FilesLoaded(versions) =>
          if (versions.nonEmpty) modelFiles.set(versions)
        case _ => ()
      }
    ),
    sourceSwitch = sourceSwitch,
    query = searchQuery,
    onSearch = () => doSearch(),
    searchHint = "Search Civitai models...",
    filters = Seq(sortAndFilters),
    emptyText = "No models found.",
    searched = searched.signal,
    cards = results.split(_.id.toString)((_, model, _) => tile(model)),
    detail = () => modelView,
    onCancel = onCancel
  )

  /** Built once, not per opened model: what it remembers of an install already
    * started must survive Back and a second look at the same model.
    */
  private lazy val versionList = CivitaiVersionList(
    versions = modelFiles.signal,
    cachedFileIds = service.cachedFileIds,
    loading = searching,
    civitaiBaseModels = civitaiBaseModels,
    installed = installed,
    onInstall = onInstall.map(install =>
      (files, grouping) => install(modelId.now(), files, grouping)
    ),
    candidates = installed
      .combineWith(modelId.signal)
      .map((known, id) => known.fromCivitaiModel(id)),
    onSelectFile = (version, file) => {
      val modelName = selectedModel.now().split("/")(1)
      val base = if (modelLabel.now().isEmpty) modelName else modelLabel.now()
      // The precision goes into the label so two registered quants of the
      // same version stay distinguishable everywhere models are listed.
      val label = file.quantization.map(q => s"$base ($q)").getOrElse(base)
      onSelect(
        modelId.now(),
        version.id.toString,
        file.id.toString,
        file.name,
        label
      )
    }
  )

  private def doSearch(): Unit = {
    val q = searchQuery.now().trim
    searched.set(true)
    service.push(
      CivitaiService.Command
        .Search(
          q,
          Some(sortFilter.now()),
          initialModelType,
          baseModelsVar.now(),
          includeNsfw.now()
        )
    )
  }

  private def sortAndFilters: HtmlElement =
    BrowserFilters.row(
      BrowserFilters.choice(
        sortOptions,
        sortFilter.now(),
        "sort order",
        picked => {
          sortFilter.set(picked)
          doSearch()
        }
      ),
      // Its own re-search is bound in the modal's mods, so none here.
      BrowserFilters.checkbox(
        "Include NSFW",
        "Civitai hides models it flags NSFW unless they are asked for, " +
          "some of them flagged safe-for-work (bugs/19)",
        includeNsfw
      )
    )

  private def tile(model: CivitaiModelListInfo): HtmlElement = {
    val examples = model.examples
    BrowserCard(
      examples = examples,
      exampleCount = examples.size,
      name = model.name,
      onOpen = () => open(model),
      installed = installed,
      installedAs = _.fromCivitaiModel(model.id.toString),
      author = model.creator.flatMap(_.username),
      downloads = Some(model.stats.downloadCount.toLong),
      likes = Some(model.stats.thumbsUpCount.toLong),
      likesLabel = "thumbs up",
      tooltip = Some(model.name)
    ).element
  }

  private def open(model: CivitaiModelListInfo): Unit = {
    // Another model, its own ticks.
    versionList.clearSelection()
    selectedModel.set(s"${model.id}/${model.name}")
    modelId.set(model.id.toString)
    modelLabel.set(model.name)
    modelFiles.set(model.modelVersions)
    service.push(CivitaiService.Command.LoadFiles(model.id))
    browsing.open()
  }

  /** The opened model: its files, what it says about itself, and what people
    * made with it (`specs/24`).
    */
  private def modelView: HtmlElement =
    div(
      BrowserTabs(
        tab,
        Tab.Files -> "Files",
        Tab.About -> "About",
        Tab.Images -> "Images"
      ),
      child <-- tab.signal.map {
        case Tab.Files => versionList.element
        case Tab.About =>
          CivitaiModelAbout(service, modelId.now(), civitaiBaseModels).element
        case Tab.Images =>
          modelId.now().toIntOption match {
            case Some(id) =>
              CivitaiImageGallery(
                service,
                id,
                modelFiles.signal,
                includeNsfw
              ).element
            case None => emptyNode
          }
      }
    )

}
