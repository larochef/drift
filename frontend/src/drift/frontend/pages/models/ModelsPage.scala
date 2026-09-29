package drift.frontend.pages.models

import drift.frontend.Page
import drift.frontend.components.Component
import drift.frontend.pages.architectures.ArchitecturesPage
import drift.frontend.pages.inference.RunConfigurationsPage
import drift.frontend.services.*
import drift.shared.RuntimeTool

import com.raquo.laminar.api.L.*

/** Everything about the models drift can run, in one place (François,
  * 2026-09-10).
  *
  * Inference, Assistant and Architectures were three sidebar entries for one
  * subject. Inference and Assistant were literally the same page with a
  * different `tool` — an image model and a chat model are both "a model, a
  * runtime, and a panel to try it in", and they were only apart because the
  * assistant arrived second. Architectures are the shapes those configurations
  * are built from, which is the same subject one level down.
  *
  * The tabs stay reachable while a session is live, which the old pages did not
  * allow: they replaced themselves with the panel, so a loaded model meant no
  * way back to the configuration that launched it. Each tab is a URL —
  * `/models/configurations`, `/models/architectures` — so a refresh or a link
  * lands on it (François, 2026-09-11).
  */
class ModelsPage(
    runConfigurationService: RunConfigurationService,
    architectureService: ArchitectureService,
    modelService: ModelService,
    cacheService: CacheService,
    downloadService: DownloadService,
    sessionService: SessionService,
    runtimeService: RuntimeService,
    assistantService: AssistantService,
    loraService: LoraService,
    projectService: ProjectService,
    browsers: BrowserServices,
    /** The path after `/models/`: its first segment names the tab. */
    section: Signal[List[String]]
) extends Component {

  private val tab: Signal[String] =
    section.map(_.headOption.getOrElse("configurations")).distinct

  private def tabLink(key: String, label: String): HtmlElement = li(
    cls <-- tab.map(current => if (current == key) "is-active" else ""),
    a(href := s"${Page.Models.path}/$key", label)
  )

  lazy val element: HtmlElement = div(
    cls := "content models-page",
    div(
      cls := "tabs",
      ul(
        tabLink("configurations", "Run configurations"),
        tabLink("architectures", "Architectures")
      )
    ),
    // Both are built once and kept: switching tabs must not throw away a
    // half-filled form, a running session's panel state, or a browser's
    // search results.
    child <-- tab.map {
      case "architectures" => architecturesTab
      case _               => configurationsTab
    }
  )

  private lazy val configurationsTab: HtmlElement =
    RunConfigurationsPage(
      runConfigurationService,
      cacheService,
      downloadService,
      sessionService,
      runtimeService,
      assistantService,
      loraService,
      projectService,
      browsers,
      tools = List(RuntimeTool.SdCpp, RuntimeTool.LlamaCpp)
    ).element

  private lazy val architecturesTab: HtmlElement =
    ArchitecturesPage(
      architectureService,
      modelService,
      cacheService,
      downloadService,
      loraService,
      browsers,
      assistantService.library.ofKind(drift.shared.PromptKind.AssistantSystem),
      runtimeService
    ).element
}
