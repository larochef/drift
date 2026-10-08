package drift.frontend

import drift.frontend.components.*
import drift.frontend.pages.*
import drift.frontend.pages.cache.ModelCachePage
import drift.frontend.pages.gallery.GalleryPage
import drift.frontend.pages.models.ModelsPage
import drift.frontend.pages.projects.{ProjectWorkspacePage, ProjectsPage}
import drift.frontend.pages.sandbox.SandboxPage
import drift.frontend.pages.settings.SettingsPage
import drift.frontend.services.*

import com.raquo.laminar.api.L.*
import frontroute.*

private class Sidebar extends Component {
  lazy val element: HtmlElement = {
    div(
      cls := "menu px-2 py-3 sidebar-menu",
      p(
        cls := "menu-label text-secondary sidebar-brand",
        img(src := "/assets/favicon.svg", alt := "", cls := "sidebar-logo"),
        "drift"
      ),
      ul(
        cls := "menu-list",
        Page.navPages.map { page =>
          li(
            a(
              href := page.path,
              cls := "text-primary bg-surface",
              span(page.icon),
              " ",
              page.label
            )
          )
        }
      )
    )
  }
}

private class Layout(
    /** Pinned to the sidebar's bottom — the machine, then the downloads. */
    sidebarFooter: Seq[Component],
    mods: Mod[HtmlElement]*
) extends Component {
  lazy val element: HtmlElement = {
    div(
      cls := "columns is-gapless",
      div(
        cls := "column is-narrow is-hidden-mobile bg-surface",
        div(
          cls := "sidebar-layout",
          Sidebar(),
          div(styleAttr := "margin-top: auto;", sidebarFooter.map(_.element))
        )
      ),
      div(
        cls := "column bg-content",
        div(
          cls := "section",
          mods
        )
      )
    )
  }
}

/** Chrome plus the route table. Pages are constructed here from services the
  * caller already built, so `main` stays responsible for wiring and this file
  * stays responsible for display. Each page mounts its own `service.effects`
  * (see `bugs/15-modelservice-effects-never-mounted.md`), so a page must keep
  * being constructed inside its route rather than hoisted out of it.
  */
class AppShell(
    modelService: ModelService,
    runConfigurationService: RunConfigurationService,
    architectureService: ArchitectureService,
    cacheService: CacheService,
    downloadService: DownloadService,
    prerequisites: LaunchPrerequisites,
    runtimeService: RuntimeService,
    authTokenService: AuthTokenService,
    promptTemplateService: PromptTemplateService,
    sessionService: SessionService,
    generationService: GenerationService,
    assistantService: AssistantService,
    projectService: ProjectService,
    historyService: HistoryService,
    postProcessService: PostProcessService,
    loraService: LoraService,
    upscalerService: UpscalerService,
    conversionService: ConversionService,
    globalDownloadsService: GlobalDownloadsService,
    machineService: MachineService,
    logService: LogService,
    statusSocketService: StatusSocketService,
    browsers: BrowserServices
) extends Component {
  lazy val element: HtmlElement =
    routes(
      Layout(
        Seq(
          MachinePanel(machineService.status),
          DownloadsPanel(globalDownloadsService)
        ),
        // Above every page, because a socket this page cannot read takes the
        // whole app's live half with it, whichever page is open.
        StaleBundleNotice(statusSocketService).element,
        // Projects are the app's front door; there is no dashboard.
        pathEnd {
          new ProjectsPage(projectService).element
        },
        // The open image is part of the address: a refresh — after a repaint
        // lands, say — comes back to it (`specs/12-gallery.md`).
        pathPrefix("gallery") {
          extractUnmatchedPath.signal { section =>
            new GalleryPage(
              historyService,
              runConfigurationService,
              sessionService,
              generationService,
              assistantService,
              projectService,
              postProcessService,
              upscalerService,
              runtimeService,
              machineService,
              prerequisites,
              section
            ).element
          }
        },
        path("projects") {
          new ProjectsPage(projectService).element
        },
        pathPrefix("projects" / segment) { id =>
          extractUnmatchedPath.signal { section =>
            new ProjectWorkspacePage(
              id,
              projectService,
              runConfigurationService,
              sessionService,
              generationService,
              assistantService,
              loraService,
              historyService,
              postProcessService,
              upscalerService,
              runtimeService,
              logService,
              machineService,
              prerequisites,
              section,
              browsers
            ).element
          }
        },
        // Trying a model, nothing kept (`specs/47-sandbox.md`); the rest of
        // the path names the kind, and the page is built once so switching
        // it keeps what is on screen until the switch clears it.
        pathPrefix("sandbox") {
          extractUnmatchedPath.signal { section =>
            new SandboxPage(
              runConfigurationService,
              sessionService,
              generationService,
              assistantService,
              loraService,
              projectService,
              historyService,
              runtimeService,
              logService,
              machineService,
              prerequisites,
              section,
              browsers
            ).element
          }
        },
        // One page for the models and how they are run. The rest of the path
        // names its tab, and arrives as a signal: the page is built once and
        // only the tab follows the URL, so switching keeps a half-filled form.
        pathPrefix("models") {
          extractUnmatchedPath.signal { section =>
            new ModelsPage(
              runConfigurationService,
              architectureService,
              modelService,
              cacheService,
              downloadService,
              sessionService,
              runtimeService,
              assistantService,
              loraService,
              projectService,
              browsers,
              section
            ).element
          }
        },
        pathPrefix("model-cache") {
          extractUnmatchedPath.signal { section =>
            new ModelCachePage(
              cacheService,
              modelService,
              architectureService,
              runConfigurationService,
              downloadService,
              loraService,
              upscalerService,
              conversionService,
              browsers,
              section
            ).element
          }
        },
        path("settings") {
          new SettingsPage(
            runtimeService,
            authTokenService,
            promptTemplateService
          ).element
        },
        noneMatched {
          NotFoundPage.element
        }
      ).element
    ).amend(
      LinkHandler.bind,
      statusSocketService.effects,
      globalDownloadsService.effects,
      // The prompt library is read by the assistant and the redraw panel on
      // every page, so it loads with the shell (`specs/32`).
      promptTemplateService.effects,
      onMountCallback(_ =>
        promptTemplateService.push(PromptTemplateService.Command.Load)
      )
    )
}
