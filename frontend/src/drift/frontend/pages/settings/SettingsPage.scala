package drift.frontend.pages.settings

import drift.frontend.components.*
import drift.frontend.services.*

import com.raquo.laminar.api.L.*

/** Settings: the runtimes drift runs its tools with (`RuntimesSection`,
  * `specs/06-sdcpp-runtime.md`, `specs/17-assistant-runtime.md`) and the API
  * tokens it downloads with (`AuthenticationSection`).
  */
class SettingsPage(
    runtimeService: RuntimeService,
    authTokenService: AuthTokenService,
    locationsService: LocationsService,
    promptTemplateService: PromptTemplateService
) extends Component {
  import RuntimeService.Command

  lazy val element: HtmlElement = div(
    cls := "content",
    onMountCallback { _ =>
      runtimeService.push(Command.Load)
      runtimeService.push(Command.LoadInstalls)
      runtimeService.push(Command.LoadReleases)
      runtimeService.push(Command.LoadTargets)
      authTokenService.push(AuthTokenService.Command.Load)
      promptTemplateService.push(PromptTemplateService.Command.Load)
    },
    h1(cls := "title text-primary", "Settings"),
    ErrorBanner(runtimeService),
    ErrorBanner(authTokenService),
    RuntimesSection(runtimeService, locationsService).element,
    AuthenticationSection(authTokenService, locationsService).element,
    PromptsSection(promptTemplateService).element
  )
}
