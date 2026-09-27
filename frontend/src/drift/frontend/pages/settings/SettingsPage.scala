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
    promptTemplateService: PromptTemplateService
) extends Component {
  import RuntimeService.Command

  lazy val element: HtmlElement = div(
    cls := "content",
    runtimeService.effects,
    authTokenService.effects,
    promptTemplateService.effects,
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
    RuntimesSection(runtimeService).element,
    AuthenticationSection(authTokenService).element,
    PromptsSection(promptTemplateService).element
  )
}
