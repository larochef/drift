package drift.frontend.pages.settings

import drift.frontend.components.Component
import drift.frontend.services.RuntimeService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Runtime management (`specs/06-sdcpp-runtime.md`,
  * `specs/17-assistant-runtime.md`): the registered runtimes of both tools with
  * their validation state and the default per tool, the installs in progress,
  * and the modal that installs a new one — picking just a tool, a release and a
  * backend, the matching TheRock build being the backend's job — or adopts a
  * directory that is already on disk.
  */
class RuntimesSection(runtimeService: RuntimeService) extends Component {

  /** Per-runtime TheRock picks on the runtime rows (for a "Change ROCm" or an
    * upgrade whose release declares another ROCm), and which rows have the
    * change panel open. Section-level so a re-rendered list keeps them.
    */
  private val theRockPicks = Var(Map.empty[String, Option[String]])
  private val changeRocmOpen = Var(Set.empty[String])

  /** Strix Halo, the target machine — detection is an open question in the
    * spec, so a predictable default fills in.
    */
  private val gfxTarget = Var("gfx1151")

  private lazy val installForm: InstallRuntimeForm =
    InstallRuntimeForm(runtimeService, gfxTarget, () => addRuntime.hide())

  private lazy val runnerForm: InstallRunnerForm =
    InstallRunnerForm(runtimeService, () => addRuntime.hide())

  private lazy val adoptForm: AdoptRuntimeForm =
    AdoptRuntimeForm(runtimeService, () => addRuntime.hide())

  private lazy val addRuntime: AddRuntimeModal =
    AddRuntimeModal(installForm, runnerForm, adoptForm)

  lazy val element: HtmlElement = div(
    div(
      cls := "level",
      div(
        cls := "level-left",
        h2(cls := "is-size-5 text-primary", "Runtimes")
      ),
      div(
        cls := "level-right",
        button(
          cls := "button is-primary is-small",
          span(cls := "plus-icon", "+"),
          " Add a runtime",
          onClick --> (_ => addRuntime.show())
        )
      )
    ),
    div(
      children <-- runtimeService.runtimes
        .combineWith(runtimeService.selection, runtimeService.releases)
        .map { (runtimes, selection, releases) =>
          if (runtimes.isEmpty)
            List(
              p(
                cls := "text-secondary",
                "No runtimes yet. \"Add a runtime\" installs a release, or adopts a directory that is already on disk."
              )
            )
          else
            runtimes.map(runtime =>
              RuntimeRow(
                runtime,
                selection,
                releases,
                runtimeService,
                theRockPicks,
                changeRocmOpen,
                gfxTarget
              ).element
            )
        }
    ),
    RuntimeInstallJobs(runtimeService).element,
    addRuntime.element
  )
}
