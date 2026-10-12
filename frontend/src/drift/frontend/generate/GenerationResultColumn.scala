package drift.frontend.generate

import drift.frontend.components.Component
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The results beside the form of a ready session: what free play is, the step
  * preview's toggle, the jobs the session still owes, the generation on show
  * and, in a project, its history.
  */
final class GenerationResultColumn(
    capabilities: Option[SessionCapabilities],
    sessionSignal: Signal[Option[Session]],
    /** The machine's load and memory, shown under the bars. */
    machine: Signal[Option[MachineStatus]],
    service: GenerationService,
    assistantService: AssistantService,
    /** The projects a free-play result can be kept into (`specs/22-…`); unused
      * inside a workspace, where results are kept by definition.
      */
    projectService: ProjectService,
    configurationLabel: () => String,
    /** Where a Keep should put the result: "" is the gallery alone. */
    keepProjectVar: Var[String],
    project: Option[GenerationPanel.ProjectBinding],
    /** Opens one of this generation's outputs full screen, on pages that have a
      * detail view to open it in — the workspace's. A result is a result
      * wherever it is clicked (`specs/19-…`).
      */
    onOpenOutput: Option[(Generation, Int) => Unit]
) extends Component {

  /** Free play: outside a project nothing is kept. */
  private val scratch: Boolean = project.isEmpty

  /** What this panel shows as its result. The session outlives any one
    * workspace — its last generation may well belong to the project the user
    * was in before, which is what a brand new project showed as its own, and
    * what came back when the image actually made in it was deleted. Inside a
    * project, only that project's.
    */
  private val panelGenerations: Signal[List[Generation]] = project match {
    case None          => service.generations
    case Some(binding) =>
      service.generations.map(
        _.filter(_.projectId.contains(binding.projectId))
      )
  }

  /** What the result shows: the job generating while there is one, else the
    * next one waiting, else the newest. Showing the newest submission meant
    * that cancelling a queued job replaced a running generation on screen with
    * that cancellation — and showing the newest job still to do, waiting ones
    * included, that queueing a job took the running one's step preview and Stop
    * away for a job with nothing to show yet. A queued job is in the list above
    * the result.
    */
  private val shownGeneration: Signal[Option[Generation]] =
    panelGenerations
      .map(all =>
        all
          .find(_.status == GenerationStatus.Generating)
          .orElse(all.find(_.status.isActive))
          .orElse(all.lastOption)
      )
      .distinct

  /** What free play is, said once above the results, with the way to empty it.
    */
  private def freePlayNotice: Node =
    if (!scratch) emptyNode
    else
      div(
        cls := "notification is-warning is-light py-2 px-3 is-size-7 mb-3",
        span(
          "Sandbox — nothing here is saved. Save a result to put it in the " +
            "gallery or a project; everything else goes when you leave."
        ),
        button(
          cls := "button is-small ml-2",
          "Clear results",
          onClick --> (_ =>
            service.push(GenerationService.Command.ClearScratch)
          )
        )
      )

  lazy val element: HtmlElement = div(
    cls := "column is-7",
    freePlayNotice,
    StepPreviewToggle(service).element,
    // Everything the session still owes that the result does not
    // already show — jobs queued behind this one, and jobs queued
    // from another project, which hold this one up just the same.
    GenerationQueue(
      service.generations,
      shownGeneration.map(_.map(_.id)),
      capabilities,
      service
    ).element,
    child <-- shownGeneration
      .map {
        case None =>
          div(
            cls := "notification py-3 px-4",
            "Nothing generated yet — the result appears here."
          )
        case Some(generation) =>
          GenerationResult(
            generation,
            capabilities,
            sessionSignal,
            machine,
            service,
            assistantService,
            projectService,
            configurationLabel,
            scratch,
            keepProjectVar,
            onOpenOutput
          ).element
      },
    project.map(_.history)
  )
}
