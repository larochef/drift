package drift.frontend.pages.projects

import drift.frontend.components.Component
import drift.frontend.pages.assistant.AssistantSessionControls
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The model bar: the pickers that launch or switch the image or video model
  * and the assistant, one per row, each with its live session's controls beside
  * it — at the top of a workspace, in the middle of a project that has nothing
  * yet (François, 2026-09-29), and at the top of the Sandbox
  * (`specs/47-sandbox.md`), so a session is driven from the same place wherever
  * it runs.
  */
class ModelBar(
    /** What is being made: text is a chat model alone, anything else an image
      * or video model and the assistant.
      */
    kind: Signal[Option[ProjectKind]],
    /** In the middle of an empty project, as the one thing to do there. */
    prominent: Boolean,
    sessionService: SessionService,
    assistantService: AssistantService,
    sessions: WorkspaceSessions,
    prerequisites: LaunchPrerequisites,
    /** The modal creating a configuration of a tool, for a project of a kind
      * (none: any), and what to do with the one created — a picker's last
      * entry.
      */
    newConfiguration: (
        RuntimeTool,
        Option[ProjectKind],
        RunConfiguration => Unit,
        () => Unit
    ) => HtmlElement,
    /** The project launching, recorded on the session; none from the Sandbox.
      */
    projectId: Option[String],
    /** Whether the live session, named by its id, may be stopped for the one
      * picked.
      */
    confirmReplace: String => Boolean,
    /** Beside the image or video picker: the live session's controls. */
    imageControls: Signal[Node],
    /** Whether the chat drawer the assistant's toggle drives is open. */
    drawerOpen: Var[Boolean],
    /** Said under the pickers, about the models they show. */
    notice: Signal[Node] = Val(emptyNode)
) extends Component {

  private def modelPicker(
      tool: RuntimeTool,
      heading: Signal[String],
      kind: Signal[Option[ProjectKind]],
      trailing: Signal[Node]
  ): HtmlElement =
    ModelPicker(
      tool,
      heading,
      kind,
      prominent,
      trailing,
      sessionService,
      sessions,
      prerequisites,
      newConfiguration,
      projectId,
      confirmReplace
    ).element

  /** The live chat model's controls; the drawer's toggle where it is the
    * assistant beside a generation panel rather than the page itself.
    */
  private def assistantControls(withDrawer: Boolean): Signal[Node] =
    sessions
      .liveKey(RuntimeTool.LlamaCpp)
      .map(_.map(_._1))
      .distinct
      .map {
        case None            => emptyNode
        case Some(sessionId) =>
          AssistantSessionControls(
            assistantService,
            () => sessionService.push(SessionService.Command.Stop(sessionId)),
            Option.when(withDrawer)(drawerOpen)
          ).element
      }

  lazy val element: HtmlElement = div(
    cls := (if (prominent) "workspace-model-bar is-prominent"
            else "workspace-model-bar"),
    // Text has no image model to pick, and its chat model is the whole of it
    // rather than an assistant (`specs/41-text-projects.md`).
    child <-- kind.map {
      case Some(ProjectKind.Text) =>
        div(
          cls := "is-flex is-flex-wrap-wrap",
          modelPicker(
            RuntimeTool.LlamaCpp,
            Val("Chat model"),
            Val(None),
            assistantControls(withDrawer = false)
          )
        )
      // One picker per row, whatever the notices beside them say: a row that
      // shortened while installing pulled the assistant up beside the image
      // model, and the header jumped about (François, 2026-09-28).
      case made =>
        div(
          cls := "is-flex is-flex-direction-column",
          styleAttr := "gap: 0.5rem;",
          modelPicker(
            RuntimeTool.SdCpp,
            Val(made match {
              case Some(ProjectKind.Video) => "Video model"
              case _                       => "Image model"
            }),
            kind,
            imageControls
          ),
          modelPicker(
            RuntimeTool.LlamaCpp,
            Val("Assistant"),
            Val(None),
            assistantControls(withDrawer = true)
          )
        )
    },
    child <-- notice
  )
}
