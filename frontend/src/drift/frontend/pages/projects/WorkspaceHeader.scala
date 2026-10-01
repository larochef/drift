package drift.frontend.pages.projects

import drift.frontend.Page
import drift.frontend.components.Component
import drift.frontend.services.*
import drift.frontend.services.ProjectService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The workspace's header: the project's name and brief, both edited in place,
  * its kind and NSFW flag, the way back to the list; and apart from it, the
  * model bar whose pickers launch or switch the image or video model and the
  * assistant — at the top of the page, or in its middle while the project has
  * nothing yet (François, 2026-09-29).
  */
class WorkspaceHeader(
    projectId: String,
    project: Signal[Option[Project]],
    /** The project as it stands, for the handlers. */
    currentProject: () => Option[Project],
    projectService: ProjectService,
    sessionService: SessionService,
    sessions: WorkspaceSessions,
    prerequisites: LaunchPrerequisites,
    /** The modal creating a configuration of a tool, for a project of a kind
      * (none: any), and what to do with the one created — a picker's last entry.
      */
    newConfiguration: (
        RuntimeTool,
        Option[ProjectKind],
        RunConfiguration => Unit,
        () => Unit
    ) => HtmlElement,
    /** Beside the image or video picker: the live session's controls. */
    imageControls: Signal[Node],
    /** Beside the assistant picker: the chat drawer's toggle, Stop and what the
      * server applied.
      */
    assistantControls: Signal[Node],
    /** Beside a text project's chat model picker: Stop and what the server
      * applied — the conversation is the page, so no drawer.
      */
    chatModelControls: Signal[Node]
) extends Component {

  // Editing state apart from the text: the signal that builds an editor
  // must not carry the draft, or every keystroke rebuilds the input and
  // drops the focus (François, 2026-09-08).
  private val labelEditing = Var(false)
  private val labelDraft = Var("")
  private val briefEditing = Var(false)
  private val briefDraft = Var("")

  private val projectKind: Signal[Option[ProjectKind]] =
    project.map(_.map(_.kind)).distinct

  private def saveLabel(): Unit = {
    val draft = labelDraft.now().trim
    if (draft.nonEmpty)
      currentProject().foreach(current =>
        projectService.push(
          Command.Update(current.id, current.copy(label = draft))
        )
      )
    labelEditing.set(false)
  }

  private def saveBrief(): Unit = {
    currentProject().foreach(current =>
      projectService.push(
        Command
          .Update(current.id, current.copy(brief = briefDraft.now().trim))
      )
    )
    briefEditing.set(false)
  }

  /** An in-place editor: the input is built once per editing session and bound
    * to its draft, so typing never rebuilds it.
    */
  private def inlineEditor(
      draft: Var[String],
      placeholderText: String,
      onSave: () => Unit,
      onCancel: () => Unit,
      small: Boolean
  ): HtmlElement = div(
    cls := "field has-addons mb-1",
    div(
      cls := "control is-expanded",
      input(
        cls := (if (small) "input is-small" else "input"),
        placeholder := placeholderText,
        controlled(value <-- draft.signal, onInput.mapToValue --> draft),
        onKeyDown.filter(_.key == "Enter") --> (_ => onSave()),
        onKeyDown.filter(_.key == "Escape") --> (_ => onCancel()),
        onMountCallback(context => context.thisNode.ref.focus())
      )
    ),
    div(
      cls := "control",
      button(
        cls := (if (small) "button is-small is-primary"
                else "button is-primary"),
        "Save",
        disabled <-- draft.signal.map(_.trim.isEmpty),
        onClick --> (_ => onSave())
      )
    ),
    div(
      cls := "control",
      button(
        cls := (if (small) "button is-small" else "button"),
        "Cancel",
        onClick --> (_ => onCancel())
      )
    )
  )

  private def modelPicker(
      tool: RuntimeTool,
      heading: Signal[String],
      kind: Signal[Option[ProjectKind]],
      prominent: Boolean,
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
      projectId = Some(projectId),
      confirmReplace = _ => true
    ).element

  lazy val element: HtmlElement = div(
    cls := "mb-3",
    div(
      cls := "level is-marginless",
      div(
        cls := "level-left",
        div(
          // The name, renamed in place; only the editing flag rebuilds this.
          child <-- project
            .map(_.map(_.label))
            .distinct
            .combineWith(labelEditing.signal)
            .map {
              case (Some(label), false) =>
                h1(
                  cls := "title text-primary mb-1",
                  label,
                  a(
                    cls := "ml-2 is-size-7",
                    title := "Rename project",
                    "rename",
                    onClick --> { _ =>
                      labelDraft.set(label)
                      labelEditing.set(true)
                    }
                  )
                )
              case (Some(_), true) =>
                inlineEditor(
                  labelDraft,
                  "Project name",
                  () => saveLabel(),
                  () => labelEditing.set(false),
                  small = false
                )
              case _ => h1(cls := "title text-primary mb-1", "Project")
            },
          child <-- project
            .map(_.map(_.brief))
            .distinct
            .combineWith(briefEditing.signal)
            .map {
              case (Some(brief), false) =>
                p(
                  cls := "text-secondary is-size-7 mb-0",
                  if (brief.isEmpty) "no brief yet" else brief,
                  a(
                    cls := "ml-2",
                    "edit",
                    onClick --> { _ =>
                      briefDraft.set(brief)
                      briefEditing.set(true)
                    }
                  )
                )
              case (Some(_), true) =>
                inlineEditor(
                  briefDraft,
                  "What is being made — the assistant reads this",
                  () => saveBrief(),
                  () => briefEditing.set(false),
                  small = true
                )
              case _ => emptyNode
            }
        )
      ),
      div(
        cls := "level-right",
        select(
          cls := "select is-small mr-3",
          title := "What the project makes — the model picker offers only " +
            "the models that make it",
          onChange.mapToValue --> Observer[String] { value =>
            ProjectKind.values
              .find(_.toString == value)
              .foreach(kind =>
                currentProject()
                  .filter(_.kind != kind)
                  .foreach(current =>
                    projectService.push(
                      Command.Update(current.id, current.copy(kind = kind))
                    )
                  )
              )
          },
          ProjectKind.values.toList.map(kind =>
            option(
              value := kind.toString,
              selected <-- project.map(_.exists(_.kind == kind)),
              s"${kind.noun.capitalize} project"
            )
          )
        ),
        label(
          cls := "checkbox text-secondary is-size-7 mr-3",
          title :=
            "An NSFW project is hidden from the list unless asked for, and its LoRA picker shows the NSFW ones",
          input(
            typ := "checkbox",
            checked <-- project.map(_.exists(_.nsfw)),
            onChange.mapToChecked --> Observer[Boolean] { checked =>
              currentProject().foreach(current =>
                projectService.push(
                  Command.Update(current.id, current.copy(nsfw = checked))
                )
              )
            }
          ),
          " NSFW"
        ),
        button(
          cls := "button is-small",
          "← Projects",
          onClick --> (_ => Page.Projects.navigate())
        )
      )
    )
  )

  /** The pickers, one per row: at the top of the page, or — `prominent` — in
    * the middle of an empty project, as the one thing to do there.
    */
  def modelBar(prominent: Boolean): HtmlElement = div(
    cls := (if (prominent) "workspace-model-bar is-prominent"
            else "workspace-model-bar"),
    // A text project has no image model to pick, and its chat model is
    // the whole of it rather than an assistant (`specs/41-text-projects.md`).
    child <-- projectKind.map {
      case Some(ProjectKind.Text) =>
        div(
          cls := "is-flex is-flex-wrap-wrap",
          modelPicker(
            RuntimeTool.LlamaCpp,
            Val("Chat model"),
            Val(None),
            prominent,
            chatModelControls
          )
        )
      // One picker per row, whatever the notices beside them say: a row that
      // shortened while installing pulled the assistant up beside the image
      // model, and the header jumped about (François, 2026-09-28).
      case kind =>
        div(
          cls := "is-flex is-flex-direction-column",
          styleAttr := "gap: 0.5rem;",
          modelPicker(
            RuntimeTool.SdCpp,
            Val(kind match {
              case Some(ProjectKind.Video) => "Video model"
              case _                       => "Image model"
            }),
            projectKind,
            prominent,
            imageControls
          ),
          modelPicker(
            RuntimeTool.LlamaCpp,
            Val("Assistant"),
            Val(None),
            prominent,
            assistantControls
          )
        )
    }
  )
}
