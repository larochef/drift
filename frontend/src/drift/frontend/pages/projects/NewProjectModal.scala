package drift.frontend.pages.projects

import drift.frontend.Page
import drift.frontend.components.{
  BrowserModal,
  Component,
  ErrorBanner,
  Showcase
}
import drift.frontend.services.ProjectService
import drift.frontend.services.ProjectService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Creating a project (`specs/19-projects-and-prompt-versions.md`): its name,
  * brief, kind and NSFW flag in a modal, from the projects page and from the
  * gallery's invitations to start one. A created project opens its workspace.
  * The fields outlive the modal, so closing and reopening keeps what was typed.
  */
class NewProjectModal(service: ProjectService, onClose: () => Unit)
    extends Component {
  private val labelVar = Var("")
  private val briefVar = Var("")
  private val nsfwVar = Var(false)
  private val kindVar = Var(ProjectKind.Image)

  /** The id sent, so only its own creation opens a workspace. */
  private var awaiting = Option.empty[String]

  private def slug(label: String): String =
    label.toLowerCase
      .map(c => if (c.isLetterOrDigit) c else '-')
      .split('-')
      .filter(_.nonEmpty)
      .mkString("-")

  private def create(): Unit = {
    val label = labelVar.now().trim
    if (label.nonEmpty) {
      val now = System.currentTimeMillis()
      val id = s"${slug(label)}-${now % 100000}"
      awaiting = Some(id)
      service.push(
        Command.Create(
          Project(
            id = id,
            label = label,
            brief = briefVar.now().trim,
            createdAt = now,
            lastUsedAt = now,
            nsfw = nsfwVar.now(),
            kind = kindVar.now()
          )
        )
      )
    }
  }

  lazy val element: HtmlElement = BrowserModal(
    title = Val("New project"),
    onCancel = onClose,
    modalMods = Seq(
      documentEvents(_.onKeyDown).filter(_.key == "Escape")
        --> (_ => onClose()),
      service.events --> Observer[ProjectService.Event] {
        case ProjectService.Event.Created(project)
            if awaiting.contains(project.id) =>
          awaiting = None
          labelVar.set("")
          briefVar.set("")
          nsfwVar.set(false)
          kindVar.set(ProjectKind.Image)
          onClose()
          Page.ProjectWorkspace(project.id).navigate()
        case _ => ()
      }
    ),
    body = Seq(
      ErrorBanner(service),
      div(
        cls := "field",
        label(cls := "label text-primary", "Name"),
        input(
          cls := "input",
          placeholder := "Fox poster",
          controlled(
            value <-- labelVar.signal,
            onInput.mapToValue --> labelVar
          ),
          onKeyDown.filter(_.key == "Enter") --> (_ => create()),
          onMountCallback(context => context.thisNode.ref.focus())
        )
      ),
      div(
        cls := "field",
        label(cls := "label text-primary", "Brief"),
        textArea(
          cls := "textarea",
          rows := 3,
          placeholder :=
            "What is being made, in your words — the assistant reads this",
          controlled(
            value <-- briefVar.signal,
            onInput.mapToValue --> briefVar
          )
        )
      ),
      div(
        cls := "field",
        label(cls := "label text-primary", "Makes"),
        div(
          cls := "control",
          ProjectKind.values.toList.map(kind =>
            label(
              cls := "radio text-primary mr-4",
              input(
                typ := "radio",
                nameAttr := "project-kind",
                checked <-- kindVar.signal.map(_ == kind),
                onChange --> (_ => kindVar.set(kind))
              ),
              s" ${kind.noun.capitalize}s"
            )
          )
        ),
        p(
          cls := "help text-secondary",
          "The workspace offers only the models that make them; " +
            "changeable later"
        )
      ),
      div(
        cls := "field",
        label(
          cls := "checkbox text-primary",
          input(
            typ := "checkbox",
            checked <-- nsfwVar.signal,
            onChange.mapToChecked --> nsfwVar
          ),
          " NSFW — hidden from the list unless asked for; its LoRA picker " +
            "shows the NSFW ones"
        )
      )
    ),
    footerRight = div(
      cls := "buttons",
      button(
        cls := "button is-success",
        span(cls := "plus-icon", "+"),
        " Create",
        disabled <-- labelVar.signal.map(_.trim.isEmpty),
        onClick --> (_ => create())
      ),
      BrowserModal.cancelButton(onClose)
    )
  ).element
}

object NewProjectModal {

  /** The invitation an empty page shows (François, 2026-09-28): what a project
    * is, in the middle of the page, with the button that starts one — and, when
    * given, the other way in — over a mosaic of what drift makes.
    */
  def invitation(
      heading: String,
      paragraphs: List[String],
      onCreate: () => Unit,
      secondary: Option[(String, () => Unit)] = None
  ): HtmlElement =
    div(
      cls := "empty-invitation",
      h2(cls := "title is-4 text-primary", heading),
      paragraphs.map(text => p(cls := "text-secondary mb-3", text)),
      div(
        cls := "buttons is-centered mt-5",
        button(
          cls := "button is-primary is-medium",
          span(cls := "plus-icon", "+"),
          " Create a project",
          onClick --> (_ => onCreate())
        ),
        secondary.map((text, action) =>
          button(
            cls := "button is-medium",
            text,
            onClick --> (_ => action())
          )
        )
      ),
      Showcase().element
    )
}
