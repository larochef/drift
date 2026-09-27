package drift.frontend.pages.projects

import drift.frontend.Page
import drift.frontend.components.{Component, ErrorBanner}
import drift.frontend.pages.gallery.GenerationCard
import drift.frontend.services.ProjectService
import drift.frontend.services.ProjectService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** The projects list (`specs/19-projects-and-prompt-versions.md`): one card per
  * project with its brief, version count and last use; a form for a new one;
  * delete with a confirmation naming what goes.
  */
class ProjectsPage(service: ProjectService) extends Component {
  private val showForm = Var(false)
  private val labelVar = Var("")
  private val briefVar = Var("")
  private val nsfwVar = Var(false)
  private val kindVar = Var(ProjectKind.Image)

  /** NSFW projects stay out of the list unless asked for (François,
    * 2026-09-08); the choice lasts for the page's life.
    */
  private val showNsfw = Var(false)

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
      service.push(
        Command.Create(
          Project(
            id = s"${slug(label)}-${now % 100000}",
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

  /** Two questions, because they have different answers (François, 2026-09-09):
    * the project always goes, and the images it made go with it only when that
    * is said out loud — deleting them removes the files and their gallery
    * entries, and nothing here is undoable.
    */
  private def confirmDelete(project: Project): Unit =
    if (
      window.confirm(
        s"Delete project '${project.label}' and its ${project.versions.size} versions?"
      )
    ) {
      val withGenerations = window.confirm(
        s"Also delete every image '${project.label}' generated?\n\n" +
          "OK deletes the image files and their gallery entries.\n" +
          "Cancel keeps them — they stay in the gallery, without a project."
      )
      service.push(Command.Delete(project.id, withGenerations))
    }

  /** One project as a tile: the newest image it made on top - what a project
    * actually looks like is the fastest way to recognise it (François,
    * 2026-09-10) - then the name, what it holds, and the actions.
    */
  private def card(project: Project): HtmlElement = div(
    cls := "card bg-card project-card",
    div(
      cls := "project-cover cursor-pointer",
      onClick --> (_ => Page.ProjectWorkspace(project.id).navigate()),
      // The placeholder sits underneath and the image covers it; a project
      // that has generated nothing answers 404, and the image hides itself
      // rather than showing the browser's broken-image glyph.
      span(
        cls := "text-secondary is-size-7 project-cover-empty",
        if (project.kind == ProjectKind.Text) "a conversation"
        else s"no ${project.kind.noun} yet"
      ),
      project.kind match {
        case ProjectKind.Image =>
          img(
            src := projectCoverPath(project.id),
            loadingAttr := "lazy",
            alt := s"Newest image of ${project.label}",
            inContext(node =>
              onError --> (_ => node.ref.style.setProperty("display", "none"))
            )
          )
        // The first frame, playing only while hovered: a grid of clips all
        // playing at once would keep the page repainting.
        case ProjectKind.Video =>
          videoTag(
            src := projectCoverPath(project.id),
            GenerationCard.preloadAttr := "metadata",
            GenerationCard.mutedAttr := true,
            GenerationCard.loopAttr := true,
            GenerationCard.playsInlineAttr := true,
            inContext(node =>
              Seq(
                onError --> (_ =>
                  node.ref.style.setProperty("display", "none")
                ),
                onMouseEnter --> (_ => { node.ref.play(); () }),
                onMouseLeave --> (_ => node.ref.pause())
              )
            )
          )
        // A conversation has no picture to show (`specs/41-text-projects.md`).
        case ProjectKind.Text => emptyNode
      },
      if (project.nsfw)
        span(cls := "tag is-danger is-small project-cover-badge", "nsfw")
      else emptyNode
    ),
    div(
      cls := "project-card-body",
      p(
        cls := "title is-6 text-primary mb-1",
        a(
          href := Page.ProjectWorkspace(project.id).path,
          cls := "text-primary",
          project.label
        )
      ),
      p(
        cls := "text-secondary is-size-7 mb-1",
        (if (project.kind == ProjectKind.Text) "text · "
         else s"${project.kind.noun} · ${project.versions.size} versions · ") +
          new scala.scalajs.js.Date(project.lastUsedAt.toDouble)
            .toLocaleDateString()
      ),
      if (project.brief.nonEmpty)
        p(cls := "text-secondary is-size-7 project-brief", project.brief)
      else emptyNode
    ),
    div(
      cls := "project-card-actions",
      // What the project makes, at a glance beside the way in (François,
      // 2026-09-15).
      span(
        cls := "project-card-kind",
        title := s"${project.kind.noun.capitalize} project",
        project.kind match {
          case ProjectKind.Image => "🖼️"
          case ProjectKind.Video => "🎬"
          case ProjectKind.Text  => "💬"
        }
      ),
      button(
        cls := "button is-primary is-small",
        "Open",
        onClick --> (_ => Page.ProjectWorkspace(project.id).navigate())
      ),
      button(
        cls := "button is-danger is-small",
        "🗑️",
        title := "Delete project",
        onClick --> (_ => confirmDelete(project))
      )
    )
  )

  private val loadingAttr =
    htmlAttr("loading", com.raquo.laminar.codecs.StringAsIsCodec)

  lazy val element: HtmlElement = div(
    cls := "content",
    service.effects,
    onMountCallback(_ => service.push(Command.Load)),
    service.events --> Observer[ProjectService.Event] {
      case ProjectService.Event.Created(project) =>
        labelVar.set("")
        briefVar.set("")
        nsfwVar.set(false)
        kindVar.set(ProjectKind.Image)
        showForm.set(false)
        Page.ProjectWorkspace(project.id).navigate()
      case _ => ()
    },
    ErrorBanner(service),
    div(
      cls := "level",
      div(cls := "level-left", h1(cls := "title text-primary", "Projects")),
      div(
        cls := "level-right",
        button(
          cls := "button is-primary",
          span(cls := "plus-icon", "+"),
          " New Project",
          onClick --> (_ => showForm.update(!_))
        )
      )
    ),
    hr(),
    label(
      cls := "checkbox text-secondary is-size-7 mb-3",
      input(
        typ := "checkbox",
        checked <-- showNsfw.signal,
        onChange.mapToChecked --> showNsfw
      ),
      " Show NSFW projects"
    ),
    child <-- showForm.signal.map {
      case false => emptyNode
      case true  =>
        div(
          cls := "box bg-card p-4 mb-4",
          div(
            cls := "field",
            label(cls := "label text-primary", "Name"),
            input(
              cls := "input",
              placeholder := "Fox poster",
              controlled(
                value <-- labelVar.signal,
                onInput.mapToValue --> labelVar
              )
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
              " NSFW — hidden from the list unless asked for; its LoRA picker shows the NSFW ones"
            )
          ),
          div(
            cls := "buttons",
            button(
              cls := "button is-success",
              "Create",
              disabled <-- labelVar.signal.map(_.trim.isEmpty),
              onClick --> (_ => create())
            ),
            button(
              cls := "button",
              "Cancel",
              onClick --> (_ => showForm.set(false))
            )
          )
        )
    },
    child <-- service.projects.combineWith(showNsfw.signal).map {
      (projects, _) =>
        if (projects.isEmpty)
          p(
            cls := "text-secondary",
            "No projects yet. A project keeps the versions of a prompt and every image they made, on whichever model."
          )
        else emptyNode
    },
    div(
      cls := "projects-grid",
      children <-- service.projects
        .combineWith(showNsfw.signal)
        .map { (projects, showNsfw) =>
          projects
            .filter(project => showNsfw || !project.nsfw)
            .map(card)
        }
    ),
    child <-- service.projects.combineWith(showNsfw.signal).map {
      (projects, showNsfw) =>
        val hidden = projects.count(project => !showNsfw && project.nsfw)
        if (hidden == 0) emptyNode
        else
          p(
            cls := "text-secondary is-size-7 mt-3",
            s"$hidden NSFW project${if (hidden == 1) "" else "s"} hidden."
          )
    }
  )
}
