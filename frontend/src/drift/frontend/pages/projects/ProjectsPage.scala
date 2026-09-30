package drift.frontend.pages.projects

import drift.frontend.Page
import drift.frontend.components.{Component, ErrorBanner}
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
  private lazy val newProject =
    NewProjectModal(service, () => showForm.set(false))

  /** NSFW projects stay out of the list unless asked for (François,
    * 2026-09-08); the choice lasts for the page's life.
    */
  private val showNsfw = Var(false)

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
        // A video project's cover is its newest video's first frame, a JPEG
        // like an image project's: the list loads no video (bug 37).
        case ProjectKind.Image | ProjectKind.Video =>
          img(
            src := projectCoverPath(project.id),
            loadingAttr := "lazy",
            alt := s"Newest ${project.kind.noun} of ${project.label}",
            inContext(node =>
              onError --> (_ => node.ref.style.setProperty("display", "none"))
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
          onClick --> (_ => showForm.set(true))
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
    child <-- showForm.signal.map(if (_) newProject.element else emptyNode),
    child <-- service.projectsLoaded
      .combineWith(service.projects.map(_.isEmpty))
      .distinct
      .map {
        case (true, true) =>
          NewProjectModal.invitation(
            "Start your first project",
            List(
              "A project is one thing you are making — a poster, a " +
                "character, a short clip. It keeps every version of your " +
                "prompt and everything each version made, whichever model " +
                "ran it, so you can go back, compare and carry on.",
              "Give it a name and a brief in your own words: the assistant " +
                "reads the brief to help you write prompts. Then pick an " +
                "image or video model at the top of the workspace — or " +
                "create one there — and generate."
            ),
            () => showForm.set(true),
            Some("Just try a model" -> (() => Page.Sandbox.navigate()))
          )
        case _ => emptyNode
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
