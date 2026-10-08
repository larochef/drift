package drift.frontend.pages.sandbox

import drift.frontend.Page
import drift.frontend.components.*
import drift.frontend.pages.projects.{WorkspaceBody, WorkspaceSessions}
import drift.frontend.services.*
import drift.shared.*

import scala.scalajs.js

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** The Sandbox (`specs/47-sandbox.md`): a model tried out, nothing kept. Shaped
  * like a project workspace without the project — no brief, no versions — with
  * an Image / Video / Text switch over that kind's workspace: the model bar
  * over the generation panel, the assistant a drawer on its right; for text,
  * the bar over the chat alone. A result is saved to the gallery or a project
  * one by one; the rest goes as the page is left.
  */
class SandboxPage(
    runConfigurationService: RunConfigurationService,
    sessionService: SessionService,
    generationService: GenerationService,
    assistantService: AssistantService,
    loraService: LoraService,
    /** The projects a result can be saved into, and who a shared model runs
      * for.
      */
    projectService: ProjectService,
    /** The gallery, which inputs are picked from (`specs/50`). */
    historyService: HistoryService,
    runtimeService: RuntimeService,
    logService: LogService,
    machineService: MachineService,
    prerequisites: LaunchPrerequisites,
    /** The path after `/sandbox/`: its first segment names the kind. */
    section: Signal[List[String]],
    browsers: BrowserServices
) extends Component {

  private val sessions =
    WorkspaceSessions(sessionService, runConfigurationService)

  // Mirrors, for the handlers.
  private val sessionsNow = Var(Map.empty[String, Session])
  private val projectsNow = Var(List.empty[Project])
  private val configurationsNow = Var(List.empty[RunConfiguration])

  /** What a live session makes, when the path does not say: images with nothing
    * live, and `None` until the configuration of the live one has loaded — the
    * page must not mount one kind's panel, handing it a staged reuse, and then
    * flip to another.
    */
  private val liveKind: Signal[Option[ProjectKind]] =
    sessionService.sessions
      .combineWith(runConfigurationService.runConfigurations)
      .combineWith(runConfigurationService.architectures)
      .map { (sessions, configurations, architectures) =>
        val live = sessions.values.filter(_.status.isActive).toList
        def kindOf(session: Session): Option[ProjectKind] =
          configurations
            .find(_.id == session.runConfigurationId)
            .flatMap(c => architectures.find(_.id == c.architectureId))
            .map(SandboxPage.kindOf)
        live
          .find(_.tool == RuntimeTool.SdCpp)
          .orElse(live.headOption)
          .fold(Option(ProjectKind.Image))(kindOf)
      }
      .distinct

  /** The kind shown; `None` while it cannot be told yet. */
  private val kind: Signal[Option[ProjectKind]] =
    section
      .map(_.headOption.flatMap(k => ProjectKind.values.find(_.noun == k)))
      .combineWith(liveKind)
      .map((named, live) => named.orElse(live))
      .distinct

  private val kindNow = Var(Option.empty[ProjectKind])

  /** Whether leaving would lose anything: a result not saved, or a chat. */
  private val unsavedResults: Signal[Boolean] =
    generationService.generations.map(_.exists(_.scratch)).distinct
  private val unsavedResultsNow = Var(false)
  private val unsaved: Signal[Boolean] =
    unsavedResults
      .combineWith(assistantService.turns.map(_.nonEmpty))
      .map(_ || _)
      .distinct
  private val unsavedNow = Var(false)

  private def projectLabel(id: String): String =
    projectsNow.now().find(_.id == id).map(_.label).getOrElse("a project")

  private def liveSession(tool: RuntimeTool): Option[Session] =
    sessionsNow.now().values.find(s => s.status.isActive && s.tool == tool)

  /** Switching kind stops the image or video model and resets the chat, once
    * the user agrees to what goes.
    */
  private def switchTo(target: ProjectKind): Unit =
    if (!kindNow.now().contains(target)) {
      val model = liveSession(RuntimeTool.SdCpp)
      val chat = assistantService.turnsNow.nonEmpty
      val losses =
        model.map(session =>
          s"stops ${configurationLabel(session.runConfigurationId)}" +
            session.projectId.fold("")(id =>
              s", which ${projectLabel(id)} is using"
            )
        ) ++
          Option.when(chat)("clears the chat") ++
          Option.when(unsavedResultsNow.now())(
            "throws away the results not saved"
          )
      if (
        losses.isEmpty ||
        dom.window.confirm(
          s"Switching to ${target.noun} ${losses.mkString(", ")}."
        )
      ) {
        model.foreach(session =>
          sessionService.push(SessionService.Command.Stop(session.id))
        )
        generationService.clearScratch()
        assistantService.clearScratch()
        Page.Sandbox.navigateTo(target)
      }
    }

  private def configurationLabel(id: String): String =
    configurationsNow
      .now()
      .find(_.id == id)
      .map(_.label)
      .getOrElse(id)

  /** A project's model stops only once the user agrees. */
  private def confirmReplace(sessionId: String): Boolean =
    sessionsNow
      .now()
      .values
      .find(_.id == sessionId)
      .flatMap(_.projectId) match {
      case None            => true
      case Some(projectId) =>
        dom.window.confirm(
          s"${projectLabel(projectId)} is using this model. Stop it and " +
            "load the one picked?"
        )
    }

  /** The models a tab runs: its own, and for images and video the assistant.
    */
  private def toolsOf(kind: ProjectKind): List[RuntimeTool] = kind match {
    case ProjectKind.Text => List(RuntimeTool.LlamaCpp)
    case _                => List(RuntimeTool.SdCpp, RuntimeTool.LlamaCpp)
  }

  /** The models the Sandbox shares with a project, said once under the pickers.
    */
  private def sharedNotice(kind: ProjectKind): Signal[Node] =
    sessionService.sessions
      .map(live =>
        toolsOf(kind).flatMap(tool =>
          live.values
            .find(s => s.status.isActive && s.tool == tool)
            .flatMap(_.projectId)
            .map(tool -> _)
        )
      )
      .distinct
      .combineWith(projectService.projects)
      .map { (shared, projects) =>
        if (shared.isEmpty) emptyNode
        else
          div(shared.map { (tool, projectId) =>
            val assistant =
              kind != ProjectKind.Text && tool == RuntimeTool.LlamaCpp
            p(
              cls := "text-secondary is-size-7 mt-2 mb-0",
              if (assistant) "The assistant is running for "
              else "Running for ",
              a(
                href := Page.ProjectWorkspace(projectId).path,
                projects
                  .find(_.id == projectId)
                  .map(_.label)
                  .getOrElse("a project")
              ),
              " — shared rather than loaded twice. " +
                (if (assistant)
                   "The chat here is the Sandbox's own, not the project's."
                 else "What you make here is still not saved.")
            )
          })
      }

  /** The workspace of a kind, with no project behind it: nothing is bound, so
    * what it makes is scratch.
    */
  private def body(kind: ProjectKind): HtmlElement =
    WorkspaceBody(
      kind = Val(Some(kind)),
      projectId = None,
      confirmReplace = confirmReplace,
      binding = () => None,
      onOpenOutput = None,
      idle = () =>
        p(
          cls := "text-secondary mt-4",
          s"Pick ${if (kind == ProjectKind.Image) "an" else "a"} " +
            s"${kind.noun} model above to try it. Nothing made here is " +
            "kept unless you save it. The assistant is optional: it talks " +
            "the prompt over with you."
        ),
      barOnTop = Val(true),
      chatIdle = "Pick a chat model above to try it. The conversation goes " +
        "when you leave.",
      freePlay = true,
      notice = sharedNotice(kind),
      sessions = sessions,
      runConfigurationService = runConfigurationService,
      sessionService = sessionService,
      generationService = generationService,
      assistantService = assistantService,
      loraService = loraService,
      projectService = projectService,
      historyService = historyService,
      runtimeService = runtimeService,
      logService = logService,
      machineService = machineService,
      prerequisites = prerequisites,
      browsers = browsers
    ).element

  private def tab(target: ProjectKind, label: String): HtmlElement = li(
    cls <-- kind.map(current =>
      if (current.contains(target)) "is-active" else ""
    ),
    a(label, onClick.preventDefault --> (_ => switchTo(target)))
  )

  /** In-app links leave through here: asked first while something would be
    * lost. The capture phase runs before frontroute's own link handling.
    */
  // Kept as JS functions: the same object has to be removed as was added.
  private val guardLinks: js.Function1[dom.MouseEvent, Unit] = event =>
    if (unsavedNow.now())
      Option(event.target)
        .collect { case element: dom.Element => element.closest("a[href]") }
        .flatMap(Option(_))
        .map(_.getAttribute("href"))
        .filter(href =>
          href.startsWith("/") && !href.startsWith(Page.Sandbox.path)
        )
        .foreach { _ =>
          if (!dom.window.confirm(SandboxPage.LeaveWarning)) {
            event.preventDefault()
            event.stopImmediatePropagation()
          }
        }

  private val guardUnload: js.Function1[dom.BeforeUnloadEvent, Unit] = event =>
    if (unsavedNow.now()) {
      event.preventDefault()
      event.returnValue = SandboxPage.LeaveWarning
    }

  lazy val element: HtmlElement = div(
    cls := "content sandbox",
    runConfigurationService.effects,
    sessionService.effects,
    generationService.effects,
    loraService.effects,
    projectService.effects,
    historyService.effects,
    prerequisites.effects,
    sessionService.sessions --> sessionsNow,
    projectService.projects --> projectsNow,
    runConfigurationService.runConfigurations --> configurationsNow,
    kind --> kindNow,
    unsaved --> unsavedNow,
    unsavedResults --> unsavedResultsNow,
    onMountCallback { _ =>
      runConfigurationService.push(RunConfigurationService.Command.Load)
      runConfigurationService.architectureService.push(
        ArchitectureService.Command.Load
      )
      sessionService.push(SessionService.Command.Load)
      projectService.push(ProjectService.Command.Load)
      loraService.push(LoraService.Command.Load)
      dom.window.addEventListener("click", guardLinks, useCapture = true)
      dom.window.addEventListener("beforeunload", guardUnload)
    },
    // Not persistent: what was not saved goes as the page is left.
    onUnmountCallback { _ =>
      dom.window.removeEventListener("click", guardLinks, useCapture = true)
      dom.window.removeEventListener("beforeunload", guardUnload)
      generationService.clearScratch()
      assistantService.clearScratch()
    },
    ErrorBanner(sessionService),
    ErrorBanner(generationService),
    div(
      cls := "level mb-3",
      div(
        cls := "level-left",
        div(
          h1(cls := "title text-primary mb-1", "Sandbox"),
          p(
            cls := "text-secondary is-size-7 mb-0",
            "Try a model. Nothing here is kept: save a result to the gallery " +
              "or a project, the rest goes when you leave."
          )
        )
      )
    ),
    div(
      cls := "tabs",
      ul(
        tab(ProjectKind.Image, "Image"),
        tab(ProjectKind.Video, "Video"),
        tab(ProjectKind.Text, "Text")
      )
    ),
    child <-- kind.map(_.fold[Node](emptyNode)(body))
  )
}

object SandboxPage {

  val LeaveWarning: String =
    "Leave the Sandbox? What you made here and did not save will be lost."

  /** What an architecture's models make: text for a chat model, video for a
    * video model that makes no images, images otherwise.
    */
  def kindOf(architecture: Architecture): ProjectKind =
    if (architecture.tool == RuntimeTool.LlamaCpp) ProjectKind.Text
    else if (
      ProjectKind.Video.accepts(architecture) &&
      !ProjectKind.Image.accepts(architecture)
    ) ProjectKind.Video
    else ProjectKind.Image
}
