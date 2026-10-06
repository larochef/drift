package drift.frontend.pages.projects

import drift.frontend.Page
import drift.frontend.components.*
import drift.frontend.pages.gallery.GenerationDetailHost
import drift.frontend.pages.generate.GenerationPanel
import drift.frontend.services.*
import drift.frontend.services.ProjectService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*

/** One project's workspace (`specs/19-projects-and-prompt-versions.md`), laid
  * out by what is loaded (François, 2026-09-29): an empty project with no model
  * is the model bar alone, in the middle of the page; a project with versions
  * and no model is the bar over its version history; with an image model up, it
  * is the bar over the generation panel, the history under its result. The
  * assistant is a drawer on the right, opened from its picker. A text project's
  * workspace is the chat column alone (`specs/41-text-projects.md`).
  *
  * The pieces live beside this file (`specs/29-split-oversized-files.md`):
  * `WorkspaceHeader`, `VersionHistory`, and — shared with the Sandbox —
  * `WorkspaceBody`, which lays out the model bar, the panel and the drawer.
  */
class ProjectWorkspacePage(
    projectId: String,
    projectService: ProjectService,
    runConfigurationService: RunConfigurationService,
    sessionService: SessionService,
    generationService: GenerationService,
    assistantService: AssistantService,
    loraService: LoraService,
    // A result opened here is the gallery's detail view, with everything it
    // offers — reuse, upscale, resize, delete — so these come along.
    historyService: HistoryService,
    postProcessService: PostProcessService,
    upscalerService: UpscalerService,
    runtimeService: RuntimeService,
    logService: LogService,
    /** What each configuration still has to download: picking one offers the
      * download rather than a launch that cannot start.
      */
    prerequisites: LaunchPrerequisites,
    /** The path past `/projects/<id>`: the generation open in the detail view,
      * and which of its outputs (`GenerationDetailHost.boundToUrl`).
      */
    section: Signal[List[String]],
    /** For a picker's new configuration: the model browsers its form opens. */
    browsers: BrowserServices
) extends Component {

  /** What the live image model suggests for the assistant (`specs/32`): its
    * configuration's override, else its architecture's default, else drift's
    * helper; nothing while no image model is up.
    */
  private val suggestedTemplate: Signal[Option[String]] =
    assistantService.suggestedTemplateId

  private val project: Signal[Option[Project]] =
    projectService.project(projectId)
  private val generations: Signal[List[Generation]] =
    projectService.generationsOf(projectId)

  /** A text project is the conversation alone (`specs/41-text-projects.md`). */
  private val isText: Signal[Boolean] =
    project.map(_.exists(_.kind == ProjectKind.Text)).distinct

  /** Mirror of the project for synchronous reads in handlers. */
  private val currentProject = Var(Option.empty[Project])

  private val sessions =
    WorkspaceSessions(sessionService, runConfigurationService)

  /** The version submissions compare against: chosen by clicking, else the
    * newest — and the version a generation just created, so the next run
    * compares to what just ran.
    *
    * It is *not* what the form follows moment to moment. Seeding the form is a
    * separate act, done when the panel appears and when the user clicks a
    * version card: a generation appends a version, and the parameters have to
    * stay exactly as they are (François, 2026-09-10).
    */
  private val selectedVersionId = Var(Option.empty[String])
  private val selectedVersion: Signal[Option[PromptVersion]] = project
    .combineWith(selectedVersionId.signal)
    .map((project, id) =>
      project.flatMap(ProjectWorkspacePage.resolveVersion(_, id))
    )
    .distinct

  /** The version ids seen so far, to tell a version that has just been appended
    * from a project reloaded for any other reason.
    */
  private val knownVersionIds = Var(Set.empty[String])

  /** The result whose detail is open, if any — the gallery's modal, on the
    * project's own generations (François, 2026-09-09).
    */
  private val openDetail = Var(Option.empty[GenerationDetailHost.Open])

  /** Clicking a version card: it becomes what submissions compare against, and
    * — the only re-seed there is once a panel is up — its recipe is laid over
    * the form.
    */
  private def selectVersion(version: PromptVersion): Unit = {
    selectedVersionId.set(Some(version.id))
    body.panel.foreach(_.applyVersion(version))
  }

  private def openOutput(generationId: String, index: Int): Unit =
    openDetail.set(Some(GenerationDetailHost.Open(generationId, index)))

  private def setCover(output: Option[GenerationOutput]): Unit =
    currentProject
      .now()
      .foreach(current =>
        projectService.push(
          Command.Update(
            current.id,
            current.copy(cover = output.map(ProjectCover.of))
          )
        )
      )

  /** Whether the project has versions; none until it has loaded, so that
    * neither the start screen nor the top bar flashes by.
    */
  private val versionsKnown: Signal[Option[Boolean]] =
    project.map(_.map(_.versions.nonEmpty)).distinct

  private def history(size: VersionHistory.Size): HtmlElement =
    VersionHistory(
      size,
      project,
      generations,
      selectedVersion,
      () => selectedVersionId.now(),
      () => currentProject.now(),
      sessions.labelOf,
      assistantService,
      historyService,
      selectVersion,
      openOutput,
      setCover
    ).element

  /** What stands where the generation panel will, with no image model live: the
    * history — or, with nothing yet, the model bar in the middle of the page.
    */
  private def idle: HtmlElement =
    div(child <-- versionsKnown.map {
      case None        => emptyNode
      case Some(true)  => history(VersionHistory.Size.Large)
      case Some(false) =>
        div(
          cls := "workspace-start",
          h2(
            cls := "title is-4 text-primary",
            child.text <-- project.map(_.map(_.kind)).distinct.map {
              case Some(ProjectKind.Video) => "Pick a video model to start"
              case _                       => "Pick an image model to start"
            }
          ),
          p(
            cls := "text-secondary mb-5",
            "Every generation becomes a version of the project. The " +
              "assistant is optional: it talks the prompt over with you."
          ),
          body.modelBar(prominent = true)
        )
    })

  private def binding: GenerationPanel.ProjectBinding =
    GenerationPanel.ProjectBinding(
      projectId = projectId,
      nsfw = project.map(_.exists(_.nsfw)).distinct,
      selectedVersion = selectedVersion,
      selectedVersionNow = () =>
        // Resolved against the project, exactly as the signal resolves it: an
        // id the project no longer has would make the backend find no parent
        // and open a second "first version" line.
        currentProject
          .now()
          .flatMap(project =>
            ProjectWorkspacePage
              .resolveVersion(project, selectedVersionId.now())
          )
          .map(_.id),
      history = history(VersionHistory.Size.Small)
    )

  private val header = WorkspaceHeader(
    project,
    () => currentProject.now(),
    projectService
  )

  /** The model bar, the generation panel and the assistant — what the Sandbox
    * runs too, here bound to the project.
    */
  private val body: WorkspaceBody = WorkspaceBody(
    kind = project.map(_.map(_.kind)).distinct,
    projectId = Some(projectId),
    confirmReplace = _ => true,
    binding = () => Some(binding),
    // The result under the form opens where every other result on this page
    // opens: the gallery's own detail view (François, 2026-09-10).
    onOpenOutput =
      Some((generation, index) => openOutput(generation.id, index)),
    idle = () => idle,
    // Up top unless it is the middle of the page already.
    barOnTop = sessions
      .liveKey(RuntimeTool.SdCpp)
      .map(_.isDefined)
      .combineWith(versionsKnown)
      .map((live, versions) => live || versions.contains(true)),
    chatIdle = "Pick a chat model above to continue the conversation; it " +
      "is kept with the project.",
    freePlay = false,
    notice = Val(emptyNode),
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
    prerequisites = prerequisites,
    browsers = browsers
  )

  lazy val element: HtmlElement = div(
    cls := "content workspace",
    GenerationDetailHost.boundToUrl(
      Page.ProjectWorkspace(projectId).path,
      section,
      openDetail
    ),
    projectService.effects,
    runConfigurationService.effects,
    sessionService.effects,
    generationService.effects,
    loraService.effects,
    historyService.effects,
    postProcessService.effects,
    upscalerService.effects,
    prerequisites.effects,
    onMountCallback { _ =>
      projectService.push(Command.Load)
      projectService.push(Command.LoadGenerations(projectId))
      // The project's own conversation, kept across visits (`specs/20`).
      assistantService.bindProject(projectId)
      runConfigurationService.push(RunConfigurationService.Command.Load)
      runConfigurationService.architectureService.push(
        drift.frontend.services.ArchitectureService.Command.Load
      )
      sessionService.push(SessionService.Command.Load)
      // What the detail view's scale rows need.
      postProcessService.push(PostProcessService.Command.LoadJobs)
      upscalerService.push(UpscalerService.Command.Load)
    },
    // A finished upscale/resize inherits this project, so it belongs beside
    // its source in the results.
    postProcessService.completions --> Observer[Generation](
      projectService.adopt
    ),
    // A deletion is the history's to do; the results only have to follow it.
    historyService.events --> Observer[HistoryService.Event] {
      case HistoryService.Event.Deleted(id) => projectService.forget(id)
      case HistoryService.Event.Imported(_) => ()
      case HistoryService.Event.Moved(list) => projectService.moved(list)
    },
    onUnmountCallback { _ =>
      assistantService.projectBrief.set(None)
      assistantService.unbindProject()
    },
    project --> currentProject,
    project.map(_.map(_.brief)).distinct --> Observer[Option[String]] { brief =>
      assistantService.projectBrief.set(brief)
    },
    isText --> assistantService.rawModel,
    // The project's prompt templates (`specs/32`): the assistant follows the
    // project's own pick, else the target architecture's default, else
    // drift's helper; a pick in the panel is saved back to the project.
    project
      .map(_.map(p => (p.assistantTemplateId, p.compactionTemplateId)))
      .combineWith(suggestedTemplate)
      .distinct --> Observer[
      (Option[(Option[String], Option[String])], Option[String])
    ] {
      case (Some((assistant, compaction)), suggested) =>
        assistantService.templateId.set(
          Some(
            assistant
              .orElse(suggested)
              .getOrElse(PromptTemplate.DefaultAssistantId)
          )
        )
        assistantService.compactionTemplateId.set(compaction)
      case _ => ()
    },
    assistantService.templateId.signal.distinct --> Observer[Option[String]] {
      chosen =>
        currentProject.now().foreach { p =>
          val stored = Some(
            p.assistantTemplateId
              .orElse(assistantService.suggestedTemplateNow)
              .getOrElse(PromptTemplate.DefaultAssistantId)
          )
          if (chosen.isDefined && chosen != stored)
            projectService.push(
              Command.Update(p.id, p.copy(assistantTemplateId = chosen))
            )
        }
    },
    // A version created by a generation becomes the selected one, so the next
    // run compares to what just ran — read off the emitted project, not a
    // mirror, which lags one propagation behind. Only a version that was not
    // there before moves the selection: a project reloaded for any other
    // reason must not pull it off the version the user picked, which would
    // make the next run append one for a recipe the project already has.
    project
      .map(_.map(_.versions).getOrElse(Nil))
      .distinct --> Observer[List[PromptVersion]] { versions =>
      val known = knownVersionIds.now()
      knownVersionIds.set(versions.map(_.id).toSet)
      if (known.isEmpty)
        // The first load is the baseline, not an arrival: follow the newest,
        // which is what the panel seeds its form from.
        selectedVersionId.set(versions.lastOption.map(_.id))
      else
        versions
          .filterNot(version => known.contains(version.id))
          .lastOption
          .foreach(version => selectedVersionId.set(Some(version.id)))
    },
    ErrorBanner(projectService),
    ErrorBanner(sessionService),
    ErrorBanner(generationService),
    GenerationDetailHost(
      openDetail,
      generations,
      historyService,
      runConfigurationService,
      sessionService,
      generationService,
      assistantService,
      postProcessService,
      upscalerService,
      runtimeService,
      prerequisites,
      projectService.projects,
      launchingProject = Some(projectId),
      // Both columns are already on screen here: staging into them is the
      // whole action, and the modal steps out of the way.
      onReuseStaged = () => {
        openDetail.set(None)
        body.panel
          .foreach(panel =>
            generationService.takePendingReuse().foreach { reuse =>
              panel.applyReuse(reuse)
              // The form now holds that generation's recipe, so the version it
              // belongs to is what the next run must compare against —
              // otherwise it appends one for a recipe the project already has.
              val staged = reuse match {
                case GenerationService.Reuse.Full(generation) => generation
                case GenerationService.Reuse.Task(generation) => generation
              }
              staged.promptVersionId
                .foreach(id => selectedVersionId.set(Some(id)))
            }
          )
      },
      onAssistantStaged = () => openDetail.set(None)
    ),
    header.element,
    body.element
  )
}

object ProjectWorkspacePage {

  /** The version an id names, falling back to the newest — the one rule, so
    * that what the form was seeded with and what the submit names cannot drift
    * apart on a stale id.
    */
  def resolveVersion(
      project: Project,
      id: Option[String]
  ): Option[PromptVersion] =
    id.flatMap(i => project.versions.find(_.id == i))
      .orElse(project.versions.lastOption)
}
