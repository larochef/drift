package drift.frontend.pages.projects

import drift.frontend.components.Component
import drift.frontend.pages.assistant.{AssistantPanel, AssistantTurns}
import drift.frontend.pages.gallery.GalleryPicker
import drift.frontend.pages.generate.GenerationPanel
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What a workspace runs, under whatever heads the page: the model bar over the
  * generation panel, the assistant a drawer on its right — or, for text, the
  * bar over the chat alone (`specs/41-text-projects.md`). A project's workspace
  * and the Sandbox (`specs/47-sandbox.md`) are both this, the Sandbox with no
  * project behind it; what differs between them comes in as parameters.
  */
class WorkspaceBody(
    /** What is being made: text is the chat alone. */
    kind: Signal[Option[ProjectKind]],
    /** The project the models are launched for; none from the Sandbox. */
    projectId: Option[String],
    /** Whether the live session, named by its id, may be stopped for the one
      * picked.
      */
    confirmReplace: String => Boolean,
    /** The project's hooks into a generation panel, built for each panel; none
      * makes its results scratch.
      */
    binding: () => Option[GenerationPanel.ProjectBinding],
    /** Opens a result's output where the page has a detail view for it. */
    onOpenOutput: Option[(Generation, Int) => Unit],
    /** What stands in for the generation panel while no image or video model is
      * live.
      */
    idle: () => HtmlElement,
    /** Whether the model bar sits at the top — not while `idle` shows it in the
      * middle of the page.
      */
    barOnTop: Signal[Boolean],
    /** Said in place of the chat while no chat model is live. */
    chatIdle: String,
    /** The chat keeps nothing and follows the live model's system template
      * rather than a project's.
      */
    freePlay: Boolean,
    /** Said under the pickers, about the models they show. */
    notice: Signal[Node],
    sessions: WorkspaceSessions,
    runConfigurationService: RunConfigurationService,
    sessionService: SessionService,
    generationService: GenerationService,
    assistantService: AssistantService,
    loraService: LoraService,
    projectService: ProjectService,
    /** The gallery, which inputs are picked from (`specs/50`); the page mounts
      * its effects.
      */
    historyService: HistoryService,
    runtimeService: RuntimeService,
    logService: LogService,
    prerequisites: LaunchPrerequisites,
    browsers: BrowserServices
) extends Component {

  private val panelNow = Var(Option.empty[GenerationPanel])

  /** The generation panel on screen, for what the page lays over its form. */
  def panel: Option[GenerationPanel] = panelNow.now()

  /** Whether the assistant drawer is open — once an assistant is live. */
  private val drawerOpen = Var(true)

  private val drawerShown: Signal[Boolean] =
    sessions
      .liveKey(RuntimeTool.LlamaCpp)
      .map(_.isDefined)
      .combineWith(drawerOpen.signal)
      .map(_ && _)
      .distinct

  /** The gallery as a source of inputs, for the form's slots and the chat's
    * attachments: opened on the project's own results in a workspace.
    */
  private val galleryPicker =
    GalleryPicker(historyService, projectService, runConfigurationService)

  private val pickFromGallery: GalleryPicker.Open =
    (accepted, multiple, onPicked) =>
      galleryPicker.open(
        GalleryPicker.Request(accepted, multiple, projectId, onPicked)
      )

  /** A picker's new configuration (François, 2026-09-28). */
  private val newConfiguration = ModelPicker.newConfiguration(
    runConfigurationService,
    loraService,
    assistantService,
    runtimeService,
    browsers
  )

  /** The pickers, one per row: at the top of the page, or — `prominent` — in
    * the middle of an empty project, as the one thing to do there.
    */
  def modelBar(prominent: Boolean): HtmlElement =
    ModelBar(
      kind,
      prominent,
      sessionService,
      assistantService,
      sessions,
      prerequisites,
      newConfiguration,
      projectId,
      confirmReplace,
      imageControls = panelNow.signal.map(
        _.fold[Node](emptyNode)(_.sessionControls)
      ),
      drawerOpen = drawerOpen,
      notice = notice
    ).element

  // ------------------------------------------------------------ generation

  private def configurationOf(
      configurationId: String
  ): Signal[Option[RunConfiguration]] =
    runConfigurationService.runConfigurations
      .map(_.find(_.id == configurationId))

  private def generationPanel(
      sessionId: String,
      configurationId: String
  ): GenerationPanel =
    GenerationPanel(
      sessionId = sessionId,
      configurationId = configurationId,
      configurationLabel = sessions.labelOf(configurationId),
      sessionSignal =
        sessionService.sessions.map(_.get(configurationId)).distinct,
      architectureId =
        configurationOf(configurationId).map(_.map(_.architectureId)).distinct,
      configurationLoras = configurationOf(configurationId)
        .map(_.map(_.loras).getOrElse(Nil))
        .distinct,
      configurationAssistantTemplateId = configurationOf(configurationId)
        .map(_.flatMap(_.assistantTemplateId))
        .distinct,
      targetArchitecture = configurationOf(configurationId)
        .combineWith(runConfigurationService.architectures)
        .map((configuration, architectures) =>
          configuration
            .flatMap(c => architectures.find(_.id == c.architectureId))
        )
        .distinct,
      service = generationService,
      assistantService = assistantService,
      loraService = loraService,
      projectService = projectService,
      logService = logService,
      onStop =
        () => sessionService.push(SessionService.Command.Stop(sessionId)),
      onRestart =
        () => sessionService.push(SessionService.Command.Restart(sessionId)),
      browsers = browsers,
      project = binding(),
      pickFromGallery = Some(pickFromGallery),
      onOpenOutput = onOpenOutput
    )

  /** Left of the drawer: the generation panel while an image or video model is
    * live, else what the page has to show without one.
    */
  private def mainColumn: HtmlElement = div(
    cls := "workspace-main",
    onUnmountCallback(_ => panelNow.set(None)),
    child <-- sessions.liveKey(RuntimeTool.SdCpp).map {
      case None =>
        panelNow.set(None)
        idle()
      case Some((sessionId, configurationId)) =>
        val panel = generationPanel(sessionId, configurationId)
        panelNow.set(Some(panel))
        panel.element
    }
  )

  // ------------------------------------------------------------- assistant

  /** The chat with the live chat model: the assistant beside a generation
    * panel, its proposals going to that panel's form, or — for text — the
    * conversation that is the page.
    */
  private def assistantColumn(text: Boolean): HtmlElement = div(
    cls := "workspace-assistant",
    child <-- sessions.liveKey(RuntimeTool.LlamaCpp).map {
      case None =>
        if (text) p(cls := "text-secondary", chatIdle) else emptyNode
      case Some((sessionId, configurationId)) =>
        AssistantPanel(
          sessionId = sessionId,
          sessionSignal =
            sessionService.sessions.map(_.get(configurationId)).distinct,
          service = assistantService,
          generationService = generationService,
          onStop =
            () => sessionService.push(SessionService.Command.Stop(sessionId)),
          proposalTarget = Option.unless(text)(
            AssistantTurns.ProposalTarget(
              apply = proposal =>
                panelNow.now() match {
                  case Some(panel) => panel.applyProposal(proposal)
                  // No image model live yet, so there is no form to fill:
                  // hold the proposal, and the panel takes it when the
                  // picker launches one.
                  case None =>
                    generationService.requestPromptProposal(proposal)
                },
              applyAndRun = proposal =>
                panelNow.now().foreach(_.applyProposalAndRun(proposal)),
              canRun = sessions
                .liveStatus(RuntimeTool.SdCpp)
                .map(_.contains(SessionStatus.Ready))
            )
          ),
          freePlay = freePlay,
          pickFromGallery = Some(pickFromGallery)
        ).element
    }
  )

  // ---------------------------------------------------------------- layout

  lazy val element: HtmlElement = div(
    galleryPicker.element,
    // An assistant just launched opens its drawer.
    sessions
      .liveKey(RuntimeTool.LlamaCpp)
      .map(_.map(_._1))
      .changes
      .filter(_.isDefined) --> Observer[Option[String]](_ =>
      drawerOpen.set(true)
    ),
    child <-- kind.map(_.contains(ProjectKind.Text)).distinct.map {
      case true =>
        div(
          modelBar(prominent = false),
          div(cls := "workspace-columns is-text", assistantColumn(text = true))
        )
      case false =>
        div(
          // Up top unless it is the middle of the page already.
          child <-- barOnTop.distinct.map {
            case true  => modelBar(prominent = false)
            case false => emptyNode
          },
          div(
            cls := "workspace-body",
            cls("has-drawer") <-- drawerShown,
            mainColumn,
            // Kept mounted while closed, so the conversation's scroll and
            // what is typed survive a toggle.
            div(
              cls := "workspace-drawer",
              display <-- drawerShown.map(shown => if (shown) "" else "none"),
              assistantColumn(text = false)
            )
          )
        )
    }
  )
}
