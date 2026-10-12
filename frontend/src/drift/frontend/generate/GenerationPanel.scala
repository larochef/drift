package drift.frontend.generate

import drift.frontend.components.*
import drift.frontend.gallery.GalleryPicker
import drift.frontend.services.*
import drift.frontend.services.GenerationService.Reuse
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The inference UI for one live session (`specs/08-inference-ui.md`): the
  * Inference page swaps to this the moment a configuration is launched, and
  * Stop returns to the configuration list.
  *
  * The form is built from the session's capabilities, never from hardcoded
  * lists, and its defaults come from `defaults_by_mode` — which `sd-server`
  * seeds from the launch argv, so they *are* the run configuration's effective
  * parameters. Modes appear as the capabilities report them: an image model
  * offers Image, a video model offers Video.
  *
  * The panel wires the pieces beside it to the session
  * (`specs/29-split-oversized-files.md`): the form's state
  * (`GenerationFormState`), what seeds it (`RecipeSeeding`), what submits it
  * (`GenerationSubmission`), the form and its sections (`GenerationForm`), its
  * LoRAs (`SessionLoras`), when it is seeded (`SessionFormSeeding`), what it
  * exchanges with the assistant (`FormAssistantLink`), the session's buttons
  * (`GenerationSessionControls`), and the result, progress and log views.
  */
class GenerationPanel(
    sessionId: String,
    /** The launched configuration — a full parameter reuse from the gallery is
      * honoured only when the recorded generation names this very one.
      */
    configurationId: String,
    configurationLabel: Signal[String],
    sessionSignal: Signal[Option[Session]],
    /** The launched configuration's architecture — what the LoRA picker filters
      * the collection by.
      */
    architectureId: Signal[Option[String]],
    service: GenerationService,
    assistantService: AssistantService,
    loraService: LoraService,
    /** The projects a free-play result can be kept into (`specs/22-…`); unused
      * inside a workspace, where results are kept by definition.
      */
    projectService: ProjectService,
    /** The session's own output (`specs/13-log-streaming.md`) - what the model
      * is doing while it loads, and why it stopped when it did.
      */
    logService: LogService,
    /** The machine's load and memory, shown under the bars. */
    machine: Signal[Option[MachineStatus]],
    onStop: () => Unit,
    /** Stops the session and launches the configuration again; the form is
      * handed to the panel the new session gets.
      */
    onRestart: () => Unit,
    /** For the LoRA picker's install button: the browsers it opens. */
    browsers: BrowserServices,
    /** Inside a project workspace (`specs/19-…`): the form follows the selected
      * version, and every submission names the project and the version it was
      * made from.
      */
    project: Option[GenerationPanel.ProjectBinding] = None,
    /** The gallery as a source of inputs (`specs/50`), on the pages that host
      * its picker.
      */
    pickFromGallery: Option[GalleryPicker.Open] = None,
    /** Opens one of this generation's outputs full screen, on pages that have a
      * detail view to open it in — the workspace's. A result is a result
      * wherever it is clicked (`specs/19-…`).
      */
    onOpenOutput: Option[(Generation, Int) => Unit] = None,
    /** The architecture this configuration runs, whose prompting notes the
      * assistant is given while this form is on screen (`specs/20`).
      */
    targetArchitecture: Signal[Option[Architecture]] = Val(
      Option.empty[Architecture]
    ),
    /** The configuration's default LoRAs (`specs/28-configuration-loras.md`),
      * what the form starts with unless a recipe of this configuration is
      * reused.
      */
    configurationLoras: Signal[List[ConfiguredLora]] = Val(
      List.empty[ConfiguredLora]
    ),
    /** The configuration's assistant template override (`specs/32`), handed to
      * the assistant beside the target architecture.
      */
    configurationAssistantTemplateId: Signal[Option[String]] = Val(
      Option.empty[String]
    )
) extends Component {
  // Mirror of `configurationLabel`, so a click handler can read it.
  private val labelNow = Var("")
  // Mirror of `configurationLoras`, so seeding can read it.
  private val configurationLorasNow = Var(List.empty[ConfiguredLora])

  /** Free play (`specs/22-free-play-and-scratch-generations.md`): a panel
    * outside a project keeps nothing. The invariant is structural rather than a
    * flag the caller passes — a generation is persisted *because* it belongs to
    * a project.
    */
  private val scratch: Boolean = project.isEmpty

  /** Where a Keep should put the result: "" is the gallery alone. */
  private val keepProjectVar = Var("")

  /** The form a restart handed over, when this panel is the relaunched
    * session's: kept as it was rather than seeded afresh.
    */
  private val carried = GenerationPanel.takeCarried(configurationId)

  private val state = carried.getOrElse(GenerationFormState())

  private val seeding = RecipeSeeding(
    state,
    configurationId,
    loraService,
    () => configurationLorasNow.now()
  )

  private val submission = GenerationSubmission(
    state,
    sessionId,
    service,
    loraService,
    scratch,
    () => submitContext
  )

  private val assistantLink = FormAssistantLink(
    state,
    assistantService,
    pickFromGallery,
    targetArchitecture,
    configurationAssistantTemplateId
  )

  /** Puts a proposal into the two prompt fields, from the assistant column of a
    * workspace or the pending hand-off of the Assistant page.
    */
  def applyProposal(proposal: PromptProposal): Unit =
    assistantLink.applyProposal(proposal)

  /** A proposal applied and run at once — the workspace chat's "Apply and run".
    */
  def applyProposalAndRun(proposal: PromptProposal): Unit = {
    applyProposal(proposal)
    handleSubmit()
  }

  /** "Reuse these parameters" for a panel that is already on screen — the
    * workspace's detail view, where nothing remounts and so nothing would pick
    * up the service's pending reuse. Does nothing until the session has
    * reported its capabilities, which is when the form exists at all.
    */
  def applyReuse(reuse: Reuse): Unit =
    service.capabilitiesOf(sessionId).foreach(seeding.applyReuse(_, reuse))

  /** Seeds the form from a version, for a panel already on screen: the
    * workspace calls this when the user picks a version card. Nothing else
    * re-seeds a panel that is up — a generation appends a version without
    * touching the form the user is holding.
    */
  def applyVersion(version: PromptVersion): Unit =
    service.capabilitiesOf(sessionId).foreach(seeding.applyVersion(_, version))

  /** What a submission says about its project: the selected version, and
    * whether the prompts are the assistant's.
    */
  private def submitContext: SubmitContext = project match {
    case None          => SubmitContext()
    case Some(binding) =>
      val assistant = assistantLink.promptsAreProposal
      SubmitContext(
        projectId = Some(binding.projectId),
        versionId = binding.selectedVersionNow(),
        origin = Some(if (assistant) "assistant" else "manual")
      )
  }

  private val capabilitiesSignal: Signal[Option[SessionCapabilities]] =
    service.capabilities.map(_.get(sessionId)).distinct

  private val loras = SessionLoras(
    state,
    seeding,
    sessionId,
    service,
    loraService,
    architectureId,
    capabilitiesSignal,
    targetArchitecture,
    browsers
  )

  private val controls =
    GenerationSessionControls(sessionId, logService, () => restart(), onStop)

  private def restart(): Unit = {
    GenerationPanel.carry(configurationId, state)
    onRestart()
  }

  private val formSeeding = SessionFormSeeding(
    state,
    seeding,
    sessionId,
    service,
    loraService,
    loras,
    project,
    capabilitiesSignal,
    carriedOver = carried.isDefined,
    applyProposal
  )

  /** The session's Log, Restart and Stop buttons, which the page places in its
    * model bar, beside the picker that names the model — a workspace and the
    * Sandbox alike.
    */
  lazy val sessionControls: HtmlElement = controls.element

  private def handleSubmit(): Unit =
    service.capabilitiesOf(sessionId).foreach(submission.submit)

  // ---------------------------------------------------------------- layout

  lazy val element: HtmlElement = div(
    cls := "content",
    configurationLabel --> labelNow,
    configurationLoras --> configurationLorasNow,
    assistantLink.bindings,
    formSeeding.bindings,
    // Ask for capabilities until they arrive — the session may still be
    // loading its model when this mounts.
    EventStream
      .periodic(2000)
      .withCurrentValueOf(sessionSignal)
      .withCurrentValueOf(capabilitiesSignal)
      .collect {
        case (_, Some(session), None)
            if session.status == SessionStatus.Ready =>
          ()
      } --> Observer[Unit](_ =>
      service.push(GenerationService.Command.LoadCapabilities(sessionId))
    ),
    onMountCallback { _ =>
      service.push(GenerationService.Command.LoadCapabilities(sessionId))
      service.push(GenerationService.Command.LoadGenerations(sessionId))
      loraService.push(LoraService.Command.Load)
    },
    onUnmountCallback(_ => logService.stop()),
    controls.log,
    child <-- sessionSignal
      .map(_.map(_.status))
      .distinct
      .combineWith(capabilitiesSignal.map(_.isDefined).distinct)
      .map {
        case (Some(SessionStatus.Starting), _) =>
          div(
            cls := "notification is-info is-light",
            p(
              cls := "has-text-weight-bold mb-1",
              "Loading the model…"
            ),
            p(
              cls := "is-size-7",
              "Large models take minutes to load. The session becomes ready " +
                "as soon as the server answers."
            ),
            LogProgressView(
              sessionSignal.map(session =>
                (session.flatMap(_.progress), session.flatMap(_.activity))
              ),
              machine
            ).element
          )
        case (Some(SessionStatus.Ready), false) =>
          div(
            cls := "notification is-info is-light",
            "Fetching what the model supports…"
          )
        case (Some(SessionStatus.Ready), true) =>
          val capabilities = service.capabilitiesOf(sessionId)
          div(
            cls := "columns",
            div(
              cls := "column is-5",
              ReuseNotice(state).element,
              capabilities
                .map(c =>
                  GenerationForm(
                    state,
                    c,
                    loras.picker,
                    loras.samplingLoras,
                    () => loras.applyLoraSampling(force = true),
                    () => loras.resetGenerationParameters(),
                    formSeeding.switchMode,
                    () => handleSubmit(),
                    assistantLink.inputTools
                  ).element
                )
                .getOrElse(emptyNode)
            ),
            GenerationResultColumn(
              capabilities,
              sessionSignal,
              machine,
              service,
              assistantService,
              projectService,
              () => labelNow.now(),
              keepProjectVar,
              project,
              onOpenOutput
            ).element
          )
        case _ => emptyNode
      }
  )
}

object GenerationPanel {

  /** A form on its way from a restarted session's panel to the next one of the
    * same configuration. One at a time: only one generation panel is up.
    */
  private var carried = Option.empty[(String, GenerationFormState)]

  private def carry(configurationId: String, state: GenerationFormState): Unit =
    carried = Some(configurationId -> state)

  private def takeCarried(
      configurationId: String
  ): Option[GenerationFormState] =
    carried.filter(_._1 == configurationId).map { (_, state) =>
      carried = None
      state
    }

  /** The workspace's hooks into the panel (`specs/19-…`): which project, and
    * which version is selected — as a signal for seeding and as a snapshot for
    * the submit.
    */
  case class ProjectBinding(
      projectId: String,
      selectedVersion: Signal[Option[PromptVersion]],
      selectedVersionNow: () => Option[String],
      /** Whether the LoRA picker offers the NSFW ones: an SFW project hides
        * them by default; the checkbox still lets the user look.
        */
      nsfw: Signal[Boolean],
      /** The project's versions and what they made, under the result. */
      history: HtmlElement
  )
}
