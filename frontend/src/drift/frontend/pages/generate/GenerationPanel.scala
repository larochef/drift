package drift.frontend.pages.generate

import drift.frontend.components.*
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
  * (`GenerationSubmission`), the form and its sections (`GenerationForm`), and
  * the result, progress and log views.
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
    onStop: () => Unit,
    /** Inside a project workspace (`specs/19-…`): the form follows the selected
      * version, and every submission names the project and the version it was
      * made from.
      */
    project: Option[GenerationPanel.ProjectBinding] = None,
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

  private val showLog = Var(false)
  private val followLog = Var(true)

  /** The assistant proposal last applied to the prompt fields — a submission
    * whose prompts still equal it is of assistant origin.
    */
  private val lastProposal = Var(Option.empty[PromptProposal])

  private val state = GenerationFormState()

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

  /** Puts a proposal into the two prompt fields, from the assistant column of a
    * workspace or the pending hand-off of the Assistant page.
    */
  def applyProposal(proposal: PromptProposal): Unit = {
    state.promptVar.set(proposal.prompt)
    state.negativePromptVar.set(proposal.negativePrompt)
    proposal.aspectRatio.foreach(applyAspectRatio)
    lastProposal.set(Some(proposal))
  }

  /** `W:H` onto the form (`specs/32`): the longer side stays, the other follows
    * the ratio, both on multiples of 16.
    */
  private def applyAspectRatio(ratio: String): Unit =
    ratio.split(":").map(_.trim.toIntOption) match {
      case Array(Some(rw), Some(rh)) if rw > 0 && rh > 0 =>
        val current = math.max(
          state.widthVar.now().trim.toIntOption.getOrElse(1024),
          state.heightVar.now().trim.toIntOption.getOrElse(1024)
        )
        val long = math.max(16, current / 16 * 16)
        val short = math.max(
          16,
          math
            .round(long.toDouble * math.min(rw, rh) / math.max(rw, rh) / 16)
            .toInt * 16
        )
        val (w, h) = if (rw >= rh) (long, short) else (short, long)
        state.widthVar.set(w.toString)
        state.heightVar.set(h.toString)
      case _ => ()
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
    * touching the form the user is holding (François, 2026-09-10).
    */
  def applyVersion(version: PromptVersion): Unit =
    service.capabilitiesOf(sessionId).foreach(seeding.applyVersion(_, version))

  /** What a submission says about its project: the selected version, and
    * whether the prompts are the assistant's.
    */
  private def submitContext: SubmitContext = project match {
    case None          => SubmitContext()
    case Some(binding) =>
      val assistant = lastProposal
        .now()
        .exists(p =>
          p.prompt == state.promptVar.now() &&
            p.negativePrompt == state.negativePromptVar.now()
        )
      SubmitContext(
        projectId = Some(binding.projectId),
        versionId = binding.selectedVersionNow(),
        origin = Some(if (assistant) "assistant" else "manual")
      )
  }

  private val capabilitiesSignal: Signal[Option[SessionCapabilities]] =
    service.capabilities.map(_.get(sessionId)).distinct

  /** What this panel shows as its result. The session outlives any one
    * workspace — its last generation may well belong to the project the user
    * was in before, which is what a brand new project showed as its own, and
    * what came back when the image actually made in it was deleted (François,
    * 2026-09-10). Inside a project, only that project's.
    */
  private val panelGenerations: Signal[List[Generation]] = project match {
    case None          => service.generations
    case Some(binding) =>
      service.generations.map(
        _.filter(_.projectId.contains(binding.projectId))
      )
  }

  /** What the result shows: the job under way while there is one, else the
    * newest. Showing the newest submission meant that cancelling a queued job
    * replaced a running generation on screen with that cancellation (François,
    * 2026-09-18).
    */
  private val shownGeneration: Signal[Option[Generation]] =
    panelGenerations
      .map(all => all.findLast(_.status.isActive).orElse(all.lastOption))
      .distinct

  private val seeded = Var(false)

  /** Whether the project's selected version has seeded this form. Once per
    * panel: a new panel is built when the model picker switches configuration,
    * which is the only moment a recipe is laid over the form on its own.
    */
  private val versionSeeded = Var(false)

  // Built once, so a mode switch keeps what is typed in its search field.
  private lazy val loraPicker = LoraPicker(
    collection = loraService.loadedLoras,
    architectureId = architectureId,
    selectedIds = state.selectedLoraIdsVar,
    strengths = state.loraStrengthsVar,
    includeNsfw = state.includeNsfwLorasVar,
    serverPaths = capabilitiesSignal.map(_.map(_.loras.map(_.path).toSet)),
    onTriggerWord = Some(insertTriggerWord)
  )

  private def insertTriggerWord(word: String): Unit =
    state.promptVar.update { prompt =>
      if (prompt.toLowerCase.contains(word.toLowerCase)) prompt
      else if (prompt.trim.isEmpty) word
      else s"${prompt.trim}, $word"
    }

  private def seedForm(capabilities: SessionCapabilities): Unit = {
    val initialMode =
      if (capabilities.supportedModes.contains(capabilities.currentMode))
        capabilities.currentMode
      else capabilities.supportedModes.headOption.getOrElse("img_gen")
    state.mode.set(initialMode)
    seeding.seedModeFields(capabilities, initialMode)
    service.takePendingReuse() match {
      case Some(reuse) => seeding.applyReuse(capabilities, reuse)
      case None        => seeding.applyConfigurationLoras()
    }
    // A proposal from the assistant page overrides whatever the form holds.
    service.takePendingPromptProposal().foreach(applyProposal)
  }

  private def switchMode(newMode: String): Unit =
    service.capabilitiesOf(sessionId).foreach { capabilities =>
      state.mode.set(newMode)
      seeding.seedModeFields(capabilities, newMode)
    }

  private def handleSubmit(): Unit =
    service.capabilitiesOf(sessionId).foreach(submission.submit)

  private def statusTag(session: Session): HtmlElement = {
    val (colour, text) = session.status match {
      case SessionStatus.Starting => ("is-info", "starting…")
      case SessionStatus.Ready    =>
        (
          "is-success",
          session.port.map(p => s"ready on :$p").getOrElse("ready")
        )
      case SessionStatus.Failed  => ("is-danger", "failed")
      case SessionStatus.Stopped => ("", "stopped")
    }
    span(cls := s"tag is-small mr-2 $colour", text)
  }

  /** What free play is, said once above the results, with the way to empty it.
    */
  private def freePlayNotice: Node =
    if (!scratch) emptyNode
    else
      div(
        cls := "notification is-warning is-light py-2 px-3 is-size-7 mb-3",
        span(
          "Free play — nothing here is saved. Keep a result to put it in the " +
            "gallery; everything else goes when the session stops."
        ),
        button(
          cls := "button is-small ml-2",
          "Clear results",
          onClick --> (_ =>
            service.push(GenerationService.Command.ClearScratch)
          )
        )
      )

  // ---------------------------------------------------------------- layout

  lazy val element: HtmlElement = div(
    cls := "content",
    configurationLabel --> labelNow,
    configurationLoras --> configurationLorasNow,
    // What the assistant edits (`specs/20`): the prompt as it stands in the
    // form right now, not a version's — a change made since the last run is
    // exactly what the user wants help with.
    state.promptVar.signal.combineWith(state.negativePromptVar.signal) -->
      Observer[(String, String)] { case (prompt, negative) =>
        assistantService.workingPrompt.set(
          Some(PromptProposal(prompt, negative))
        )
      },
    // What the model is and how hard it follows the negative prompt, beside
    // the prompt itself (`specs/20`).
    targetArchitecture --> assistantService.targetArchitecture.writer,
    configurationAssistantTemplateId -->
      assistantService.configurationTemplateId.writer,
    state.cfgVar.signal.map(
      _.trim.toDoubleOption
    ) --> assistantService.workingCfg.writer,
    // The size, for a template that wants the aspect ratio (`specs/32`).
    state.widthVar.signal
      .combineWith(state.heightVar.signal)
      .map((w, h) =>
        for {
          width <- w.trim.toIntOption
          height <- h.trim.toIntOption
        } yield (width, height)
      ) --> assistantService.workingSize.writer,
    onUnmountCallback { _ =>
      assistantService.workingPrompt.set(None)
      assistantService.targetArchitecture.set(None)
      assistantService.configurationTemplateId.set(None)
      assistantService.workingCfg.set(None)
      assistantService.workingSize.set(None)
    },
    // The project's flag sets the LoRA picker's NSFW checkbox; the user can
    // still tick it.
    project
      .map(binding =>
        binding.nsfw.distinct --> state.includeNsfwLorasVar.writer
      )
      .getOrElse(emptyMod),
    // The selected version seeds the form once, when this panel appears — on
    // mount, and again when the model picker builds a new one, which is the
    // only reset the user wants. After that the form is theirs: a generation
    // appends a version without touching what they are holding, and only a
    // click on a version card lays a recipe over it (François, 2026-09-10).
    project
      .map(_.selectedVersion)
      .getOrElse(Val(Option.empty[PromptVersion]))
      .combineWith(capabilitiesSignal)
      .map((version, capabilities) => version.zip(capabilities))
      --> Observer[Option[(PromptVersion, SessionCapabilities)]] {
        case Some((version, capabilities)) if !versionSeeded.now() =>
          versionSeeded.set(true)
          seeded.set(true)
          seeding.applyVersion(capabilities, version)
          // As when the defaults seed the form, a proposal handed over from the
          // Assistant page wins over what was recorded.
          service.takePendingPromptProposal().foreach(applyProposal)
        case _ => ()
      },
    // Seed the form exactly once, from the first capabilities available — a
    // signal, not `.changes`, because they may already be loaded when this
    // mounts (the user navigated away and back).
    capabilitiesSignal --> Observer[Option[SessionCapabilities]] {
      case Some(capabilities) if !seeded.now() =>
        seeded.set(true)
        seedForm(capabilities)
      case _ => ()
    },
    // A reuse may name LoRAs before the collection has been listed.
    loraService.loadedLoras --> Observer[Option[List[Lora]]](
      seeding.resolvePendingLoras
    ),
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
    // Only in free play: inside a workspace the page already mounts these,
    // and a second mount subscribes every stream twice.
    if (scratch) projectService.effects else emptyMod,
    onMountCallback { _ =>
      service.push(GenerationService.Command.LoadCapabilities(sessionId))
      service.push(GenerationService.Command.LoadGenerations(sessionId))
      loraService.push(LoraService.Command.Load)
      if (scratch) projectService.push(ProjectService.Command.Load)
    },
    div(
      cls := "level panel-header",
      div(
        cls := "level-left",
        h1(
          cls := "title text-primary",
          child.text <-- configurationLabel
        ),
        child <-- sessionSignal.map {
          case Some(session) => statusTag(session)
          case None          => emptyNode
        }
      ),
      div(
        cls := "level-right",
        button(
          cls := "button mr-2",
          child.text <-- showLog.signal.map(open =>
            if (open) "▼ Log" else "▶ Log"
          ),
          onClick --> { _ =>
            val opening = !showLog.now()
            showLog.set(opening)
            if (opening) logService.follow(sessionId) else logService.stop()
          }
        ),
        button(
          cls := "button is-warning",
          "⏹ Stop session",
          onClick --> (_ => onStop())
        )
      )
    ),
    hr(),
    onUnmountCallback(_ => logService.stop()),
    child <-- showLog.signal.map {
      case true =>
        SessionLog(logService, followLog, () => showLog.set(false)).element
      case false => emptyNode
    },
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
              )
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
                    loraPicker,
                    switchMode,
                    () => handleSubmit()
                  ).element
                )
                .getOrElse(emptyNode)
            ),
            div(
              cls := "column is-7",
              freePlayNotice,
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
                      service,
                      assistantService,
                      projectService,
                      () => labelNow.now(),
                      scratch,
                      keepProjectVar,
                      onOpenOutput
                    ).element
                }
            )
          )
        case _ => emptyNode
      }
  )
}

object GenerationPanel {

  /** The workspace's hooks into the panel (`specs/19-…`): which project, and
    * which version is selected — as a signal for seeding and as a snapshot for
    * the submit.
    */
  case class ProjectBinding(
      projectId: String,
      selectedVersion: Signal[Option[PromptVersion]],
      selectedVersionNow: () => Option[String],
      /** Whether the LoRA picker offers the NSFW ones: an SFW project hides
        * them by default (François, 2026-09-08); the checkbox still lets the
        * user look.
        */
      nsfw: Signal[Boolean]
  )
}
