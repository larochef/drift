package drift.frontend.generate

import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** When the form of a live session is seeded, and from what: once from the
  * session's defaults or the project's selected version, again on a mode
  * switch, and whenever the selected LoRAs' settings change.
  */
final class SessionFormSeeding(
    state: GenerationFormState,
    seeding: RecipeSeeding,
    sessionId: String,
    service: GenerationService,
    loraService: LoraService,
    loras: SessionLoras,
    /** Inside a project workspace (`specs/19-…`): the form follows the selected
      * version.
      */
    project: Option[GenerationPanel.ProjectBinding],
    capabilitiesSignal: Signal[Option[SessionCapabilities]],
    /** Whether the form is one a restart handed over: kept as it was rather
      * than seeded afresh.
      */
    carriedOver: Boolean,
    /** Puts an assistant's proposal into the prompt fields. */
    applyProposal: PromptProposal => Unit
) {

  private val seeded = Var(carriedOver)

  /** Whether the project's selected version has seeded this form. Once per
    * panel: a new panel is built when the model picker switches configuration,
    * which is the only moment a recipe is laid over the form on its own.
    */
  private val versionSeeded = Var(carriedOver)

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

  def switchMode(newMode: String): Unit =
    service.capabilitiesOf(sessionId).foreach { capabilities =>
      state.mode.set(newMode)
      seeding.seedModeFields(capabilities, newMode)
      // the selection is the same, so nothing else lays its settings again
      seeding.applyLoraSampling(
        capabilities,
        newMode,
        loras.selectedSampling,
        force = false
      )
    }

  /** What seeds the form while the panel is mounted. */
  lazy val bindings: Modifier[HtmlElement] = List[Modifier[HtmlElement]](
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
    // click on a version card lays a recipe over it.
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
    loras.followsSelection,
    // A reuse may name LoRAs before the collection has been listed.
    loraService.loadedLoras --> Observer[Option[List[Lora]]](
      seeding.resolvePendingLoras
    )
  )
}
