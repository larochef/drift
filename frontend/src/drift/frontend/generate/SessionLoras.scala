package drift.frontend.generate

import drift.frontend.components.*
import drift.frontend.models.architectures.LoraSamplingFields
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The LoRAs of the form on a live session: the picker that selects them, and
  * the sampling settings the selected ones carry, laid over the form as one
  * more layer of defaults (`specs/49-lora-sampling-settings.md`).
  */
final class SessionLoras(
    state: GenerationFormState,
    seeding: RecipeSeeding,
    sessionId: String,
    service: GenerationService,
    loraService: LoraService,
    /** The launched configuration's architecture — what the LoRA picker filters
      * the collection by.
      */
    architectureId: Signal[Option[String]],
    capabilitiesSignal: Signal[Option[SessionCapabilities]],
    targetArchitecture: Signal[Option[Architecture]],
    /** For the LoRA picker's install button: the browsers it opens. */
    browsers: BrowserServices
) {

  // Built once, so a mode switch keeps what is typed in its search field.
  lazy val picker = LoraPicker(
    collection = loraService.loadedLoras,
    architectureId = architectureId,
    selectedIds = state.selectedLoraIdsVar,
    strengths = state.loraStrengthsVar,
    includeNsfw = state.includeNsfwLorasVar,
    serverPaths = capabilitiesSignal.map(_.map(_.loras.map(_.path).toSet)),
    onTriggerWord = Some(insertTriggerWord),
    // Installed for the architecture; the running server lists it after a
    // Restart, which the picker's "needs restart" mark says.
    headerAction = child <-- targetArchitecture.map(
      _.map(architecture =>
        LoraInstallButton(browsers, architecture, loraService).element
      ).getOrElse(emptyNode)
    ),
    // A LoRA's own settings, typed where they are tried: saved on the LoRA,
    // they reach the form as its layer of defaults does.
    samplingEditor = Some(lora =>
      div(
        child <-- targetArchitecture
          .map(_.exists(LoraSamplingFields.twoExperts))
          .distinct
          .map(twoExperts =>
            LoraSamplingFields(lora, twoExperts, loraService).element
          )
      )
    )
  )

  private def insertTriggerWord(word: String): Unit =
    state.promptVar.update { prompt =>
      if (prompt.toLowerCase.contains(word.toLowerCase)) prompt
      else if (prompt.trim.isEmpty) word
      else s"${prompt.trim}, $word"
    }

  private def carrying(ids: List[String], loras: List[Lora]): List[Lora] = {
    val byId = loras.map(lora => lora.id -> lora).toMap
    ids.flatMap(byId.get).filter(_.sampling.nonEmpty)
  }

  /** The selected LoRAs that carry sampling settings, in selection order
    * (`specs/49-lora-sampling-settings.md`).
    */
  val samplingLoras: Signal[List[Lora]] =
    state.selectedLoraIdsVar.signal
      .combineWith(loraService.loadedLoras)
      .map((ids, loaded) => carrying(ids, loaded.getOrElse(Nil)))

  /** The selected LoRAs' settings into the form: over the fields that still
    * hold a default, or over all of them when the user asks (`force`).
    */
  def applyLoraSampling(force: Boolean): Unit =
    service.capabilitiesOf(sessionId).foreach { capabilities =>
      seeding.applyLoraSampling(
        capabilities,
        state.mode.now(),
        selectedSampling,
        force
      )
    }

  def resetGenerationParameters(): Unit =
    service.capabilitiesOf(sessionId).foreach { capabilities =>
      seeding.resetGenerationParameters(
        capabilities,
        state.mode.now(),
        selectedSampling
      )
    }

  def selectedSampling: LoraSampling =
    LoraSampling.of(
      carrying(
        state.selectedLoraIdsVar.now(),
        loraService.loadedLorasNow.getOrElse(Nil)
      )
    )

  // A LoRA is one more layer of defaults (`specs/49`): whenever the
  // selection's settings, the mode or the session's defaults change, the
  // fields the user has not changed follow.
  lazy val followsSelection: Modifier[HtmlElement] =
    samplingLoras
      .map(LoraSampling.of)
      .combineWith(state.mode.signal, capabilitiesSignal)
      .distinct --> Observer[
      (LoraSampling, String, Option[SessionCapabilities])
    ] {
      case (sampling, mode, Some(capabilities)) =>
        seeding.applyLoraSampling(capabilities, mode, sampling, force = false)
      case _ => ()
    }
}
