package drift.frontend.generate

import drift.frontend.gallery.GalleryPicker
import drift.frontend.services.AssistantService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What passes between the generation form and the assistant: the inputs the
  * form shows it, what it is told of the form while one is on screen
  * (`specs/20`), and the proposals it hands back.
  */
final class FormAssistantLink(
    state: GenerationFormState,
    assistantService: AssistantService,
    /** The gallery as a source of inputs (`specs/50`), on the pages that host
      * its picker.
      */
    pickFromGallery: Option[GalleryPicker.Open],
    /** The architecture this configuration runs, whose prompting notes the
      * assistant is given while this form is on screen (`specs/20`).
      */
    targetArchitecture: Signal[Option[Architecture]],
    /** The configuration's assistant template override (`specs/32`), handed to
      * the assistant beside the target architecture.
      */
    configurationAssistantTemplateId: Signal[Option[String]]
) {

  /** The assistant proposal last applied to the prompt fields — a submission
    * whose prompts still equal it is of assistant origin.
    */
  private val lastProposal = Var(Option.empty[PromptProposal])

  /** The gallery entries picked into this form, by the URL their slot holds:
    * what the assistant is told of one it is shown.
    */
  private var pickedOutputs = Map.empty[String, GalleryPicker.Picked]

  val inputTools = InputTools(
    fromGallery = pickFromGallery.map(open =>
      (accepted, multiple, onPicked) =>
        open(
          accepted,
          multiple,
          picked => {
            pickedOutputs ++= picked.map(entry => entry.output.url -> entry)
            onPicked(picked.map(_.output.url))
          }
        )
    ),
    askAssistant = Some(askAssistant)
  )

  /** An input shown to the assistant: a gallery entry as the gallery stages it,
    * with its parameters; anything else uploaded as the file it is.
    */
  private def askAssistant(source: String): Unit =
    pickedOutputs.get(source) match {
      case Some(picked) =>
        assistantService.attach(
          AssistantService.outputAttachment(
            picked.generation,
            picked.output,
            picked.configurationLabel
          )
        )
      case None => assistantService.uploadFrom(source, "input image")
    }

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

  /** Whether the form's prompts are still the proposal last applied. */
  def promptsAreProposal: Boolean =
    lastProposal
      .now()
      .exists(p =>
        p.prompt == state.promptVar.now() &&
          p.negativePrompt == state.negativePromptVar.now()
      )

  /** Keeps the assistant told of the form while it is mounted, and tells it
    * nothing once it is gone.
    */
  lazy val bindings: Modifier[HtmlElement] = List[Modifier[HtmlElement]](
    // What the assistant edits (`specs/20`): the prompt as it stands in the
    // form right now, not a version's — a change made since the last run is
    // exactly what the user wants help with.
    state.promptVar.signal.combineWith(state.negativePromptVar.signal) -->
      Observer[(String, String)] { case (prompt, negative) =>
        assistantService.workingPrompt.set(
          Some(PromptProposal(prompt, negative, aspectRatio = None))
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
    }
  )
}
