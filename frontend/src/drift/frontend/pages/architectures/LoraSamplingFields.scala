package drift.frontend.pages.architectures

import drift.frontend.components.Component
import drift.frontend.services.LoraService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The sampling a LoRA was made for (`specs/49-lora-sampling-settings.md`),
  * typed from its author's page: every field optional, an empty one leaving
  * that setting to the layers below. Each is saved as it is left; a value that
  * does not read is not saved.
  */
class LoraSamplingFields(
    lora: Lora,
    /** Whether the architecture runs two experts (wan 2.2): its LoRAs then set
      * the high-noise expert's steps and CFG too, and no sigmas.
      */
    twoExperts: Boolean,
    service: LoraService
) extends Component {

  private val sampling = lora.sampling

  /** An emptied field is "not set"; anything else must read as `A`. */
  private def optional[A](
      text: String,
      read: String => Option[A]
  ): Option[Option[A]] =
    if (text.isEmpty) Some(None) else read(text).map(Some(_))

  private def positive(text: String): Option[Int] =
    text.toIntOption.filter(_ > 0)

  private def cell(
      labelText: String,
      current: String,
      width: String,
      kind: String,
      tip: String
  )(updated: String => Option[LoraSampling]): HtmlElement =
    div(
      cls := "field mr-2 mb-1",
      label(cls := "label is-size-7 text-secondary mb-0", labelText),
      input(
        cls := "input is-small",
        styleAttr := s"width: $width;",
        typ := kind,
        defaultValue := current,
        title := tip,
        onChange.mapToValue --> { value =>
          updated(value.trim).foreach(next =>
            service.push(
              LoraService.Command.Update(lora.copy(sampling = next))
            )
          )
        }
      )
    )

  private def shown[A](value: Option[A]): String = value.fold("")(_.toString)

  lazy val element: HtmlElement =
    div(
      cls := "mt-2",
      div(
        cls := "is-flex is-flex-wrap-wrap",
        cell(
          "Steps",
          shown(sampling.steps),
          "4.5rem",
          "number",
          "The steps it was made for: more is fine, fewer break the image"
        )(optional(_, positive).map(steps => sampling.copy(steps = steps))),
        cell("CFG", shown(sampling.cfg), "4.5rem", "number", "Its CFG scale")(
          optional(_, _.toDoubleOption).map(cfg => sampling.copy(cfg = cfg))
        ),
        cell(
          "Flow shift",
          shown(sampling.flowShift),
          "4.5rem",
          "number",
          "How far its steps lean toward the noisy end; a turbo LoRA " +
            "usually wants about 3"
        )(
          optional(_, _.toDoubleOption.filter(_ > 0))
            .map(shift => sampling.copy(flowShift = shift))
        ),
        Option.when(twoExperts)(
          cell(
            "High-noise steps",
            shown(sampling.highNoiseSteps),
            "4.5rem",
            "number",
            "The high-noise expert's steps"
          )(
            optional(_, positive)
              .map(steps => sampling.copy(highNoiseSteps = steps))
          )
        ),
        Option.when(twoExperts)(
          cell(
            "High-noise CFG",
            shown(sampling.highNoiseCfg),
            "4.5rem",
            "number",
            "The high-noise expert's CFG scale"
          )(
            optional(_, _.toDoubleOption)
              .map(cfg => sampling.copy(highNoiseCfg = cfg))
          )
        ),
        cell(
          "Distilled guidance",
          shown(sampling.distilledGuidance),
          "4.5rem",
          "number",
          "For the models that embed a guidance scale (FLUX)"
        )(
          optional(_, _.toDoubleOption)
            .map(guidance => sampling.copy(distilledGuidance = guidance))
        ),
        cell(
          "Sampler",
          shown(sampling.sampler),
          "7rem",
          "text",
          "A sampler's name as the generation form lists it, when the " +
            "LoRA needs one"
        )(text => Some(sampling.copy(sampler = Some(text).filter(_.nonEmpty)))),
        cell(
          "Scheduler",
          shown(sampling.scheduler),
          "7rem",
          "text",
          "A scheduler's name as the generation form lists it, when the " +
            "LoRA needs one"
        )(text =>
          Some(sampling.copy(scheduler = Some(text).filter(_.nonEmpty)))
        ),
        Option.unless(twoExperts)(
          cell(
            "Sigmas",
            SampleParameters.sigmasText(sampling.sigmas),
            "16rem",
            "text",
            "Its exact noise levels, when its page lists them " +
              "(1.0, 0.9375, 0.875, …): they replace the schedule and are " +
              "their own step count"
          )(
            SampleParameters
              .sigmasOf(_)
              .map(sigmas => sampling.copy(sigmas = sigmas))
          )
        )
      ),
      p(
        cls := "help text-secondary",
        "What the LoRA was made for, from its author's page. Selecting the " +
          "LoRA puts these in the generation form, where they can still be " +
          "changed; an empty field leaves that setting alone."
      )
    )
}
