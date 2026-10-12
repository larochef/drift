package drift.frontend.generate

import drift.shared.*

/** One sampling field of the form, said once: how its text is read from a block
  * of sampling parameters — a session's defaults, a recorded request, the
  * selected LoRAs' settings over the defaults — and how what the form holds
  * goes back into one, for the request. Seeding, reuse, the LoRAs' layer and
  * the submission all go through the same declaration.
  */
final case class SamplingField(
    name: String,
    read: (SampleParameters, SamplingField.Offer) => String,
    /** The block with what the form holds: text that does not read leaves the
      * block as it was.
      */
    write: (SampleParameters, String) => SampleParameters
)

object SamplingField {

  /** The samplers and schedulers a field may show: a model's own, or any — for
    * a session's defaults, which are shown as the session gives them.
    */
  final case class Offer(
      sampler: String => Boolean,
      scheduler: String => Boolean
  )
  object Offer {
    val any: Offer = Offer(_ => true, _ => true)
    def of(capabilities: SessionCapabilities): Offer =
      Offer(capabilities.samplers.contains, capabilities.schedulers.contains)
  }

  private def number(text: String): Option[Double] = text.trim.toDoubleOption

  /** An empty sampler, scheduler or flow shift is "(model default)": the field
    * is left out of the request.
    */
  private def chosen(text: String): Option[String] =
    Some(text.trim).filter(_.nonEmpty)

  private val steps = SamplingField(
    "steps",
    (sample, _) => sample.sampleSteps.toString,
    (sample, text) =>
      sample.copy(sampleSteps =
        text.trim.toIntOption.getOrElse(sample.sampleSteps)
      )
  )

  private val cfg = SamplingField(
    "cfg",
    (sample, _) => sample.guidance.txtCfg.toString,
    (sample, text) =>
      sample.copy(guidance =
        sample.guidance
          .copy(txtCfg = number(text).getOrElse(sample.guidance.txtCfg))
      )
  )

  /** What the model's main pass runs with. Custom sigmas are their own step
    * count, so they are written after the steps; a list that does not read
    * keeps the block's.
    */
  val Main: List[SamplingField] = List(
    steps,
    cfg,
    SamplingField(
      "flowShift",
      (sample, _) => sample.flowShift.fold("")(_.toString),
      (sample, text) => sample.copy(flowShift = number(text))
    ),
    SamplingField(
      "sigmas",
      (sample, _) => SampleParameters.sigmasText(sample.customSigmas),
      (sample, text) => {
        val sigmas =
          SampleParameters.sigmasOf(text).getOrElse(sample.customSigmas)
        sample.copy(
          customSigmas = sigmas,
          sampleSteps =
            if (sigmas.isEmpty) sample.sampleSteps
            else SampleParameters.sigmaSteps(sigmas)
        )
      }
    ),
    SamplingField(
      "sampler",
      (sample, offer) =>
        sample.sampleMethod.filter(offer.sampler).getOrElse(""),
      (sample, text) => sample.copy(sampleMethod = chosen(text))
    ),
    SamplingField(
      "scheduler",
      (sample, offer) => sample.scheduler.filter(offer.scheduler).getOrElse(""),
      (sample, text) => sample.copy(scheduler = chosen(text))
    ),
    SamplingField(
      "distilledGuidance",
      (sample, _) => sample.guidance.distilledGuidance.toString,
      (sample, text) =>
        sample.copy(guidance =
          sample.guidance.copy(distilledGuidance =
            number(text).getOrElse(sample.guidance.distilledGuidance)
          )
        )
    )
  )

  /** What a two-expert model's high-noise pass runs with: its own steps and
    * CFG; the sampler and scheduler are the model's business, so they ride
    * along unchanged.
    */
  val HighNoise: List[SamplingField] = List(
    steps.copy(name = "highNoiseSteps"),
    cfg.copy(name = "highNoiseCfg")
  )

  val names: Set[String] = (Main ++ HighNoise).map(_.name).toSet
}
