package drift.runner.server

import drift.runner.diffusion.*

/** What a request starts from when it does not say: the launch flags' values,
  * else the model's own. The capabilities document announces them and the
  * requests fall back on them.
  */
final private[server] class LaunchDefaults(
    options: ImageOptions,
    pipeline: Pipeline.Opened
) {

  /** The steps a request takes when neither it nor the flags say. */
  val steps = options.steps.getOrElse(pipeline.defaultSteps)

  /** The flow shift a request starts from: the launch's, else sd-server's
    * default — or none for a family that derives its own.
    */
  val defaultShift: Option[Double] = options.flowShift.orElse(
    pipeline match {
      case image: ImagePipeline if image.ownShift => None
      case _ => Some(FlowSchedule.DefaultShift)
    }
  )

  /** `--sigmas`, for a model that takes them. */
  val launchSigmas: Seq[Double] = pipeline match {
    case image: ImagePipeline if image.takesSigmas => options.sigmas
    case _                                         => Nil
  }

  /** `--sampling-method`, when it is one of the model's samplers. */
  def launchSampler(pipeline: ImagePipeline): Option[String] =
    options.sampler.filter(pipeline.samplers.contains)
}
