package drift.runner.diffusion

import drift.runner.ops.Ops
import drift.runner.weights.WeightSource

import java.nio.file.Path

/** A model family end to end, as a launch opens it: one that draws images, one
  * that draws videos, or the upscaler.
  */
trait Pipeline extends AutoCloseable {

  /** The family's name, as messages give it. */
  def family: String

  /** The jobs it runs, as the native API names them. */
  def mode: String

  /** The steps a request takes when neither it nor the launch says. */
  def defaultSteps: Int
}

object Pipeline {

  /** Any of the three a launch opens. */
  type Opened = ImagePipeline | VideoPipeline | SeedVr2Pipeline

  /** The files a launch gives a model, by what each is. */
  final case class ModelFiles(
      diffusionModel: Path,
      highNoiseModel: Option[Path],
      vae: Option[Path],
      textEncoder: Option[Path],
      t5: Option[Path],
      tokenizer: Option[Path],
      textEncoderVision: Option[Path],
      connectors: Option[Path],
      lladaParts: Map[String, Path],
      audioVae: Option[Path],
      controlNet: Option[Path],
      fps: Option[Int]
  ) {

    /** A file the family cannot do without, or the launch's error naming the
      * flag it comes by.
      */
    def needed(file: Option[Path], flag: String, what: String): Path =
      file.getOrElse(
        throw new IllegalArgumentException(
          s"${diffusionModel.getFileName} needs $what: pass $flag <file>"
        )
      )

    def neededVae: Path = needed(vae, "--vae", "a VAE")
    def neededTextEncoder: Path = needed(textEncoder, "--llm", "a text encoder")
  }

  object ModelFiles {

    /** A diffusion model alone, the other files to add. */
    def of(diffusionModel: Path): ModelFiles =
      ModelFiles(
        diffusionModel,
        highNoiseModel = None,
        vae = None,
        textEncoder = None,
        t5 = None,
        tokenizer = None,
        textEncoderVision = None,
        connectors = None,
        lladaParts = Map.empty,
        audioVae = None,
        controlNet = None,
        fps = None
      )
  }

  /** A family of models: the model kinds it runs, as drift's architectures name
    * them (`drift.protocol.RunnerModelKinds`), how its diffusion model is
    * recognised by its weights, and how its pipeline is opened from the
    * launch's files.
    */
  final case class Family(
      name: String,
      kinds: Seq[String],
      holds: WeightSource => Boolean,
      open: (Ops, ModelFiles) => Opened,
      takesControlNet: Boolean = false
  )

  /** Every family the runner runs, in the order they are tried: the first whose
    * weights the file holds is the file's.
    */
  val Families: Seq[Family] =
    (SeedVr2Pipeline.Family +: VideoPipeline.Families) ++ ImagePipeline.Families

  /** The family `diffusionModel` is of, if any. */
  def familyOf(ops: Ops, diffusionModel: Path): Option[Family] = {
    val source = WeightSource.open(ops, diffusionModel)
    try Families.find(_.holds(source))
    finally source.close()
  }

  /** The pipeline of the family the diffusion model is of. A file no family
    * recognises is refused, with the families the runner does run; so is a
    * ControlNet for a family that takes none.
    */
  def open(ops: Ops, files: ModelFiles): Opened = {
    val family = familyOf(ops, files.diffusionModel).getOrElse(
      throw new IllegalArgumentException(
        s"${files.diffusionModel.getFileName} is none of the models the " +
          s"runner runs (${Families.map(_.name).mkString(", ")})"
      )
    )
    if (files.controlNet.isDefined && !family.takesControlNet)
      throw new IllegalArgumentException(
        s"--control-net: ${family.name} takes no ControlNet"
      )
    family.open(ops, files)
  }
}
