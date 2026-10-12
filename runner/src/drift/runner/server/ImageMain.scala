package drift.runner.server

import drift.runner.diffusion.*
import drift.runner.native.HipRuntime
import drift.runner.ops.{HipOps, MatVecInputs}

/** The runner as drift launches an image configuration: `sd-server`'s flags and
  * native API (`specs/42`, step 12). drift installs a folder whose `sd-server`
  * starts this main (the backend's `RunnerFiles`, `specs/43`). Like sd-server
  * it loads the model before it listens: the first answer to the capabilities
  * probe means ready.
  */
object ImageMain {

  val Version = "drift runner 0.1.0 (images)"

  /** The model kinds its families run (`Pipeline.Families`), as drift's
    * architectures name them (`specs/43`).
    */
  val ModelKinds: Seq[String] = Pipeline.Families.flatMap(_.kinds)

  def main(arguments: Array[String]): Unit = {
    if (arguments.contains("--version")) RunnerIdentity.version(Version)
    if (arguments.contains("--model-kinds"))
      RunnerIdentity.modelKinds(ModelKinds)
    if (arguments.contains("--help") || arguments.contains("-h")) {
      println(
        s"$Version: sd-server's native API on drift's own engine (${Pipeline.Families.map(_.name).mkString(", ")})\n" +
          "  --diffusion-model FILE  --vae FILE  --llm FILE  (or --model FILE, one file)  --tokenizer FILE  --listen-ip HOST  --listen-port PORT\n" +
          "  LLaDA-Image: --llada-queryformer FILE  --llada-text-projection FILE  --llada-sigvq FILE  (or --embeddings-connectors FILE)\n" +
          "  -W WIDTH  -H HEIGHT  --steps N  --cfg-scale S  --guidance G  --flow-shift MU  -s SEED  --video-frames N  --audio-vae FILE\n" +
          "  --high-noise-diffusion-model FILE  --t5xxl FILE  --high-noise-steps N  --high-noise-cfg-scale S  --moe-boundary B  --fps N"
      )
      sys.exit(0)
    }
    ImageOptions.parse(arguments.toSeq) match {
      case Left(problem) =>
        System.err.println(s"error: $problem")
        sys.exit(1)
      case Right(options) =>
        println(s"$Version: ${options.diffusionModel}")
        options.notes.foreach(println)
        val hip = HipRuntime.fromEnvironment()
        println(s"device: ${hip.deviceName} (${hip.rocmRoot})")
        val ops = new HipOps(hip, MatVecInputs.Float)
        val started = System.nanoTime()
        val pipeline = Pipeline.open(
          ops,
          Pipeline.ModelFiles(
            diffusionModel = options.diffusionModel,
            highNoiseModel = options.highNoiseModel,
            vae = options.vae,
            textEncoder = options.llm,
            t5 = options.t5xxl,
            tokenizer = options.tokenizer,
            textEncoderVision = options.llmVision,
            connectors = options.connectors,
            lladaParts = options.lladaParts,
            audioVae = options.audioVae,
            controlNet = options.controlNet,
            fps = options.fps
          )
        )
        val family = pipeline.family
        println(
          f"$family loaded in ${(System.nanoTime() - started) / 1e9}%.1f s"
        )
        pipeline match {
          case image: ImagePipeline =>
            if (options.sigmas.nonEmpty && !image.takesSigmas)
              println(
                s"--sigmas accepted and ignored: $family runs its own schedule"
              )
            if (options.guidance.isDefined && !image.takesGuidance)
              println(
                s"--guidance ${options.guidance.get} accepted and ignored: $family has no guidance embedding"
              )
          case video: VideoPipeline =>
            options.highNoiseModel.foreach(file =>
              println(
                if (video.hasHighNoiseExpert) s"high-noise expert: $file"
                else
                  s"--high-noise-diffusion-model $file accepted and ignored: $family has one model"
              )
            )
            if (video.makesSoundtrack && !video.decodesSoundtrack)
              println(s"no --audio-vae: $family's videos are silent")
            if (!video.makesSoundtrack)
              options.audioVae.foreach(file =>
                println(
                  s"--audio-vae $file accepted and ignored: $family makes no soundtrack"
                )
              )
          case _: SeedVr2Pipeline => ()
        }
        new ImageServer(options, pipeline, ops).serve()
    }
  }
}
