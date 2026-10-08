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

  /** The diffusion families it draws (`diffusion/ImagePipeline`), as drift's
    * architectures name their model kind (`specs/43`).
    */
  val ModelKinds: Seq[String] =
    Seq(
      "krea2",
      "flux2-klein",
      "flux2-dev",
      "qwen-image-2.1",
      "pid-flux2",
      "pid-flux1",
      "pid-qwen-image",
      "hidream-o1",
      "mage-flow",
      "nucleus-image",
      "llada-image",
      "grn",
      "minimax-h3",
      "wan-2.2-14b",
      "ltx-2.5",
      "seedvr2"
    )

  def main(arguments: Array[String]): Unit = {
    if (arguments.contains("--version")) RunnerIdentity.version(Version)
    if (arguments.contains("--model-kinds"))
      RunnerIdentity.modelKinds(ModelKinds)
    if (arguments.contains("--help") || arguments.contains("-h")) {
      println(
        s"$Version: sd-server's native API on drift's own engine (Krea 2, FLUX.2 [klein] and [dev], Qwen Image 2.1, PiD, HiDream O1, Mage-Flow, Nucleus-Image, LLaDA-Image, GRN; MiniMax H3, Wan 2.2 A14B and LTX 2.5 video; SeedVR2 upscaling)\n" +
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
        val upscaler = SeedVr2Pipeline.holds(ops, options.diffusionModel)
        val video =
          !upscaler && VideoPipeline.holds(ops, options.diffusionModel)
        println(
          f"  model recognized: ${(System.nanoTime() - started) / 1e9}%.1f s"
        )
        val pipeline: ImagePipeline | VideoPipeline | SeedVr2Pipeline =
          if (upscaler)
            new SeedVr2Pipeline(
              ops,
              options.diffusionModel,
              options.vae.getOrElse(
                throw new IllegalArgumentException("SeedVR2 needs its --vae")
              )
            )
          else if (video)
            VideoPipeline.open(
              ops,
              options.diffusionModel,
              options.highNoiseModel,
              options.vae,
              options.llm,
              options.t5xxl,
              options.tokenizer,
              options.fps,
              options.audioVae,
              options.controlNet
            )
          else if (options.controlNet.isDefined)
            throw new IllegalArgumentException(
              "--control-net: the runner's image models take no ControlNet"
            )
          else
            ImagePipeline.open(
              ops,
              options.diffusionModel,
              options.vae,
              options.llm,
              options.tokenizer,
              options.llmVision,
              options.connectors,
              options.lladaParts
            )
        val family = pipeline match {
          case image: ImagePipeline      => image.family
          case video: VideoPipeline      => video.family
          case upscaler: SeedVr2Pipeline => upscaler.family
        }
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
        new ImageServer(options, pipeline).start()
        println(s"listening on http://${options.host}:${options.port}")
    }
  }
}
