package drift.runner.server

import drift.runner.diffusion.ImagePipeline
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
      "pid-qwen-image"
    )

  def main(arguments: Array[String]): Unit = {
    if (arguments.contains("--version")) RunnerIdentity.version(Version)
    if (arguments.contains("--model-kinds"))
      RunnerIdentity.modelKinds(ModelKinds)
    if (arguments.contains("--help") || arguments.contains("-h")) {
      println(
        s"$Version: sd-server's native API on drift's own engine (Krea 2, FLUX.2 [klein] and [dev], Qwen Image 2.1, PiD)\n" +
          "  --diffusion-model FILE  --vae FILE  --llm FILE  --listen-ip HOST  --listen-port PORT\n" +
          "  -W WIDTH  -H HEIGHT  --steps N  --cfg-scale S  --guidance G  --flow-shift MU  -s SEED"
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
        val pipeline = ImagePipeline.open(
          ops,
          options.diffusionModel,
          options.vae,
          options.llm,
          options.tokenizer,
          options.llmVision
        )
        println(
          f"${pipeline.family} loaded in ${(System.nanoTime() - started) / 1e9}%.1f s"
        )
        if (options.guidance.isDefined && !pipeline.takesGuidance)
          println(
            s"--guidance ${options.guidance.get} accepted and ignored: ${pipeline.family} has no guidance embedding"
          )
        new ImageServer(options, pipeline).start()
        println(s"listening on http://${options.host}:${options.port}")
    }
  }
}
