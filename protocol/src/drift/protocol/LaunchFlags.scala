package drift.protocol

/** The flags drift's runner takes when it is started as an image, video or
  * upscale server, by their sd-server names: one list for the runner, which
  * refuses any other by name, and for the backend, whose launches are checked
  * against it.
  */
object ImageServerFlags {

  /** Read: each takes one value. */
  val Valued: Set[String] = Set(
    "--diffusion-model",
    "--model",
    "-m",
    "--vae",
    "--llm",
    "--listen-ip",
    "--listen-port",
    "-W",
    "--width",
    "-H",
    "--height",
    "--steps",
    "--cfg-scale",
    "--guidance",
    "--flow-shift",
    "-s",
    "--seed",
    "-p",
    "--prompt",
    "-n",
    "--negative-prompt",
    "--lora-model-dir",
    "--tokenizer",
    "--llm_vision",
    "--audio-vae",
    "--video-frames",
    "--high-noise-diffusion-model",
    "--t5xxl",
    "--high-noise-steps",
    "--high-noise-cfg-scale",
    "--moe-boundary",
    "--fps",
    "--control-net",
    "--sigmas",
    "--sampling-method",
    "--embeddings-connectors",
    "--llada-queryformer",
    "--llada-text-projection",
    "--llada-sigvq"
  )

  /** Accepted and ignored: they change how sd-cpp runs, not the image. The
    * value is how many values the flag takes.
    */
  val Harmless: Map[String, Int] = Map(
    "--diffusion-fa" -> 0,
    "--mmap" -> 0,
    "--offload-to-cpu" -> 0,
    "--vae-conv-direct" -> 0,
    "--diffusion-conv-direct" -> 0,
    "--cache-mode" -> 1, // spectrum and the like: the runner computes every step
    "--scheduler" -> 1,
    "-t" -> 1,
    "--threads" -> 1,
    "--rng" -> 1,
    "--sampler-rng" -> 1,
    "--vae-format" -> 1, // the runner reads the VAE's layout from its weights
    "--high-noise-sampling-method" -> 1, // Euler, as for the low-noise steps
    "--high-noise-scheduler" -> 1,
    "-v" -> 0,
    "--verbose" -> 0,
    "--color" -> 0
  )

  /** Refused until the runner does what they ask, with why. */
  val NotYet: Map[String, String] = Map(
    "--vae-tiling" -> "tiled VAE decoding comes later",
    "--taesd" -> "TAESD previews come later"
  )

  /** Whether a launch may carry the flag. */
  def accepts(flag: String): Boolean =
    Valued(flag) || Harmless.contains(flag)
}
