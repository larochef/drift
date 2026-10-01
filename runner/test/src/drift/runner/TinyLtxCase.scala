package drift.runner

import drift.runner.diffusion.Lora
import drift.runner.formats.SafetensorsModel
import drift.runner.models.{
  Gemma4Text,
  Ltx2,
  Ltx2Layout,
  LtxConnector,
  LtxTextFeatures,
  LtxVideoVae,
  WeightSource
}
import drift.runner.tensor.{DType, Shape, Tensor}
import drift.runner.ops.Ops

/** The golden tiny LTX 2.5 pieces (`fixtures/tiny_diffusion.py`) on a backend,
  * against transformers' and diffusers'.
  */
object TinyLtxCase {

  private def relative(actual: Array[Float], expected: Array[Float]): Double = {
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** Gemma 4's every hidden state, the last one normed. */
  def gemmaError(ops: Ops): Double = {
    val source = WeightSource.open(
      ops,
      Fixtures.path("tiny/gemma4_text/model.safetensors")
    )
    val model = Gemma4Text(ops, source, "model.")
    try
      Fixtures.withSafetensors("tiny/gemma4_text/expected.safetensors") {
        golden =>
          val states = model.encode(golden("ids").decode().map(_.toInt))
          try relative(ops.toFloats(states), golden("hidden").decode())
          finally ops.release(states)
      }
    finally model.close()
  }

  /** The text features and both connectors: the worst of the video's and the
    * audio's errors.
    */
  def connectorsError(ops: Ops): Double = {
    val source = WeightSource.open(
      ops,
      Fixtures.path("tiny/ltx_connectors/model.safetensors")
    )
    val features = new LtxTextFeatures(ops, source, states = 3, hidden = 32)
    val video = LtxConnector(ops, source, "video_embeddings_connector")
    val audio = LtxConnector(ops, source, "audio_embeddings_connector")
    try
      Fixtures.withSafetensors("tiny/ltx_connectors/expected.safetensors") {
        golden =>
          val states =
            ops.fromFloats(Shape.of(30, 32), golden("states").decode())
          val (v, a) = features(states)
          val (videoOut, audioOut) = (video(v, 16), audio(a, 16))
          try
            math.max(
              relative(ops.toFloats(videoOut), golden("video").decode()),
              relative(ops.toFloats(audioOut), golden("audio").decode())
            )
          finally Seq(states, v, a, videoOut, audioOut).foreach(ops.release)
      }
    finally {
      features.release()
      video.release()
      audio.release()
      source.close()
    }
  }

  /** One step of `model` over the fixture's inputs (`golden`: `video`, `audio`,
    * `text`, `audio_text`, `timestep`), the video tokens in `held` conditions:
    * the worst of the video's and the audio's velocity errors against
    * `expected` (the video's and the audio's names).
    */
  private def stepError(
      ops: Ops,
      model: Ltx2,
      golden: SafetensorsModel,
      layout: Ltx2Layout,
      expected: (String, String),
      held: Seq[(Long, Long)] = Nil
  ): Double = {
    val c = model.config
    val video = ops.fromFloats(
      Shape.of(layout.videoTokens, c.videoChannels),
      golden("video").decode()
    )
    val audio = ops.fromFloats(
      Shape.of(layout.audioFrames, c.audioChannels),
      golden("audio").decode()
    )
    val text =
      ops.fromFloats(Shape.of(16, c.hidden), golden("text").decode())
    val audioText = ops.fromFloats(
      Shape.of(16, c.audioHidden),
      golden("audio_text").decode()
    )
    val (videoOut, audioOut) =
      (
        ops.allocate(DType.F32, video.shape),
        ops.allocate(DType.F32, audio.shape)
      )
    try {
      model.velocity(
        layout,
        video,
        audio,
        text,
        audioText,
        golden("timestep").decode().head,
        videoOut,
        audioOut,
        conditioned = held
      )
      math.max(
        relative(ops.toFloats(videoOut), golden(expected._1).decode()),
        relative(ops.toFloats(audioOut), golden(expected._2).decode())
      )
    } finally
      Seq(video, audio, text, audioText, videoOut, audioOut).foreach(
        ops.release
      )
  }

  /** One step of the joint transformer: the worst of the video's and the
    * audio's velocity errors.
    */
  def transformerError(ops: Ops): Double = {
    val model =
      Ltx2.open(ops, Fixtures.path("tiny/ltx_transformer/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/ltx_transformer/expected.safetensors") {
        golden =>
          val Seq(frames, height, width, audioFrames) =
            golden("shape").decode().map(_.toInt).toSeq
          val layout = Ltx2Layout(
            frames,
            height,
            width,
            audioFrames,
            golden("fps").decode().head
          )
          stepError(ops, model, golden, layout, ("video_out", "audio_out"))
      }
    finally model.close()
  }

  /** One conditioned step: 3 latent frames of 3 × 4, the first held, then a
    * still keyframe at pixel frame 13 and a clip of 2 latent frames from pixel
    * frame 9 appended and held; the positions the runner's own.
    */
  def conditionError(ops: Ops): Double = {
    val model =
      Ltx2.open(ops, Fixtures.path("tiny/ltx_transformer/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/ltx_condition/expected.safetensors") {
        golden =>
          val fps = 16.0
          val layout = Ltx2Layout(
            3,
            3,
            4,
            9,
            fps,
            Ltx2Layout.keyframe(1, 3, 4, 13, single = true, fps) ++
              Ltx2Layout.keyframe(2, 3, 4, 9, single = false, fps)
          )
          stepError(
            ops,
            model,
            golden,
            layout,
            ("video_out", "audio_out"),
            held = Seq((0L, 12L), (36L, 36L))
          )
      }
    finally model.close()
  }

  /** The fixture's LoRA (transformer and connectors, both namings) at 0.8: the
    * worst of the velocities' and the connector rows' errors with it, and of
    * the velocities' once it is removed again.
    */
  def loraError(ops: Ops): Double = {
    val path = Fixtures.path("tiny/ltx_lora/model.safetensors")
    val model = Ltx2.open(ops, path)
    val source = WeightSource.open(ops, path)
    val features = new LtxTextFeatures(ops, source, states = 3, hidden = 32)
    val lora = Lora.open(ops, Fixtures.path("tiny/ltx_lora/lora.safetensors"))
    try
      Fixtures.withSafetensors("tiny/ltx_lora/expected.safetensors") { golden =>
        assert(lora.unread.isEmpty, s"not LoRA pairs: ${lora.unread}")
        val layout = Ltx2Layout(2, 3, 4, 9, 16.0)
        val (video, audio) = model.connectors.get
        def textError(expected: (String, String)): Double = {
          val states =
            ops.fromFloats(Shape.of(30, 32), golden("states").decode())
          val (v, a) = features(states)
          val (videoOut, audioOut) = (video(v, 16), audio(a, 16))
          try
            math.max(
              relative(ops.toFloats(videoOut), golden(expected._1).decode()),
              relative(ops.toFloats(audioOut), golden(expected._2).decode())
            )
          finally
            Seq(states, v, a, videoOut, audioOut).foreach(ops.release)
        }
        val unmatched = model.useLoras(Seq(lora -> 0.8f))
        assert(unmatched.isEmpty, s"unmatched LoRA targets: $unmatched")
        val withLora = Seq(
          stepError(ops, model, golden, layout, ("lora_video", "lora_audio")),
          textError(("lora_text_video", "lora_text_audio"))
        )
        model.useLoras(Nil)
        val back = Seq(
          stepError(ops, model, golden, layout, ("video_out", "audio_out")),
          textError(("text_video", "text_audio"))
        )
        (withLora ++ back).max
      }
    finally {
      lora.close()
      features.release()
      source.close()
      model.close()
    }
  }

  /** The conv VAE encoder: 9 frames of 64 × 64 into 2 latent frames, and one
    * into one; the worst error.
    */
  def encoderError(ops: Ops): Double = {
    val vae = LtxVideoVae.open(
      ops,
      Fixtures.path("tiny/ltx_video_encoder/model.safetensors")
    )
    try
      Fixtures.withSafetensors("tiny/ltx_video_encoder/expected.safetensors") {
        golden =>
          def error(frames: Int, name: String): Double = {
            val pixels = golden(name).decode()
            val inputs = (0 until frames).map(i =>
              ops.fromFloats(
                Shape.of(64, 64, 3),
                pixels.slice(i * 64 * 64 * 3, (i + 1) * 64 * 64 * 3)
              )
            )
            val latents: Seq[Tensor] =
              try vae.encode(inputs)
              finally inputs.foreach(ops.release)
            try {
              assert(
                latents.size == (frames - 1) / 8 + 1,
                s"${latents.size} latent frames"
              )
              relative(
                latents.flatMap(ops.toFloats).toArray,
                golden(s"${name}_latents").decode()
              )
            } finally latents.foreach(ops.release)
          }
          math.max(error(9, "clip"), error(1, "still"))
      }
    finally vae.close()
  }

  /** The conv VAE decoder's frames, 3 latent frames into 17. */
  def vaeError(ops: Ops): Double = {
    val vae = LtxVideoVae.open(
      ops,
      Fixtures.path("tiny/ltx_video_vae/model.safetensors")
    )
    try
      Fixtures.withSafetensors("tiny/ltx_video_vae/expected.safetensors") {
        golden =>
          val latents = golden("latents").decode()
          val frames = (0 until 3).map(i =>
            ops.fromFloats(
              Shape.of(2, 2, 32),
              latents.slice(i * 128, (i + 1) * 128)
            )
          )
          val decoded = scala.collection.mutable.ArrayBuffer.empty[Float]
          try vae.decode(frames, rgb => decoded ++= ops.toFloats(rgb))
          finally frames.foreach(ops.release)
          assert(
            vae.frameCount(3) == 17 && decoded.size == 17 * 64 * 64 * 3,
            s"${decoded.size} values"
          )
          relative(decoded.toArray, golden("frames").decode())
      }
    finally vae.close()
  }
}
