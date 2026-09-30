package drift.runner

import drift.runner.models.{
  Gemma4Text,
  Ltx2,
  Ltx2Layout,
  LtxConnector,
  LtxTextFeatures,
  LtxVideoVae,
  WeightSource
}
import drift.runner.tensor.{DType, Shape}
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

  /** One step of the joint transformer: the worst of the video's and the
    * audio's velocity errors.
    */
  def transformerError(ops: Ops): Double = {
    val model =
      Ltx2.open(ops, Fixtures.path("tiny/ltx_transformer/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/ltx_transformer/expected.safetensors") {
        golden =>
          val c = model.config
          val Seq(frames, height, width, audioFrames) =
            golden("shape").decode().map(_.toInt).toSeq
          val layout = Ltx2Layout(
            frames,
            height,
            width,
            audioFrames,
            golden("fps").decode().head
          )
          val video = ops.fromFloats(
            Shape.of(layout.videoTokens, c.videoChannels),
            golden("video").decode()
          )
          val audio = ops.fromFloats(
            Shape.of(audioFrames, c.audioChannels),
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
              audioOut
            )
            math.max(
              relative(ops.toFloats(videoOut), golden("video_out").decode()),
              relative(ops.toFloats(audioOut), golden("audio_out").decode())
            )
          } finally
            Seq(video, audio, text, audioText, videoOut, audioOut).foreach(
              ops.release
            )
      }
    finally model.close()
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
