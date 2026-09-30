package drift.runner

import drift.runner.models.{Umt5, Wan, WanVideoVae}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny Wan pieces (`fixtures/tiny_diffusion.py`) on a backend:
  * UMT5's hidden states, the I2V transformer's velocity, and the Wan 2.1 VAE
  * encoding and decoding video through its causal streams, against
  * transformers' and diffusers'.
  */
object TinyWanCase {

  private def relative(actual: Array[Float], expected: Array[Float]): Double = {
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  def umt5Error(ops: Ops): Double = {
    val encoder = Umt5.open(ops, Fixtures.path("tiny/umt5/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/umt5/expected.safetensors") { golden =>
        val ids = golden("ids").decode().map(_.toInt)
        val hidden = encoder.encode(ids)
        try relative(ops.toFloats(hidden), golden("hidden").decode())
        finally ops.release(hidden)
      }
    finally encoder.close()
  }

  def velocityError(ops: Ops): Double = {
    val model = Wan.open(ops, Fixtures.path("tiny/wan/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/wan/expected.safetensors") { golden =>
        val c = model.config
        val (frames, height, width) = (3, 8, 12)
        val (rows, columns) = (height / 2, width / 2)
        val tokens = frames * rows * columns
        // latents [T, h, w, 36] → per frame packed rows, noise then condition
        val latents = golden("latents").decode()
        def packed(from: Int, channels: Int) = {
          val values = new Array[Float](tokens * channels * 4)
          for {
            t <- 0 until frames
            y <- 0 until height
            x <- 0 until width
            ch <- 0 until channels
          } {
            val row = (t * rows + y / 2) * columns + x / 2
            val feature = ch * 4 + (y % 2) * 2 + x % 2
            values(row * channels * 4 + feature) =
              latents(((t * height + y) * width + x) * 36 + from + ch)
          }
          ops.fromFloats(Shape.of(tokens, channels * 4L), values)
        }
        val noise = packed(0, c.outChannels)
        val condition = packed(c.outChannels, c.inChannels - c.outChannels)
        val text = model.text(
          ops.fromFloats(Shape.of(8, c.textWidth), golden("text").decode())
        )
        val out = ops.allocate(DType.F32, noise.shape)
        try {
          model.velocity(
            noise,
            Some(condition),
            text,
            golden("timestep").decode().head,
            frames,
            rows,
            columns,
            out
          )
          // the expected velocity [16, T, h, w] as the same packed rows
          val expected = golden("velocity").decode()
          val actual = ops.toFloats(out)
          val reordered = new Array[Float](actual.length)
          for {
            t <- 0 until frames
            y <- 0 until height
            x <- 0 until width
            ch <- 0 until c.outChannels
          } {
            val row = (t * rows + y / 2) * columns + x / 2
            val feature = ch * 4 + (y % 2) * 2 + x % 2
            reordered(((t * height + y) * width + x) * c.outChannels + ch) =
              actual(row * c.outChannels * 4 + feature)
          }
          relative(reordered, expected)
        } finally text.release()
      }
    finally model.close()
  }

  /** The worst errors of the encoded latents and the decoded frames. */
  def videoVaeErrors(ops: Ops): (Double, Double) = {
    val vae = WanVideoVae.open(
      ops,
      Fixtures.path("tiny/wan_video_vae/model.safetensors")
    )
    try
      Fixtures.withSafetensors("tiny/wan_video_vae/expected.safetensors") {
        golden =>
          val video = golden("video").decode()
          val frames = (0 until 9).map(i =>
            ops.fromFloats(
              Shape.of(32, 32, 3),
              video.slice(i * 32 * 32 * 3, (i + 1) * 32 * 32 * 3)
            )
          )
          val encoded = vae.encode(frames)
          val encodedError = relative(
            encoded.flatMap(ops.toFloats).toArray,
            golden("encoded").decode()
          )
          (frames ++ encoded).foreach(ops.release)
          val latents = golden("latents").decode()
          val latentFrames = (0 until 3).map(i =>
            ops.fromFloats(
              Shape.of(4, 4, 16),
              latents.slice(i * 256, (i + 1) * 256)
            )
          )
          val decoded = scala.collection.mutable.ArrayBuffer.empty[Float]
          vae.decode(
            latentFrames,
            rgb =>
              decoded ++= ops
                .toFloats(rgb)
                .map(v => math.max(-1f, math.min(1f, v)))
          )
          latentFrames.foreach(ops.release)
          (encodedError, relative(decoded.toArray, golden("decoded").decode()))
      }
    finally vae.close()
  }
}
