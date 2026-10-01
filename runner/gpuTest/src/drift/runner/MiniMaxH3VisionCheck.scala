package drift.runner

import drift.runner.models.QwenVision
import drift.runner.ops.CpuOps
import drift.runner.vision.PreparedImage

import java.nio.file.{Files, Paths}
import java.nio.{ByteBuffer, ByteOrder}

/** Reads the vision tower out of MiniMax H3's text encoder (`visual.` in a GGUF
  * without metadata) on the reference backend and sees one 64 × 96 pattern,
  * `(7x + 13y + 50c) mod 256`: `TEXT_ENCODER [OUT]`. Prints the configuration
  * read off the weights and the tokens' shape; with an `OUT` folder writes the
  * tokens and the deepstack rows there (raw little-endian F32: `tokens`,
  * `deepstack`) for a comparison with transformers.
  */
object MiniMaxH3VisionCheck {

  def main(arguments: Array[String]): Unit = {
    val textEncoder = arguments(0)
    val ops = new CpuOps
    val tower = QwenVision.fromWeights(ops, Paths.get(textEncoder), "visual.")
    try {
      println(tower.config)
      val (height, width) = (64, 96)
      val pixels = Array.tabulate(height * width * 3) { i =>
        val (y, x, c) = (i / (width * 3), i / 3 % width, i % 3)
        ((7 * x + 13 * y + 50 * c) % 256) / 127.5f - 1f
      }
      val sizing = drift.runner.vision.ImageSizing.qwen(16, 2)
      val seen = tower.encode(
        PreparedImage(sizing.patches(pixels, height, width), 4, 6, 2, "pattern")
      )
      println(
        s"tokens ${seen.tokens.shape}, deepstack ${seen.deepstack.map(_.shape)}"
      )
      arguments.lift(1).map(Paths.get(_)).foreach { folder =>
        def write(name: String, values: Array[Float]) = {
          val buffer = ByteBuffer
            .allocate(4 * values.length)
            .order(ByteOrder.LITTLE_ENDIAN)
          buffer.asFloatBuffer().put(values)
          Files.write(folder.resolve(name), buffer.array())
        }
        write("tokens", ops.toFloats(seen.tokens))
        write("deepstack", seen.deepstack.flatMap(ops.toFloats).toArray)
      }
    } finally tower.close()
  }
}
