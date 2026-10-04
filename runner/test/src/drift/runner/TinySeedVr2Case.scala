package drift.runner

import drift.runner.models.{SeedVr2, SeedVr2Window, SeedVr2Windows}
import drift.runner.ops.Ops
import drift.runner.tensor.Shape

/** The golden tiny SeedVR2 transformer (`fixtures/tiny_seedvr2.py`, the
  * reference's 3B `NaDiT`) on a backend: one velocity at timestep 1000 and one
  * at 637.5 over 5 latent frames of 40 × 48 (a 5 × 20 × 24 token grid, with
  * several windows on every axis); and the reference's windows for a list of
  * grids.
  */
object TinySeedVr2Case {

  /** `[frames, height, width, channels]` latents as packed patches: token
    * `(t, y / 2, x / 2)`, feature `c × 4 + (y % 2) × 2 + x % 2`.
    */
  private def packed(values: Array[Float], dimensions: Seq[Int]) = {
    val Seq(frames, height, width, channels) = dimensions
    val features = channels * 4
    Array.tabulate(frames * height / 2 * width / 2 * features) { i =>
      val (token, feature) = (i / features, i % features)
      val (c, p) = (feature / 4, feature % 4)
      val (t, cell) =
        (token / (height / 2 * width / 2), token % (height / 2 * width / 2))
      val (y, x) =
        (cell / (width / 2) * 2 + p / 2, cell % (width / 2) * 2 + p % 2)
      values(((t * height + y) * width + x) * channels + c)
    }
  }

  private def relativeError(actual: Array[Float], expected: Array[Float]) = {
    assert(actual.length == expected.length)
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** The velocities' errors at timesteps 1000 and 637.5, each relative to its
    * largest value.
    */
  def velocityErrors(
      ops: Ops,
      folder: String = "tiny/seedvr2"
  ): (Double, Double) = {
    val model = SeedVr2.open(ops, Fixtures.path(s"$folder/model.safetensors"))
    try
      Fixtures.withSafetensors(s"$folder/expected.safetensors") { golden =>
        val dimensions = golden("latent").shape.dimensions.map(_.toInt)
        val Seq(frames, height, width, channels) = dimensions
        val count = frames * height / 2 * width / 2
        val tokens = ops.fromFloats(
          Shape.of(count, channels * 4L),
          packed(golden("latent").decode(), dimensions)
        )
        val embedding =
          ops.fromFloats(golden("text").shape, golden("text").decode())
        val length = golden("text").shape.dimensions.head
        val text =
          ops.allocate(embedding.dtype, Shape.of(length, model.config.hidden))
        val out = ops.allocate(
          embedding.dtype,
          Shape.of(count, model.config.outputChannels * 4L)
        )
        try {
          model.encodeText(embedding, text)
          def error(timestep: Float, name: String) = {
            model.velocity(
              tokens,
              text,
              timestep,
              frames,
              height / 2,
              width / 2,
              out
            )
            relativeError(
              ops.toFloats(out),
              packed(
                golden(name).decode(),
                golden(name).shape.dimensions.map(_.toInt)
              )
            )
          }
          (error(1000f, "velocity"), error(637.5f, "velocity_mid"))
        } finally Seq(tokens, embedding, text, out).foreach(ops.release)
      }
    finally model.close()
  }

  /** The grids whose windows differ from the reference's, with the method. */
  def windowMismatches: Seq[String] =
    Fixtures.withSafetensors("tiny/seedvr2_rules/expected.safetensors") {
      golden =>
        val grids = golden("grids").decode().map(_.toInt).grouped(3).toSeq
        for {
          (grid, index) <- grids.zipWithIndex
          (name, windows) <- Seq(
            "plain" -> SeedVr2Windows.plain,
            "shifted" -> SeedVr2Windows.shifted
          )
          expected = golden(s"windows.$index.$name")
            .decode()
            .map(_.toInt)
            .grouped(6)
            .map(w => SeedVr2Window(w(0), w(1), w(2), w(3), w(4), w(5)))
            .toSeq
          if windows(grid(0), grid(1), grid(2)) != expected
        } yield s"${grid.mkString(" × ")} $name"
    }
}
