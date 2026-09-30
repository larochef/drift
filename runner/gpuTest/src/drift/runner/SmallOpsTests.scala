package drift.runner

import utest.*

import drift.runner.ops.{
  Activation,
  CpuOps,
  HipOps,
  MatVecInputs,
  Ops,
  Rope,
  RopeLayout,
  RopeSections
}
import drift.runner.tensor.{Comparison, DType, Shape, Tolerance}

/** The elementwise, row-wise and RoPE kernels against the `Cpu` backend, on
  * awkward shapes.
  */
object SmallOpsTests extends TestSuite {

  private val Shapes = Seq(
    Shape.of(1, 1),
    Shape.of(3, 7),
    Shape.of(2, 255),
    Shape.of(5, 4097),
    Shape.of(17, 2560),
    Shape.of(1, 2048) // one long row: more threads
  )

  /** Runs `run` on both backends for each shape and compares. */
  private def againstCpu(shapes: Seq[Shape], tolerance: Tolerance)(
      run: (Ops, Shape) => Array[Float]
  ): Unit = {
    val cpu = new CpuOps
    val hip = new HipOps(Gpu.hip, MatVecInputs.Int8)
    try
      shapes.foreach { shape =>
        val comparison =
          Comparison.of(run(cpu, shape), run(hip, shape), tolerance)
        if (!comparison.passed) println(s"  $shape: ${comparison.summary}")
        assert(comparison.passed)
      }
    finally {
      hip.close()
      cpu.close()
    }
  }

  private def input(ops: Ops, shape: Shape, seed: Long) =
    ops.fromFloats(
      shape,
      TestData.gaussian(seed, shape.elementCount.toInt).map(_ * 3)
    )

  private def unary(
      f: (Ops, drift.runner.tensor.Tensor, drift.runner.tensor.Tensor) => Unit,
      tolerance: Tolerance
  ) =
    againstCpu(Shapes, tolerance) { (ops, shape) =>
      val out = ops.allocate(DType.F32, shape)
      f(ops, input(ops, shape, 1), out)
      ops.toFloats(out)
    }

  private val Close = Tolerance(1e-6, 1e-5)

  val tests = Tests {
    test("mul") {
      againstCpu(Shapes, Tolerance.Exact) { (ops, shape) =>
        val out = ops.allocate(DType.F32, shape)
        ops.mul(input(ops, shape, 1), input(ops, shape, 2), out)
        ops.toFloats(out)
      }
    }
    test("scale") {
      unary((ops, x, out) => ops.scale(x, 0.37f, out), Tolerance.Exact)
    }
    test("addRow") {
      againstCpu(Shapes, Tolerance.Exact) { (ops, shape) =>
        val out = ops.allocate(DType.F32, shape)
        ops.addRow(
          input(ops, shape, 1),
          input(ops, Shape.of(shape.last), 2),
          out
        )
        ops.toFloats(out)
      }
    }
    test("activations") {
      Activation.values.foreach(kind =>
        unary((ops, x, out) => ops.activation(kind, x, out), Close)
      )
    }
    test("gated") {
      Seq(Activation.Silu, Activation.GeluTanh).foreach { kind =>
        againstCpu(Shapes, Close) { (ops, shape) =>
          val out = ops.allocate(DType.F32, shape)
          ops.gated(kind, input(ops, shape, 1), input(ops, shape, 2), out)
          ops.toFloats(out)
        }
      }
    }
    test("conversions round trip the same way") {
      Seq(DType.F16, DType.BF16).foreach { dtype =>
        againstCpu(Shapes, Tolerance.Exact) { (ops, shape) =>
          val (half, back) =
            (ops.allocate(dtype, shape), ops.allocate(DType.F32, shape))
          ops.convert(input(ops, shape, 1), half)
          ops.convert(half, back)
          ops.toFloats(back)
        }
      }
    }
    test("rmsNorm with Gemma's offset") {
      againstCpu(Shapes, Tolerance(1e-5, 1e-5)) { (ops, shape) =>
        val out = ops.allocate(DType.F32, shape)
        ops.rmsNorm(
          input(ops, shape, 1),
          input(ops, Shape.of(shape.last), 2),
          1e-6f,
          1f,
          out
        )
        ops.toFloats(out)
      }
    }
    test("layerNorm") {
      for {
        weight <- Seq(false, true)
        bias <- Seq(false, true)
      }
        againstCpu(Shapes, Tolerance(1e-5, 1e-5)) { (ops, shape) =>
          val out = ops.allocate(DType.F32, shape)
          val row = Shape.of(shape.last)
          ops.layerNorm(
            input(ops, shape, 1),
            Option.when(weight)(input(ops, row, 2)),
            Option.when(bias)(input(ops, row, 3)),
            1e-5f,
            out
          )
          ops.toFloats(out)
        }
    }
    test("group RMS norm, a weight row per stream") {
      againstCpu(Seq(Shape.of(3, 4 * 64), Shape.of(1, 4 * 2560)), Close) {
        (ops, shape) =>
          val out = ops.allocate(DType.F32, shape)
          val weight = input(ops, Shape.of(shape.last), 2)
          ops.groupRmsNorm(input(ops, shape, 1), weight, 4, 1e-6f, 1f, out)
          ops.toFloats(out)
      }
    }
    test("gated RMS norm with a sigmoid gate") {
      againstCpu(Seq(Shape.of(6, 32), Shape.of(3, 128)), Close) {
        (ops, shape) =>
          val out = ops.allocate(DType.F32, shape)
          ops.gatedRmsNorm(
            input(ops, shape, 1),
            input(ops, Shape.of(shape.last), 2),
            input(ops, shape, 3),
            Activation.Sigmoid,
            1e-6f,
            out
          )
          ops.toFloats(out)
      }
    }
    test("streams: mix, combine and the n-gram gate") {
      againstCpu(Seq(Shape.of(3, 64), Shape.of(2, 2560)), Close) {
        (ops, shape) =>
          val wide = Shape.of(shape.dimensions.head, 4 * shape.last)
          val mixed = ops.allocate(DType.F32, shape)
          ops.streamsMix(input(ops, wide, 1), input(ops, wide, 2), 4, mixed)
          val x = input(ops, wide, 3)
          ops.streamsCombine(
            x,
            mixed,
            input(ops, Shape.of(shape.dimensions.head, 4), 4)
          )
          val gated = ops.allocate(DType.F32, wide)
          ops.pleGate(input(ops, wide, 5), x, mixed, gated)
          ops.toFloats(mixed) ++ ops.toFloats(x) ++ ops.toFloats(gated)
      }
    }
    test("a dilated causal convolution, its state carried") {
      againstCpu(Seq(Shape.of(5, 24), Shape.of(1, 24)), Close) { (ops, shape) =>
        val channels = shape.last
        val state = input(ops, Shape.of(9, channels), 2)
        val history =
          ops.allocate(DType.F32, Shape.of(shape.dimensions.head, 9, channels))
        val out = ops.allocate(DType.F32, shape)
        ops.causalConv(
          input(ops, shape, 1),
          input(ops, Shape.of(channels, 4), 3),
          3,
          state,
          Some(history),
          out
        )
        ops.toFloats(out) ++ ops.toFloats(state) ++ ops.toFloats(history)
      }
    }
    test("n-gram rows, cut at EOS, the predecessors carried") {
      val hash = drift.runner.models.NgramTables.hash(3, 2, 7, 320, 1234, 50, 0)
      againstCpu(Seq(Shape.of(1, 6)), Tolerance(0, 0)) { (ops, _) =>
        val state = ops.fromFloats(Shape.of(2), Array(0f, 0f))
        val history = ops.allocate(DType.F32, Shape.of(6, 2))
        val rows = ops.allocate(DType.I32, Shape.of(6, hash.heads))
        val second = ops.allocate(DType.I32, Shape.of(3, hash.heads))
        ops.ngramRows(
          ops.fromInts(Shape.of(6), Array(11, 7, 12, 13, 300, 7)),
          state,
          hash,
          Some(history),
          rows
        )
        ops.ngramRows(
          ops.fromInts(Shape.of(3), Array(14, 15, 16)),
          state,
          hash,
          None,
          second
        )
        val ints = (ops.toInts(rows) ++ ops.toInts(second)).map(_.toFloat)
        ints ++ ops.toFloats(state) ++ ops.toFloats(history)
      }
    }
    test("embedding rows of F32 narrower than a block") {
      againstCpu(Seq(Shape.of(10, 16), Shape.of(4, 160)), Close) {
        (ops, shape) =>
          val out = ops.allocate(DType.F32, Shape.of(3, shape.last))
          ops.embedding(
            input(ops, shape, 1),
            ops.fromInts(Shape.of(3), Array(2, 0, 3)),
            out
          )
          ops.toFloats(out)
      }
    }
    test("AdaLN modulation and a gated residual") {
      againstCpu(Seq(Shape.of(5, 64), Shape.of(3, 6144)), Close) {
        (ops, shape) =>
          val cols = Shape.of(shape.last)
          val out = ops.allocate(DType.F32, shape)
          ops.modulate(
            input(ops, shape, 1),
            input(ops, cols, 2),
            input(ops, cols, 3),
            out
          )
          ops.gatedAdd(out, input(ops, shape, 4), input(ops, cols, 5))
          ops.toFloats(out)
      }
    }
    test("3×3 convolutions at strides 1 and 2 (BF16 operands)") {
      // [H, W, in]; the weight [out, in × 9] scaled to keep sums near 1
      againstCpu(
        Seq(Shape.of(6, 10, 16), Shape.of(4, 4, 3)),
        Tolerance(5e-2, 2e-2)
      ) { (ops, shape) =>
        val Seq(height, width, in) = shape.dimensions
        val weight = ops.allocate(DType.BF16, Shape.of(8, in * 9))
        ops.convert(
          ops.fromFloats(
            Shape.of(8, in * 9),
            TestData.gaussian(2, 8 * in.toInt * 9).map(_ * 0.1f)
          ),
          weight
        )
        Seq(1, 2).flatMap { stride =>
          val out =
            ops.allocate(
              DType.F32,
              Shape.of(height / stride, width / stride, 8)
            )
          ops.conv3x3(
            input(ops, shape, 1),
            weight,
            input(ops, Shape.of(8), 3),
            out,
            stride
          )
          ops.toFloats(out)
        }.toArray
      }
    }
    test("1-D convolutions: dilated, strided, and a transposed one's overlap-add") {
      // [T, in]; the weight [out, in × taps] scaled to keep sums near 1
      againstCpu(
        Seq(Shape.of(37, 16), Shape.of(300, 3)),
        Tolerance(5e-2, 2e-2)
      ) { (ops, shape) =>
        val Seq(length, in) = shape.dimensions
        def weight(out: Long, columns: Long) = {
          val bf16 = ops.allocate(DType.BF16, Shape.of(out, columns))
          ops.convert(
            ops.fromFloats(
              Shape.of(out, columns),
              TestData.gaussian(2, (out * columns).toInt).map(_ * 0.1f)
            ),
            bf16
          )
          bf16
        }
        val x = input(ops, shape, 1)
        val bias = input(ops, Shape.of(8), 3)
        // 7 taps at dilation 3, "same"; 12 at stride 5 padded 7 before
        val same = ops.allocate(DType.F32, Shape.of(length, 8))
        ops.conv1d(x, weight(8, in * 7), Some(bias), 7, 3, 1, 9, 9, same)
        val strided =
          ops.allocate(DType.F32, Shape.of((length + 7 - 12) / 5 + 1, 8))
        ops.conv1d(x, weight(8, in * 12), None, 12, 1, 5, 7, 0, strided)
        // a transposed convolution of 9 taps at stride 5 into 8 channels
        val columns = ops.allocate(DType.F32, Shape.of(length, 8 * 9))
        ops.conv1d(x, weight(72, in), None, 1, 1, 1, 0, 0, columns)
        val upsampled = ops.allocate(DType.F32, Shape.of(length * 5, 8))
        ops.overlapAdd(columns, 9, 5, 2, bias, upsampled)
        Seq(same, strided, upsampled).flatMap(ops.toFloats).toArray
      }
    }
    test("BigVGAN's anti-aliased SnakeBeta") {
      againstCpu(
        Seq(Shape.of(1, 5), Shape.of(3, 7), Shape.of(257, 96)),
        Tolerance(1e-4, 1e-4)
      ) {
        (ops, shape) =>
          val channels = Shape.of(shape.last)
          val out = ops.allocate(DType.F32, shape)
          ops.antiAliasedSnake(
            input(ops, shape, 1),
            ops.fromFloats(
              channels,
              TestData.gaussian(2, shape.last.toInt).map(v => math.exp(0.3 * v).toFloat)
            ),
            ops.fromFloats(
              channels,
              TestData.gaussian(3, shape.last.toInt).map(v => math.exp(-0.3 * v).toFloat)
            ),
            input(ops, Shape.of(12), 4),
            input(ops, Shape.of(12), 5),
            out
          )
          ops.toFloats(out)
      }
    }
    test("group norm over pixels, then patches packed and columns joined") {
      againstCpu(
        Seq(Shape.of(3000, 64), Shape.of(25, 32)),
        Tolerance(1e-4, 1e-4)
      ) { (ops, shape) =>
        val channels = Shape.of(shape.last)
        val normed = ops.allocate(DType.F32, shape)
        // an offset mean, as a VAE's activations have
        val x = input(ops, shape, 1)
        ops.addRow(x, input(ops, channels, 4), x)
        ops.groupNorm(
          x,
          32,
          input(ops, channels, 2),
          input(ops, channels, 3),
          1e-6f,
          normed
        )
        val image = normed.view(shape.dimensions.head / 5, 5, shape.last)
        val packed = ops.allocate(
          DType.F32,
          Shape.of(shape.dimensions.head / 25, shape.last * 5 * 5)
        )
        ops.packPatches(image, 5, packed)
        val joined = ops.allocate(
          DType.F32,
          Shape.of(shape.dimensions.head, 2 * shape.last + 3)
        )
        ops.concatColumns(
          Seq(
            normed,
            input(ops, Shape.of(shape.dimensions.head, 3), 5),
            normed
          ),
          joined
        )
        ops.toFloats(packed) ++ ops.toFloats(joined)
      }
    }
    test("short attention: many sequences at once, grouped heads") {
      // [S, B, heads, D]: 12 layers of 7 tokens, 20 heads; one sequence of 33
      Seq((12, 7, 20, 20), (33, 1, 4, 2)).foreach { (s, b, heads, kvHeads) =>
        // float sums of 128 products inside a softmax, in another order
        againstCpu(Seq(Shape.of(s, b, heads, 128)), Tolerance(5e-5, 1e-4)) {
          (ops, shape) =>
            val kv = Shape.of(s, b, kvHeads, 128)
            val out = ops.allocate(DType.F32, shape)
            ops.shortAttention(
              input(ops, shape, 1),
              input(ops, kv, 2),
              input(ops, kv, 3),
              0.0883f,
              out
            )
            ops.toFloats(out)
        }
      }
    }
    test("argmax, the first among equals") {
      againstCpu(Shapes :+ Shape.of(1, 248320), Tolerance(0, 0)) {
        (ops, shape) =>
          val values = TestData
            .gaussian(3, shape.elementCount.toInt)
            .map(v => math.round(v * 4).toFloat) // ties
          Array(ops.argmax(ops.fromFloats(shape, values)).toFloat)
      }
    }
    test("candidates, the lower index first among equals") {
      val cases = Seq(
        (Shape.of(1, 248320), 40),
        (Shape.of(3, 248320), 1), // greedy
        (Shape.of(3, 248320), 256),
        (Shape.of(2, 5000), 1024),
        (Shape.of(3, 7), 256), // narrower than the count
        (Shape.of(1, 1), 1)
      )
      cases.foreach { (shape, count) =>
        againstCpu(Seq(shape), Tolerance(0, 0)) { (ops, shape) =>
          // rounded for ties, some negative zeros, both crossing the cut
          val values = TestData
            .gaussian(5, shape.elementCount.toInt)
            .map(v => math.round(v * 8) / 2f)
            .map(v => if (v == 0) -0f else v)
          ops
            .candidates(ops.fromFloats(shape, values), count)
            .flatMap(c => c.ids.map(_.toFloat) ++ c.values)
            .toArray
        }
      }
    }
    test("softmax") {
      againstCpu(Shapes, Tolerance(1e-7, 1e-5)) { (ops, shape) =>
        val out = ops.allocate(DType.F32, shape)
        ops.softmax(input(ops, shape, 1), 0.125f, out)
        ops.toFloats(out)
      }
    }
    test("RoPE layouts, partial rotary and mRoPE") {
      val ropes = Seq(
        Rope(1e4f, 128, RopeLayout.Neox, RopeSections.Single),
        Rope(1e4f, 128, RopeLayout.Interleaved, RopeSections.Single),
        Rope(
          1e7f,
          64,
          RopeLayout.Neox,
          RopeSections.Single
        ), // Qwen 3.8: a quarter of 256
        Rope(1e7f, 64, RopeLayout.Neox, RopeSections.Contiguous(11, 11, 10)),
        Rope(1e7f, 64, RopeLayout.Neox, RopeSections.Interleaved(11, 11, 10)),
        // Krea 2: three axes of 32, 48 and 48 values, each its own frequencies
        Rope(
          1e3f,
          128,
          RopeLayout.Interleaved,
          RopeSections.Axes(Seq(16, 24, 24))
        ),
        // FLUX.2: four axes of 32 values
        Rope(
          2e3f,
          128,
          RopeLayout.Interleaved,
          RopeSections.Axes(Seq(16, 16, 16, 16))
        )
      )
      ropes.foreach { rope =>
        val dimension = if (rope.rotaryDimensions == 64) 256 else 128
        val shape = Shape.of(37, 3, dimension)
        // The angle is position × inverse frequency in float; the GPU's powf
        // and the JVM's pow may differ by an ulp in the frequency, which at
        // position 4096 moves the angle by up to ~1e-3: values of about 10
        // then move by ~1e-2 at worst. A wrong pair or section moves them by
        // their own size.
        againstCpu(Seq(shape), Tolerance(1e-2, 0)) { (ops, shape) =>
          val count = rope.sections.positionsPerToken
          val positions = Array.tabulate(count * 37)(i => (i * 131) % 4096)
          val out = ops.allocate(DType.F32, shape)
          val positionShape =
            if (count == 1) Shape.of(37) else Shape.of(count, 37)
          ops.rope(
            input(ops, shape, 1),
            ops.fromInts(positionShape, positions),
            rope,
            out
          )
          ops.toFloats(out)
        }
      }
    }
  }
}
