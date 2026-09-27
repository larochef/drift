package drift.runner

import utest.*

import drift.runner.ops.{
  Activation,
  Attention,
  CpuOps,
  DeltaRule,
  ExpertActivations,
  ExpertProjections,
  HipOps,
  MatVecInputs,
  Ops,
  QueryKeyNorms,
  Rope,
  RopeLayout,
  RopeSections,
  RowNorm,
  SharedExpert
}
import drift.runner.tensor.{
  Comparison,
  DType,
  RocmFp4,
  Shape,
  Tensor,
  Tolerance
}

/** The kernels that do several operations in one launch (`HipOps`) against the
  * launches they replace, which they must equal bit for bit: a verification's
  * tokens then still get exactly a decode step's results. And against the `Cpu`
  * backend where the operation is new.
  */
object FusedOpsTests extends TestSuite {

  private val Hidden = 2048

  private def exactly(
      label: String,
      expected: Array[Float],
      actual: Array[Float]
  ): Unit = {
    val differing = expected.indices.count(i =>
      java.lang.Float.floatToIntBits(expected(i)) !=
        java.lang.Float.floatToIntBits(actual(i))
    )
    println(s"  $label: $differing of ${expected.length} differ")
    assert(differing == 0)
  }

  private def withHip(body: HipOps => Unit): Unit = {
    val hip = new HipOps(Gpu.hip, MatVecInputs.Float)
    try body(hip)
    finally hip.close()
  }

  private def floats(ops: Ops, rows: Int, cols: Int, seed: Long): Tensor =
    ops.fromFloats(Shape.of(rows, cols), TestData.gaussian(seed, rows * cols))

  /** A copy of `x` that no kernel wrote quantized: its products quantize it
    * themselves.
    */
  private def copied(ops: Ops, x: Tensor): Tensor = {
    val copy = ops.allocate(DType.F32, x.shape)
    ops.copy(x, copy)
    copy
  }

  /** The products of ROCmFP4 `weight` over what a norm-like kernel wrote
    * (`write` gives it), and over a copy: the kernel's quantization of x must
    * be quantize_x's.
    */
  private def checkQuantizedInput(
      hip: HipOps,
      label: String,
      weight: Tensor,
      write: () => Tensor
  ): Unit = {
    val n = weight.shape.dimensions.head
    val x = write()
    val rows = x.shape.elementCount / weight.shape.last
    val view = x.view(rows, weight.shape.last)
    val (fused, separate) =
      (
        hip.allocate(DType.F32, Shape.of(rows, n)),
        hip.allocate(DType.F32, Shape.of(rows, n))
      )
    hip.linear(view, weight, fused)
    hip.linear(copied(hip, view), weight, separate)
    exactly(label, hip.toFloats(separate), hip.toFloats(fused))
  }

  val tests = Tests {
    test("products over x quantized by the kernel that wrote it") {
      withHip { hip =>
        Seq(RocmFp4.Fast, RocmFp4.Dual).foreach { dtype =>
          Seq(1, 3).foreach { t =>
            def weight(k: Int, seed: Int) = hip.fromBytes(
              dtype,
              Shape.of(1536, k),
              TestData.quantized(dtype, 1536, k, seed)
            )
            val (narrow, wide) = (weight(Hidden, 1), weight(2 * Hidden, 2))
            val norm = floats(hip, 1, Hidden, 3).view(Hidden)
            // a first product quantizes x itself, and so tells the norms to
            hip.linear(
              floats(hip, t, Hidden, 4),
              narrow,
              hip.allocate(DType.F32, Shape.of(t, 1536))
            )
            checkQuantizedInput(
              hip,
              s"$dtype rmsNorm, $t rows",
              narrow,
              () => {
                val out = hip.allocate(DType.F32, Shape.of(t, Hidden))
                hip.rmsNorm(floats(hip, t, Hidden, 5), norm, 1e-6f, 0, out)
                out
              }
            )
            checkQuantizedInput(
              hip,
              s"$dtype addRmsNorm, $t rows",
              narrow,
              () => {
                val out = hip.allocate(DType.F32, Shape.of(t, Hidden))
                hip.addRmsNorm(
                  floats(hip, t, Hidden, 6),
                  floats(hip, t, Hidden, 7),
                  RowNorm(norm, 1e-6f, 0, out)
                )
                out
              }
            )
            checkQuantizedInput(
              hip,
              s"$dtype gatedRmsNorm, $t rows",
              wide,
              () => {
                val heads = t * 2 * Hidden / 128
                val out = hip.allocate(DType.F32, Shape.of(heads, 128))
                hip.gatedRmsNorm(
                  floats(hip, heads, 128, 8),
                  floats(hip, 1, 128, 9).view(128),
                  floats(hip, heads, 128, 10),
                  Activation.Silu,
                  1e-6f,
                  out
                )
                out
              }
            )
            checkQuantizedInput(
              hip,
              s"$dtype gated, $t rows",
              wide,
              () => {
                val out = hip.allocate(DType.F32, Shape.of(t, 2 * Hidden))
                hip.gated(
                  Activation.Sigmoid,
                  floats(hip, t, 2 * Hidden, 11),
                  floats(hip, t, 2 * Hidden, 12),
                  out
                )
                out
              }
            )
          }
        }
      }
    }
    test("the residual stream's update and norm") {
      val cpu = new CpuOps
      try
        withHip { hip =>
          Seq(1, 3).foreach { t =>
            val k = 8
            def run(ops: Ops, fused: Boolean): Seq[Array[Float]] = {
              val residual = floats(ops, t, Hidden, 20)
              val y = floats(ops, t, Hidden, 21)
              val normed = floats(ops, t, Hidden, 22)
              val norm = RowNorm(
                floats(ops, 1, Hidden, 23).view(Hidden),
                1e-6f,
                0,
                normed
              )
              if (fused) ops.addRmsNorm(residual, y, norm)
              else {
                ops.add(residual, y, residual)
                ops.rmsNorm(residual, norm.weight, norm.epsilon, 0, normed)
              }
              val shared = SharedExpert(
                floats(ops, t, Hidden, 24),
                normed,
                floats(ops, 1, Hidden, 25).view(Hidden)
              )
              val expertOut = floats(ops, t * k, Hidden, 26)
              val weights = floats(ops, t, k, 27)
              val next = RowNorm(
                floats(ops, 1, Hidden, 28).view(Hidden),
                1e-6f,
                0,
                normed
              )
              if (fused)
                ops.moeCombineNorm(
                  expertOut,
                  weights,
                  Some(shared),
                  residual,
                  next
                )
              else {
                ops.moeCombine(
                  expertOut,
                  weights,
                  Some(shared),
                  Some(residual),
                  residual
                )
                ops.rmsNorm(residual, next.weight, next.epsilon, 0, normed)
              }
              Seq(ops.toFloats(residual), ops.toFloats(normed))
            }
            val separate = run(hip, fused = false)
            val fused = run(hip, fused = true)
            exactly(s"residual, $t rows", separate(0), fused(0))
            exactly(s"normed, $t rows", separate(1), fused(1))
            run(cpu, fused = true).zip(fused).foreach { (expected, actual) =>
              assert(
                Comparison.of(expected, actual, Tolerance(1e-4, 1e-4)).passed
              )
            }
          }
        }
      finally cpu.close()
    }
    test("a router and its routing") {
      withHip { hip =>
        Seq(1, 3, 8).foreach { t =>
          val (experts, k) = (256, 8)
          val router = floats(hip, experts, Hidden, 30)
          val x = floats(hip, t, Hidden, 31 + t)
          def run(fused: Boolean): Seq[Array[Float]] = {
            val logits = hip.allocate(DType.F32, Shape.of(t, experts))
            val ids = hip.allocate(DType.I32, Shape.of(t, k))
            val weights = hip.allocate(DType.F32, Shape.of(t, k))
            if (fused) hip.routeLinear(x, router, logits, ids, weights)
            else {
              hip.linear(x, router, logits)
              hip.route(logits, ids, weights)
            }
            Seq(
              hip.toFloats(logits),
              hip.toInts(ids).map(_.toFloat),
              hip.toFloats(weights)
            )
          }
          run(false)
            .zip(run(true))
            .zip(Seq("logits", "ids", "weights"))
            .foreach { case ((expected, actual), label) =>
              exactly(s"$label, $t rows", expected, actual)
            }
        }
      }
    }
    test("ROCmFP4 experts in one launch") {
      withHip { hip =>
        Seq(RocmFp4.Fast, RocmFp4.Dual).foreach { dtype =>
          val (experts, k, intermediate) = (16, 4, 512)
          def quantized(rows: Int, cols: Int, seed: Int, count: Int = 1) =
            hip.fromBytes(
              dtype,
              if (count == 1) Shape.of(rows, cols)
              else Shape.of(count, rows, cols),
              TestData.quantized(dtype, count * rows, cols, seed)
            )
          val weights = ExpertProjections(
            floats(hip, experts, Hidden, 40),
            quantized(intermediate, Hidden, 41, experts),
            quantized(intermediate, Hidden, 42, experts),
            quantized(Hidden, intermediate, 43, experts),
            quantized(intermediate, Hidden, 44).view(1, intermediate, Hidden),
            quantized(intermediate, Hidden, 45).view(1, intermediate, Hidden),
            quantized(Hidden, intermediate, 46),
            floats(hip, 1, Hidden, 48).view(Hidden)
          )
          def activations(t: Int) = ExpertActivations(
            hip.allocate(DType.F32, Shape.of(t, experts)),
            hip.allocate(DType.I32, Shape.of(t, k)),
            hip.allocate(DType.F32, Shape.of(t, k)),
            hip.allocate(DType.F32, Shape.of(t * k, intermediate)),
            hip.allocate(DType.F32, Shape.of(t * k, Hidden)),
            hip.fromInts(Shape.of(t), Array.fill(t)(0)),
            hip.allocate(DType.F32, Shape.of(t, intermediate)),
            hip.allocate(DType.F32, Shape.of(t, Hidden))
          )
          def outputs(into: ExpertActivations) =
            Seq(into.gated, into.expertOut, into.sharedGated, into.sharedOut)
              .map(hip.toFloats)
          val labels =
            Seq("gated", "expert outputs", "shared gated", "shared output")
          val x3 = floats(hip, 3, Hidden, 47)
          val together = activations(3)
          hip.expertOutputs(x3, weights, together)
          Seq(1, 3).foreach { t =>
            val x = x3.rows(0, t)
            // each part as its own launch
            val separate = activations(t)
            val slots = separate.ids.view(t * k)
            hip.routeLinear(
              x,
              weights.router,
              separate.logits,
              separate.ids,
              separate.routeWeights
            )
            hip.expertsGatedLinear(
              x,
              weights.gate,
              weights.up,
              slots,
              separate.gated
            )
            hip.expertsGatedLinear(
              x,
              weights.sharedGate,
              weights.sharedUp,
              separate.sharedIds,
              separate.sharedGated
            )
            hip.expertsLinear(
              separate.gated,
              weights.down,
              slots,
              separate.expertOut
            )
            hip.linear(
              separate.sharedGated,
              weights.sharedDown,
              separate.sharedOut
            )
            val fused = activations(t)
            hip.expertOutputs(x, weights, fused)
            outputs(separate).zip(outputs(fused)).zip(labels).foreach {
              case ((expected, actual), label) =>
                exactly(s"$dtype $label, $t tokens", expected, actual)
            }
            // a token's results do not depend on the tokens beside it
            if (t == 1)
              outputs(fused).zip(outputs(together)).zip(labels).foreach {
                case ((alone, all), label) =>
                  exactly(
                    s"$dtype $label, the first of 3 tokens alone",
                    alone,
                    all.take(alone.length)
                  )
              }
            // and summed into the residual stream, normed, with the product
            // that follows over what the sum quantized
            def summed(together: Boolean): Seq[Array[Float]] = {
              val residual = floats(hip, t, Hidden, 49)
              val normed = hip.allocate(DType.F32, Shape.of(t, Hidden))
              val norm = RowNorm(
                floats(hip, 1, Hidden, 50).view(Hidden),
                1e-6f,
                0,
                normed
              )
              val into = activations(t)
              if (together) hip.mixtureNorm(x, weights, into, residual, norm)
              else {
                hip.expertOutputs(x, weights, into)
                hip.moeCombineNorm(
                  into.expertOut,
                  into.routeWeights,
                  Some(SharedExpert(into.sharedOut, x, weights.sharedRouter)),
                  residual,
                  norm
                )
              }
              // rows enough for the int8 products
              val product =
                hip.allocate(DType.F32, Shape.of(t, experts * intermediate))
              hip.linear(
                normed,
                weights.gate.view(experts * intermediate, Hidden),
                product
              )
              Seq(residual, normed, product).map(hip.toFloats)
            }
            summed(false)
              .zip(summed(true))
              .zip(Seq("residual", "normed", "product after"))
              .foreach { case ((expected, actual), label) =>
                exactly(s"$dtype summed $label, $t tokens", expected, actual)
              }
          }
        }
      }
    }
    test("a gated attention's inputs") {
      withHip { hip =>
        val (heads, kvHeads, d, pageSize) = (16, 2, 256, 16)
        val rope =
          Rope(1e7f, 64, RopeLayout.Neox, RopeSections.Interleaved(11, 11, 10))
        Seq(1, 3).foreach { t =>
          val start = 14 // the tokens cross a page
          val queryAndGate =
            floats(hip, t, heads * 2 * d, 60).view(t, heads, 2 * d)
          val keys = floats(hip, t, kvHeads * d, 61).view(t, kvHeads, d)
          val values = floats(hip, t, kvHeads * d, 62).view(t, kvHeads, d)
          val norms = QueryKeyNorms(
            floats(hip, 1, d, 63).view(d),
            floats(hip, 1, d, 64).view(d),
            1e-6f,
            0,
            rope
          )
          val positions = hip.fromInts(
            Shape.of(3, t),
            Array.tabulate(3 * t)(i => start + i % t + i / t)
          )
          val pageTable = hip.fromInts(Shape.of(4), Array(3, 0, 2, 1))
          def run(fused: Boolean): Seq[Array[Float]] = {
            val cache = hip.allocateCache(4, pageSize, kvHeads, d)
            hip.zero(cache.keys)
            hip.zero(cache.values)
            def scratch(h: Int) = hip.allocate(DType.F32, Shape.of(t, h, d))
            val (queries, gate, rotatedKeys, rotatedQueries) =
              (scratch(heads), scratch(heads), scratch(kvHeads), scratch(heads))
            // the keys are normed in place by the separate steps
            val ownKeys = copied(hip, keys)
            if (fused)
              hip.attentionInputs(
                queryAndGate,
                ownKeys,
                values,
                norms,
                positions,
                cache,
                pageTable,
                start,
                queries,
                gate,
                rotatedKeys,
                rotatedQueries
              )
            else {
              hip.splitHalves(queryAndGate, queries, gate)
              hip.rmsNorm(
                queries.view(t * heads, d),
                norms.queryNorm,
                1e-6f,
                0,
                queries.view(t * heads, d)
              )
              hip.rmsNorm(
                ownKeys.view(t * kvHeads, d),
                norms.keyNorm,
                1e-6f,
                0,
                ownKeys.view(t * kvHeads, d)
              )
              hip.rope(queries, positions, rope, rotatedQueries)
              hip.rope(ownKeys, positions, rope, rotatedKeys)
              hip.cacheWrite(rotatedKeys, values, cache, pageTable, start)
            }
            val attended = scratch(heads)
            hip.attention(
              rotatedQueries,
              cache,
              pageTable,
              start,
              start + t,
              Attention(
                (1 / math.sqrt(d)).toFloat,
                causal = true,
                None,
                None,
                None
              ),
              attended
            )
            Seq(rotatedQueries, gate, attended).map(hip.toFloats)
          }
          run(false)
            .zip(run(true))
            .zip(Seq("queries", "gate", "attention over the cache"))
            .foreach { case ((expected, actual), label) =>
              exactly(s"$label, $t tokens", expected, actual)
            }
        }
      }
    }
    test("the convolution and the delta rule") {
      withHip { hip =>
        Seq(true, false).foreach { tiled =>
          // Qwen 3.6's heads: more blocks than run at once, as the model has
          val rule = DeltaRule(16, 32, 128, tiled)
          val width = rule.qkvWidth
          val state = Shape.of(rule.valueHeads, 128, 128)
          Seq(1, 3).foreach { t =>
            def run(fused: Boolean): Seq[Array[Float]] = {
              val qkv = floats(hip, t, width, 70)
              val convWeight = floats(hip, width, 4, 71)
              val convState = floats(hip, 3, width, 72)
              val convHistory = hip.allocate(DType.F32, Shape.of(t, 3, width))
              val convolved = hip.allocate(DType.F32, Shape.of(t, width))
              val (a, b) = (
                floats(hip, t, rule.valueHeads, 73),
                floats(hip, t, rule.valueHeads, 74)
              )
              val decay = hip.fromFloats(
                Shape.of(rule.valueHeads),
                TestData.gaussian(75, rule.valueHeads).map(v => -math.abs(v))
              )
              val dtBias =
                floats(hip, 1, rule.valueHeads, 76).view(rule.valueHeads)
              val deltaState = hip.fromFloats(
                state,
                TestData.gaussian(77, state.elementCount.toInt).map(_ * 0.1f)
              )
              val history =
                hip.allocate(DType.F32, Shape(t.toLong +: state.dimensions))
              val out =
                hip.allocate(DType.F32, Shape.of(t, rule.valueHeads, 128))
              if (fused)
                hip.convolvedDeltaRule(
                  qkv,
                  convWeight,
                  convState,
                  Some(convHistory),
                  convolved,
                  a,
                  b,
                  decay,
                  dtBias,
                  deltaState,
                  rule,
                  Some(history),
                  out
                )
              else {
                hip.causalConv(
                  qkv,
                  convWeight,
                  1,
                  convState,
                  Some(convHistory),
                  convolved
                )
                hip.gatedDeltaRule(
                  convolved,
                  a,
                  b,
                  decay,
                  dtBias,
                  deltaState,
                  rule,
                  Some(history),
                  out
                )
              }
              Seq(out, convState, convHistory, deltaState, history).map(
                hip.toFloats
              )
            }
            run(false)
              .zip(run(true))
              .zip(Seq("out", "conv state", "conv history", "state", "history"))
              .foreach { case ((expected, actual), label) =>
                exactly(s"$label, $t tokens, tiled $tiled", expected, actual)
              }
          }
        }
      }
    }
    test("a prompt's ROCmFP4 experts on the matrix units") {
      // x rounds to F16 (2⁻¹¹ of itself), the weights decode exactly, the sums
      // are F32: each product within 2⁻¹¹ of Σ |w x|, and float noise
      val cpu = new CpuOps
      try
        withHip { hip =>
          Seq(RocmFp4.Fast, RocmFp4.Dual).foreach { dtype =>
            val (experts, n, k, tokens, used) = (6, 48, 2048, 20, 4)
            val slots = tokens * used
            val ids = Array.tabulate(slots)(s => (s * 5 + s / used) % experts)
            val gateBytes = TestData.quantized(dtype, experts * n, k, 80)
            val upBytes = TestData.quantized(dtype, experts * n, k, 81)
            val x = TestData.gaussian(82, tokens * k)
            def products(
                ops: Ops,
                bytes: Array[Byte],
                xs: Array[Float],
                rows: Int,
                absolute: Boolean
            ): Array[Float] = {
              val weights = ops.fromBytes(dtype, Shape.of(experts, n, k), bytes)
              val input =
                if (absolute)
                  ops.fromFloats(Shape.of(rows, k), xs.map(math.abs))
                else ops.fromFloats(Shape.of(rows, k), xs)
              val out = ops.allocate(DType.F32, Shape.of(slots, n))
              if (absolute) {
                val decoded = dtype.decode(
                  java.lang.foreign.MemorySegment.ofArray(bytes),
                  experts * n * k
                )
                val positive =
                  ops.fromFloats(Shape.of(experts, n, k), decoded.map(math.abs))
                ops.expertsLinear(
                  input,
                  positive,
                  ops.fromInts(Shape.of(slots), ids),
                  out
                )
              } else
                ops.expertsLinear(
                  input,
                  weights,
                  ops.fromInts(Shape.of(slots), ids),
                  out
                )
              ops.toFloats(out)
            }
            def check(
                label: String,
                actual: Array[Float],
                expected: Array[Float],
                bounds: Array[Double]
            ): Unit = {
              val worst = actual.indices
                .map(i => math.abs(actual(i) - expected(i)) / bounds(i))
                .max
              println(
                f"  $dtype $label: worst error at ${worst * 100}%.1f%% of its bound"
              )
              assert(worst <= 1.0)
            }
            val bound = 2.0 / 2048
            val gate = products(cpu, gateBytes, x, tokens, absolute = false)
            val up = products(cpu, upBytes, x, tokens, absolute = false)
            val gateSums = products(cpu, gateBytes, x, tokens, absolute = true)
            val upSums = products(cpu, upBytes, x, tokens, absolute = true)
            def errors(sums: Array[Float]) = sums.map(v => bound * v + 1e-5)
            check(
              "experts",
              products(hip, gateBytes, x, tokens, absolute = false),
              gate,
              errors(gateSums)
            )
            val (gateErrors, upErrors) = (errors(gateSums), errors(upSums))
            val gated = hip.allocate(DType.F32, Shape.of(slots, n))
            hip.expertsGatedLinear(
              hip.fromFloats(Shape.of(tokens, k), x),
              hip.fromBytes(dtype, Shape.of(experts, n, k), gateBytes),
              hip.fromBytes(dtype, Shape.of(experts, n, k), upBytes),
              hip.fromInts(Shape.of(slots), ids),
              gated
            )
            def silu(v: Double) = v / (1 + math.exp(-v))
            check(
              "experts gated",
              hip.toFloats(gated),
              gate.indices
                .map(i => silu(gate(i)) * up(i))
                .map(_.toFloat)
                .toArray,
              gate.indices.map { i =>
                1.1 * math.abs(up(i)) * gateErrors(i) + (math.abs(
                  silu(gate(i))
                ) + 1.1 * gateErrors(i)) * upErrors(i) + 1e-5
              }.toArray
            )
          }
        }
      finally cpu.close()
    }
    test("ROCmFP4 rows of a few blocks, several to a wave") {
      withHip { hip =>
        Seq(RocmFp4.Fast, RocmFp4.Dual).foreach { dtype =>
          Seq(256, 512, 1024).foreach { k =>
            // 2051 rows are not a multiple of the rows walked at once: one to a wave
            val bytes = TestData.quantized(dtype, 2051, k, 50)
            val x = floats(hip, 2, k, 51)
            val grouped = hip.allocate(DType.F32, Shape.of(2, 2048))
            val alone = hip.allocate(DType.F32, Shape.of(2, 2051))
            hip.linear(
              x,
              hip.fromBytes(
                dtype,
                Shape.of(2048, k),
                bytes.take(dtype.byteSize(2048L * k).toInt)
              ),
              grouped
            )
            hip.linear(x, hip.fromBytes(dtype, Shape.of(2051, k), bytes), alone)
            val expected = hip.toFloats(alone)
            exactly(
              s"$dtype [2, $k]·[2048, $k]ᵀ",
              expected.take(2048) ++ expected.slice(2051, 2051 + 2048),
              hip.toFloats(grouped)
            )
          }
        }
      }
    }
  }
}
