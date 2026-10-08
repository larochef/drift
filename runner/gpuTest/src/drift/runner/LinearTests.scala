package drift.runner

import utest.*

import drift.runner.ops.{
  ConvRotLevels,
  ConvRotParts,
  CpuOps,
  HipOps,
  MatVecInputs,
  Ops
}
import drift.runner.tensor.{
  Comparison,
  DType,
  GgmlQuants,
  IQuants,
  KQuants,
  RocmFp4,
  Shape,
  Tolerance
}

/** The matrix-vector kernels against the `Cpu` backend's decode-and-dot: float
  * kernels to float noise, integer kernels within the rounding of x.
  */
object LinearTests extends TestSuite {

  /** `(M, N, K)` cases: one row, odd row counts, rows shorter than a chunk, a
    * model's width; several vectors per pass (up to 8), and the GEMM path
    * beyond.
    */
  private def cases(dtype: DType): Seq[(Int, Int, Int)] = {
    val small = if (dtype.blockElements == 256) 256 else 64
    Seq(
      (1, 1, small),
      (1, 7, 512),
      (1, 33, 640 max small * 3),
      (1, 64, 2560),
      (2, 9, 4096),
      (3, 20, 1024),
      (8, 16, 2560),
      (9, 24, 512),
      (40, 72, 2560),
      (1, 1030, 1024) // past the split kernel's rows: the direct kernels
    )
      .filter((_, _, k) =>
        k % dtype.blockElements == 0 && dtype.byteSize(k) % 4 == 0
      ) ++
      // F32 rows not in whole blocks of 32 (MiniMax H3's ControlNet input,
      // 196): the split kernel, and hipBLAS on weights converted as a whole
      Option
        .when(dtype == DType.F32)(
          Seq((4, 24, 196), (56, 40, 196), (300, 16, 68))
        )
        .toSeq
        .flatten
  }

  /** How far an integer kernel may land from the exact product: it multiplies
    * with x rounded per 32 values to `max|x| / 127` steps, so each product is
    * off by at most `|w_k| × step / 2`. Plus float noise on the outputs' scale.
    */
  private def quantizationBound(
      weights: Array[Float],
      x: Array[Float],
      m: Int,
      n: Int,
      k: Int
  ): Array[Double] = {
    val halfSteps = Array.tabulate(m * k) { index =>
      val start = index / 32 * 32
      (start until start + 32).map(i => math.abs(x(i))).max / 127.0 / 2
    }
    Array.tabulate(m * n) { index =>
      val (sample, row) = (index / n, index % n)
      (0 until k)
        .map(c => math.abs(weights(row * k + c)) * halfSteps(sample * k + c))
        .sum
    }
  }

  /** `Σ |w_k x_k|` per output: the GEMM path rounds x and w to F16, so each
    * product is off by at most about `2 × 2⁻¹¹` of itself; the result is then
    * rounded to F16 once more.
    */
  private def productSums(
      weights: Array[Float],
      x: Array[Float],
      m: Int,
      n: Int,
      k: Int
  ): Array[Double] =
    Array.tabulate(m * n) { index =>
      val (sample, row) = (index / n, index % n)
      (0 until k)
        .map(c => math.abs(weights(row * k + c) * x(sample * k + c)).toDouble)
        .sum
    }

  /** `integer`: the type quantizes x when `inputs` is `Int8`. */
  private def check(
      dtype: DType,
      integer: Boolean,
      inputs: MatVecInputs
  ): Unit = {
    val cpu = new CpuOps
    val hip = new HipOps(Gpu.hip, inputs)
    try
      cases(dtype).zipWithIndex.foreach { case ((m, n, k), index) =>
        val weights = TestData.quantized(dtype, n, k, seed = index)
        val x = TestData.gaussian(100 + index, m * k)
        def run(ops: Ops): Array[Float] = {
          val out = ops.allocate(DType.F32, Shape.of(m, n))
          ops.linear(
            ops.fromFloats(Shape.of(m, k), x),
            ops.fromBytes(dtype, Shape.of(n, k), weights),
            out
          )
          ops.toFloats(out)
        }
        val expected = run(cpu)
        val actual = run(hip)
        // float sums in another order: errors scale with the outputs' size
        val scale = expected.map(math.abs).max.toDouble
        val decoded = dtype.decode(
          java.lang.foreign.MemorySegment.ofArray(weights),
          n * k
        )
        def within(path: String, bounds: Array[Double]): Unit = {
          val worst = expected.indices.map { i =>
            math.abs(actual(i) - expected(i)) / (bounds(i) + 1e-5 * scale)
          }.max
          println(
            f"  $dtype [$m, $k]·[$n, $k]ᵀ ($path): worst error at ${worst * 100}%.1f%% of its bound"
          )
          assert(worst <= 1.0)
        }
        if (m >= hip.GemmMinimumRows) {
          // BF16 weights meet x rounded to BF16 (2⁻⁸), the others to F16 (2⁻¹¹)
          val unit = math.pow(2, if (dtype == DType.BF16) -8 else -11)
          within(
            "GEMM",
            productSums(decoded, x, m, n, k)
              .zip(expected)
              .map((sum, y) => (sum * 2 + math.abs(y)) * unit * 1.01)
          )
        } else if (
          integer && inputs == MatVecInputs.Int8 ||
          rocmFp4Int8(dtype, k) && n > hip.SplitMaximumRows
        )
          within("int8 x", quantizationBound(decoded, x, m, n, k))
        else {
          val comparison =
            Comparison.of(expected, actual, Tolerance(1e-5 * scale, 0))
          println(s"  $dtype [$m, $k]·[$n, $k]ᵀ: ${comparison.summary}")
          assert(comparison.passed)
        }
      }
    finally {
      hip.close()
      cpu.close()
    }
  }

  /** ROCmFP4 rows of whole 256s meet int8 x whatever the inputs (`HipOps`). */
  private def rocmFp4Int8(dtype: DType, k: Int): Boolean =
    (dtype == RocmFp4.Dual || dtype == RocmFp4.Fast) && k % 256 == 0

  private val types: Seq[(String, DType, Boolean)] = Seq(
    ("Q8_0", GgmlQuants.Q8_0, true),
    ("Q4_0", GgmlQuants.Q4_0, true),
    ("Q4_K", KQuants.Q4_K, true),
    ("Q6_K", KQuants.Q6_K, true),
    ("Q5_1", GgmlQuants.Q5_1, true),
    ("Q5_K", KQuants.Q5_K, true),
    ("IQ4_NL", IQuants.IQ4_NL, true),
    ("IQ4_XS", IQuants.IQ4_XS, true),
    ("ROCmFP4", RocmFp4.Dual, true),
    ("ROCmFP4 fast", RocmFp4.Fast, true),
    ("F32", DType.F32, false),
    ("F16", DType.F16, false),
    ("BF16", DType.BF16, false)
  )

  /** ComfyUI's rotated linears read in place (`Ops.rotated`), int8 and 4-bit:
    * the GPU's decode is the `Cpu` backend's bit for bit, and the product the
    * BF16 GEMM's at any row count.
    */
  private def checkRotated(fourBit: Boolean): Unit = {
    val cpu = new CpuOps
    val hip = new HipOps(Gpu.hip, MatVecInputs.Float)
    try
      Seq(
        (1, 3, 256),
        (5, 33, 768),
        (40, 72, 2560),
        (64, 300, 4096)
      ).zipWithIndex
        .foreach { case ((m, n, k), index) =>
          val random = new java.util.Random(index)
          val codes = new Array[Byte](if (fourBit) n * k / 2 else n * k)
          random.nextBytes(codes)
          val scales = Array.fill(n)(0.0005f + random.nextFloat() * 0.002f)
          val codebook = Array.tabulate(16)(i => (i - 7.5f) / 7.6f)
          // E4M3 relative scales of 64 to 240: some levels clamp at ±127
          val relative = Array.fill(n * k / 16)(
            ((13 + random.nextInt(2)) << 3 | random.nextInt(8)).toByte
          )
          val x = TestData.gaussian(200 + index, m * k)
          def run(ops: Ops): (Array[Float], Array[Float]) = {
            val stored =
              if (fourBit) Shape.of(n, k / 2) else Shape.of(n, k)
            val weight = ops.rotated(
              ops.fromBytes(DType.I8, stored, codes),
              ConvRotParts(
                ops.fromFloats(Shape.of(n), scales),
                Option.when(fourBit)(
                  ConvRotLevels(
                    ops.fromFloats(Shape.of(16), codebook),
                    ops.fromBytes(DType.F8E4M3, Shape.of(n, k / 16), relative),
                    16
                  )
                )
              )
            )
            val bf16 = ops.allocate(DType.BF16, Shape.of(n, k))
            val decoded = ops.allocate(DType.F32, Shape.of(n, k))
            ops.convert(weight, bf16)
            ops.convert(bf16, decoded)
            val out = ops.allocate(DType.F32, Shape.of(m, n))
            ops.linear(ops.fromFloats(Shape.of(m, k), x), weight, out)
            (ops.toFloats(decoded), ops.toFloats(out))
          }
          val (expectedWeights, expected) = run(cpu)
          val (weights, actual) = run(hip)
          assert(weights.sameElements(expectedWeights))
          val scale = expected.map(math.abs).max.toDouble
          val bounds = productSums(expectedWeights, x, m, n, k)
            .zip(expected)
            .map((sum, y) => (sum * 2 + math.abs(y)) * math.pow(2, -8) * 1.01)
          val worst = expected.indices.map { i =>
            math.abs(actual(i) - expected(i)) / (bounds(i) + 1e-5 * scale)
          }.max
          println(
            f"  rotated ${if (fourBit) "4-bit" else "int8"} [$m, $k]·[$n, $k]ᵀ: worst error at ${worst * 100}%.1f%% of its bound"
          )
          assert(worst <= 1.0)
        }
    finally {
      hip.close()
      cpu.close()
    }
  }

  /** The experts' products (`expertsLinear`, `expertsGatedLinear`) as the `Cpu`
    * backend computes them: rows shorter than a wave's chunk (several rows per
    * wave), a model's width, one x per token or per slot, and enough slots to
    * be grouped by expert.
    */
  private def checkExperts(dtype: DType): Unit = {
    val cpu = new CpuOps
    val hip = new HipOps(Gpu.hip, MatVecInputs.Float)
    val experts = 5
    try
      Seq(
        (6, 2, 512),
        (6, 6, 512),
        (4, 4, 2048),
        (6, 2, 256 max dtype.blockElements),
        // a prompt's many slots, grouped by expert
        (200, 25, 512),
        (130, 130, 512),
        (96, 12, 2048)
      )
        .filter((_, _, k) => dtype.byteSize(k) % 4 == 0)
        .zipWithIndex
        .foreach { case ((slots, xRows, k), index) =>
          val n = 24
          val gate = TestData.quantized(dtype, experts * n, k, seed = index)
          val up = TestData.quantized(dtype, experts * n, k, seed = 50 + index)
          val x = TestData.gaussian(100 + index, xRows * k)
          val ids = Array.tabulate(slots)(s => (s * 3 + index) % experts)
          def run(ops: Ops, gated: Boolean): Array[Float] = {
            val shape = Shape.of(experts, n, k)
            val out = ops.allocate(DType.F32, Shape.of(slots, n))
            val input = ops.fromFloats(Shape.of(xRows, k), x)
            val chosen = ops.fromInts(Shape.of(slots), ids)
            if (gated)
              ops.expertsGatedLinear(
                input,
                ops.fromBytes(dtype, shape, gate),
                ops.fromBytes(dtype, shape, up),
                chosen,
                out
              )
            else
              ops.expertsLinear(
                input,
                ops.fromBytes(dtype, shape, gate),
                chosen,
                out
              )
            ops.toFloats(out)
          }
          Seq(false, true).foreach { gated =>
            val expected = run(cpu, gated)
            val scale = expected.map(math.abs).max.toDouble
            val comparison =
              Comparison.of(
                expected,
                run(hip, gated),
                Tolerance(1e-5 * scale, 0)
              )
            println(
              s"  $dtype experts${
                  if (gated) " gated" else ""
                } $slots slots of [$n, $k], $xRows x: ${comparison.summary}"
            )
            assert(comparison.passed)
          }
        }
    finally {
      hip.close()
      cpu.close()
    }
  }

  /** `linears` over mixed types, as a layer's input projections: the products
    * of one type share launches (four at most), the others run alone.
    */
  private def checkSeveral(m: Int, k: Int): Unit = {
    val cpu = new CpuOps
    val hip = new HipOps(Gpu.hip, MatVecInputs.Float)
    val matrices = Seq(
      (KQuants.Q4_K, 40),
      (KQuants.Q6_K, 9),
      (KQuants.Q4_K, 7),
      (KQuants.Q4_K, 1),
      (KQuants.Q4_K, 33),
      (KQuants.Q4_K, 16),
      (KQuants.Q6_K, 24),
      (DType.F32, 5),
      (RocmFp4.Fast, 36),
      (RocmFp4.Fast, 2),
      (RocmFp4.Dual, 12),
      (RocmFp4.Dual, 30)
    )
    val weights = matrices.zipWithIndex.map { case ((dtype, n), index) =>
      TestData.quantized(dtype, n, k, seed = index)
    }
    val x = TestData.gaussian(7, m * k)
    def run(ops: Ops): Seq[Array[Float]] = {
      val outs =
        matrices.map((_, n) => ops.allocate(DType.F32, Shape.of(m, n)))
      ops.linears(
        ops.fromFloats(Shape.of(m, k), x),
        matrices.zip(weights).map { case ((dtype, n), bytes) =>
          ops.fromBytes(dtype, Shape.of(n, k), bytes)
        },
        outs
      )
      outs.map(ops.toFloats)
    }
    try
      run(cpu).zip(run(hip)).zip(matrices).zip(weights).foreach {
        case (((expected, actual), (dtype, n)), bytes) =>
          val scale = expected.map(math.abs).max.toDouble
          if (rocmFp4Int8(dtype, k)) {
            val decoded =
              dtype.decode(
                java.lang.foreign.MemorySegment.ofArray(bytes),
                n * k
              )
            val bounds = quantizationBound(decoded, x, m, n, k)
            val worst = expected.indices.map { i =>
              math.abs(actual(i) - expected(i)) / (bounds(i) + 1e-5 * scale)
            }.max
            println(
              f"  $dtype [$m, $k]·[$n, $k]ᵀ among several (int8 x): worst error at ${worst * 100}%.1f%% of its bound"
            )
            assert(worst <= 1.0)
          } else {
            val comparison =
              Comparison.of(expected, actual, Tolerance(1e-5 * scale, 0))
            println(
              s"  $dtype [$m, $k]·[$n, $k]ᵀ among several: ${comparison.summary}"
            )
            assert(comparison.passed)
          }
      }
    finally {
      hip.close()
      cpu.close()
    }
  }

  val tests = Tests {
    test("several products") {
      checkSeveral(1, 2048)
      checkSeveral(3, 512)
      checkSeveral(2, 4096)
    }
    test("experts") {
      types.filter(_._3).foreach((_, dtype, _) => checkExperts(dtype))
    }
    test("int8 inputs") {
      types.foreach((_, dtype, integer) =>
        check(dtype, integer, MatVecInputs.Int8)
      )
    }
    test("rotated linears, int8 and 4-bit, read in place") {
      checkRotated(fourBit = false)
      checkRotated(fourBit = true)
    }
    test("float inputs") {
      types.foreach((_, dtype, integer) =>
        check(dtype, integer, MatVecInputs.Float)
      )
    }
  }
}
