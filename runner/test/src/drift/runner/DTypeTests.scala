package drift.runner

import utest.*

import drift.runner.tensor.{
  Comparison,
  DType,
  GgmlQuants,
  IQuants,
  KQuants,
  RocmFp4,
  Tolerance
}

/** Every decoder against values computed by another implementation: gguf-py for
  * GGML's types, the fork's C reference for ROCmFP4, ml_dtypes for fp8. Exact:
  * the decoders repeat the reference's arithmetic, so they agree to the bit.
  */
object DTypeTests extends TestSuite {

  private def checkQuant(dtype: DType): Unit =
    Fixtures.withSafetensors(s"quants/${dtype.name.toLowerCase}.safetensors") {
      fixture =>
        val blocks = fixture("blocks")
        assert(blocks.shape.last == dtype.blockBytes)
        val expected = fixture("expected").decode()
        val actual = dtype.decode(blocks.bytes, expected.length)
        val comparison = Comparison.of(expected, actual, Tolerance.Exact)
        if (!comparison.passed) println(s"  $dtype: ${comparison.summary}")
        assert(comparison.passed)
    }

  val tests = Tests {
    test("GGML quants") {
      test("Q4_0") { checkQuant(GgmlQuants.Q4_0) }
      test("Q4_1") { checkQuant(GgmlQuants.Q4_1) }
      test("Q5_0") { checkQuant(GgmlQuants.Q5_0) }
      test("Q5_1") { checkQuant(GgmlQuants.Q5_1) }
      test("Q8_0") { checkQuant(GgmlQuants.Q8_0) }
    }
    test("K-quants") {
      test("Q2_K") { checkQuant(KQuants.Q2_K) }
      test("Q3_K") { checkQuant(KQuants.Q3_K) }
      test("Q4_K") { checkQuant(KQuants.Q4_K) }
      test("Q5_K") { checkQuant(KQuants.Q5_K) }
      test("Q6_K") { checkQuant(KQuants.Q6_K) }
    }
    test("I-quants") {
      test("IQ4_NL") { checkQuant(IQuants.IQ4_NL) }
      test("IQ4_XS") { checkQuant(IQuants.IQ4_XS) }
    }
    test("ROCmFP4") {
      test("dual scale") { checkQuant(RocmFp4.Dual) }
      test("fast") { checkQuant(RocmFp4.Fast) }
      test("half scales are half of unsigned E4M3") {
        assert(RocmFp4.halfScale(0x00) == 0f)
        assert(
          RocmFp4.halfScale(0x01) == java.lang.Math.scalb(1f, -10)
        ) // smallest subnormal
        assert(RocmFp4.halfScale(0x38) == 0.5f) // E4M3 1.0
        assert(RocmFp4.halfScale(0x7e) == 224f) // E4M3 448, the largest
        assert(RocmFp4.halfScale(0x7f) == 0f) // NaN code: the reference gives 0
      }
    }
    test("fp8, all 256 codes") {
      Fixtures.withSafetensors("fp8.safetensors") { fixture =>
        val codes = java.lang.foreign.MemorySegment
          .ofArray((0 until 256).map(_.toByte).toArray)
        for (
          (dtype, name) <- Seq(DType.F8E4M3 -> "e4m3", DType.F8E5M2 -> "e5m2")
        ) {
          val comparison =
            Comparison.of(
              fixture(name).decode(),
              dtype.decode(codes, 256),
              Tolerance.Exact
            )
          assert(comparison.passed)
        }
      }
    }
    test("byte sizes follow the blocks") {
      assert(KQuants.Q4_K.byteSize(512) == 288)
      assert(RocmFp4.Dual.byteSize(64) == 36)
      assert(RocmFp4.Fast.byteSize(64) == 34)
      val error =
        intercept[IllegalArgumentException](GgmlQuants.Q8_0.byteSize(33))
      assert(error.getMessage.contains("not whole Q8_0 blocks"))
    }
    test("type lookups") {
      assert(DType.fromGgmlId(12) == Right(KQuants.Q4_K))
      assert(DType.fromGgmlId(100) == Right(RocmFp4.Dual))
      assert(
        DType.fromGgmlId(16) == Left("GGML type IQ2_XXS is not supported yet")
      )
      assert(DType.fromGgmlId(999) == Left("unknown GGML type id 999"))
      assert(DType.fromSafetensorsName("F8_E4M3FN") == Right(DType.F8E4M3))
      assert(DType.fromSafetensorsName("I64").isLeft)
    }
  }
}
