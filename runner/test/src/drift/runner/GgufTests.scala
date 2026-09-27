package drift.runner

import java.lang.foreign.MemorySegment
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.{ByteBuffer, ByteOrder}

import utest.*

import drift.runner.formats.{FormatException, Gguf, GgufFile, GgufValue}
import drift.runner.tensor.{
  Comparison,
  DType,
  GgmlQuants,
  KQuants,
  RocmFp4,
  Shape,
  Tolerance
}

/** `tiny.gguf` was written by gguf-py; its values by the references. */
object GgufTests extends TestSuite {

  private lazy val bytes = Files.readAllBytes(Fixtures.path("tiny.gguf"))

  private def read(content: Array[Byte]): GgufFile =
    Gguf.read(MemorySegment.ofArray(content), "tiny")

  /** A copy of the fixture with the type id of tensor `name` replaced. */
  private def withType(name: String, typeId: Int): Array[Byte] = {
    val copy = bytes.clone()
    val key = name.getBytes(StandardCharsets.UTF_8)
    // a tensor info: u64 name length, the name, u32 rank, u64 dims, u32 type
    val at = copy.indexOfSlice(key) + key.length
    val rank =
      ByteBuffer.wrap(copy, at, 4).order(ByteOrder.LITTLE_ENDIAN).getInt
    ByteBuffer
      .wrap(copy, at + 4 + 8 * rank, 4)
      .order(ByteOrder.LITTLE_ENDIAN)
      .putInt(typeId)
    copy
  }

  val tests = Tests {
    test("metadata of every value type") {
      val file = read(bytes)
      assert(file.version == 3)
      assert(file.architecture == "llama")
      assert(file.long("test.u8") == 200)
      assert(file.long("test.i8") == -100)
      assert(file.long("test.u16") == 60000)
      assert(file.long("test.i16") == -30000)
      assert(file.long("test.u32") == 4000000000L)
      assert(file.long("test.i32") == -2000000000)
      assert(file.double("test.f32") == 1.5)
      assert(
        file.metadata("test.u64") == GgufValue.Integer(BigInt(2).pow(63) + 5)
      )
      assert(file.long("test.i64") == -(1L << 62))
      assert(file.double("test.f64") == 2.25)
      assert(file.boolean("test.bool"))
      assert(file.string("test.string") == "héllo")
      assert(file.strings("test.strings") == Seq("a", "bc"))
      assert(
        file.metadata("test.ints") ==
          GgufValue.Array(Vector(1, 2, 3).map(n => GgufValue.Integer(n)))
      )
      assert(file.alignment == 32)
    }
    test("tensors decode to the references' values") {
      val file = read(bytes)
      assert(file("f32").shape == Shape.of(3, 5)) // numpy's order, not GGUF's
      assert(file("f16").dtype == DType.F16)
      assert(file("q8_0").dtype == GgmlQuants.Q8_0)
      assert(file("q4_k").dtype == KQuants.Q4_K)
      assert(file("q4_0_rocmfp4").dtype == RocmFp4.Dual)
      assert(file("q4_k").shape == Shape.of(1, 512))
      Fixtures.withSafetensors("tiny.expected.safetensors") { expected =>
        assert(file.tensors.keySet == expected.tensors.keySet)
        file.tensors.values.foreach { tensor =>
          val comparison =
            Comparison.of(
              expected(tensor.name).decode(),
              tensor.decode(),
              Tolerance.Exact
            )
          assert(comparison.passed)
        }
      }
    }
    test("an unsupported type fails only when that tensor is asked for") {
      // IQ2_XXS blocks are 256 elements: the 512-element q4_k tensor stays whole
      val file = read(withType("q4_k", 16))
      assert(file("q8_0").dtype == GgmlQuants.Q8_0)
      val error = intercept[NoSuchElementException](file("q4_k"))
      assert(error.getMessage == "q4_k: GGML type IQ2_XXS is not supported yet")
    }
    test("a type whose blocks do not fit the file's spans is refused") {
      // Q8_0 blocks read as ROCmFP4: 34-byte blocks taken for 18-byte ones
      val error = intercept[FormatException](read(withType("q8_0", 100)))
      assert(error.getMessage.contains("q8_0 is Q4_0_ROCMFP4"))
      assert(error.getMessage.contains("but the file gives it"))
    }
    test("ROCmFP4 with a scale byte no ROCmFP4 file holds is refused") {
      val copy = bytes.clone()
      val file = read(bytes)
      val tensor = file("q4_0_rocmfp4")
      copy((tensor.fileOffset + 16).toInt) = 0x90.toByte // block 0, first scale
      val error = intercept[FormatException](read(copy))
      assert(error.getMessage.contains("scale byte 0x90"))
    }
    test("broken files are refused with a reason") {
      def refusal(content: Array[Byte]) =
        intercept[FormatException](read(content)).getMessage
      assert(refusal(bytes.take(bytes.length - 40)).contains("past the end"))
      assert(refusal(bytes.take(200)).contains("truncated"))
      assert(
        refusal(
          "GGML0000".getBytes(StandardCharsets.US_ASCII)
        ) == "tiny: not a GGUF file"
      )
      val bigEndian = bytes.clone()
      ByteBuffer.wrap(bigEndian, 4, 4).order(ByteOrder.BIG_ENDIAN).putInt(3)
      assert(refusal(bigEndian).contains("big-endian"))
    }
  }
}
