package drift.runner

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.{ByteBuffer, ByteOrder}

import utest.*

import drift.runner.formats.{FormatException, Safetensors, SafetensorsModel}
import drift.runner.tensor.{Comparison, DType, Shape, Tolerance}

object SafetensorsTests extends TestSuite {

  /** A safetensors file from a header and a data section. */
  private def fileOf(
      header: String,
      data: Array[Byte]
  ): java.lang.foreign.MemorySegment = {
    val json = header.getBytes(StandardCharsets.UTF_8)
    val buffer = ByteBuffer
      .allocate(8 + json.length + data.length)
      .order(ByteOrder.LITTLE_ENDIAN)
    buffer.putLong(json.length.toLong).put(json).put(data)
    java.lang.foreign.MemorySegment.ofArray(buffer.array())
  }

  val tests = Tests {
    test("every dtype decodes to the values numpy and ml_dtypes wrote") {
      Fixtures.withSafetensors("mixed.safetensors") { file =>
        Fixtures.withSafetensors("mixed.expected.safetensors") { expected =>
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
          assert(file("bf16").dtype == DType.BF16)
          assert(file("f16").shape == Shape.of(2, 3))
        }
      }
    }
    test("metadata and scale tensors") {
      val mapped =
        new drift.runner.tensor.MappedFile(Fixtures.path("mixed.safetensors"))
      try {
        val file = Safetensors.read(mapped.segment, "mixed")
        assert(file.metadata == Map("format" -> "pt", "note" -> "fixture"))
        assert(
          file.scaleOf("layer.weight").map(_.name) == Some("layer.weight_scale")
        )
        assert(file.scaleOf("f32").isEmpty)
      } finally mapped.close()
    }
    test("a sharded model reads through its index") {
      Fixtures.withSafetensors("sharded/model.safetensors.index.json") {
        model =>
          assert(model.files.size == 2)
          assert(model.tensors.keySet == Set("a.weight", "b.weight", "c.bias"))
          assert(
            model("a.weight")
              .decode()
              .sameElements(Array(0f, 1f, 2f, 3f, 4f, 5f))
          )
          assert(model("c.bias").decode().sameElements(Array(7f)))
      }
    }
    test("an index naming a missing shard is refused") {
      val folder = Files.createTempDirectory("sharded")
      try {
        val index = folder.resolve("model.safetensors.index.json")
        Files.writeString(
          index,
          """{"weight_map": {"x": "model-00001-of-00001.safetensors"}}"""
        )
        val error = assertThrows[FormatException](SafetensorsModel.open(index))
        assert(error.getMessage.contains("which is missing"))
      } finally {
        Files.list(folder).forEach(Files.delete)
        Files.delete(folder)
      }
    }
    test("an unsupported dtype fails only when that tensor is asked for") {
      val file = Safetensors.read(
        fileOf(
          """{"ids":{"dtype":"I64","shape":[1],"data_offsets":[0,8]},"x":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}""",
          new Array[Byte](12)
        ),
        "test"
      )
      assert(file("x").decode().sameElements(Array(0f)))
      val error = assertThrows[NoSuchElementException](file("ids"))
      assert(error.getMessage == "ids: safetensors dtype I64 is not supported")
    }
    test("broken files are refused with a reason") {
      def refusal(header: String, data: Int) =
        assertThrows[FormatException](
          Safetensors.read(fileOf(header, new Array[Byte](data)), "test")
        ).getMessage
      assert(
        refusal("""{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}""", 4)
          .contains("past the end")
      )
      assert(
        refusal("""{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}""", 8)
          .contains("holds 8 bytes")
      )
      assert(refusal("""[1, 2]""", 0).contains("not a JSON object"))
      val truncated = java.lang.foreign.MemorySegment
        .ofArray(Array[Byte](100, 0, 0, 0, 0, 0, 0, 0, '{'))
      assert(
        assertThrows[FormatException](
          Safetensors.read(truncated, "test")
        ).getMessage.contains("runs past")
      )
    }
  }
}
