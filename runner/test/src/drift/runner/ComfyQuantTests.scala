package drift.runner

import utest.*

import drift.runner.formats.{ComfyQuant, FormatException}
import drift.runner.models.{ComfyQuantStorage, WeightSource}
import drift.runner.ops.CpuOps
import drift.runner.tensor.{
  Comparison,
  ConvRot,
  DType,
  Shape,
  Storage,
  Tensor,
  Tolerance
}

import java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED

object ComfyQuantTests extends TestSuite {

  val tests = Tests {
    test("int8 and 4-bit linears decode to the weights numpy unrotated") {
      Fixtures.withSafetensors("comfy_quant.safetensors") { file =>
        Fixtures.withSafetensors("comfy_quant.expected.safetensors") {
          expected =>
            val weights = ComfyQuant.weights(file)
            assert(weights.keySet == expected.tensors.keySet)
            // the 4-bit weight is stored two columns a byte
            assert(file("four.weight").shape == Shape.of(3, 256))
            weights.foreach { (name, weight) =>
              assert(weight.shape == expected(name).shape)
              val rows = weight.shape.dimensions(0).toInt
              val comparison = Comparison.of(
                expected(name).decode(),
                weight.rows(0, rows),
                Tolerance(1e-5, 1e-4)
              )
              assert(comparison.passed)
              // a later row alone is the same row
              val columns = weight.shape.dimensions(1).toInt
              val last = Comparison.of(
                expected(name).decode().drop((rows - 1) * columns),
                weight.rows(rows - 1, 1),
                Tolerance(1e-5, 1e-4)
              )
              assert(last.passed)
            }
        }
      }
    }
    test("rotated weights left in the file are the decoded ones, bit for bit") {
      val ops = new CpuOps
      def bits(tensor: Tensor): Seq[Short] = {
        val bf16 =
          if (tensor.dtype == DType.BF16) tensor
          else {
            val decoded = ops.allocate(DType.BF16, tensor.shape)
            ops.convert(tensor, decoded)
            decoded
          }
        val Storage.Host(segment) = bf16.storage: @unchecked
        segment
          .asSlice(bf16.byteOffset, bf16.byteSize)
          .toArray(JAVA_SHORT_UNALIGNED)
          .toSeq
      }
      def opened[A](storage: ComfyQuantStorage)(body: WeightSource => A): A = {
        val source = WeightSource
          .open(ops, Fixtures.path("comfy_quant.safetensors"), storage)
        try body(source)
        finally source.close()
      }
      try
        opened(ComfyQuantStorage.Decoded) { decoded =>
          val names = decoded.names.filter(_.endsWith(".weight")).toSeq
          opened(ComfyQuantStorage.InPlace) { inPlace =>
            assert(names.exists(inPlace.linear(_).dtype == ConvRot.Codes))
            assert(names.exists(inPlace.linear(_).dtype == ConvRot.Nibbles))
            names.foreach { name =>
              assert(bits(inPlace.linear(name)) == bits(decoded(name)))
              // asked as anything but a linear, a weight comes decoded
              assert(inPlace(name).dtype == DType.BF16)
              assert(bits(inPlace(name)) == bits(decoded(name)))
              val rows = decoded.shape(name).dimensions.head.toInt
              assert(
                inPlace.hostRows(name, 0, rows).map(CpuOps.toBfloat16).toSeq ==
                  bits(decoded(name))
              )
            }
          }
          opened(ComfyQuantStorage.DecodedOnGpu) { onGpu =>
            names.foreach { name =>
              assert(onGpu(name).dtype == DType.BF16)
              assert(bits(onGpu(name)) == bits(decoded(name)))
            }
          }
        }
      finally ops.close()
    }
    test("the rotation is its own inverse") {
      val random = new java.util.Random(3)
      val values = Array.fill(512)(random.nextGaussian().toFloat)
      val twice = values.clone()
      ComfyQuant.unrotate(twice, 0, 512, 256)
      assert(!Comparison.of(values, twice, Tolerance(1e-3, 1e-3)).passed)
      ComfyQuant.unrotate(twice, 0, 512, 256)
      assert(Comparison.of(values, twice, Tolerance(1e-5, 1e-4)).passed)
    }
    test("a ConvRot group the runner does not undo is refused") {
      val marker =
        """{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":64}"""
      val header =
        s"""{"m.weight":{"dtype":"I8","shape":[1,64],"data_offsets":[0,64]},""" +
          s""""m.weight_scale":{"dtype":"F32","shape":[1,1],"data_offsets":[64,68]},""" +
          s""""m.comfy_quant":{"dtype":"U8","shape":[${marker.length}],"data_offsets":[68,${68 + marker.length}]}}"""
      val json = header.getBytes("UTF-8")
      val buffer = java.nio.ByteBuffer
        .allocate(8 + json.length + 68 + marker.length)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
      buffer.putLong(json.length.toLong).put(json)
      buffer.position(8 + json.length + 68)
      buffer.put(marker.getBytes("UTF-8"))
      val file = drift.runner.formats.Safetensors
        .read(java.lang.foreign.MemorySegment.ofArray(buffer.array()), "test")
      val error = assertThrows[FormatException](ComfyQuant.weights(file))
      assert(error.getMessage.contains("64"))
    }
  }
}
