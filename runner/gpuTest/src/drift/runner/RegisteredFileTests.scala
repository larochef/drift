package drift.runner

import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.nio.file.Files
import java.nio.{ByteBuffer, ByteOrder}

import utest.*

import drift.runner.native.RegisteredFile
import drift.runner.ops.{CpuOps, HipOps, MatVecInputs}
import drift.runner.tensor.{DType, Shape, Storage, Tensor}

/** A mapped file the GPU reads in place, with no copy. */
object RegisteredFileTests extends TestSuite {

  val tests = Tests {
    test("both backends read a registered file's values in place") {
      val values = TestData.gaussian(6, 4096)
      val bytes =
        ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN)
      values.foreach(bytes.putFloat)
      val path = Files.createTempFile("registered", ".bin")
      Files.write(path, bytes.array())
      val file = new RegisteredFile(Gpu.hip, path)
      val hip = new HipOps(Gpu.hip, MatVecInputs.Int8)
      val cpu = new CpuOps
      try {
        assert(file.host.toArray(JAVA_FLOAT).sameElements(values))
        val shape = Shape.of(values.length)
        val mapped = Tensor(
          DType.F32,
          shape,
          Storage.Registered(file.host, file.device),
          0
        )
        // mapped + 0, computed by the GPU straight from the mapping
        val out = hip.allocate(DType.F32, shape)
        hip.add(
          mapped,
          hip.fromFloats(shape, Array.fill(values.length)(0f)),
          out
        )
        assert(hip.toFloats(out).sameElements(values))
        assert(cpu.toFloats(mapped).sameElements(values))
      } finally {
        cpu.close()
        hip.close()
        file.close()
        Files.delete(path)
      }
    }
  }
}
