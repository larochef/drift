package drift.runner

import drift.runner.models.{ComfyQuantStorage, WeightSource}
import drift.runner.ops.{CpuOps, HipOps, MatVecInputs}
import drift.runner.tensor.{ConvRot, DType, Shape, Storage}

import java.nio.file.{Path, Paths}

/** ComfyUI's rotated linears read where the file holds them against decoded at
  * load (`bugs/53`), on a real checkpoint: the GPU's decode against the CPU's
  * value by value on the smallest and the largest layer of each kind, the
  * decode's speed, and for each way of holding the weights the load, the
  * memory, and a pass over every layer — what each forward pass adds.
  *
  * `./mill runner.gpuTest.runMain drift.runner.ConvRotBenchmark <checkpoint>
  * [cpu]` — `cpu` also times the decode on the CPU.
  */
object ConvRotBenchmark {

  private val hip = Gpu.hip

  private def timed[A](body: => A): (A, Double) = {
    hip.synchronize()
    val started = System.nanoTime()
    val result = body
    hip.synchronize()
    (result, (System.nanoTime() - started) / 1e9)
  }

  private def weightNames(source: WeightSource): Seq[String] =
    source.names.toSeq.filter(_.endsWith(".weight")).sorted

  def main(arguments: Array[String]): Unit = {
    val path = Paths.get(
      arguments.headOption.getOrElse(sys.error("usage: <checkpoint> [cpu]"))
    )
    val ops = new HipOps(hip, MatVecInputs.Float)
    try {
      layers(ops, path)
      val storages =
        Seq(ComfyQuantStorage.InPlace, ComfyQuantStorage.DecodedOnGpu) ++
          Option.when(arguments.lift(1).contains("cpu"))(
            ComfyQuantStorage.Decoded
          )
      storages.foreach(checkpoint(ops, path, _))
    } finally ops.close()
  }

  /** The smallest and the largest layer of each kind decoded by the GPU against
    * the CPU's decode of the same file, the largest ones timed.
    */
  private def layers(ops: HipOps, path: Path): Unit = {
    val cpu = new CpuOps
    val inPlace = WeightSource.open(ops, path, ComfyQuantStorage.InPlace)
    val reference = WeightSource.open(cpu, path, ComfyQuantStorage.Decoded)
    try
      Seq(ConvRot.Codes, ConvRot.Nibbles).foreach { kind =>
        val ofKind = weightNames(inPlace)
          .map(name => name -> inPlace.linear(name))
          .filter(_._2.dtype == kind)
          .sortBy(_._2.shape.elementCount)
        println(
          f"${ofKind.size} $kind linears, ${ofKind.map(_._2.shape.elementCount).sum / 1e9}%.2f G values"
        )
        Seq(ofKind.headOption, ofKind.lastOption).flatten.distinct.foreach {
          (name, weight) =>
            val decoded = ops.allocate(DType.BF16, weight.shape)
            val floats = ops.allocate(DType.F32, weight.shape)
            ops.convert(weight, decoded)
            ops.convert(decoded, floats)
            val values = ops.toFloats(floats)
            val expected = reference(name)
            val Storage.Host(segment) = expected.storage: @unchecked
            val bits = segment
              .asSlice(expected.byteOffset, expected.byteSize)
              .toArray(java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED)
            var (different, index) = (0, 0)
            while (index < values.length) {
              if (CpuOps.toBfloat16(values(index)) != bits(index))
                different += 1
              index += 1
            }
            println(
              s"  $name ${weight.shape}: $different of ${values.length} values differ"
            )
            val passes = 20
            ops.convert(weight, decoded)
            val (_, seconds) =
              timed((0 until passes).foreach(_ => ops.convert(weight, decoded)))
            val count = weight.shape.elementCount.toDouble * passes
            println(
              f"    ${seconds / passes * 1000}%.2f ms a decode, ${count / seconds / 1e9}%.1f G values/s, " +
                f"${2 * count / seconds / 1e9}%.0f GB/s of BF16"
            )
            ops.release(floats)
            ops.release(decoded)
        }
      }
    finally {
      reference.close()
      inPlace.close()
      cpu.close()
    }
  }

  /** The whole checkpoint held one way: the load, the GPU memory it takes, and
    * with weights left in the file a decode of every layer.
    */
  private def checkpoint(
      ops: HipOps,
      path: Path,
      storage: ComfyQuantStorage
  ): Unit = {
    val (source, opening) = timed(WeightSource.open(ops, path, storage))
    try {
      val (tensors, loading) = timed(weightNames(source).map(source.linear(_)))
      val allocated = tensors
        .filter(_.storage.isInstanceOf[Storage.Device])
        .map(_.byteSize)
        .sum
      println(
        f"$storage: opened in $opening%.1f s, ${tensors.size} weights in $loading%.1f s, " +
          f"${allocated / 1e9}%.1f GB allocated on the GPU"
      )
      val rotated = tensors.filter(tensor => ConvRot.stored(tensor.dtype))
      if (rotated.nonEmpty) {
        val scratch = ops.allocate(
          DType.BF16,
          Shape.of(rotated.map(_.shape.elementCount).max)
        )
        (1 to 3).foreach { pass =>
          val (_, seconds) = timed(
            rotated.foreach(weight =>
              ops.convert(weight, scratch.copy(shape = weight.shape))
            )
          )
          val values = rotated.map(_.shape.elementCount).sum
          println(
            f"  pass $pass: every layer decoded in $seconds%.2f s " +
              f"(${values / seconds / 1e9}%.1f G values/s)"
          )
        }
        ops.release(scratch)
      }
    } finally source.close()
  }
}
