package drift.runner

import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.lang.foreign.{Arena, MemorySegment}
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, Path, Paths, StandardOpenOption}

import drift.runner.native.{Dim3, KernelArgument, KernelModule, RegisteredFile}

/** How fast the GPU streams through device memory (`hipMalloc`) against a
  * registered file mapping, the question step 1 of `specs/42` settles.
  *
  * `./mill runner.gpuTest.runMain drift.runner.MemoryBandwidth <folder> [GiB]`
  * — the folder receives a test file of that size, all 1.0f, kept for reuse.
  */
object MemoryBandwidth {

  private val Blocks = 2048
  private val Iterations = 20

  def main(arguments: Array[String]): Unit = {
    val folder = Paths.get(
      arguments.headOption.getOrElse(sys.error("usage: <folder> [GiB]"))
    )
    val gibibytes = arguments.lift(1).fold(1)(_.toInt)
    val bytes = gibibytes.toLong << 30
    val hip = Gpu.hip
    println(
      s"${hip.deviceName}, ${bytes >> 20} MiB per pass, $Iterations passes"
    )

    val module = new KernelModule(hip, "bandwidth")
    val readSum = module.function("read_sum_f32x4")
    val partials = hip.allocate(Blocks * 4L)

    def measure(label: String, source: MemorySegment): Unit = {
      val (start, end) = (hip.createEvent(), hip.createEvent())
      def pass(): Unit = hip.launch(
        readSum,
        Dim3(Blocks),
        Dim3(256),
        0,
        MemorySegment.NULL,
        KernelArgument.Pointer(source),
        KernelArgument.I64(bytes / 16),
        KernelArgument.Pointer(partials)
      )
      pass() // warm-up: first touch, page tables, clocks
      hip.record(start, MemorySegment.NULL)
      (1 to Iterations).foreach(_ => pass())
      hip.record(end, MemorySegment.NULL)
      val milliseconds = hip.elapsedMilliseconds(start, end)
      val arena = Arena.ofConfined()
      val sum =
        try {
          val host = arena.allocate(Blocks * 4L, 64)
          hip.copy(host, partials, Blocks * 4L)
          host.toArray(JAVA_FLOAT).map(_.toDouble).sum
        } finally arena.close()
      val gigabytesPerSecond =
        bytes.toDouble * Iterations / (milliseconds / 1000) / 1e9
      println(
        f"$label%-22s $gigabytesPerSecond%7.1f GB/s  (sum ${sum}%.0f, expected ${bytes / 4}%d)"
      )
      hip.destroyEvent(start)
      hip.destroyEvent(end)
    }

    val device = hip.allocate(bytes)
    val arena = Arena.ofConfined()
    try {
      val ones = arena.allocate(bytes, 64)
      var offset = 0L
      while (offset < bytes) { ones.set(JAVA_FLOAT, offset, 1f); offset += 4 }
      hip.copy(device, ones, bytes)
    } finally arena.close()
    measure("device (hipMalloc)", device)
    hip.free(device)

    val file = testFile(folder.resolve(s"bandwidth-$gibibytes-gib.bin"), bytes)
    val registered = new RegisteredFile(hip, file)
    try measure("registered mapping", registered.device)
    finally registered.close()

    hip.free(partials)
    module.close()
  }

  private def testFile(path: Path, bytes: Long): Path = {
    if (!Files.exists(path) || Files.size(path) != bytes) {
      Files.createDirectories(path.getParent)
      val chunk =
        ByteBuffer.allocate(1 << 20).order(java.nio.ByteOrder.LITTLE_ENDIAN)
      while (chunk.hasRemaining) chunk.putFloat(1f)
      val channel = FileChannel.open(
        path,
        StandardOpenOption.CREATE,
        StandardOpenOption.WRITE,
        StandardOpenOption.TRUNCATE_EXISTING
      )
      try {
        var written = 0L
        while (written < bytes) {
          chunk.rewind()
          written += channel.write(chunk)
        }
      } finally channel.close()
    }
    path
  }
}
