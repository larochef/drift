package drift.backend.session

import java.nio.file.Files

import utest.*

/** The machine's figures for the status socket: the idle time of a GPU that
  * waits, and the driver's files.
  */
object MachineMonitorTests extends TestSuite {

  private val Second = 1000000000L
  private val Gigabyte = 1000L * 1000 * 1000

  val tests = Tests {

    test("the idle time grows in steps while the GPU waits, and ends with it") {
      var time = 0L
      var percent = 2
      val monitor = MachineMonitor(
        thresholdPercent = () => 10,
        readGpu =
          () => Some(MachineMonitor.Gpu(Some(percent), Some(60 * Gigabyte))),
        readMemory = () => Some(MemoryHeadroom(127 * Gigabyte, 8 * Gigabyte)),
        now = () => time
      )
      assert(monitor.status.isEmpty)
      monitor.sample()
      val first = monitor.status.get
      assert(first.gpuIdleSeconds == 0)
      assert(first.memoryLow)
      assert(first.gpuMemoryBytes.contains(60 * Gigabyte))
      time = 12 * Second
      monitor.sample()
      assert(monitor.status.get.gpuIdleSeconds == 10)
      time = 16 * Second
      percent = 95
      monitor.sample()
      val busy = monitor.status.get
      assert(busy.gpuIdleSeconds == 0)
      assert(busy.gpuPercent.contains(95))
      assert(busy.history.map(_.gpuPercent) == List(Some(2), Some(2), Some(95)))
      assert(busy.history.forall(_.memoryLeftBytes == 8 * Gigabyte))
    }

    test("the history keeps the last five minutes, the newest last") {
      var percent = 0
      val monitor = MachineMonitor(
        thresholdPercent = () => 10,
        readGpu = () => Some(MachineMonitor.Gpu(Some(percent % 100), None)),
        readMemory = () => Some(MemoryHeadroom(127 * Gigabyte, 50 * Gigabyte)),
        now = () => 0L
      )
      (1 to 200).foreach { reading =>
        percent = reading
        monitor.sample()
      }
      val status = monitor.status.get
      assert(!status.memoryLow)
      assert(status.history.size == 151)
      assert(status.history.last.gpuPercent.contains(0))
      assert(status.history.head.gpuPercent.contains(50))
    }

    test("the GPU is read from the first card that reports a load") {
      val root = Files.createTempDirectory("drm")
      val other = Files.createDirectories(root.resolve("card0/device"))
      Files.writeString(other.resolve("vendor"), "0x10de\n")
      val device = Files.createDirectories(root.resolve("card1/device"))
      Files.writeString(device.resolve("gpu_busy_percent"), "34\n")
      Files.writeString(device.resolve("mem_info_vram_used"), "2000000000\n")
      Files.writeString(device.resolve("mem_info_gtt_used"), "58000000000\n")
      Files.createDirectories(root.resolve("card1-eDP-1"))
      val gpu = MachineMonitor.readGpu(root).get
      assert(gpu.percent.contains(34))
      assert(gpu.memoryBytes.contains(60 * Gigabyte))
      assert(MachineMonitor.readGpu(root.resolve("card0")).isEmpty)
    }
  }
}
