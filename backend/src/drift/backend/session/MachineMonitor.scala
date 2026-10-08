package drift.backend.session

import drift.backend.Background
import drift.shared.{MachineSample, MachineStatus}

import java.nio.file.*
import scala.jdk.CollectionConverters.*
import scala.util.Try

/** The machine's load and memory for the status socket: read every
  * [[MachineStatus.SampleSeconds]] by a watch of its own, whether a page is
  * open or not, so the history a page gets on opening already covers the
  * minutes before — when a run that looks stuck was noticed.
  *
  * The GPU's figures come from the amdgpu driver's files; with another driver
  * they are absent and only the memory is reported.
  */
class MachineMonitor(
    /** The memory warning's threshold, as the settings have it now. */
    thresholdPercent: () => Int,
    readGpu: () => Option[MachineMonitor.Gpu] = () => MachineMonitor.readGpu(),
    readMemory: () => Option[MemoryHeadroom] = () => MemoryHeadroom.read(),
    now: () => Long = () => System.nanoTime()
) {
  import MachineMonitor.*

  private var idleSince = Option.empty[Long]
  private var history = Vector.empty[MachineSample]
  private var last = Option.empty[MachineStatus]

  /** The latest reading; `None` before the first, and where the machine's
    * memory cannot be read.
    */
  def status: Option[MachineStatus] = synchronized(last)

  /** Reads the machine every [[MachineStatus.SampleSeconds]], for as long as
    * drift runs.
    */
  def watch(background: Background): Unit =
    background.start("drift-machine") {
      while (true) {
        sample()
        Thread.sleep(MachineStatus.SampleSeconds * 1000L)
      }
    }

  /** One reading, added to the history. */
  def sample(): Unit = synchronized {
    val time = now()
    last = readMemory().map { memory =>
      val gpu = readGpu()
      idleSince =
        if (gpu.flatMap(_.percent).exists(_ < IdlePercent))
          idleSince.orElse(Some(time))
        else None
      val idleSeconds =
        idleSince.fold(0L)(since => (time - since) / 1000000000L)
      val left = rounded(memory.freeBytes)
      history = (history :+ MachineSample(gpu.flatMap(_.percent), left))
        .takeRight(HistorySamples)
      MachineStatus(
        gpuPercent = gpu.flatMap(_.percent),
        gpuIdleSeconds =
          (idleSeconds / MachineStatus.IdleStepSeconds * MachineStatus.IdleStepSeconds).toInt,
        gpuMemoryBytes = gpu.flatMap(_.memoryBytes).map(rounded),
        memoryTotalBytes = memory.totalBytes,
        memoryLeftBytes = left,
        memoryLow =
          memory.freeBytes * 100 < memory.totalBytes * thresholdPercent(),
        history = history.toList
      )
    }
  }
}

object MachineMonitor {

  /** What the driver says of the GPU, each figure where it has it. */
  case class Gpu(percent: Option[Int], memoryBytes: Option[Long])

  private val HistorySamples =
    MachineStatus.HistorySeconds / MachineStatus.SampleSeconds + 1

  /** Under this load the GPU counts as having nothing to do. */
  private val IdlePercent = 10

  /** To 100 MB: finer, the figure would change at every reading and cross the
    * socket each time for nothing anyone reads.
    */
  private def rounded(bytes: Long): Long = {
    val step = 100L * 1000 * 1000
    (bytes + step / 2) / step * step
  }

  private def number(file: Path): Option[Long] =
    Try(Files.readString(file).trim.toLong).toOption

  /** The first card whose driver reports a load (amdgpu's `gpu_busy_percent`),
    * its memory the dedicated and the shared together.
    */
  def readGpu(root: Path = Paths.get("/sys/class/drm")): Option[Gpu] =
    Try {
      val listing = Files.list(root)
      try
        listing
          .iterator()
          .asScala
          .filter(_.getFileName.toString.matches("card\\d+"))
          .map(_.resolve("device"))
          .toList
          .sortBy(_.toString)
      finally listing.close()
    }.getOrElse(Nil)
      .find(device => Files.isReadable(device.resolve("gpu_busy_percent")))
      .map { device =>
        val memory = List("mem_info_vram_used", "mem_info_gtt_used")
          .flatMap(name => number(device.resolve(name)))
        Gpu(
          number(device.resolve("gpu_busy_percent")).map(_.toInt),
          Option.when(memory.nonEmpty)(memory.sum)
        )
      }
}
