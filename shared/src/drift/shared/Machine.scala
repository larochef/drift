package drift.shared

/** What the machine is doing, now: in the sidebar on every page and under every
  * running bar, so a run that crawls can be told from one that computes.
  *
  * The memory left is not the kernel's `MemAvailable` but what is really spare
  * once the weights are counted (free pages plus the file cache nothing maps):
  * on a machine whose GPU shares the system memory, the figure that says
  * whether a run has room.
  */
case class MachineStatus(
    /** The GPU's load; `None` where the driver does not report one. */
    gpuPercent: Option[Int],
    /** For how long the GPU has had next to nothing to do, in steps of
      * [[MachineStatus.IdleStepSeconds]]; 0 while it works.
      */
    gpuIdleSeconds: Int,
    /** The memory the GPU holds, every process together. */
    gpuMemoryBytes: Option[Long],
    memoryTotalBytes: Long,
    memoryLeftBytes: Long,
    /** Under the threshold the memory warning uses. */
    memoryLow: Boolean,
    /** The last [[MachineStatus.HistorySeconds]], oldest first, one sample
      * every [[MachineStatus.SampleSeconds]] and this status's own figures
      * last: what the sidebar's graphs draw.
      */
    history: List[MachineSample]
)

/** The machine at one moment of its history. */
case class MachineSample(gpuPercent: Option[Int], memoryLeftBytes: Long)

object MachineStatus {

  /** The idle time moves in steps, so an idle machine sends a message every few
    * seconds and not at every sample.
    */
  val IdleStepSeconds = 5

  /** From this long idle beside a running bar, the line says so: shorter pauses
    * are a run's ordinary host work between two GPU passes.
    */
  val IdleNoticeSeconds = 15

  /** How often the machine is read. */
  val SampleSeconds = 2

  /** How far back the history goes. */
  val HistorySeconds = 300

  def gigabytes(bytes: Long): Long = math.round(bytes / 1e9)
}
