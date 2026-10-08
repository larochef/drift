package drift.frontend.components

import drift.shared.MachineStatus

import com.raquo.laminar.api.L.*

/** The machine in one line, under a running bar: whether the GPU works and what
  * memory is left, where the user looks when a run seems stuck. A GPU that has
  * waited for a while is said in so many words — the run is then reading from
  * disk, or working on the processor, not computing.
  */
class MachineLine(status: Signal[Option[MachineStatus]]) extends Component {

  lazy val element: HtmlElement = div(
    child <-- status.distinct.map {
      case None          => emptyNode
      case Some(machine) =>
        val gpu = machine.gpuPercent.map(percent =>
          MachineLine
            .idle(machine)
            .fold(s"GPU $percent %")(idle => s"GPU $idle")
        )
        val memory =
          s"${MachineStatus.gigabytes(machine.memoryLeftBytes)} of " +
            s"${MachineStatus.gigabytes(machine.memoryTotalBytes)} GB of " +
            "memory left"
        p(
          cls := "is-size-7 mt-1 mb-0",
          cls := (if (MachineLine.idle(machine).isDefined || machine.memoryLow)
                    "has-text-warning"
                  else "text-secondary"),
          (gpu.toList :+ memory).mkString(" · ")
        )
    }
  )
}

object MachineLine {

  /** "idle for 40 s", once the GPU has waited long enough to be worth saying.
    */
  def idle(machine: MachineStatus): Option[String] =
    Option.when(machine.gpuIdleSeconds >= MachineStatus.IdleNoticeSeconds)(
      s"idle for ${machine.gpuIdleSeconds} s"
    )
}
