package drift.frontend.components

import drift.shared.{MachineSample, MachineStatus}

import com.raquo.laminar.api.L.*

/** The machine's corner of the sidebar, on every page: the GPU's load and the
  * memory really left over the last minutes, and what the GPU holds. A box of
  * its own, so it is not read as one more download. Always there, its frames
  * drawn: a figure the backend has not given yet, or that the driver does not
  * report, reads as a dash over an empty graph.
  */
class MachinePanel(status: Signal[Option[MachineStatus]]) extends Component {

  private def history(
      point: (MachineStatus, MachineSample) => Option[MachineGraph.Point]
  ): Signal[Vector[Option[MachineGraph.Point]]] =
    status.map(
      _.fold(Vector.empty)(machine =>
        machine.history.toVector.map(point(machine, _))
      )
    )

  private def figure(
      label: String,
      text: Signal[String],
      graph: MachineGraph
  ): HtmlElement =
    div(
      cls := "machine-figure",
      div(
        cls := "is-flex",
        span(cls := "text-secondary is-flex-grow-1", label),
        span(cls := "text-primary", child.text <-- text)
      ),
      graph.element
    )

  private def text(read: MachineStatus => String): Signal[String] =
    status.map(_.fold(MachinePanel.Unknown)(read)).distinct

  lazy val element: HtmlElement =
    div(
      cls := "machine-panel",
      p(cls := "machine-heading", "Machine"),
      figure(
        "GPU",
        text(_.gpuPercent.fold(MachinePanel.Unknown)(percent => s"$percent %")),
        MachineGraph(
          "gpu",
          history((_, sample) =>
            sample.gpuPercent.map(percent =>
              MachineGraph.Point(percent.toDouble, s"$percent %")
            )
          )
        )
      ),
      // The curve rises as the memory fills: high is nothing left.
      figure(
        "Memory left",
        text(machine =>
          s"${MachineStatus.gigabytes(machine.memoryLeftBytes)} of " +
            s"${MachineStatus.gigabytes(machine.memoryTotalBytes)} GB"
        ),
        MachineGraph(
          "memory",
          history((machine, sample) =>
            Some(
              MachineGraph.Point(
                100.0 * (machine.memoryTotalBytes - sample.memoryLeftBytes) /
                  machine.memoryTotalBytes.max(1L),
                s"${MachineStatus.gigabytes(sample.memoryLeftBytes)} GB left"
              )
            )
          )
        )
      ),
      p(
        cls := "text-secondary mb-0",
        title := "the memory the GPU holds, every process together: " +
          "the loaded models and what they compute on",
        child.text <-- text(machine =>
          machine.gpuMemoryBytes.fold(MachinePanel.Unknown)(bytes =>
            s"${MachineStatus.gigabytes(bytes)} GB"
          )
        ).map(held => s"On the GPU: $held")
      )
    )
}

object MachinePanel {

  /** A figure nothing has given. */
  val Unknown = "–"
}
