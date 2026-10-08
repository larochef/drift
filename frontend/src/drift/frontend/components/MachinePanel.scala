package drift.frontend.components

import drift.shared.{MachineSample, MachineStatus}

import com.raquo.laminar.api.L.*

/** The machine's corner of the sidebar, on every page: the GPU's load and the
  * memory really left over the last minutes, and what the GPU holds. A box of
  * its own, so it is not read as one more download. Hidden until the backend
  * has said anything.
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
    status.map(_.fold("")(read)).distinct

  lazy val element: HtmlElement =
    div(
      cls := "machine-panel",
      cls("is-hidden") <-- status.map(_.isEmpty).distinct,
      p(cls := "machine-heading", "Machine"),
      figure(
        "GPU",
        text(_.gpuPercent.fold("")(percent => s"$percent %")),
        MachineGraph(
          "gpu",
          history((_, sample) =>
            sample.gpuPercent.map(percent =>
              MachineGraph.Point(percent.toDouble, s"$percent %")
            )
          )
        )
      ).amend(
        cls("is-hidden") <-- status.map(_.forall(_.gpuPercent.isEmpty)).distinct
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
        cls("is-hidden") <-- status
          .map(_.forall(_.gpuMemoryBytes.isEmpty))
          .distinct,
        title := "the memory the GPU holds, every process together: " +
          "the loaded models and what they compute on",
        child.text <-- text(machine =>
          machine.gpuMemoryBytes.fold("")(bytes =>
            s"On the GPU: ${MachineStatus.gigabytes(bytes)} GB"
          )
        )
      )
    )
}
