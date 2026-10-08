package drift.frontend.components

import drift.shared.MachineStatus

import com.raquo.laminar.api.L.*

/** A figure of the machine over its last minutes: the curve takes the colour of
  * the third each value sits in (low, middle, high), two dashed lines mark
  * where it changes, a thin line each minute and one at 0 and at the top. Under
  * the pointer, the value at that moment and how long ago it was.
  */
class MachineGraph(
    /** Told apart from the other graphs on the page (the colours' gradient). */
    name: String,
    /** Oldest first, one every `MachineStatus.SampleSeconds`, the newest at the
      * right edge; `None` where there was no reading.
      */
    points: Signal[Vector[Option[MachineGraph.Point]]]
) extends Component {
  import MachineGraph.*

  private val gradient = s"machine-zones-$name"

  /** The sample under the pointer, counted back from the newest. */
  private val hovered = Var(Option.empty[Int])

  private val shown: Signal[Option[(Point, Int)]] =
    hovered.signal.combineWith(points).map { (back, all) =>
      back.flatMap(b => all.lift(all.size - 1 - b).flatten.map(_ -> b))
    }

  private def x(back: Int): Int = Width - back * MachineStatus.SampleSeconds

  private def y(level: Double): Double =
    math.round((Bottom - level.max(0).min(100) / 100 * (Bottom - Top)) * 10) /
      10.0

  private def path(all: Vector[Option[Point]]): String = {
    val drawn = new StringBuilder
    var drawing = false
    all.zipWithIndex.foreach { (point, index) =>
      point match {
        case Some(value) =>
          drawn
            .append(if (drawing) 'L' else 'M')
            .append(x(all.size - 1 - index))
            .append(' ')
            .append(y(value.level))
            .append(' ')
          drawing = true
        case None => drawing = false
      }
    }
    drawn.toString
  }

  private def ago(back: Int): String = {
    val seconds = back * MachineStatus.SampleSeconds
    if (seconds == 0) "now"
    else if (seconds < 60) s"$seconds s ago"
    else
      s"${seconds / 60}:${if (seconds % 60 < 10) "0" else ""}${seconds % 60} ago"
  }

  lazy val element: HtmlElement = div(
    cls := "machine-graph",
    svg.svg(
      svg.viewBox := s"0 0 $Width $Height",
      svg.preserveAspectRatio := "none",
      svg.defs(
        svg.linearGradient(
          svg.idAttr := gradient,
          svg.gradientUnits := "userSpaceOnUse",
          svg.x1 := "0",
          svg.y1 := Bottom.toString,
          svg.x2 := "0",
          svg.y2 := Top.toString,
          Zones.flatMap((zone, from, to) =>
            Seq(from, to).map(offset =>
              svg.stop(
                svg.offsetAttr := offset,
                svg.cls := s"machine-zone-$zone"
              )
            )
          )
        )
      ),
      (0 to MachineStatus.HistorySeconds by 60).map(second =>
        svg.line(
          svg.cls := "minute",
          svg.x1 := second.toString,
          svg.y1 := Top.toString,
          svg.x2 := second.toString,
          svg.y2 := Bottom.toString
        )
      ),
      // the frame's other two sides: nothing and everything
      Seq(Bottom, Top).map(edge =>
        svg.line(
          svg.cls := "minute",
          svg.x1 := "0",
          svg.y1 := edge.toString,
          svg.x2 := Width.toString,
          svg.y2 := edge.toString
        )
      ),
      Seq(100.0 / 3, 200.0 / 3).map(level =>
        svg.line(
          svg.cls := "threshold",
          svg.x1 := "0",
          svg.y1 := y(level).toString,
          svg.x2 := Width.toString,
          svg.y2 := y(level).toString
        )
      ),
      svg.path(
        svg.cls := "curve",
        svg.stroke := s"url(#$gradient)",
        svg.d <-- points.map(path)
      )
    ),
    children <-- shown.map {
      case None                => Nil
      case Some((point, back)) =>
        val left = x(back) * 100.0 / Width
        List(
          span(cls := "cursor", styleAttr := s"left: $left%;"),
          span(
            cls := s"marker machine-fill-${zoneOf(point.level)}",
            styleAttr := s"left: $left%; top: ${y(point.level) * 100 / Height}%;"
          ),
          // kept inside the graph's width: the sidebar clips what leaves it
          span(
            cls := "tip",
            styleAttr := s"left: ${left.max(30).min(70)}%;",
            s"${point.text} · ${ago(back)}"
          )
        )
    },
    onMouseMove --> { event =>
      val bounds = element.ref.getBoundingClientRect()
      if (bounds.width > 0) {
        val fraction =
          ((event.clientX - bounds.left) / bounds.width).max(0.0).min(1.0)
        hovered.set(
          Some(
            math
              .round((1 - fraction) * Width / MachineStatus.SampleSeconds)
              .toInt
          )
        )
      }
    },
    onMouseLeave.mapTo(None) --> hovered
  )
}

object MachineGraph {

  /** A value on the graph: how high it sits, 0 to 100, and how it reads. */
  case class Point(level: Double, text: String)

  // The drawing's own units: a second across, and the room the curve's
  // thickness needs above and below.
  private val Width = MachineStatus.HistorySeconds
  private val Height = 40
  private val Top = 2
  private val Bottom = 38

  /** The thirds, bottom up, as the gradient's stops: each keeps one colour. */
  private val Zones = Seq(
    ("low", "0", "0.3333"),
    ("middle", "0.3333", "0.6667"),
    ("high", "0.6667", "1")
  )

  private def zoneOf(level: Double): String =
    if (level < 100.0 / 3) "low"
    else if (level < 200.0 / 3) "middle"
    else "high"
}
