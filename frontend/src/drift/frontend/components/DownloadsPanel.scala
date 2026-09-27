package drift.frontend.components

import drift.frontend.services.{ActiveDownload, GlobalDownloadsService}
import drift.shared.LaunchBlocker

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.StringAsIsCodec

/** The downloads corner of the sidebar, on every page: each current or pending
  * transfer with a progress bar, whichever page started it. Hidden while
  * nothing is moving.
  */
class DownloadsPanel(service: GlobalDownloadsService) extends Component {
  import DownloadsPanel.*

  private def row(item: ActiveDownload): HtmlElement = {
    val percent = item.totalBytes
      .filter(_ > 0)
      .map(total => (item.downloadedBytes * 100 / total).min(100))
    val sizeText = item.totalBytes
      .filter(_ > 0)
      .map(total =>
        item.unit match {
          case Some(unit) => s"${item.downloadedBytes} / $total $unit"
          case None       =>
            s"${LaunchBlocker.humanBytes(item.downloadedBytes)} / ${LaunchBlocker.humanBytes(total)}"
        }
      )
      .getOrElse(item.detail)
    div(
      cls := "mb-2",
      p(
        cls := "is-size-7 text-primary mb-0",
        styleAttr :=
          "white-space: nowrap; overflow: hidden; text-overflow: ellipsis;",
        title := s"${item.label}${
            if (item.detail.nonEmpty) s" — ${item.detail}" else ""
          }",
        s"${kindIcon(item.kind)} ${item.label}"
      ),
      // No value at all renders Bulma's indeterminate animation — honest for
      // queued/unpacking states with no byte counts.
      percent match {
        case Some(p) =>
          progressTag(
            cls := "progress is-info is-small mb-0",
            valueAttr := p.toString,
            maxAttr := "100"
          )
        case None =>
          progressTag(cls := "progress is-info is-small mb-0")
      },
      p(
        cls := "is-size-7 text-secondary mb-0",
        percent.map(p => s"$p% — ").getOrElse("") + sizeText
      )
    )
  }

  lazy val element: HtmlElement =
    div(
      cls := "px-2 py-2",
      styleAttr := "max-width: 230px;",
      child <-- service.activeDownloads.map {
        case Nil   => emptyNode
        case items =>
          // Only the transfers actually moving get a row — their number is
          // bounded by the download concurrency. The queue behind them is a
          // count, so a bulk install cannot grow the panel without limit.
          val (queued, transferring) = items.partition(_.queued)
          div(
            p(
              cls := "menu-label text-secondary mb-1",
              s"Downloads (${items.size})"
            ),
            transferring.map(row),
            if (queued.nonEmpty)
              p(
                cls := "is-size-7 text-secondary mb-0",
                s"${queued.size} pending download" +
                  (if (queued.size == 1) "" else "s")
              )
            else emptyNode
          )
      }
    )
}

object DownloadsPanel {

  // <progress> is not in Laminar's default bundle; its value/max are attrs.
  private val progressTag = htmlTag("progress")
  private val valueAttr = htmlAttr("value", StringAsIsCodec)
  private val maxAttr = htmlAttr("max", StringAsIsCodec)

  private def kindIcon(kind: String): String = kind match {
    case "model"      => "🧠"
    case "lora"       => "🎨"
    case "runtime"    => "⚙️"
    case "conversion" => "🔁"
    case _            => "⬇"
  }
}
