package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** The cache tag plus, where it makes sense, a download or cancel control.
  * Shared by the architecture cards and the Model Cache page so the two cannot
  * drift apart.
  *
  * An active download takes precedence over the (stale until reloaded) cache
  * state; an absent status means the cache has not reported yet, which is not
  * the same as "missing" and must not look like it.
  */
class CacheIndicator(
    modelId: String,
    status: Option[ModelCacheStatus],
    job: Option[DownloadJob],
    onDownload: String => Unit,
    onCancelDownload: String => Unit
) extends Component {
  lazy val element: HtmlElement =
    job.filter(_.state.isActive) match {
      case Some(active) =>
        val percent = active.totalBytes
          .filter(_ > 0)
          .map(total => s"${(active.downloadedBytes * 100 / total).toInt}%")
          .getOrElse(LaunchBlocker.humanBytes(active.downloadedBytes))
        span(
          span(cls := "tag is-small is-info", s"downloading $percent"),
          button(
            cls := "button is-small ml-1",
            "✕",
            title := "Cancel download",
            onClick --> (_ => onCancelDownload(modelId))
          )
        )
      case None =>
        status match {
          case None =>
            // Not a control: the cache has not reported on this model yet.
            span(
              cls := "tag is-small",
              title := "cache status not loaded yet",
              "…"
            )
          case Some(reported) =>
            reported.state match {
              case CacheState.Cached =>
                span(
                  cls := "tag is-small is-success",
                  reported.bytes
                    .map(LaunchBlocker.humanBytes)
                    .getOrElse("on disk")
                )
              case CacheState.Missing =>
                span(
                  span(cls := "tag is-small is-warning", "not downloaded"),
                  button(
                    cls := "button is-small is-info ml-1",
                    "⬇",
                    title := "Download",
                    onClick --> (_ => onDownload(modelId))
                  ),
                  job.flatMap(_.error) match {
                    // The reason in full: a failure the user has to hover to
                    // find reads as "nothing happened".
                    case Some(reason) =>
                      span(
                        span(cls := "tag is-small is-danger ml-1", "failed"),
                        span(cls := "is-size-7 has-text-danger ml-1", reason)
                      )
                    case None => emptyNode
                  }
                )
              case CacheState.Broken =>
                span(cls := "tag is-small is-danger", "path missing")
            }
        }
    }
}
