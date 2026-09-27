package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.frontend.services.GenerationService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the session still has to do (`specs/08-inference-ui.md`): the job under
  * way and the ones waiting behind it, each with a stop of its own.
  *
  * The result under the form shows the newest submission, which is not
  * necessarily the one running — submit three and the panel shows the third,
  * queued, while the first is generating. Without this list the running job has
  * no button, and the waiting ones cannot be dropped at all, though that is the
  * cheapest cancel sd-cpp offers.
  *
  * The jobs listed are the **session's**, not the project's: a job queued from
  * another project holds this one up just the same, and dropping it is how the
  * queue clears.
  */
class GenerationQueue(
    /** Every generation of the watched session, oldest first. */
    generations: Signal[List[Generation]],
    /** The generation the result below is showing, which carries its own stop.
      */
    shown: Signal[Option[String]],
    capabilities: Option[SessionCapabilities],
    service: GenerationService
) extends Component {

  private val waiting: Signal[List[Generation]] =
    generations.map(_.filter(_.status.isActive)).distinct

  /** What the list has to show: nothing when the single active job is the one
    * the result already shows — a list of one would repeat its button. Any
    * other job, including one active in another project, stays listed: it holds
    * this session up and is nowhere else on screen.
    */
  private val rows: Signal[List[Generation]] =
    waiting
      .combineWith(shown)
      .map((jobs, shownId) =>
        if (jobs.sizeIs == 1 && shownId.contains(jobs.head.id)) Nil else jobs
      )
      .distinct

  private def row(generation: Generation, position: Int): HtmlElement = {
    val running = generation.status == GenerationStatus.Generating
    div(
      cls := "queue-row is-flex is-align-items-center is-size-7 py-1",
      span(
        cls := s"tag is-small mr-2 ${if (running) "is-info" else "is-light"}",
        if (running) "running" else s"#$position"
      ),
      span(
        cls := "text-secondary is-flex-grow-1 text-break mr-2",
        GenerationQueue.summaryOf(generation)
      ),
      GenerationCancel.stopButton(
        service,
        capabilities,
        generation,
        label = "⏹",
        classes = "button is-small",
        hint =
          if (running) "stop this generation"
          else "drop this job from the queue"
      )
    )
  }

  lazy val element: HtmlElement = div(
    child <-- rows.map {
      case Nil  => emptyNode
      case jobs =>
        div(
          // `bg-card`, as every other box in the app: a bare Bulma box comes
          // out unreadable on this theme.
          cls := "box bg-card p-2 mb-2",
          p(
            cls := "is-size-7 has-text-weight-bold mb-1",
            if (jobs.size == 1) "1 job on this session"
            else s"${jobs.size} jobs on this session"
          ),
          jobs.zipWithIndex.map((generation, index) =>
            row(generation, index + 1)
          )
        )
    }
  )
}

object GenerationQueue {

  /** A job in one line: its prompt, or what it is when there is none. */
  def summaryOf(generation: Generation): String = {
    val prompt = generation.imageParameters
      .map(_.prompt)
      .orElse(generation.videoParameters.map(_.prompt))
      .map(_.trim)
      .filter(_.nonEmpty)
      .getOrElse(generation.kind)
    if (prompt.length > 90) prompt.take(90) + "…" else prompt
  }
}
