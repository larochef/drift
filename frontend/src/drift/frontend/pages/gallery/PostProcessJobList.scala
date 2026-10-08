package drift.frontend.pages.gallery

import drift.frontend.components.{Component, LogProgressView}
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** The post-processing jobs on the shown output: running with their progress
  * and a way to stop them, failed with the tail of the process output, or done
  * with the result to open. A finished card closes with its ×, so a second
  * redraw of the same output is not read as the first one's result.
  */
class PostProcessJobList(
    image: Signal[Option[GenerationOutput]],
    jobs: Signal[List[PostProcessJob]],
    machine: Signal[Option[MachineStatus]],
    onOpen: String => Unit,
    onCancel: String => Unit,
    /** Stops a tiled job after the tile in flight, keeping its tiles — or,
      * forced, as soon as that tile can be dropped
      * (`specs/40-pause-and-resume.md`).
      */
    onPause: (String, Boolean) => Unit,
    onResume: String => Unit,
    onDismiss: String => Unit
) extends Component {

  private val progressTag = htmlTag("progress")

  private def jobLabel(job: PostProcessJob): String = job.kind match {
    case "pid"                      => "PiD upscale"
    case SeedVr2UpscaleRequest.Kind => "SeedVR2 upscale"
    case "redraw"                   => "Redraw"
    case "edit"                     => "Edit"
    case _                          => "Upscale"
  }

  /** Closes a finished job's card. Only the notice goes: the job stays in the
    * list the service holds, and its result stays in the gallery. A running job
    * has no × — its card carries the only way to stop it.
    */
  private def closeButton(job: PostProcessJob): HtmlElement =
    button(
      cls := "delete is-small ml-2",
      title := "close this notice — the result stays in the gallery",
      onClick --> (_ => onDismiss(job.id))
    )

  private def jobRow(job: PostProcessJob): HtmlElement = job.state match {
    case PostProcessState.Running =>
      div(
        cls := "notification is-info is-light py-2 px-3 is-size-7 mb-1",
        job.progress match {
          case Some(progress) =>
            List(
              span(
                if (progress.completed < progress.total)
                  s"${jobLabel(job)} running — tile ${progress.completed + 1} of ${progress.total}" +
                    PostProcessJobList.remaining(job, progress)
                else
                  s"${jobLabel(job)} running — blending ${progress.total} tiles"
              ),
              progressTag(
                cls := "progress is-small is-info mt-1",
                value := progress.completed.toString,
                maxAttr := progress.total.toString
              )
            )
          // ESRGAN alone runs blind: sd-cli prints no bar for it
          case None if job.kind == "upscale" =>
            List(
              span(
                s"${jobLabel(job)} running… (sd-cli reports no progress; a " +
                  "4× pass on a 1024² image takes about a minute)"
              ),
              progressTag(cls := "progress is-small is-info mt-1")
            )
          // a job that is one run on a server (SeedVR2): its own bar, read
          // from the log below, is the progress
          case None => List(span(s"${jobLabel(job)} running"))
        },
        // What the run inside the current tile is doing, the way the inference
        // panel shows it: the bar sd-cpp is drawing, or its last line.
        LogProgressView(
          jobs.map(all =>
            all.find(_.id == job.id) match {
              case Some(live) => (live.logProgress, live.activity)
              case None       => (job.logProgress, job.activity)
            }
          ),
          machine
        ).element,
        div(
          cls := "mt-1 buttons are-small",
          button(
            cls := "button is-small",
            "Stop",
            title := "kills the sd-cli run, or stops the server the job " +
              "runs on; what was written is removed",
            onClick --> (_ => onCancel(job.id))
          ),
          // Only a tiled job has tiles to keep; the others end in one run.
          if (job.progress.isEmpty) emptyNode
          else if (job.pauseRequested)
            button(
              cls := "button is-small is-warning",
              "Force pause",
              title := "drops the tile in flight instead of waiting for it; " +
                "that tile runs again on resume",
              onClick --> (_ =>
                if (
                  window.confirm(
                    "Stop the tile it is working on?\n\nEverything already " +
                      "done is kept, and that tile is run again when you " +
                      "resume."
                  )
                ) onPause(job.id, true)
              )
            )
          else
            button(
              cls := "button is-small",
              "Pause",
              title := "stops after the tile it is on, keeps every tile it " +
                "has and frees the GPU; Resume carries on where it stopped, " +
                "a drift restart in between or not",
              onClick --> (_ => onPause(job.id, false))
            )
        ),
        // The ask lands at once, the tile it waits on can be minutes: say so
        // rather than leave the click looking lost.
        if (job.pauseRequested)
          p(
            cls := "mt-1",
            "Pausing — it stops when this tile is done. Force pause drops " +
              "that tile instead."
          )
        else emptyNode
      )
    case PostProcessState.Paused =>
      div(
        cls := "notification is-warning is-light py-2 px-3 is-size-7 mb-1",
        p(
          cls := "has-text-weight-bold mb-1",
          job.progress.fold(s"${jobLabel(job)} paused")(progress =>
            s"${jobLabel(job)} paused — ${progress.completed} of " +
              s"${progress.total} tiles done"
          )
        ),
        job.progress.map(progress =>
          progressTag(
            cls := "progress is-small is-warning mt-1",
            value := progress.completed.toString,
            maxAttr := progress.total.toString
          )
        ),
        div(
          cls := "mt-1 buttons are-small",
          button(
            cls := "button is-small is-link",
            "Resume",
            title := "starts a server again and carries on at the first tile " +
              "it does not have",
            onClick --> (_ => onResume(job.id))
          ),
          button(
            cls := "button is-small",
            "Cancel",
            title := "drops the tiles it kept and forgets the job",
            onClick --> (_ => onCancel(job.id))
          )
        )
      )
    case PostProcessState.Failed =>
      div(
        cls := "notification is-danger is-light py-2 px-3 is-size-7 mb-1",
        div(
          cls := "is-flex is-align-items-center",
          p(
            cls := "has-text-weight-bold mb-1 is-flex-grow-1",
            s"${jobLabel(job)} failed: " +
              job.error.getOrElse("no reason given")
          ),
          closeButton(job)
        ),
        if (job.outputTail.isEmpty) emptyNode
        else
          pre(
            cls := "is-size-7",
            styleAttr := "white-space: pre-wrap; word-break: break-all; background: transparent; padding: 0; max-height: 12rem; overflow: auto;",
            job.outputTail.mkString("\n")
          )
      )
    case PostProcessState.Cancelled =>
      div(
        cls := "notification is-warning is-light py-2 px-3 is-size-7 mb-1",
        div(
          cls := "is-flex is-align-items-center",
          span(
            cls := "is-flex-grow-1",
            s"${jobLabel(job)} cancelled — nothing was kept."
          ),
          closeButton(job)
        )
      )
    case PostProcessState.Completed =>
      div(
        cls := "notification is-success is-light py-2 px-3 is-size-7 mb-1",
        div(
          cls := "is-flex is-align-items-center",
          span(
            cls := "is-flex-grow-1",
            s"${jobLabel(job)} done — the result is a new entry beside " +
              "this one. ",
            job.result.map(result =>
              button(
                cls := "button is-small ml-2",
                "Open result",
                onClick --> (_ => onOpen(result.id))
              )
            )
          ),
          closeButton(job)
        )
      )
  }

  lazy val element: HtmlElement = div(
    children <-- jobs
      .combineWith(image)
      .map { (all, shown) =>
        shown.toList.flatMap(output =>
          all.filter(job =>
            job.sourceDate == output.date &&
              job.sourceFileName == output.fileName
          )
        )
      }
      .distinct
      .map(_.map(jobRow))
  )
}

object PostProcessJobList {

  /** What is left of a tiled job, once a tile has been timed: the tiles still
    * to run at what a tile has cost this run (`specs/15-post-hoc-resize.md`).
    * Nothing until then — a first estimate from no measurement at all would
    * only be a guess dressed as a number.
    */
  def remaining(job: PostProcessJob, progress: PostProcessProgress): String =
    job.secondsPerTile.fold("") { seconds =>
      val left = (progress.total - progress.completed).max(0) * seconds
      s", about ${duration(left)} left"
    }

  /** A duration in words, to the unit that matters: "40 s", "6 min", "1 h 20".
    */
  def duration(seconds: Double): String = {
    val whole = math.round(seconds).toInt
    if (whole < 90) s"$whole s"
    else if (whole < 3600) s"${math.round(whole / 60.0).toInt} min"
    else {
      val hours = whole / 3600
      val minutes = math.round((whole % 3600) / 60.0).toInt
      if (minutes == 0) s"$hours h" else f"$hours h $minutes%02d"
    }
  }
}
