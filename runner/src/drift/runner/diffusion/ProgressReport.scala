package drift.runner.diffusion

import drift.protocol.{JobBar, JobProgress}

import java.util.Locale

/** What the job under way says of its progress, said once for the two who read
  * it: the log, where a person follows it — a line, a bar redrawn — and the
  * job's document, where drift reads it without parsing the log.
  *
  * A pipeline, or the server around it, says where the job is (an image of the
  * batch, a stage, a sampling pass) and how far (a bar); whoever follows the
  * job is told of each change (`listener`).
  */
object ProgressReport {

  /** Told of each change, by the server for the job it runs. */
  @volatile var listener: JobProgress => Unit = _ => ()

  private var current = JobProgress()

  private def tell(next: JobProgress): Unit = synchronized {
    current = next
    listener(next)
  }

  /** A job begins: nothing is known of it yet. */
  def begin(): Unit = tell(JobProgress())

  /** The image of the batch being made, from 1. */
  def image(index: Int, count: Int, detail: String): Unit = {
    println(s"generating image $index/$count $detail".trim)
    tell(current.copy(bar = None, image = Some(index), images = Some(count)))
  }

  /** The stage the job enters, from 1. */
  def stage(index: Int, total: Int, name: String): Unit = {
    println(s"stage $index/$total: $name")
    tell(
      current.copy(
        bar = None,
        stage = Some(index),
        stages = Some(total),
        stageName = Some(name)
      )
    )
  }

  /** The sampling pass that starts: its name, for a model that has several. */
  def pass(name: Option[String]): Unit = {
    println(
      name.fold("sampling using Euler method")(pass =>
        s"sampling($pass) using Euler method"
      )
    )
    tell(current.copy(bar = None, pass = name))
  }

  /** How far what is under way is: drawn as the log's bar, and kept. */
  def bar(
      phase: String,
      done: Double,
      total: Int,
      rate: Double,
      unit: String
  ): Unit = {
    val filled =
      if (total > 0) math.min(50, math.max(0, (50 * done / total).toInt))
      else 50
    val count =
      if (done == done.floor) done.toInt.toString
      else String.format(Locale.ROOT, "%.2f", done)
    println(
      String.format(
        Locale.ROOT,
        "  |%s%s| %s/%d - %.2f%s",
        "=" * filled,
        " " * (50 - filled),
        count,
        total,
        rate,
        unit
      )
    )
    tell(current.copy(bar = Some(JobBar(phase, done, total, rate, unit))))
  }
}
