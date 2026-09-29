package drift.backend

import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger
import ox.*
import ox.channels.Channel

/** Where drift's background work runs: forks of the application's ox scope
  * (`main`), started from whatever thread asks — a request's, or another
  * fork's. Nothing drift starts outlives that scope, and nothing is a thread
  * of its own.
  *
  * Each piece of work is named for the logs, and fails alone: its fork is
  * unsupervised, so what it throws — logged — ends that work and never the
  * scope, and drift with it. That includes an interruption: a stalled
  * download is interrupted on purpose (`StallWatch`).
  */
final class Background(runner: InScopeRunner) {

  /** Starts `work` in a fork of its own. Returns at once. */
  def start(name: String)(work: => Unit): Unit =
    runner.async(forkUnsupervised(Background.guarded(name)(work)).discard)
}

object Background {

  /** The application's background, for the scope it is called in. */
  def apply()(using Ox): Background = new Background(inScopeRunner())

  private val logger = Logger[Background]

  /** Runs `work` under `name`, logging what it throws rather than passing it
    * on. An interruption ends it quietly: the scope ending, or the work
    * cancelled.
    */
  def guarded(name: String)(work: => Unit): Unit = {
    Thread.currentThread().setName(name)
    try work
    catch {
      case NonFatal(err)           => logger.error(s"$name failed", err)
      case _: InterruptedException => logger.debug(s"$name interrupted")
    }
  }
}

/** A fixed number of forks taking the work handed to them in turn, in the
  * order it was handed over: at most `workers` pieces run at once, and the
  * rest wait their turn without holding anything.
  *
  * Each piece runs in a fork of its own that its worker waits for, so an
  * interruption aimed at the work — a stalled download's — ends that piece
  * and never the worker; the worker itself is only interrupted by the scope
  * ending.
  */
final class WorkQueue(background: Background, name: String, workers: Int) {
  // Unbounded: handing work over never waits.
  private val queue = Channel.unlimited[() => Unit]

  (1 to workers).foreach(_ =>
    background.start(name) {
      forever {
        val work = queue.receive()
        unsupervised(
          forkUnsupervised(Background.guarded(name)(work())).join()
        )
      }
    }
  )

  /** Queues `work`. Returns at once. */
  def submit(work: => Unit): Unit = queue.send(() => work)
}
