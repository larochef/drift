package drift.backend.download

import drift.backend.Background

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}
import scala.concurrent.duration.DurationInt
import scala.util.control.NonFatal

import ox.*

/** Interrupts the thread of one transfer when its reads stop for `stallMillis`,
  * or at once when it is cancelled.
  *
  * The JDK HttpClient has no read timeout on a response body: a connection that
  * dies without a reset — a network hiccup, a NAT dropping it — leaves the read
  * waiting forever, and the transfer holds its host slot with it. Seen on
  * 2026-09-28: two model downloads and a TheRock archive frozen together while
  * the network served new connections at full speed, and a cancel that never
  * took effect because the flag is only read between reads. Interrupting the
  * blocked read makes it throw; the chunk retries on a fresh connection.
  *
  * Taken from `StallWatches` on the transfer's own thread; `close` in a
  * `finally`.
  */
final private[download] class StallWatch(
    isCancelled: () => Boolean,
    stallMillis: Long,
    watches: StallWatches
) {
  private val worker = Thread.currentThread()
  private val lastRead = AtomicLong(System.currentTimeMillis())
  private val tripped = AtomicBoolean(false)

  /** Interrupts the transfer if it has gone silent or been cancelled since. */
  private[download] def check(): Unit =
    if (
      !tripped.get() && (isCancelled() ||
        System.currentTimeMillis() - lastRead.get() > stallMillis)
    ) {
      tripped.set(true)
      worker.interrupt()
    }

  /** Data arrived. */
  def touch(): Unit = lastRead.set(System.currentTimeMillis())

  /** The watch interrupted the transfer for silence, not for a cancel. */
  def stalled: Boolean = tripped.get() && !isCancelled()

  /** Stops watching, and clears the interrupt this watch raised so the thread
    * can retry.
    */
  def close(): Unit = {
    watches.forget(this)
    if (tripped.get()) Thread.interrupted()
  }
}

/** Every transfer's `StallWatch`, checked once a second by one fork of
  * `background`, however many transfers there are.
  */
final class StallWatches(background: Background) {
  private val active = ConcurrentHashMap.newKeySet[StallWatch]()

  background.start("drift-download-stall-watch") {
    forever {
      sleep(1.second)
      // One watch failing to check must not stop the others being checked.
      active.forEach(watch =>
        try watch.check()
        catch { case NonFatal(_) => () }
      )
    }
  }

  /** A watch over the transfer running on the calling thread. */
  private[download] def watch(
      isCancelled: () => Boolean,
      stallMillis: Long
  ): StallWatch = {
    val watch = StallWatch(isCancelled, stallMillis, this)
    active.add(watch)
    watch
  }

  private[download] def forget(watch: StallWatch): Unit =
    active.remove(watch).discard
}
