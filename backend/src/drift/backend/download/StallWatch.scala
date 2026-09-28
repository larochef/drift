package drift.backend.download

import java.util.concurrent.{Executors, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}

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
  * Built on the transfer's own thread; `close` in a `finally`.
  */
final private[download] class StallWatch(
    isCancelled: () => Boolean,
    stallMillis: Long
) {
  private val worker = Thread.currentThread()
  private val lastRead = AtomicLong(System.currentTimeMillis())
  private val tripped = AtomicBoolean(false)

  private val check = StallWatch.scheduler.scheduleAtFixedRate(
    () =>
      if (
        !tripped.get() && (isCancelled() ||
          System.currentTimeMillis() - lastRead.get() > stallMillis)
      ) {
        tripped.set(true)
        worker.interrupt()
      },
    1,
    1,
    TimeUnit.SECONDS
  )

  /** Data arrived. */
  def touch(): Unit = lastRead.set(System.currentTimeMillis())

  /** The watch interrupted the transfer for silence, not for a cancel. */
  def stalled: Boolean = tripped.get() && !isCancelled()

  /** Stops watching, and clears the interrupt this watch raised so the thread
    * can retry.
    */
  def close(): Unit = {
    check.cancel(false)
    if (tripped.get()) Thread.interrupted()
  }
}

object StallWatch {
  private val scheduler = Executors.newSingleThreadScheduledExecutor { task =>
    val thread = Thread(task, "drift-download-stall-watch")
    thread.setDaemon(true)
    thread
  }
}
