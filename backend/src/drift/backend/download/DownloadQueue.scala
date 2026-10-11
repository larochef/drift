package drift.backend.download

import drift.backend.{Background, WorkQueue}
import drift.shared.DownloadState

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** One transfer as its queue knows it: where it stands, how far it is, and why
  * it failed. Times are milliseconds.
  */
case class Transfer(
    state: DownloadState,
    downloadedBytes: Long,
    totalBytes: Option[Long],
    error: Option[String],
    startedAt: Option[Long],
    completedAt: Option[Long]
)

/** A queue of downloads, each known by its key — a model, a LoRA's file, an
  * upscaler: at most `workers` transfers at once (saturating the link with
  * several 15 GB fetches helps nobody), one transfer per key at a time, and the
  * list of them all for whoever shows it.
  *
  * What a transfer fetches, and where to, is its caller's; the queue owns what
  * every transfer shares: waiting its turn, its progress, being cancelled, and
  * how it ended.
  */
final class DownloadQueue[K](
    background: Background,
    name: String,
    workers: Int = 2
) {
  private val logger = Logger[DownloadQueue[?]]

  final private class Entry(
      @volatile var transfer: Transfer,
      val cancelled: AtomicBoolean = AtomicBoolean(false)
  )
  private val entries = ConcurrentHashMap[K, Entry]()

  private val work = WorkQueue(background, name, workers)

  private def now: Option[Long] = Some(System.currentTimeMillis())

  def all: Map[K, Transfer] =
    entries.asScala.view.mapValues(_.transfer).toMap

  def get(key: K): Option[Transfer] = Option(entries.get(key)).map(_.transfer)

  /** Stops a transfer: a queued one at once, a running one at its next read
    * (its partial file is its fetcher's to keep for a resume). `None` for a key
    * the queue does not know.
    */
  def cancel(key: K): Option[Transfer] =
    Option(entries.get(key)).map { entry =>
      entry.cancelled.set(true)
      if (entry.transfer.state == DownloadState.Queued)
        entry.transfer = entry.transfer
          .copy(state = DownloadState.Cancelled, completedAt = now)
      entry.transfer
    }

  /** Lets go of what the queue knows of `key` — its subject is gone. A transfer
    * still running is told to stop first.
    */
  def forget(key: K): Unit =
    Option(entries.remove(key)).foreach(_.cancelled.set(true))

  /** Records a transfer that needed no work: the file is already there, or
    * nothing could be fetched for the reason given.
    */
  def settled(
      key: K,
      state: DownloadState,
      error: Option[String] = None,
      bytes: Option[Long] = None
  ): Transfer = {
    val transfer = Transfer(
      state,
      downloadedBytes = bytes.getOrElse(0L),
      totalBytes = bytes,
      error = error,
      startedAt = None,
      completedAt = now
    )
    entries.put(key, Entry(transfer))
    transfer
  }

  /** Queues a transfer for `key`, unless one is queued or running already: the
    * transfer as it stands either way. `fetch` does the work when its turn
    * comes, given a way to know it was cancelled and a way to say how far it is
    * (bytes done, and the total once known).
    */
  def queue(key: K, totalBytes: Option[Long] = None)(
      fetch: (() => Boolean, (Long, Option[Long]) => Unit) => DownloadOutcome
  ): Transfer = synchronized {
    get(key).filter(_.state.isActive).getOrElse {
      val entry = Entry(
        Transfer(DownloadState.Queued, 0L, totalBytes, None, None, None)
      )
      entries.put(key, entry)
      work.submit(run(key, entry, fetch))
      entry.transfer
    }
  }

  private def run(
      key: K,
      entry: Entry,
      fetch: (() => Boolean, (Long, Option[Long]) => Unit) => DownloadOutcome
  ): Unit =
    // Cancelled while it waited: it never starts.
    if (!entry.cancelled.get()) {
      entry.transfer = entry.transfer
        .copy(state = DownloadState.Downloading, startedAt = now)
      val outcome =
        try
          fetch(
            () => entry.cancelled.get(),
            (done, total) =>
              entry.transfer = entry.transfer.copy(
                downloadedBytes = done,
                totalBytes = total.orElse(entry.transfer.totalBytes)
              )
          )
        catch {
          case NonFatal(err) =>
            logger.warn(s"$name: the download of $key blew up", err)
            DownloadOutcome.Failed(
              Option(err.getMessage).getOrElse(err.toString)
            )
        }
      entry.transfer = outcome match {
        case DownloadOutcome.Completed(_, bytes) =>
          entry.transfer.copy(
            state = DownloadState.Completed,
            downloadedBytes = bytes,
            totalBytes = entry.transfer.totalBytes.orElse(Some(bytes)),
            completedAt = now
          )
        case DownloadOutcome.Cancelled =>
          entry.transfer
            .copy(state = DownloadState.Cancelled, completedAt = now)
        case DownloadOutcome.Failed(reason) =>
          logger.warn(s"$name: the download of $key failed: $reason")
          entry.transfer.copy(
            state = DownloadState.Failed,
            error = Some(reason),
            completedAt = now
          )
      }
    }
}
