package drift.backend.session

import drift.shared.*

import java.util.concurrent.{ConcurrentHashMap, LinkedBlockingQueue}
import java.util.concurrent.atomic.AtomicLong
import scala.jdk.CollectionConverters.*

/** One session's captured output (`specs/13-log-streaming.md`).
  *
  * Bounded on purpose: a verbose load can print a lot, and the point of this
  * buffer is the live view, not the archive - the mirrored file on disk is the
  * archive and outlives it. Opening the view mid-load replays what is held here
  * and then follows, so nothing is missed by arriving late.
  *
  * sd-cpp redraws its progress bars in place, which arrives as one line per
  * redraw: those never accumulate here. Each replaces the last, so the tail
  * stays readable and the buffer is not spent on a bar nobody scrolls back to.
  */
final class SessionLog(capacity: Int = SessionLog.DefaultCapacity) {
  private val lines = scala.collection.mutable.ArrayDeque.empty[LogLine]
  private var lastWasProgress = false
  @volatile private var currentProgress: Option[SessionProgress] = None
  @volatile private var lastError: Option[String] = None
  @volatile private var lastActivity: Option[String] = None
  @volatile private var samplingPass: Option[String] = None

  private val subscribers =
    ConcurrentHashMap[Long, LinkedBlockingQueue[LogLine]]()
  private val nextSubscriberId = AtomicLong(0)

  /** Where the current phase has got to, or `None` between phases. */
  def progress: Option[SessionProgress] = currentProgress

  /** The last non-progress line, shortened: what the session is doing when
    * there is no bar. sd-cpp prefixes its level and source position and then
    * prints absolute paths, so the noise is dropped from the front and the
    * length capped - this is a status line, not the log.
    */
  def activity: Option[String] = lastActivity

  /** The most recent line that read like a failure - what a failed session
    * shows instead of making the user scroll a thousand lines.
    */
  def errorLine: Option[String] = lastError

  def snapshot: List[LogLine] = synchronized(lines.toList)

  /** Takes one captured line. Progress replaces the previous progress line
    * rather than being appended after it; any other line ends the burst, which
    * is what clears the bar when a phase finishes.
    */
  def append(text: String): Unit = {
    val cleaned = LogProgress.clean(text)
    if (cleaned.nonEmpty) {
      val line = LogLine(System.currentTimeMillis(), cleaned)
      LogProgress.parseWithRest(cleaned) match {
        case Some((progress, rest)) =>
          currentProgress = Some(
            if (progress.kind == ProgressKind.Sampling)
              progress.copy(note = samplingPass)
            else progress
          )
          synchronized {
            if (lastWasProgress && lines.nonEmpty) lines.remove(lines.size - 1)
            push(line)
            lastWasProgress = true
          }
          // sd-cpp sometimes glues the next log line onto the end of a redraw;
          // it is narrative, so it goes in as one.
          if (rest.nonEmpty) append(rest)
        case None =>
          LogProgress.samplingPassOf(cleaned).foreach(samplingPass = _)
          // A narrative line means the bar it followed is finished with.
          currentProgress = None
          if (LogProgress.looksLikeError(cleaned)) lastError = Some(cleaned)
          lastActivity = Some(SessionLog.summarise(cleaned))
          synchronized {
            push(line)
            lastWasProgress = false
          }
      }
      subscribers.values.asScala.foreach(queue => queue.offer(line))
    }
  }

  private def push(line: LogLine): Unit = {
    lines.append(line)
    while (lines.size > capacity) lines.remove(0)
  }

  /** Follows the log from now on. The queue is bounded: a subscriber that stops
    * draining loses lines rather than growing without limit, since a stalled
    * browser must never hold the reader thread up.
    */
  def subscribe(): (Long, LinkedBlockingQueue[LogLine]) = {
    val id = nextSubscriberId.incrementAndGet()
    val queue = LinkedBlockingQueue[LogLine](SessionLog.SubscriberQueueSize)
    subscribers.put(id, queue)
    (id, queue)
  }

  def unsubscribe(id: Long): Unit = subscribers.remove(id)

  /** The session is over: wake every follower so its stream can end. */
  def close(): Unit = {
    subscribers.keys.asScala.toList.foreach(unsubscribe)
    currentProgress = None
  }
}

object SessionLog {

  /** A few thousand lines: enough to hold a whole load with `-v` on, small
    * enough that a dozen dead sessions cost nothing worth measuring.
    */
  val DefaultCapacity: Int = 4000

  val SubscriberQueueSize: Int = 2000

  private val Prefix = """^\[[A-Z]+\s*\]\s*[\w.\-]+:\d+\s*-\s*""".r

  /** One log line as a status line: sd-cpp's `[INFO   ] file.cpp:123  - ` is
    * noise in front of the only part worth reading, and its paths are long
    * enough to fill a column on their own.
    */
  def summarise(line: String): String = {
    val body = Prefix.replaceFirstIn(line, "")
    if (body.length <= 110) body else body.take(109) + "…"
  }
}
