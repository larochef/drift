package drift.backend.session

import drift.shared.LogLine

import scala.concurrent.duration.DurationInt

import ox.*
import ox.channels.Source
import ox.flow.Flow
import utest.*

/** Following a session's log (`specs/13-log-streaming.md`): the log view's
  * stream is a follower's source, so it has to end when the session does and
  * must never hold the reader up.
  */
object SessionLogTests extends TestSuite {

  /** What a follower got, once its source is done. */
  private def texts(following: Source[LogLine]): List[String] =
    timeout(5.seconds)(Flow.fromSource(following).runToList()).map(_.text)

  val tests = Tests {

    test("a follower gets the lines appended once it follows, then ends") {
      val log = SessionLog()
      log.append("before")
      val (_, following) = log.subscribe()
      log.append("one")
      log.append("two")
      log.close()
      assert(texts(following) == List("one", "two"))
    }

    test("following a closed log ends at once") {
      val log = SessionLog()
      log.append("done")
      log.close()
      val (_, following) = log.subscribe()
      assert(texts(following) == Nil)
    }

    test("a follower that stops reading loses lines, not the reader") {
      val log = SessionLog()
      val (_, following) = log.subscribe()
      val lines = SessionLog.SubscriberQueueSize + 10
      timeout(5.seconds)((1 to lines).foreach(index => log.append(s"$index")))
      log.close()
      assert(texts(following).size == SessionLog.SubscriberQueueSize)
    }

    test("a follower that leaves is let go") {
      val log = SessionLog()
      val (id, following) = log.subscribe()
      log.unsubscribe(id)
      log.append("after")
      assert(texts(following) == Nil)
    }
  }
}
