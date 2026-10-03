package drift.backend.session

import drift.shared.{BatchProgress, LogLine}

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

    test("a batch is followed image by image, in either engine's words") {
      val log = SessionLog()
      assert(log.batch.isEmpty)
      log.append(
        "[INFO ] stable-diffusion.cpp:3210 - generating image: 1/4 - seed 42"
      )
      assert(log.batch == Some(BatchProgress(1, 4)))
      log.append("generating image 2/4 (seed 43)")
      assert(log.batch == Some(BatchProgress(2, 4)))
    }

    test("the batch outlives the bars and the narrative inside an image") {
      val log = SessionLog()
      log.append("generating image 3/4 (seed 44)")
      log.append("  |=====>          | 3/20 - 1.42it/s")
      log.append(
        "[INFO ] stable-diffusion.cpp:3300 - sampling completed, taking 14.10s"
      )
      assert(log.batch == Some(BatchProgress(3, 4)))
      assert(log.batch.map(_.completed) == Some(2))
    }

    test("the job's closing line ends the batch, and so does the session") {
      val log = SessionLog()
      log.append("generating image 4/4 (seed 45)")
      log.append(
        "[INFO ] stable-diffusion.cpp:3400 - generate_image completed in 61.20s"
      )
      assert(log.batch.isEmpty)
      log.append("generating image 1/2 (seed 7)")
      log.close()
      assert(log.batch.isEmpty)
    }
  }
}
