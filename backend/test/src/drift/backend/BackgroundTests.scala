package drift.backend

import java.util.concurrent.atomic.AtomicInteger
import scala.concurrent.duration.DurationInt

import ox.*
import ox.channels.Channel
import utest.*

/** Where every job, download and monitor of drift runs (`Background`,
  * `WorkQueue`).
  *
  * Worth testing because all of drift lives in one ox scope: a background
  * failure that reached it would end drift, and a queue that lost a worker
  * would stop downloading without a word.
  */
object BackgroundTests extends TestSuite {

  private def next[T](channel: Channel[T]): T =
    timeout(5.seconds)(channel.receive())

  /** Throws the interruption a stalled download is ended with (`StallWatch`).
    */
  private def interrupted(): Unit = {
    Thread.currentThread().interrupt()
    sleep(1.second)
  }

  val tests = Tests {

    test("work started in the background runs") {
      supervised {
        val done = Channel.unlimited[String]
        Background().start("test")(done.send("ran"))
        assert(next(done) == "ran")
      }
    }

    test("a failure ends that work, not the scope") {
      supervised {
        val background = Background()
        val done = Channel.unlimited[String]
        background.start("failing")(throw RuntimeException("boom"))
        background.start("interrupted")(interrupted())
        background.start("after")(done.send("ran"))
        assert(next(done) == "ran")
      }
    }

    test("a queue runs its work in order, one worker at a time") {
      supervised {
        val queue = WorkQueue(Background(), "test", 1)
        val done = Channel.unlimited[Int]
        (1 to 5).foreach(index => queue.submit(done.send(index)))
        assert((1 to 5).map(_ => next(done)) == (1 to 5))
      }
    }

    test("a queue runs no more than its workers at once") {
      supervised {
        val queue = WorkQueue(Background(), "test", 2)
        val running = AtomicInteger(0)
        val most = AtomicInteger(0)
        val done = Channel.unlimited[Unit]
        (1 to 6).foreach(_ =>
          queue.submit {
            most.accumulateAndGet(running.incrementAndGet(), math.max)
            sleep(50.millis)
            running.decrementAndGet()
            done.send(())
          }
        )
        (1 to 6).foreach(_ => next(done))
        assert(most.get() == 2)
      }
    }

    test("a piece that fails or is interrupted costs the queue no worker") {
      supervised {
        val queue = WorkQueue(Background(), "test", 1)
        val done = Channel.unlimited[String]
        queue.submit(throw RuntimeException("boom"))
        queue.submit(interrupted())
        queue.submit(done.send("ran"))
        assert(next(done) == "ran")
      }
    }
  }
}
