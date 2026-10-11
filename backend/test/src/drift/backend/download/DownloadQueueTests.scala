package drift.backend.download

import drift.backend.Background
import drift.shared.DownloadState

import java.nio.file.Paths
import scala.concurrent.duration.DurationInt

import ox.*
import ox.channels.Channel
import utest.*

/** What every download shares (`DownloadQueue`): its turn, its progress, its
  * cancel and its end — the fetching itself is given.
  */
object DownloadQueueTests extends TestSuite {

  private def next[T](channel: Channel[T]): T =
    timeout(5.seconds)(channel.receive())

  /** Waits until the transfer of `key` satisfies `reached`. */
  private def eventually(queue: DownloadQueue[String], key: String)(
      reached: Transfer => Boolean
  ): Transfer =
    timeout(5.seconds) {
      var transfer = queue.get(key)
      while (!transfer.exists(reached)) {
        sleep(5.millis)
        transfer = queue.get(key)
      }
      transfer.get
    }

  private def ended(queue: DownloadQueue[String], key: String): Transfer =
    eventually(queue, key)(!_.state.isActive)

  val tests = Tests {
    test("a transfer runs, says how far it is, and ends completed") {
      supervised {
        val queue = DownloadQueue[String](Background(), "test")
        val half = Channel.unlimited[Unit]
        val go = Channel.unlimited[Unit]
        queue.queue("a") { (_, progress) =>
          progress(50, Some(100))
          half.send(())
          next(go)
          DownloadOutcome.Completed(Paths.get("/nowhere"), 100)
        }
        next(half)
        val running = queue.get("a").get
        assert(running.state == DownloadState.Downloading)
        assert(running.downloadedBytes == 50, running.totalBytes == Some(100))
        go.send(())
        val done = ended(queue, "a")
        assert(done.state == DownloadState.Completed)
        assert(done.downloadedBytes == 100, done.completedAt.nonEmpty)
      }
    }
    test("a failure and a fetch that throws both end failed, with why") {
      supervised {
        val queue = DownloadQueue[String](Background(), "test")
        queue.queue("refused")((_, _) => DownloadOutcome.Failed("refused"))
        queue.queue("broken")((_, _) => throw RuntimeException("broken pipe"))
        assert(ended(queue, "refused").error == Some("refused"))
        val broken = ended(queue, "broken")
        assert(broken.state == DownloadState.Failed)
        assert(broken.error == Some("broken pipe"))
      }
    }
    test("a key has one transfer at a time, and a new one once it ended") {
      supervised {
        val queue = DownloadQueue[String](Background(), "test")
        val go = Channel.unlimited[Unit]
        val runs = Channel.unlimited[Int]
        def fetch(run: Int)(
            cancelled: () => Boolean,
            progress: (Long, Option[Long]) => Unit
        ): DownloadOutcome = {
          runs.send(run)
          next(go)
          DownloadOutcome.Completed(Paths.get("/nowhere"), 1)
        }
        queue.queue("a")(fetch(1))
        assert(next(runs) == 1)
        // asked again while it runs: the same transfer, no second fetch
        assert(queue.queue("a")(fetch(2)).state == DownloadState.Downloading)
        go.send(())
        ended(queue, "a")
        queue.queue("a")(fetch(3))
        assert(next(runs) == 3)
        go.send(())
        ended(queue, "a")
      }
    }
    test("a queued transfer is cancelled at once and never starts") {
      supervised {
        val queue = DownloadQueue[String](Background(), "test", workers = 1)
        val go = Channel.unlimited[Unit]
        val started = Channel.unlimited[String]
        def held(key: String)(
            cancelled: () => Boolean,
            progress: (Long, Option[Long]) => Unit
        ): DownloadOutcome = {
          started.send(key)
          next(go)
          DownloadOutcome.Completed(Paths.get("/nowhere"), 1)
        }
        queue.queue("first")(held("first"))
        queue.queue("second")(held("second"))
        assert(next(started) == "first")
        assert(queue.get("second").get.state == DownloadState.Queued)
        assert(queue.cancel("second").get.state == DownloadState.Cancelled)
        go.send(())
        ended(queue, "first")
        // the worker is free and the second never ran
        queue.queue("third")(held("third"))
        assert(next(started) == "third")
        go.send(())
        ended(queue, "third")
        assert(queue.get("second").get.state == DownloadState.Cancelled)
      }
    }
    test("a running transfer is told it was cancelled, and ends so") {
      supervised {
        val queue = DownloadQueue[String](Background(), "test")
        val started = Channel.unlimited[Unit]
        queue.queue("a") { (cancelled, _) =>
          started.send(())
          while (!cancelled()) sleep(5.millis)
          DownloadOutcome.Cancelled
        }
        next(started)
        queue.cancel("a")
        assert(ended(queue, "a").state == DownloadState.Cancelled)
        assert(queue.cancel("unknown").isEmpty)
      }
    }
    test("what needed no fetching is recorded as it is") {
      supervised {
        val queue = DownloadQueue[String](Background(), "test")
        val there =
          queue.settled("there", DownloadState.Completed, bytes = Some(7))
        assert(there.downloadedBytes == 7, there.totalBytes == Some(7))
        queue.settled("local", DownloadState.Failed, Some("nothing to fetch"))
        assert(queue.all.keySet == Set("there", "local"))
        assert(queue.get("local").get.error == Some("nothing to fetch"))
      }
    }
  }
}
