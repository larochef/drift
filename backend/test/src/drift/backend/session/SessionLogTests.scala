package drift.backend.session

import drift.protocol.{JobBar, JobProgress}
import drift.shared.LogProgress
import drift.shared.{
  BatchProgress,
  LogLine,
  ProgressKind,
  SessionProgress,
  StageProgress
}

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

    test("a video's decode has its own bar after the last step") {
      val log = SessionLog()
      log.append(
        "  |==================================================| 8/8 - 7.90s/it"
      )
      assert(log.progress.map(_.kind) == Some(ProgressKind.Sampling))
      log.append("sampling done, decoding 60 latent frames (VAE)")
      assert(log.progress.isEmpty)
      log.append(
        "  |=================                                 | 34/100 - 1.87%/s"
      )
      assert(
        log.progress == Some(
          SessionProgress(ProgressKind.Decoding, 34, 100, "1.87%/s")
        )
      )
    }

    test("the runner's inputs being read, and a long step redrawn within") {
      val log = SessionLog()
      log.append("reading 1 references: 239 × 768 × 768 (VAE)")
      log.append(
        "  |==========                                        | 48/240 - 0.61tiles/s"
      )
      assert(
        log.progress == Some(
          SessionProgress(ProgressKind.Preparing, 48, 240, "0.61tiles/s")
        )
      )
      log.append("sampling using Euler method")
      log.append(
        "  |=====================                             | 3.42/8 - 58.80s/it"
      )
      assert(
        log.progress == Some(
          SessionProgress(ProgressKind.Sampling, 3, 8, "58.80s/it", 0.42)
        )
      )
      assert(log.progress.map(_.percent) == Some(43))
      // 4.58 steps at 58.8 s
      assert(
        log.progress.flatMap(_.remaining) ==
          Some("about 4 min left for this step")
      )
    }

    test("a video's stages, from its opening line to its closing one") {
      val log = SessionLog()
      log.append("generating 56 frames of 512 × 512 (seed 42)")
      assert(log.stage.isEmpty)
      log.append("stage 1/5: reading the prompt")
      log.append("stage 2/5: sampling")
      log.append(
        "  |======                                            | 1/8 - 12.40s/it"
      )
      assert(log.stage == Some(StageProgress(2, 5, "sampling")))
      assert(log.stage.map(_.completed) == Some(1))
      log.append("generate_video completed in 140.20s")
      assert(log.stage.isEmpty)
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
    test("a job's own report reads as the bar its log would have drawn") {
      val told = JobProgress(
        bar = Some(JobBar(JobBar.Sampling, 2.5, 8, 1.25, "s/it")),
        pass = Some("high noise")
      )
      val bar = SessionManager.barOf(told).get
      assert(bar.kind == ProgressKind.Sampling)
      assert(bar.done == 2, bar.total == 8, bar.within == 0.5)
      assert(bar.note == Some("high noise"))
      // the time left is worked out from the rate, as for a log's bar
      assert(bar.remainingSeconds.exists(left => math.abs(left - 6.875) < 1e-6))
      // the same bar, as the runner prints it for the log
      val drawn = LogProgress.parse("  |===  | 2.50/8 - 1.25s/it").get
      assert(drawn.done == bar.done, drawn.within == bar.within)
      assert(drawn.detail == bar.detail)
      assert(SessionManager.barOf(JobProgress()).isEmpty)
      val decode =
        JobProgress(bar = Some(JobBar(JobBar.Decoding, 40, 100, 2, "%/s")))
      assert(
        SessionManager.barOf(decode).map(_.kind) == Some(ProgressKind.Decoding)
      )
    }
  }
}
