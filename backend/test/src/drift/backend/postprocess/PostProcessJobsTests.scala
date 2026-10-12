package drift.backend.postprocess

import drift.backend.Background
import drift.backend.outputs.GenerationHistory
import drift.backend.storage.StorageService
import drift.protocol.{JobBar, JobProgress}
import drift.shared.*

import java.nio.file.Files
import scala.concurrent.duration.DurationInt

import ox.*
import ox.channels.Channel
import utest.*

/** A post-processing job from its start to its end (`PostProcessJobs`): what it
  * is while its thread runs, how a failure and a cancel are told apart, and
  * what is left of it once the thread has ended.
  *
  * Worth testing because a job's state is written from its own thread, from the
  * request that cancels it and from whatever lists it, and a cancel that reads
  * as a failure shows the user an error for what they asked.
  */
object PostProcessJobsTests extends TestSuite {

  private def jobsIn(using Ox): PostProcessJobs = {
    val outputs = Files.createTempDirectory("jobs-outputs")
    val storage = StorageService(Files.createTempDirectory("jobs-config"))
    storage.init
    PostProcessJobs(
      outputs,
      Files.createTempDirectory("jobs-logs"),
      GenerationHistory(outputs, _ => ()),
      storage,
      Background()
    )
  }

  private val source = {
    val output = GenerationOutput(
      date = "2026-01-01",
      fileName = "g1-0.png",
      url = "/api/outputs/2026-01-01/g1-0.png",
      mimeType = "image/png",
      format = "png"
    )
    PostProcessSource(
      "2026-01-01",
      "g1-0.png",
      Files.createTempFile("g1-0", ".png"),
      Generation(
        id = "g1",
        sessionId = "",
        runConfigurationId = "configuration",
        kind = "image",
        status = GenerationStatus.Completed,
        submittedAt = 0L,
        outputs = List(output),
        importedFileName = None,
        inputSources = Nil,
        warnings = Nil
      ),
      output
    )
  }

  private def next[T](channel: Channel[T]): T =
    timeout(5.seconds)(channel.receive())

  /** Waits for what a job's thread does last: letting its record go. */
  private def ended(jobs: PostProcessJobs, id: String): Unit =
    timeout(5.seconds) {
      while (jobs.isRunning(id)) sleep(5.millis)
    }

  private def stateOf(jobs: PostProcessJobs, id: String): PostProcessJob =
    jobs.list.find(_.id == id).get

  val tests = Tests {
    test("a job runs, says what its server job reports, and fails by name") {
      supervised {
        val jobs = jobsIn
        val reported = Channel.unlimited[Unit]
        val go = Channel.unlimited[Unit]
        val job = jobs.start("upscale", source) { job =>
          jobs.report(
            job,
            JobProgress(
              bar = Some(JobBar(JobBar.Sampling, 2.5, 4, 1.5, "s/it")),
              pass = None,
              image = None,
              images = None,
              stage = None,
              stages = None,
              stageName = None
            )
          )
          reported.send(())
          next(go)
          jobs.doneWaiting(job)
          jobs.fail(job, "the server went away")
        }
        next(reported)
        val running = stateOf(jobs, job.id)
        assert(running.state == PostProcessState.Running)
        val bar = running.logProgress.get
        assert(bar.kind == ProgressKind.Sampling, bar.done == 2, bar.total == 4)
        go.send(())
        ended(jobs, job.id)
        val failed = stateOf(jobs, job.id)
        assert(
          failed.state == PostProcessState.Failed,
          failed.error.contains("the server went away"),
          failed.logProgress.isEmpty
        )
      }
    }
    test("a cancelled job that then fails is a cancel, without a reason") {
      supervised {
        val jobs = jobsIn
        val started = Channel.unlimited[Unit]
        val job = jobs.start("upscale", source) { job =>
          started.send(())
          timeout(5.seconds) {
            while (!jobs.isCancelled(job)) sleep(5.millis)
          }
          jobs.fail(job, "killed")
        }
        next(started)
        assert(jobs.cancel(job.id))
        ended(jobs, job.id)
        val cancelled = stateOf(jobs, job.id)
        assert(
          cancelled.state == PostProcessState.Cancelled,
          cancelled.error.isEmpty
        )
        // nothing left to cancel
        assert(!jobs.cancel(job.id))
      }
    }
    test("a run that ends does not take away the record of the one after") {
      supervised {
        val jobs = jobsIn
        val firstMayEnd = Channel.unlimited[Unit]
        val secondStarted = Channel.unlimited[Unit]
        val secondMayEnd = Channel.unlimited[Unit]
        val job = jobs.start("redraw", source)(_ => next(firstMayEnd))
        // the same job carried on while its first run is still unwinding
        jobs.start("redraw", source, resuming = Some(job.id)) { _ =>
          secondStarted.send(())
          next(secondMayEnd)
        }
        next(secondStarted)
        firstMayEnd.send(())
        sleep(100.millis)
        assert(jobs.isRunning(job.id))
        secondMayEnd.send(())
        ended(jobs, job.id)
      }
    }
  }
}
