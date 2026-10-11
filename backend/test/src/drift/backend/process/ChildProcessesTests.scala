package drift.backend.process

import java.nio.file.Files
import drift.backend.Background

import scala.concurrent.duration.DurationInt

import ox.*
import utest.*

/** The processes drift starts (`ChildProcesses`): known while they live,
  * stopped together, and never waited on for ever.
  */
object ChildProcessesTests extends TestSuite {

  val tests = Tests {

    test("a tool's answer: its exit code and what it printed") {
      val answer = ChildProcesses
        .answer(List("sh", "-c", "printf hello; exit 3"), 5.seconds, 1024)
      assert(answer.map(_.exit) == Right(3))
      assert(answer.map(_.text) == Right("hello"))
    }

    test("a talkative tool is not blocked by what is not kept") {
      // Far more than a pipe holds: unread, the tool would never end.
      val answer = ChildProcesses.answer(
        List("sh", "-c", "head -c 4000000 /dev/zero; printf done >&2"),
        10.seconds,
        most = 16
      )
      assert(answer.map(_.exit) == Right(0))
      assert(answer.map(_.bytes.length) == Right(16))
    }

    test("a tool's answer is there though a helper it started lives on") {
      val started = System.nanoTime()
      val answer = ChildProcesses
        .answer(List("sh", "-c", "sleep 30 & printf started"), 10.seconds, 64)
      assert(answer.map(_.text) == Right("started"))
      assert((System.nanoTime() - started) / 1000000 < 5000)
    }

    test("a tool that outlasts its time is stopped, and that is the reason") {
      val started = System.nanoTime()
      val answer =
        ChildProcesses.answer(List("sleep", "60"), 300.millis, 1024)
      assert(answer.isLeft)
      assert((System.nanoTime() - started) / 1000000 < 15000)
    }

    test("a run followed to its end: every line, then the exit code") {
      supervised {
        val logFile = Files.createTempFile("child-processes", ".log")
        val lines = List.newBuilder[String]
        val process =
          ChildProcesses.start(List("sh", "-c", "echo one; echo two; exit 2"))
        val exit =
          ChildProcesses.runToEnd("test", process, logFile, None, Background())(
            lines += _
          )
        assert(exit == Some(2))
        assert(lines.result().take(2) == List("one", "two"))
      }
    }

    test("a run that outlasts its time is stopped and has no exit code") {
      supervised {
        val logFile = Files.createTempFile("child-processes", ".log")
        val process = ChildProcesses.start(List("sleep", "60"))
        val exit =
          ChildProcesses.runToEnd(
            "test",
            process,
            logFile,
            Some(300.millis),
            Background()
          )(_ => ())
        assert(exit.isEmpty)
        assert(!process.isAlive)
      }
    }

    test("a run ends though something else still holds its output open") {
      supervised {
        // The child leaves a grandchild holding the pipe, as a tool that
        // starts a helper does: the run must end with the child, not with the
        // helper seven seconds later.
        val logFile = Files.createTempFile("child-processes", ".log")
        val process =
          ChildProcesses.start(List("sh", "-c", "sleep 7 & echo started"))
        val started = System.nanoTime()
        val exit =
          ChildProcesses.runToEnd("test", process, logFile, None, Background())(
            _ => ()
          )
        assert(exit == Some(0))
        assert((System.nanoTime() - started) / 1000000 < 6500)
      }
    }

    test("stopping everything stops what was started") {
      val one = ChildProcesses.start(List("sleep", "60"))
      val two = ChildProcesses.start(List("sleep", "60"))
      ChildProcesses.stopAll()
      assert(!one.isAlive && !two.isAlive)
    }
  }
}
