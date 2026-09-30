package drift.backend.process

import drift.backend.Background

import java.nio.file.Files
import scala.concurrent.duration.DurationInt

import ox.*
import ox.channels.Channel
import utest.*

/** A child process's output (`ProcessOutput`): mirrored verbatim to its log,
  * handed over line by line — sd-cpp's bar redraws, ended by a carriage return,
  * included — and done once the stream ends.
  */
object ProcessOutputTests extends TestSuite {

  private def printing(text: String): Process =
    ProcessBuilder("printf", "%b", text).redirectErrorStream(true).start()

  val tests = Tests {

    test("drained on the calling thread, line by line and verbatim") {
      val logFile = Files.createTempFile("process-output", ".log")
      val lines = List.newBuilder[String]
      ProcessOutput.drain("test", printing("one\ntwo\rthree"), logFile)(
        lines += _
      )
      assert(lines.result() == List("one", "two", "three"))
      assert(Files.readString(logFile) == "one\ntwo\rthree")
    }

    test("captured in the background, it says when the stream has ended") {
      supervised {
        val logFile = Files.createTempFile("process-output", ".log")
        val lines = Channel.unlimited[String]
        val ended = Channel.unlimited[Unit]
        ProcessOutput.capture(
          "test",
          printing("only\n"),
          logFile,
          Background()
        )(lines.send)(() => ended.send(()))
        timeout(5.seconds)(ended.receive())
        assert(lines.receive() == "only")
      }
    }
  }
}
