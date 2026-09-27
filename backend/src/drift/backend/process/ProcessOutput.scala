package drift.backend.process

import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import java.nio.file.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** A child process's merged output, drained on a thread of its own: mirrored
  * verbatim to a log file and handed over line by line — a session's sd-server
  * (`specs/13-log-streaming.md`) and a conversion's sd-cli
  * (`specs/25-model-conversion.md`) both read their progress bars this way.
  *
  * The draining is the part that cannot be skipped. A full stdout pipe that
  * nobody reads blocks the child, which is the classic way to make a load
  * appear to hang at 40% - drift reads the pipe instead of handing it to the
  * OS, so this thread is the child's only outlet.
  *
  * Splitting treats a carriage return as a terminator as well as a newline,
  * because that is how sd-cpp redraws its progress bars: without it, a whole
  * generation's worth of bar arrives as one line after the work is over, which
  * is exactly when it stops being progress.
  */
object ProcessOutput {
  private val logger = Logger("drift.backend.process.ProcessOutput")

  /** Starts the draining thread `name`: every line (or bar redraw) goes to
    * `onLine`, and `onDone` runs once the stream has ended, whatever the
    * reason.
    */
  def capture(
      name: String,
      process: Process,
      logFile: Path
  )(onLine: String => Unit)(onDone: () => Unit): Thread = {
    val thread = Thread(
      () => {
        val reader =
          InputStreamReader(process.getInputStream, StandardCharsets.UTF_8)
        try {
          val writer = Files.newBufferedWriter(
            logFile,
            StandardCharsets.UTF_8,
            StandardOpenOption.CREATE,
            StandardOpenOption.APPEND
          )
          try {
            val chunk = new Array[Char](4096)
            val pending = StringBuilder()
            var read = reader.read(chunk)
            while (read != -1) {
              // Mirrored verbatim, carriage returns and escapes included, so
              // the file reads exactly as the terminal would have shown it.
              writer.write(chunk, 0, read)
              writer.flush()
              var index = 0
              while (index < read) {
                chunk(index) match {
                  case '\n' | '\r' =>
                    onLine(pending.toString)
                    pending.clear()
                  case character => pending.append(character)
                }
                index += 1
              }
              read = reader.read(chunk)
            }
            onLine(pending.toString)
          } finally writer.close()
        } catch {
          case NonFatal(err) =>
            logger.warn(s"$name: reading the output stopped: ${err.getMessage}")
        } finally {
          reader.close()
          onDone()
        }
      },
      name
    )
    thread.setDaemon(true)
    thread.start()
    thread
  }
}
