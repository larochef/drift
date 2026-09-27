package drift.backend.session

import drift.backend.process.ProcessOutput

import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.*
import scala.util.control.NonFatal

/** A session process's output (`specs/13-log-streaming.md`): drained into its
  * log file and ring buffer while it runs, and its tail read back when it
  * fails.
  */
private[session] object SessionOutput {

  /** Drains the process's merged output on a thread of its own
    * ([[ProcessOutput.capture]]): mirrored to the session's log file, split
    * into lines for the ring buffer, its progress bars read on the way.
    */
  def capture(
      sessionId: String,
      process: Process,
      logFile: Path,
      log: SessionLog
  ): Unit = {
    ProcessOutput.capture(s"session-log-$sessionId", process, logFile)(
      log.append
    )(() => log.close())
    ()
  }

  /** The last few lines of a session's log — enough to name the error without
    * shipping a verbose multi-minute load. Reads only the file's end; a `-v`
    * load can produce megabytes.
    */
  def tail(logFile: Path, lines: Int = 10): String =
    try {
      val size = Files.size(logFile)
      val channel = FileChannel.open(logFile, StandardOpenOption.READ)
      try {
        val buffer = ByteBuffer.allocate(8 * 1024)
        channel.position(math.max(0L, size - buffer.capacity))
        channel.read(buffer)
        String(
          buffer.array(),
          0,
          buffer.position(),
          StandardCharsets.UTF_8
        ).linesIterator.toList
          .takeRight(lines)
          .mkString("\n")
      } finally channel.close()
    } catch { case NonFatal(_) => "" }
}
