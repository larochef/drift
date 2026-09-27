package drift.backend.session

import drift.shared.*

import java.net.*
import java.net.http.*
import java.util.concurrent.TimeUnit
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** What starting and stopping a server process needs, whatever it serves: a
  * free port, the readiness probe, and a terminate that escalates.
  */
private[session] object ServerProcesses {
  private val logger = Logger("drift.backend.session.ServerProcesses")

  // Readiness probes only; connections are to localhost, so short is honest.
  private val probeClient = HttpClient
    .newBuilder()
    .connectTimeout(java.time.Duration.ofSeconds(2))
    .build()

  /** A free port from the configured range, other than the `promised` ones —
    * ports handed to servers that have not bound them yet, so a bind test alone
    * would hand the same port out twice.
    */
  def freePort(settings: SessionSettings, promised: Set[Int]): Option[Int] =
    (settings.portRangeStart to settings.portRangeEnd)
      .find(port => !promised.contains(port) && bindable(port))

  private def bindable(port: Int): Boolean =
    try {
      val socket = ServerSocket()
      try {
        socket.bind(InetSocketAddress(InetAddress.getLoopbackAddress, port))
        true
      } finally socket.close()
    } catch { case NonFatal(_) => false }

  /** llama-server answers `/health` with 503 while the model loads and 200 once
    * it serves; sd-server has no health route, its capabilities listing is the
    * proof.
    */
  def answersProbe(port: Int, tool: RuntimeTool): Boolean =
    try {
      val path = tool match {
        case RuntimeTool.SdCpp    => "/sdcpp/v1/capabilities"
        case RuntimeTool.LlamaCpp => "/health"
      }
      val request = HttpRequest
        .newBuilder(URI.create(s"http://127.0.0.1:$port$path"))
        .timeout(java.time.Duration.ofSeconds(2))
        .GET()
        .build()
      probeClient
        .send(request, HttpResponse.BodyHandlers.discarding())
        .statusCode == 200
    } catch { case NonFatal(_) => false }

  /** SIGTERM, then SIGKILL after the grace period. */
  def terminate(process: Process): Unit = {
    process.destroy()
    if (!process.waitFor(SessionManager.GraceSeconds, TimeUnit.SECONDS)) {
      logger.warn(
        s"Pid ${process.pid()} ignored SIGTERM for ${SessionManager.GraceSeconds}s; killing"
      )
      process.destroyForcibly()
      process.waitFor(5, TimeUnit.SECONDS)
    }
  }
}
