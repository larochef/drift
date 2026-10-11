package drift.backend.session

import drift.protocol.{CapabilitiesHead, JobRoutes}
import drift.shared.*

import java.net.*
import java.net.http.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString

/** What starting and stopping a server process needs, whatever it serves: a
  * free port, the readiness probe, and a terminate that escalates.
  */
private[session] object ServerProcesses {
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

  /** Whether a server started now could listen on the port. Address reuse is
    * switched off for the test — Java switches it on by default — so that a
    * port whose last server closed its connections less than a minute ago, and
    * which the kernel still holds, counts as taken: a server that cannot bind
    * it may still print that it listens, and then exit.
    */
  private[session] def bindable(port: Int): Boolean =
    try {
      val socket = ServerSocket()
      try {
        socket.setReuseAddress(false)
        socket.bind(InetSocketAddress(InetAddress.getLoopbackAddress, port))
        true
      } finally socket.close()
    } catch { case NonFatal(_) => false }

  /** Why a server that answers on `port` cannot be driven, if it cannot: a
    * drift runner speaking another version of the job protocol than this drift
    * — an older build left installed, or one started by hand. Asked of a server
    * expected to be the runner (`runner`), one that says no version is that
    * old.
    */
  def protocolMismatch(port: Int, runner: Boolean): Option[String] =
    try {
      val request = HttpRequest
        .newBuilder(
          URI.create(s"http://127.0.0.1:$port${JobRoutes.Capabilities}")
        )
        .timeout(java.time.Duration.ofSeconds(5))
        .GET()
        .build()
      val body =
        probeClient.send(request, HttpResponse.BodyHandlers.ofString()).body()
      readFromString[CapabilitiesHead](body).drift.map(_.protocol) match {
        case Some(JobRoutes.Version) => None
        case Some(other)             =>
          Some(
            s"the drift runner speaks version $other of the job protocol and this drift version ${JobRoutes.Version}: reinstall the runner from Settings"
          )
        case None if runner =>
          Some(
            s"the drift runner says no protocol version, and this drift speaks version ${JobRoutes.Version}: reinstall the runner from Settings"
          )
        case None => None
      }
    } catch { case NonFatal(_) => None }

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
}
