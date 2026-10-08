package drift.frontend.services

import drift.shared.*

import scala.scalajs.js.timers
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** The status WebSocket — the push replacement for the per-service status
  * polls. See `StatusUpdate` in the shared module for the protocol: full topic
  * snapshots, sent only when a topic changes, everything on (re)connect.
  *
  * Mounted once by `AppShell` like `GlobalDownloadsService` (mounting twice
  * would open two sockets). The consuming services subscribe to the topic
  * streams inside their own `effects`, so a service still only listens while
  * some page has it mounted — exactly the scope its poll used to have — and
  * pages keep seeding state through the REST loads they already push on mount.
  *
  * A dropped connection retries every two seconds until the backend answers;
  * the on-connect snapshot then replaces whatever the client missed.
  */
class StatusSocketService {

  private val updateBus = new EventBus[StatusUpdate]

  private val _undecodable = Var(false)

  /** Whether the last message could not be read. Every topic travels in one
    * message, so a bundle that cannot decode one decodes none of them: the app
    * goes quiet while looking merely idle, and only a reload cures it. The
    * shell says so rather than leaving the page to look like nothing is
    * happening (François, 2026-09-20).
    */
  val undecodable: Signal[Boolean] = _undecodable.signal.distinct

  private def topic[A](select: StatusUpdate => Option[A]): EventStream[A] =
    updateBus.events.map(select).collect { case Some(value) => value }

  val sessions: EventStream[List[Session]] = topic(_.sessions)
  val downloads: EventStream[List[DownloadJob]] = topic(_.downloads)
  val loraDownloads: EventStream[List[LoraDownloadJob]] =
    topic(_.loraDownloads)
  val upscalerDownloads: EventStream[List[UpscalerDownloadJob]] =
    topic(_.upscalerDownloads)
  val runtimeInstalls: EventStream[List[RuntimeInstallJob]] =
    topic(_.runtimeInstalls)

  /** Keyed by session id; carries only the sessions whose list changed. */
  val generations: EventStream[Map[String, List[Generation]]] =
    topic(_.generations)
  val postProcessJobs: EventStream[List[PostProcessJob]] =
    topic(_.postProcessJobs)
  val conversions: EventStream[List[ConversionJob]] = topic(_.conversions)
  val machine: EventStream[MachineStatus] = topic(_.machine)

  private var socket: Option[dom.WebSocket] = None
  private var mounted = false
  private var reopenHandle: Option[timers.SetTimeoutHandle] = None

  private def open(): Unit = {
    val protocol =
      if (dom.window.location.protocol == "https:") "wss" else "ws"
    val ws = new dom.WebSocket(
      s"$protocol://${dom.window.location.host}$statusSocketPath"
    )
    socket = Some(ws)
    // A message that does not decode is loud — in the console, and on the
    // page: a schema mismatch (a stale bundle against a backend that has been
    // restarted on a newer build) silences every topic at once, and the socket
    // itself stays up, so nothing recovers on its own.
    ws.onmessage = message =>
      try {
        val update =
          readFromString[StatusUpdate](message.data.asInstanceOf[String])
        if (_undecodable.now()) _undecodable.set(false)
        updateBus.writer.onNext(update)
      } catch {
        case NonFatal(err) =>
          dom.console.error(
            s"Status socket: this page cannot read the backend's messages " +
              s"(${err.getMessage}). Reload it."
          )
          _undecodable.set(true)
      }
    // Fires for failed connection attempts too, so this alone is the whole
    // retry loop.
    ws.onclose = _ => scheduleReopen()
  }

  private def scheduleReopen(): Unit =
    if (mounted && reopenHandle.isEmpty)
      reopenHandle = Some(timers.setTimeout(2000) {
        reopenHandle = None
        if (mounted) open()
      })

  val effects: Modifier[HtmlElement] = onMountUnmountCallback(
    _ => {
      mounted = true
      open()
    },
    _ => {
      mounted = false
      reopenHandle.foreach(timers.clearTimeout)
      reopenHandle = None
      // `onclose` fires for this close too; `mounted` gates the reopen.
      socket.foreach(_.close())
      socket = None
    }
  )
}
