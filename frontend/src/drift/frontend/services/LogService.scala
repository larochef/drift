package drift.frontend.services

import drift.shared.*

import scala.scalajs.js
import scala.util.{Failure, Success}

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** A session's output as it arrives (`specs/13-log-streaming.md`).
  *
  * The same NDJSON-over-fetch reader the assistant chat uses - the server
  * replays its buffer before following, so opening the view halfway through a
  * load shows what already happened rather than only what happens next.
  *
  * The lines are held here rather than in the view, so closing and reopening
  * the panel does not lose them, and bounded at the same order as the server's
  * ring buffer: the browser must not grow without limit on a chatty load.
  */
class LogService extends ServiceErrors {
  import scala.concurrent.ExecutionContext.Implicits.global

  private val _lines = Var(List.empty[LogLine])
  private val _following = Var(Option.empty[String])

  /** The captured lines, oldest first. */
  val lines: Signal[List[LogLine]] = _lines.signal

  /** Which session is being followed, if any. */
  val following: Signal[Option[String]] = _following.signal

  private var controller: Option[dom.AbortController] = None

  /** Follows a session from its buffered beginning. Following a second session
    * abandons the first: one log view, one stream.
    */
  def follow(sessionId: String): Unit = {
    if (!_following.now().contains(sessionId)) {
      stop()
      _lines.set(List.empty)
      _following.set(Some(sessionId))
      val abort = new dom.AbortController()
      controller = Some(abort)
      dom
        .fetch(
          s"/api/sessions/$sessionId/logs",
          new dom.RequestInit {
            method = dom.HttpMethod.GET
            signal = abort.signal
          }
        )
        .toFuture
        .onComplete {
          case Success(response) if response.status != 200 =>
            reportFailure(
              "Following the log",
              s"the session has no captured output (${response.status})."
            )
            _following.set(None)
          case Success(response) =>
            clearError()
            val reader = response.body.getReader()
            val decoder = LogService.TextDecoder()
            var buffered = ""
            def pump(): Unit =
              reader.read().toFuture.onComplete {
                case Success(chunk) if chunk.done => _following.set(None)
                case Success(chunk)               =>
                  buffered += decoder.decode(
                    chunk.value,
                    js.Dynamic.literal(stream = true)
                  )
                  val split = buffered.split("\n", -1)
                  buffered = split.last
                  val parsed = split.init
                    .map(_.trim)
                    .filter(_.nonEmpty)
                    .flatMap(line =>
                      try Some(readFromString[LogLine](line))
                      catch { case _: Throwable => None }
                    )
                    .toList
                  if (parsed.nonEmpty)
                    _lines.update(held =>
                      (held ++ parsed).takeRight(LogService.MaximumLines)
                    )
                  pump()
                case Failure(_) => _following.set(None)
              }
            pump()
          case Failure(err) =>
            if (!abort.signal.aborted)
              reportFailure("Following the log", err)
            _following.set(None)
        }
    }
  }

  /** Stops following, without dropping what was already read. */
  def stop(): Unit = {
    controller.foreach(_.abort())
    controller = None
    _following.set(None)
  }

  val effects: Modifier[HtmlElement] = emptyMod
}

object LogService {
  val MaximumLines: Int = 4000

  /** The browser's `TextDecoder`, which the dom bindings do not expose. */
  @js.native
  @scalajs.js.annotation.JSGlobal("TextDecoder")
  class TextDecoder extends js.Object {
    def decode(input: js.Any, options: js.Any): String = js.native
  }
}
