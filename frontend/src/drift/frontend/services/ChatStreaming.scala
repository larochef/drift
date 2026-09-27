package drift.frontend.services

import drift.shared.*

import scala.concurrent.ExecutionContext
import scala.scalajs.js
import scala.scalajs.js.annotation.JSGlobal
import scala.util.{Failure, Success}

import com.github.plokhotnyuk.jsoniter_scala.core.{
  readFromString,
  writeToString
}
import org.scalajs.dom

/** The browser's `TextDecoder`, which the dom bindings do not expose: decodes
  * the streamed body chunk by chunk, keeping split multi-byte characters across
  * reads with `stream = true`.
  */
@js.native
@JSGlobal("TextDecoder")
private class TextDecoder extends js.Object {
  def decode(input: js.typedarray.ArrayBufferView, options: js.Object): String =
    js.native
}

/** One chat reply streamed over HTTP (`specs/21-assistant-page.md`): the
  * request posted to the session's chat route, and its NDJSON body read chunk
  * by chunk into `ChatEvent`s.
  */
private[services] object ChatStreaming {

  /** Posts `request` and hands each event to `onEvent` as it arrives; `onEnd`
    * then hears how the body ended — "stopped" when it simply ran out or was
    * aborted through `controller`, else why the request failed.
    */
  def start(
      sessionId: String,
      request: ChatRequest,
      controller: dom.AbortController
  )(onEvent: ChatEvent => Unit)(onEnd: String => Unit)(using
      ExecutionContext
  ): Unit =
    dom
      .fetch(
        assistantChatPath(sessionId),
        new dom.RequestInit {
          method = dom.HttpMethod.POST
          headers = js.Dictionary("Content-Type" -> "application/json")
          body = writeToString(request)
          signal = controller.signal
        }
      )
      .toFuture
      .onComplete {
        case Success(response) if response.status != 200 =>
          response.text().toFuture.onComplete {
            case Success(text) => onEnd(s"chat refused: ${text.take(300)}")
            case Failure(err)  => onEnd(s"chat refused: ${err.getMessage}")
          }
        case Success(response) =>
          val reader = response.body.getReader()
          val decoder = new TextDecoder()
          var buffered = ""
          def pump(): Unit =
            reader.read().toFuture.onComplete {
              case Success(chunk) if chunk.done =>
                if (buffered.trim.nonEmpty)
                  onEvent(readFromString[ChatEvent](buffered.trim))
                onEnd("stopped")
              case Success(chunk) =>
                buffered += decoder.decode(
                  chunk.value,
                  js.Dynamic.literal(stream = true)
                )
                val lines = buffered.split("\n", -1)
                buffered = lines.last
                lines.init
                  .map(_.trim)
                  .filter(_.nonEmpty)
                  .foreach(line => onEvent(readFromString[ChatEvent](line)))
                pump()
              case Failure(_) => onEnd("stopped")
            }
          pump()
        case Failure(err) =>
          onEnd(
            if (controller.signal.aborted) "stopped"
            else s"no reply — ${err.getMessage}"
          )
      }
}
