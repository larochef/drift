package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.model.StatusCode
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Talking to a live llama.cpp session
  * (`specs/18-assistant-models-and-sessions.md`). The browser never learns the
  * server's port: chat goes through drift, like generation does.
  *
  * Media never travels inline from the browser (`specs/21-assistant-page.md`):
  * a generated output is referenced by its date and file name, anything else is
  * uploaded once and referenced by id, and the backend reads, scales and
  * encodes the bytes for the model. Videos of tens of megabytes are expected.
  */
sealed trait ChatAttachment derives CanEqual

/** A persisted generation output, under the outputs root. */
case class OutputAttachment(date: String, fileName: String)
    extends ChatAttachment

/** A file uploaded through `uploadAssistantAttachment`. */
case class UploadAttachment(id: String) extends ChatAttachment

object ChatAttachment {
  given Schema[ChatAttachment] = Schema.derived
  // Discriminated by a lowercase `"type"` that may sit anywhere in the object,
  // like `ModelSource`.
  given JsonValueCodec[ChatAttachment] = JsonCodecMaker.make(
    CodecMakerConfig
      .withAdtLeafClassNameMapper(n =>
        JsonCodecMaker.simpleClassName(n).toLowerCase
      )
      .withRequireDiscriminatorFirst(false)
  )
}

/** One turn. Attachments become `image_url` (or video) parts. */
case class ChatMessage(
    role: String,
    text: String,
    attachments: List[ChatAttachment] = Nil
)

case class ChatRequest(
    messages: List[ChatMessage],
    maxTokens: Option[Int] = None,
    temperature: Option[Double] = None
)
object ChatMessage {
  given Schema[ChatMessage] = Schema.derived
}

object ChatRequest {
  given Schema[ChatRequest] = Schema.derived
  given JsonValueCodec[ChatRequest] = JsonCodecMaker.make(
    CodecMakerConfig
      .withAdtLeafClassNameMapper(n =>
        JsonCodecMaker.simpleClassName(n).toLowerCase
      )
      .withRequireDiscriminatorFirst(false)
  )
}

/** One event of a streamed reply: a piece of the answer, a piece of the model's
  * thinking, or the end — with the token counts the server reported, which the
  * conversation spec turns into a context meter.
  */
case class ChatEvent(
    content: Option[String] = None,
    reasoning: Option[String] = None,
    done: Boolean = false,
    promptTokens: Option[Int] = None,
    completionTokens: Option[Int] = None,
    /** Prefill (the prompt, images included) and generation: duration and rate
      * of each, from the server's final chunk.
      */
    promptMillis: Option[Double] = None,
    completionMillis: Option[Double] = None,
    promptTokensPerSecond: Option[Double] = None,
    completionTokensPerSecond: Option[Double] = None,
    error: Option[String] = None
)
object ChatEvent {
  given JsonValueCodec[ChatEvent] = JsonCodecMaker.make
}

/** What the server actually applied, read from `/props` once it is ready: the
  * context size (`-c` may be clamped or, at 0, taken from the model) and
  * whether a projector is loaded.
  */
case class AssistantProperties(
    contextSize: Int,
    vision: Boolean,
    modelPath: String
)
object AssistantProperties {
  given JsonValueCodec[AssistantProperties] = JsonCodecMaker.make
}

/** A file stored by `uploadAssistantAttachment`, served back by
  * `getAssistantUpload` for previews.
  */
case class UploadedAttachment(
    id: String,
    name: String,
    mimeType: String,
    sizeBytes: Long
)
object UploadedAttachment {
  given JsonValueCodec[UploadedAttachment] = JsonCodecMaker.make
}

private val assistantBase = endpoint.in("api")

/** Refused, with the reason, unless the session is a ready assistant. */
val getAssistantProperties
    : PublicEndpoint[String, String, AssistantProperties, Any] =
  assistantBase.get
    .in("assistant" / path[String]("sessionId") / "properties")
    .errorOut(stringBody)
    .out(jsonBody[AssistantProperties])

/** Stores a file for later attachment. The body is the raw bytes; the name and
  * mime type ride as query parameters, so a browser can send a `File` as is.
  */
val uploadAssistantAttachment: PublicEndpoint[
  (String, String, Array[Byte]),
  String,
  UploadedAttachment,
  Any
] =
  assistantBase.post
    .in("assistant" / "uploads")
    .in(query[String]("name"))
    .in(query[String]("type"))
    .in(byteArrayBody)
    .errorOut(stringBody)
    .out(jsonBody[UploadedAttachment])

/** A finished reply's Markdown as HTML (`specs/20`): the body is the reply,
  * the answer what the backend's commonmark rendered and jsoup cleaned — the
  * model cards' pipeline, so the page never renders Markdown itself.
  */
val renderAssistantMarkdown: PublicEndpoint[String, Unit, String, Any] =
  assistantBase.post
    .in("assistant" / "markdown")
    .in(stringBody)
    .out(stringBody)

/** Serves an uploaded file back, for the preview beside the composer. */
val getAssistantUpload
    : PublicEndpoint[String, Unit, (Array[Byte], String), Any] =
  assistantBase.get
    .in("assistant" / "uploads" / path[String]("id"))
    .errorOut(statusCode(StatusCode.NotFound))
    .out(byteArrayBody)
    .out(header[String]("Content-Type"))

/** The chat: `POST` a [[ChatRequest]] here and read the response as it streams
  * — one [[ChatEvent]] as JSON per line, the last with `done` or `error`.
  * Aborting the request cancels the generation. Defined on the backend, since
  * its streamed body is server-specific.
  */
def assistantChatPath(sessionId: String): String =
  s"/api/assistant/$sessionId/chat"
