package drift.backend.routes

import drift.backend.assistant.{AssistantMedia, AssistantProxy}
import drift.shared.*

import java.nio.charset.StandardCharsets
import java.nio.file.Files

import com.github.plokhotnyuk.jsoniter_scala.core.writeToString
import ox.Chunk
import sttp.shared.Identity
import sttp.tapir.*
import sttp.tapir.json.jsoniter.*
import sttp.tapir.server.ServerEndpoint
import sttp.tapir.server.netty.sync.OxStreams

def assistantEndpoints(
    proxy: AssistantProxy,
    media: AssistantMedia
): List[ServerEndpoint[Any, Identity]] =
  List(
    getAssistantProperties.serverLogic[Identity](proxy.properties),
    uploadAssistantAttachment.serverLogic[Identity] { (name, mimeType, bytes) =>
      media.store(name, mimeType, bytes)
    },
    getAssistantUpload.serverLogic[Identity] { id =>
      media.uploaded(id) match {
        case Some(file) =>
          Right((Files.readAllBytes(file), AssistantMedia.mimeFor(file)))
        case None => Left(())
      }
    }
  )

/** The chat (`assistantChatPath` in the shared module): a `POST` of the
  * [[ChatRequest]], answered by a body that streams one [[ChatEvent]] as JSON
  * per line and ends after `done` or `error`. The request is ordinary HTTP, so
  * its size is bounded only like any other body — attachments are references
  * anyway — and a client that aborts cancels the generation.
  */
def assistantChatEndpoint(
    proxy: AssistantProxy
): ServerEndpoint[OxStreams, Identity] =
  endpoint.post
    .in("api" / "assistant" / path[String]("sessionId") / "chat")
    .in(jsonBody[ChatRequest])
    .out(
      streamTextBody(OxStreams)(
        CodecFormat.TextPlain(),
        Some(StandardCharsets.UTF_8)
      )
    )
    .serverLogicSuccess[Identity] { (sessionId, request) =>
      proxy
        .chat(sessionId, request)
        .map(event =>
          Chunk.fromArray(
            (writeToString(event) + "\n").getBytes(StandardCharsets.UTF_8)
          )
        )
    }
