package drift.runner.server

import drift.runner.decode.*

import java.nio.charset.StandardCharsets.UTF_8
import scala.util.control.NonFatal

import ox.*
import ox.flow.Flow
import sttp.model.StatusCode
import sttp.shared.Identity
import sttp.tapir.*
import sttp.tapir.server.ServerEndpoint
import sttp.tapir.server.netty.sync.{NettySyncServer, OxStreams}

/** llama-server's HTTP API over a `ChatEngine`: `/health` (503 while the model
  * loads), `/props`, `/v1/models`, `/lora-adapters`, `/tokenize` and
  * `/v1/chat/completions`, streamed as server-sent events or whole. Served as
  * the backend serves its own API: tapir on Netty, on ox.
  */
final class ChatServer(options: ServerOptions, load: () => ChatEngine) {

  @volatile private var engine = Option.empty[ChatEngine]
  @volatile private var failure = Option.empty[String]
  private val name = options.model.getFileName.toString

  /** An answer whole: its status and its JSON. */
  private type Whole = (StatusCode, String)

  private def whole(status: Int, value: ujson.Value): Whole =
    StatusCode(status) -> ujson.write(value)

  private val answer = statusCode.and(stringJsonBody)

  /** A completion's answer: whole, or a stream of server-sent events — its
    * status, its content type and its bytes as they come.
    */
  private type Reply = (StatusCode, String, Flow[Chunk[Byte]])

  private def replied(status: Int, value: ujson.Value): Reply =
    (
      StatusCode(status),
      "application/json",
      Flow.fromValues(Chunk.fromArray(ujson.write(value).getBytes(UTF_8)))
    )

  /** What a request throws is the answer's reason, as llama-server words it. */
  private def guarded(path: String)(answer: => Whole): Whole =
    try answer
    catch {
      case NonFatal(error) =>
        System.err.println(s"$path: $error")
        whole(
          500,
          OpenAi.error(500, Option(error.getMessage).getOrElse(error.toString))
        )
    }

  private def ready[A](unready: ujson.Value => A)(body: ChatEngine => A): A =
    engine match {
      case Some(loaded) => body(loaded)
      case None         =>
        unready(OpenAi.error(503, failure.getOrElse("Loading model")))
    }

  private val endpoints: List[ServerEndpoint[OxStreams, Identity]] = List(
    endpoint.get.in("health").out(answer).serverLogicSuccess[Identity] { _ =>
      (engine, failure) match {
        case (Some(_), _)        => whole(200, ujson.Obj("status" -> "ok"))
        case (None, Some(error)) => whole(500, OpenAi.error(500, error))
        case _ => whole(503, OpenAi.error(503, "Loading model"))
      }
    },
    endpoint.get.in("props").out(answer).serverLogicSuccess[Identity] { _ =>
      whole(
        200,
        ujson.Obj(
          "default_generation_settings" ->
            ujson.Obj("n_ctx" -> options.context),
          "modalities" -> ujson.Obj("vision" -> options.visionModel.isDefined),
          "model_path" -> options.model.toString,
          "build_info" -> Main.Version
        )
      )
    },
    endpoint.get.in("v1" / "models").out(answer).serverLogicSuccess[Identity] {
      _ =>
        whole(
          200,
          ujson.Obj(
            "object" -> "list",
            "data" -> ujson.Arr(
              ujson
                .Obj("id" -> name, "object" -> "model", "owned_by" -> "drift")
            )
          )
        )
    },
    endpoint.get
      .in("lora-adapters")
      .out(answer)
      .serverLogicSuccess[Identity](_ => whole(200, ujson.Arr())),
    endpoint.post
      .in("tokenize")
      .in(stringBody)
      .out(answer)
      .serverLogicSuccess[Identity] { text =>
        guarded("/tokenize")(ready(whole(503, _)) { loaded =>
          whole(
            200,
            ujson.Obj(
              "tokens" -> ujson.Arr(
                loaded
                  .tokenize(ujson.read(text)("content").str)
                  .map(ujson.Num(_))*
              )
            )
          )
        })
      },
    endpoint.post
      .in("v1" / "chat" / "completions")
      .in(stringBody)
      .out(statusCode)
      .out(header[String]("Content-Type"))
      .out(streamBinaryBody(OxStreams)(CodecFormat.OctetStream()))
      .serverLogicSuccess[Identity] { text =>
        try ready(replied(503, _))(completion(text, _))
        catch {
          case NonFatal(error) =>
            System.err.println(s"/v1/chat/completions: $error")
            replied(
              500,
              OpenAi.error(
                500,
                Option(error.getMessage).getOrElse(error.toString)
              )
            )
        }
      }
  )

  /** Starts serving, and loading the model beside: `/health` answers 503 until
    * it is in. Returns once the port is bound; the scope keeps both.
    */
  def start()(using Ox): Unit = {
    NettySyncServer()
      .addEndpoints(endpoints)
      .host(options.host)
      .port(options.port)
      .start()
      .discard
    forkUser(
      try engine = Some(load())
      catch {
        case NonFatal(error) =>
          failure = Some(Option(error.getMessage).getOrElse(error.toString))
          System.err.println(s"loading ${options.model} failed: ${failure.get}")
      }
    ).discard
  }

  private def completion(text: String, engine: ChatEngine): Reply = {
    val body = ujson.read(text)
    OpenAi.request(body, engine.context, engine.sees) match {
      case Left(problem)  => replied(400, OpenAi.error(400, problem))
      case Right(request) =>
        val id = s"chatcmpl-${java.util.UUID.randomUUID().toString.take(12)}"
        val stream = body.obj.get("stream").exists(_.bool)
        if (stream) {
          val includeUsage = body.obj
            .get("stream_options")
            .flatMap(_.objOpt)
            .exists(_.get("include_usage").exists(_.bool))
          // The reply is generated as the stream is read: each piece is sent
          // as it is made, and a client that stops reading stops the reply.
          val events = Flow.usingEmit[Chunk[Byte]] { emit =>
            val listener = new StreamingListener(emit.apply, id)
            listener.event(
              OpenAi.chunk(
                id,
                name,
                ujson.Obj("role" -> "assistant", "content" -> ""),
                None
              )
            )
            try {
              val result = chat(engine, request, listener)
              listener.event(
                OpenAi.chunk(id, name, ujson.Obj(), Some(result.finishReason))
              )
              if (includeUsage) {
                val last = OpenAi.chunk(id, name, ujson.Obj(), None)
                last("choices") = ujson.Arr()
                last("usage") = OpenAi.usage(result)
                last("timings") = OpenAi.timings(result)
                listener.event(last)
              }
              listener.send("data: [DONE]\n\n")
            } catch {
              case error: IllegalArgumentException =>
                listener.event(OpenAi.error(400, error.getMessage))
            }
          }
          (StatusCode.Ok, "text/event-stream", events)
        } else {
          val reasoningText = new StringBuilder
          val contentText = new StringBuilder
          val result = chat(
            engine,
            request,
            new ChatListener {
              def reasoning(text: String): Unit = reasoningText ++= text
              def content(text: String): Unit = contentText ++= text
              def cancelled: Boolean = false
            }
          )
          val message =
            ujson.Obj("role" -> "assistant", "content" -> contentText.toString)
          if (reasoningText.nonEmpty)
            message("reasoning_content") = reasoningText.toString
          replied(
            200,
            ujson.Obj(
              "id" -> id,
              "object" -> "chat.completion",
              "created" -> System.currentTimeMillis() / 1000,
              "model" -> name,
              "choices" -> ujson.Arr(
                ujson.Obj(
                  "index" -> 0,
                  "message" -> message,
                  "finish_reason" -> result.finishReason
                )
              ),
              "usage" -> OpenAi.usage(result),
              "timings" -> OpenAi.timings(result)
            )
          )
        }
    }
  }

  /** `engine.chat`, its timings printed as llama-server prints them. */
  private def chat(
      engine: ChatEngine,
      request: ChatRequest,
      listener: ChatListener
  ): ChatResult = {
    val result = engine.chat(request, listener)
    println(OpenAi.timingReport(result))
    result
  }

  /** Streams each piece as a delta. A send that fails — the client closed the
    * stream: drift's Stop — cancels the reply, and nothing more is sent.
    */
  final private class StreamingListener(emit: Chunk[Byte] => Unit, id: String)
      extends ChatListener {
    @volatile private var gone = false

    def send(text: String): Unit =
      if (!gone)
        try emit(Chunk.fromArray(text.getBytes(UTF_8)))
        catch { case _: Throwable => gone = true }

    def event(value: ujson.Value): Unit =
      send(s"data: ${ujson.write(value)}\n\n")

    private def delta(field: String, text: String): Unit =
      event(OpenAi.chunk(id, name, ujson.Obj(field -> text), None))

    def reasoning(text: String): Unit = delta("reasoning_content", text)
    def content(text: String): Unit = delta("content", text)
    def cancelled: Boolean = gone
  }
}
