package drift.runner.server

import drift.runner.decode.*

import java.io.{IOException, OutputStream}
import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets.UTF_8
import java.util.concurrent.Executors

import com.sun.net.httpserver.{HttpExchange, HttpServer}

/** llama-server's HTTP API over a `ChatEngine`: `/health` (503 while the model
  * loads), `/props`, `/v1/models`, `/lora-adapters`, `/tokenize` and
  * `/v1/chat/completions`, streamed as server-sent events or whole. The JDK's
  * own HTTP server: one localhost client, no dependency.
  */
final class ChatServer(options: ServerOptions, load: () => ChatEngine) {

  @volatile private var engine = Option.empty[ChatEngine]
  @volatile private var failure = Option.empty[String]
  private val name = options.model.getFileName.toString
  private val server =
    HttpServer.create(new InetSocketAddress(options.host, options.port), 64)

  server.setExecutor(Executors.newFixedThreadPool(4))
  route("/health") { exchange =>
    (engine, failure) match {
      case (Some(_), _) => json(exchange, 200, ujson.Obj("status" -> "ok"))
      case (None, Some(error)) => json(exchange, 500, OpenAi.error(500, error))
      case _ => json(exchange, 503, OpenAi.error(503, "Loading model"))
    }
  }
  route("/props") { exchange =>
    json(
      exchange,
      200,
      ujson.Obj(
        "default_generation_settings" -> ujson.Obj("n_ctx" -> options.context),
        "modalities" -> ujson.Obj("vision" -> options.visionModel.isDefined),
        "model_path" -> options.model.toString,
        "build_info" -> Main.Version
      )
    )
  }
  route("/v1/models") { exchange =>
    json(
      exchange,
      200,
      ujson.Obj(
        "object" -> "list",
        "data" -> ujson.Arr(
          ujson.Obj("id" -> name, "object" -> "model", "owned_by" -> "drift")
        )
      )
    )
  }
  route("/lora-adapters")(exchange => json(exchange, 200, ujson.Arr()))
  route("/tokenize") { exchange =>
    ready(exchange) { engine =>
      val body =
        ujson.read(new String(exchange.getRequestBody.readAllBytes(), UTF_8))
      json(
        exchange,
        200,
        ujson.Obj(
          "tokens" -> ujson.Arr(
            engine.tokenize(body("content").str).map(ujson.Num(_))*
          )
        )
      )
    }
  }
  route("/v1/chat/completions")(exchange =>
    ready(exchange)(completion(exchange, _))
  )

  def start(): Unit = {
    server.start()
    val loader = new Thread(() =>
      try engine = Some(load())
      catch {
        case error: Throwable =>
          failure = Some(Option(error.getMessage).getOrElse(error.toString))
          System.err.println(s"loading ${options.model} failed: ${failure.get}")
      }
    )
    loader.start()
  }

  private def route(path: String)(handle: HttpExchange => Unit): Unit =
    server.createContext(
      path,
      exchange =>
        try handle(exchange)
        catch {
          case error: IOException => () // the client went away
          case error: Throwable   =>
            System.err.println(s"$path: $error")
            try
              json(
                exchange,
                500,
                OpenAi.error(
                  500,
                  Option(error.getMessage).getOrElse(error.toString)
                )
              )
            catch { case _: Throwable => () }
        } finally exchange.close()
    )

  private def ready(exchange: HttpExchange)(body: ChatEngine => Unit): Unit =
    engine match {
      case Some(e) => body(e)
      case None    =>
        json(
          exchange,
          503,
          OpenAi.error(503, failure.getOrElse("Loading model"))
        )
    }

  private def json(
      exchange: HttpExchange,
      status: Int,
      value: ujson.Value
  ): Unit = {
    val bytes = ujson.write(value).getBytes(UTF_8)
    exchange.getResponseHeaders.set("Content-Type", "application/json")
    exchange.sendResponseHeaders(status, bytes.length.toLong)
    exchange.getResponseBody.write(bytes)
  }

  private def completion(exchange: HttpExchange, engine: ChatEngine): Unit = {
    val body =
      ujson.read(new String(exchange.getRequestBody.readAllBytes(), UTF_8))
    OpenAi.request(body, engine.context, engine.sees) match {
      case Left(problem)  => json(exchange, 400, OpenAi.error(400, problem))
      case Right(request) =>
        val id = s"chatcmpl-${java.util.UUID.randomUUID().toString.take(12)}"
        val stream = body.obj.get("stream").exists(_.bool)
        if (stream) {
          exchange.getResponseHeaders.set("Content-Type", "text/event-stream")
          exchange.getResponseHeaders.set("Cache-Control", "no-cache")
          exchange.sendResponseHeaders(200, 0)
          val out = exchange.getResponseBody
          val listener = new StreamingListener(out, id)
          event(
            out,
            OpenAi.chunk(
              id,
              name,
              ujson.Obj("role" -> "assistant", "content" -> ""),
              None
            )
          )
          try {
            val result = chat(engine, request, listener)
            event(
              out,
              OpenAi.chunk(id, name, ujson.Obj(), Some(result.finishReason))
            )
            val includeUsage = body.obj
              .get("stream_options")
              .flatMap(_.objOpt)
              .exists(_.get("include_usage").exists(_.bool))
            if (includeUsage) {
              val last = OpenAi.chunk(id, name, ujson.Obj(), None)
              last("choices") = ujson.Arr()
              last("usage") = OpenAi.usage(result)
              last("timings") = OpenAi.timings(result)
              event(out, last)
            }
            out.write("data: [DONE]\n\n".getBytes(UTF_8))
            out.flush()
          } catch {
            case error: IllegalArgumentException =>
              event(out, OpenAi.error(400, error.getMessage))
          }
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
          json(
            exchange,
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

  private def event(out: OutputStream, value: ujson.Value): Unit = {
    out.write(s"data: ${ujson.write(value)}\n\n".getBytes(UTF_8))
    out.flush()
  }

  /** Streams each piece as a delta; a failed write (the client closed the
    * stream: drift's Stop) cancels the reply.
    */
  final private class StreamingListener(out: OutputStream, id: String)
      extends ChatListener {
    @volatile private var gone = false
    private def send(field: String, text: String): Unit =
      if (!gone)
        try event(out, OpenAi.chunk(id, name, ujson.Obj(field -> text), None))
        catch { case _: IOException => gone = true }
    def reasoning(text: String): Unit = send("reasoning_content", text)
    def content(text: String): Unit = send("content", text)
    def cancelled: Boolean = gone
  }
}
