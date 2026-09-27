package drift.runner

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}

import scala.jdk.CollectionConverters.*

import drift.runner.decode.ChatEngine
import drift.runner.ops.{HipOps, MatVecInputs}
import drift.runner.server.{ChatServer, ServerOptions}

/** The server on a real model, driven as drift's assistant proxy drives
  * llama-server: health until loaded, `/props`, a streamed reply with usage and
  * timings, reasoning on and off, and a stream dropped half way.
  *
  * `./mill runner.gpuTest.runMain drift.runner.ServerSmoke <model.gguf>`
  */
object ServerSmoke {

  private val client = HttpClient.newHttpClient()
  private val port = 18094

  private def get(path: String) =
    client.send(
      HttpRequest
        .newBuilder(URI.create(s"http://127.0.0.1:$port$path"))
        .build(),
      HttpResponse.BodyHandlers.ofString()
    )

  private def post(body: ujson.Value) =
    HttpRequest
      .newBuilder(URI.create(s"http://127.0.0.1:$port/v1/chat/completions"))
      .POST(HttpRequest.BodyPublishers.ofString(ujson.write(body)))
      .build()

  private def user(text: String) = ujson.Obj(
    "role" -> "user",
    "content" -> ujson.Arr(ujson.Obj("type" -> "text", "text" -> text))
  )

  private def request(text: String, thinking: Boolean, maxTokens: Int) =
    conversation(Seq(user(text)), thinking, maxTokens)

  private def conversation(
      messages: Seq[ujson.Value],
      thinking: Boolean,
      maxTokens: Int
  ) =
    ujson.Obj(
      "messages" -> ujson.Arr.from(messages),
      "stream" -> true,
      "stream_options" -> ujson.Obj("include_usage" -> true),
      "max_tokens" -> maxTokens,
      "temperature" -> 0,
      "chat_template_kwargs" -> ujson.Obj("enable_thinking" -> thinking)
    )

  def main(arguments: Array[String]): Unit = {
    val options = ServerOptions
      .parse(
        Seq(
          "-m",
          arguments(0),
          "-c",
          "4096",
          "--port",
          port.toString,
          "-ngl",
          "99"
        )
      )
      .toOption
      .get
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    val server = new ChatServer(
      options,
      () =>
        new ChatEngine(
          ops,
          options.model,
          options.draftModel,
          options.visionModel,
          options.context,
          options.drafts
        )
    )
    server.start()
    var loading = 0
    while (get("/health").statusCode() == 503) {
      loading += 1; Thread.sleep(100)
    }
    println(
      s"health 200 after ${loading * 100} ms of 503; props: ${get("/props").body()}"
    )

    // a follow-up with thinking off: the template rewrites the reply (its empty
    // reasoning dropped), and the turn resumes from the reply's start
    val question = user("Name a prime number above 50. The number only.")
    val turns = Seq.newBuilder[ujson.Value]
    turns += question
    for (turn <- 0 until 2) {
      val body = ujson.read(
        client
          .send(
            HttpRequest
              .newBuilder(
                URI.create(s"http://127.0.0.1:$port/v1/chat/completions")
              )
              .POST(
                HttpRequest.BodyPublishers.ofString(
                  ujson.write(
                    ujson.Obj(
                      "messages" -> ujson.Arr.from(turns.result()),
                      "max_tokens" -> 20,
                      "temperature" -> 0,
                      "chat_template_kwargs" -> ujson.Obj(
                        "enable_thinking" -> false
                      )
                    )
                  )
                )
              )
              .build(),
            HttpResponse.BodyHandlers.ofString()
          )
          .body()
      )
      val reply = body("choices")(0)("message")("content").str
      println(
        s"follow-up turn $turn: ${ujson.write(reply)}, prompt ${body("usage")("prompt_tokens").num.toInt} tokens, cached ${body("timings")("cache_n").num.toInt}"
      )
      turns += ujson.Obj("role" -> "assistant", "content" -> reply)
      turns += user("And another one?")
    }

    for (thinking <- Seq(false, true)) {
      val events = client
        .send(
          post(
            request(
              "What is 17 * 3? Answer with the number only.",
              thinking,
              300
            )
          ),
          HttpResponse.BodyHandlers.ofLines()
        )
        .body()
        .iterator()
        .asScala
        .filter(_.startsWith("data: "))
        .map(_.drop(6))
        .toVector
      val chunks = events.takeWhile(_ != "[DONE]").map(ujson.read(_))
      def deltas(field: String) = chunks
        .flatMap(c =>
          c("choices").arr
            .flatMap(_.obj.get("delta"))
            .flatMap(_.obj.get(field))
            .map(_.str)
        )
        .mkString
      val finish = chunks.flatMap(
        _("choices").arr
          .flatMap(_.obj.get("finish_reason"))
          .filter(_ != ujson.Null)
          .map(_.str)
      )
      val usage = chunks.flatMap(_.obj.get("usage")).headOption
      val timings = chunks.flatMap(_.obj.get("timings")).headOption
      println(
        s"thinking $thinking: reasoning ${ujson.write(deltas("reasoning_content").take(120))}"
      )
      println(
        s"  content ${ujson.write(deltas("content"))}, finish $finish, ended with [DONE]: ${events.last == "[DONE]"}"
      )
      println(
        s"  usage ${usage.map(ujson.write(_))}, decode ${timings.map(_("predicted_per_second").num.round)} tok/s"
      )
    }

    // drift's Stop closes the stream: the reply must end, and the next request run
    val started = System.nanoTime()
    val dropped = client
      .send(
        post(
          request(
            "Count from 1 to 500, comma separated.",
            thinking = false,
            2000
          )
        ),
        HttpResponse.BodyHandlers.ofInputStream()
      )
      .body()
    dropped.readNBytes(4000)
    dropped.close()
    val next = client.send(
      post(request("Say hi.", thinking = false, 20)),
      HttpResponse.BodyHandlers.ofString()
    )
    println(
      f"after a dropped stream the next reply came ${(System.nanoTime() - started) / 1e9}%.1f s later: ${next.statusCode()}"
    )
    sys.exit(0)
  }
}
