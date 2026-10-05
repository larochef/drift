package drift.backend.assistant

import drift.backend.session.SessionManager
import drift.backend.storage.StorageService
import drift.shared.*

import java.net.URI
import java.net.http.*
import java.nio.file.{Path, Paths}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.{named, JsonCodecMaker}
import com.typesafe.scalalogging.Logger
import ox.flow.Flow

// llama-server's OpenAI-shaped request and streamed chunks, reduced to what
// drift writes and reads. Fields that must always be written carry no default
// (jsoniter omits a field equal to its default).
private case class ImageUrl(url: String)
private case class OpenAiPart(
    @named("type") kind: String,
    text: Option[String] = None,
    image_url: Option[ImageUrl] = None,
    /** Unverified against a llama.cpp build with video input: the part name
      * follows the OpenAI-style convention the vision parts use.
      */
    video_url: Option[ImageUrl] = None
)
private case class OpenAiMessage(role: String, content: List[OpenAiPart])
private case class StreamOptions(include_usage: Boolean)

/** One adapter of llama-server's per-request `lora` list: its id as
  * `/lora-adapters` numbers it, at a scale.
  */
private case class AdapterScale(id: Int, scale: Double)
private case class OpenAiRequest(
    messages: List[OpenAiMessage],
    stream: Boolean,
    stream_options: StreamOptions,
    max_tokens: Option[Int] = None,
    temperature: Option[Double] = None,
    lora: Option[List[AdapterScale]] = None
)
private case class LoadedAdapter(id: Int, path: String)
private case class Delta(
    content: Option[String] = None,
    reasoning_content: Option[String] = None
)
private case class Choice(
    delta: Option[Delta] = None,
    finish_reason: Option[String] = None
)
private case class Usage(
    prompt_tokens: Option[Int] = None,
    completion_tokens: Option[Int] = None
)

/** llama-server's timings, on the final chunk: the prefill (prompt) and
  * generation (predicted) phases, each with its token count, duration and rate.
  */
private case class Timings(
    prompt_n: Option[Int] = None,
    predicted_n: Option[Int] = None,
    prompt_ms: Option[Double] = None,
    predicted_ms: Option[Double] = None,
    prompt_per_second: Option[Double] = None,
    predicted_per_second: Option[Double] = None
)
private case class Chunk(
    choices: List[Choice] = Nil,
    usage: Option[Usage] = None,
    timings: Option[Timings] = None
)
private case class ErrorDetail(message: Option[String] = None)
private case class ErrorBody(error: Option[ErrorDetail] = None)
private case class GenerationSettings(n_ctx: Option[Int] = None)
private case class Modalities(vision: Option[Boolean] = None)
private case class Props(
    default_generation_settings: Option[GenerationSettings] = None,
    modalities: Option[Modalities] = None,
    model_path: Option[String] = None
)

/** One question answered whole (`AssistantProxy.ask`): no stream, and the chat
  * template told not to think first — a model that thinks spent its tokens
  * before the answer one time in three (`specs/52-auto-redraw.md`).
  */
private case class TemplateArguments(enable_thinking: Boolean)
private case class AskRequest(
    messages: List[OpenAiMessage],
    stream: Boolean,
    max_tokens: Int,
    temperature: Double,
    chat_template_kwargs: TemplateArguments
)
private case class AnswerMessage(content: Option[String] = None)
private case class AnswerChoice(message: Option[AnswerMessage] = None)
private case class AnswerBody(choices: List[AnswerChoice] = Nil)

private given JsonValueCodec[AskRequest] = JsonCodecMaker.make
private given JsonValueCodec[AnswerBody] = JsonCodecMaker.make
private given JsonValueCodec[OpenAiRequest] = JsonCodecMaker.make
private given JsonValueCodec[Chunk] = JsonCodecMaker.make
private given JsonValueCodec[ErrorBody] = JsonCodecMaker.make
private given JsonValueCodec[Props] = JsonCodecMaker.make
private given JsonValueCodec[List[LoadedAdapter]] = JsonCodecMaker.make

/** drift's side of a live llama-server
  * (`specs/18-assistant-models-and-sessions.md`): reads what the server applied
  * from `/props`, and relays a chat completion as a stream of [[ChatEvent]]s.
  * The browser never talks to the server's port itself, as with generation.
  */
final class AssistantProxy(
    sessionManager: SessionManager,
    media: AssistantMedia,
    storage: StorageService,
    /** Where the adapters llama-server was launched with live, to match them
      * with the configuration's LoRAs.
      */
    lorasRoot: Path
) {
  private val logger = Logger[AssistantProxy]

  private val client = HttpClient
    .newBuilder()
    .connectTimeout(java.time.Duration.ofSeconds(5))
    .build()

  def properties(sessionId: String): Either[String, AssistantProperties] =
    sessionManager.assistantPort(sessionId).flatMap { port =>
      try {
        val request = HttpRequest
          .newBuilder(URI.create(s"http://127.0.0.1:$port/props"))
          .timeout(java.time.Duration.ofSeconds(5))
          .GET()
          .build()
        val response =
          client.send(request, HttpResponse.BodyHandlers.ofString())
        if (response.statusCode != 200)
          Left(s"/props answered ${response.statusCode}")
        else {
          val props = readFromString[Props](response.body)
          Right(
            AssistantProperties(
              contextSize = props.default_generation_settings
                .flatMap(_.n_ctx)
                .getOrElse(0),
              vision = props.modalities.flatMap(_.vision).getOrElse(false),
              modelPath = props.model_path.getOrElse("")
            )
          )
        }
      } catch {
        case NonFatal(err) =>
          Left(
            s"/props failed: ${Option(err.getMessage).getOrElse(err.toString)}"
          )
      }
    }

  /** One streamed completion. Ends with a `done` event carrying the token
    * counts, or an `error` event; cancelling the flow (the client closing its
    * socket) closes the server connection, which stops the generation.
    */
  def chat(sessionId: String, request: ChatRequest): Flow[ChatEvent] =
    sessionManager.assistantPort(sessionId).flatMap { port =>
      // Attachments are read and encoded before anything is sent: a missing
      // file is refused by name rather than half a request reaching the model.
      val messages = request.messages.foldRight(
        Right(List.empty[OpenAiMessage]): Either[String, List[OpenAiMessage]]
      ) { (message, rest) =>
        for {
          tail <- rest
          head <- toOpenAi(message)
        } yield head :: tail
      }
      for {
        encoded <- messages
        adapters <- adaptersFor(sessionId, port)
      } yield (port, encoded, adapters)
    } match {
      case Left(reason) =>
        Flow.fromValues(ChatEvent(error = Some(reason), done = true))
      case Right((port, messages, adapters)) =>
        Flow.usingEmit { emit =>
          val body = writeToString(
            OpenAiRequest(
              messages = messages,
              stream = true,
              stream_options = StreamOptions(include_usage = true),
              max_tokens = request.maxTokens,
              temperature = request.temperature,
              lora = Option.when(adapters.nonEmpty)(adapters)
            )
          )
          val http = HttpRequest
            .newBuilder(
              URI.create(s"http://127.0.0.1:$port/v1/chat/completions")
            )
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build()
          try {
            val response =
              client.send(http, HttpResponse.BodyHandlers.ofLines())
            val lines = response.body()
            try
              if (response.statusCode != 200) {
                val text = lines.iterator().asScala.mkString("\n")
                emit(
                  ChatEvent(
                    error = Some(errorMessage(response.statusCode, text)),
                    done = true
                  )
                )
              } else {
                var promptTokens = Option.empty[Int]
                var completionTokens = Option.empty[Int]
                var timings = Option.empty[Timings]
                var finished = false
                val iterator = lines.iterator().asScala
                while (!finished && iterator.hasNext) {
                  val line = iterator.next()
                  if (line.startsWith("data:")) {
                    val payload = line.stripPrefix("data:").trim
                    if (payload == "[DONE]") finished = true
                    else if (payload.nonEmpty) {
                      val chunk = readFromString[Chunk](payload)
                      chunk.usage.foreach { usage =>
                        promptTokens = usage.prompt_tokens.orElse(promptTokens)
                        completionTokens =
                          usage.completion_tokens.orElse(completionTokens)
                      }
                      chunk.timings.foreach { reported =>
                        timings = Some(reported)
                        promptTokens = promptTokens.orElse(reported.prompt_n)
                        completionTokens =
                          completionTokens.orElse(reported.predicted_n)
                      }
                      chunk.choices.flatMap(_.delta).foreach { delta =>
                        val content = delta.content.filter(_.nonEmpty)
                        val reasoning =
                          delta.reasoning_content.filter(_.nonEmpty)
                        if (content.isDefined || reasoning.isDefined)
                          emit(
                            ChatEvent(content = content, reasoning = reasoning)
                          )
                      }
                    }
                  }
                }
                // A rate the server did not state is derived from count and
                // duration, so a build that omits one still shows both phases.
                def rate(
                    stated: Option[Double],
                    count: Option[Int],
                    millis: Option[Double]
                ): Option[Double] =
                  stated.orElse(
                    count
                      .zip(millis.filter(_ > 0))
                      .map((n, ms) => n * 1000.0 / ms)
                  )
                emit(
                  ChatEvent(
                    done = true,
                    promptTokens = promptTokens,
                    completionTokens = completionTokens,
                    promptMillis = timings.flatMap(_.prompt_ms),
                    completionMillis = timings.flatMap(_.predicted_ms),
                    promptTokensPerSecond = rate(
                      timings.flatMap(_.prompt_per_second),
                      timings.flatMap(_.prompt_n),
                      timings.flatMap(_.prompt_ms)
                    ),
                    completionTokensPerSecond = rate(
                      timings.flatMap(_.predicted_per_second),
                      timings.flatMap(_.predicted_n),
                      timings.flatMap(_.predicted_ms)
                    )
                  )
                )
              }
            finally lines.close()
          } catch {
            case NonFatal(err) =>
              logger.warn(s"Chat on session $sessionId failed", err)
              emit(
                ChatEvent(
                  error = Some(
                    s"chat failed: ${Option(err.getMessage).getOrElse(err.toString)}"
                  ),
                  done = true
                )
              )
          }
        }
    }

  /** The session's run configuration's default LoRAs as llama-server adapters
    * (`specs/35-assistant-loras.md`): launched unapplied, each is applied to
    * this request at the configuration's strength. A default LoRA the server
    * did not load — installed after the launch, or not GGUF — refuses the
    * request by name, as a tiled job refuses a missing default LoRA.
    */
  private def adaptersFor(
      sessionId: String,
      port: Int
  ): Either[String, List[AdapterScale]] =
    sessionManager.list
      .find(_.id == sessionId)
      .flatMap(session =>
        storage.get[RunConfiguration](
          "run-configurations",
          session.runConfigurationId
        )
      )
      .map(_.loras)
      .filter(_.nonEmpty) match {
      case None           => Right(Nil)
      case Some(defaults) =>
        loadedAdapters(port).flatMap { loaded =>
          defaults.foldRight(Right(Nil): Either[String, List[AdapterScale]]) {
            (configured, rest) =>
              for {
                tail <- rest
                lora <- storage
                  .get[Lora]("loras", configured.loraId)
                  .toRight(
                    s"default LoRA '${configured.loraId}' is not installed"
                  )
                files = lora.files.filter(file =>
                  LoraAdapters.isAdapterFile(file.fileName)
                )
                _ <- Either.cond(
                  files.nonEmpty,
                  (),
                  s"LoRA '${lora.label}' has no GGUF file, and llama.cpp loads GGUF adapters only"
                )
                ids <- files.foldRight(Right(Nil): Either[String, List[Int]]) {
                  (file, found) =>
                    for {
                      others <- found
                      id <- loaded
                        .get(
                          normalized(
                            lorasRoot.resolve(lora.storagePathOf(file))
                          )
                        )
                        .toRight(
                          s"LoRA '${lora.label}' is not loaded by this session " +
                            "(installed after it started?): restart the session"
                        )
                    } yield id :: others
                }
              } yield ids.map(AdapterScale(_, configured.strength)) ++ tail
          }
        }
    }

  /** What `/lora-adapters` lists, by normalized path. */
  private def loadedAdapters(port: Int): Either[String, Map[Path, Int]] =
    try {
      val request = HttpRequest
        .newBuilder(URI.create(s"http://127.0.0.1:$port/lora-adapters"))
        .timeout(java.time.Duration.ofSeconds(5))
        .GET()
        .build()
      val response = client.send(request, HttpResponse.BodyHandlers.ofString())
      if (response.statusCode != 200)
        Left(s"/lora-adapters answered ${response.statusCode}")
      else
        Right(
          readFromString[List[LoadedAdapter]](response.body)
            .map(adapter => normalized(Paths.get(adapter.path)) -> adapter.id)
            .toMap
        )
    } catch {
      case NonFatal(err) =>
        Left(
          s"/lora-adapters failed: ${Option(err.getMessage).getOrElse(err.toString)}"
        )
    }

  private def normalized(path: Path): Path = path.toAbsolutePath.normalize

  private def toOpenAi(message: ChatMessage): Either[String, OpenAiMessage] =
    message.attachments
      .foldRight(
        Right(List.empty[OpenAiPart]): Either[String, List[OpenAiPart]]
      ) { (attachment, rest) =>
        for {
          tail <- rest
          encoded <- media.encode(attachment)
        } yield {
          val url = Some(ImageUrl(encoded.dataUrl))
          (if (encoded.isVideo) OpenAiPart("video_url", video_url = url)
           else OpenAiPart("image_url", image_url = url)) :: tail
        }
      }
      .map(parts =>
        OpenAiMessage(
          role = message.role,
          content = OpenAiPart("text", text = Some(message.text)) :: parts
        )
      )

  /** The first live assistant session that reads images, oldest first — or why
    * there is none.
    */
  def visionSession: Either[String, String] =
    sessionManager.list
      .filter(session =>
        session.tool == RuntimeTool.LlamaCpp &&
          session.status == SessionStatus.Ready
      )
      .sortBy(_.startedAt)
      .find(session => properties(session.id).exists(_.vision))
      .map(_.id)
      .toRight(
        "no running assistant reads images: start one whose model has a " +
          "vision projector"
      )

  /** One question about one picture, answered whole and without thinking:
    * `system`, then `text` under the picture (a data URL). Blocks until the
    * answer is in.
    */
  def ask(
      sessionId: String,
      system: String,
      text: String,
      pictureUrl: String,
      maxTokens: Int,
      temperature: Double
  ): Either[String, String] =
    sessionManager.assistantPort(sessionId).flatMap { port =>
      val body = writeToString(
        AskRequest(
          messages = List(
            OpenAiMessage(
              "system",
              List(OpenAiPart("text", text = Some(system)))
            ),
            OpenAiMessage(
              "user",
              List(
                OpenAiPart("image_url", image_url = Some(ImageUrl(pictureUrl))),
                OpenAiPart("text", text = Some(text))
              )
            )
          ),
          stream = false,
          max_tokens = maxTokens,
          temperature = temperature,
          chat_template_kwargs = TemplateArguments(enable_thinking = false)
        )
      )
      try {
        val response = client.send(
          HttpRequest
            .newBuilder(
              URI.create(s"http://127.0.0.1:$port/v1/chat/completions")
            )
            .timeout(java.time.Duration.ofMinutes(15))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build(),
          HttpResponse.BodyHandlers.ofString()
        )
        if (response.statusCode != 200)
          Left(errorMessage(response.statusCode, response.body))
        else
          readFromString[AnswerBody](response.body).choices.headOption
            .flatMap(_.message)
            .flatMap(_.content)
            .filter(_.trim.nonEmpty)
            .toRight("the assistant answered nothing")
      } catch {
        case NonFatal(err) =>
          Left(
            s"asking the assistant failed: ${Option(err.getMessage).getOrElse(err.toString)}"
          )
      }
    }

  private def errorMessage(status: Int, text: String): String =
    (try readFromString[ErrorBody](text).error.flatMap(_.message)
    catch { case NonFatal(_) => None })
      .map(message => s"llama-server answered $status: $message")
      .getOrElse(s"llama-server answered $status: ${text.take(300)}")
}
