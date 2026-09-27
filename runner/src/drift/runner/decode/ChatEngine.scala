package drift.runner.decode

import drift.runner.formats.{Gguf, GgufValue}
import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.text.*
import drift.runner.vision.ImageSizing

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.time.LocalDateTime

/** One chat request: the conversation as OpenAI messages (an image part as
  * `{"type": "image"}`), the pictures in the order they appear with a key of
  * each, the sampling, and what ends the reply.
  */
final case class ChatRequest(
    messages: ujson.Value,
    tools: Option[ujson.Value],
    maxTokens: Option[Int],
    sampling: Sampling,
    stopStrings: Seq[String],
    /** Extra template variables, e.g. `enable_thinking`. */
    templateVariables: Map[String, ujson.Value],
    images: Seq[(BufferedImage, String)]
)

/** Where a reply's text goes as it is produced. */
trait ChatListener {
  def reasoning(text: String): Unit
  def content(text: String): Unit

  /** Asked between tokens: true stops the reply (the client went away). */
  def cancelled: Boolean
}

final case class ChatResult(
    promptTokens: Int,
    /** The prompt's first tokens already in the sequence (not run again). */
    cachedTokens: Int,
    completionTokens: Int,
    /** "stop" (an end token or a stop string) or "length". */
    finishReason: String,
    promptMilliseconds: Double,
    predictedMilliseconds: Double,
    drafted: Int,
    accepted: Int
)

/** The library core of the runner (`specs/42`, step 7): a model with its own
  * tokenizer and chat template, one sequence, replies streamed as reasoning and
  * content. The HTTP server is a thin shell over it; drift could as well call
  * it directly. With `drafts`, a model carrying an MTP layer (in its own file
  * or in `draftModel`) drafts that many tokens per step. With a `visionModel`
  * (an mmproj) it sees images: the template's image placeholder becomes as many
  * tokens as the image has, their embeddings given by the vision tower.
  */
final class ChatEngine(
    ops: Ops,
    val modelPath: Path,
    draftModel: Option[Path],
    visionModel: Option[Path],
    val context: Int,
    drafts: Int
) extends AutoCloseable {

  private val model: CausalModel = Models.open(ops, modelPath, draftModel)

  private val vision: Option[QwenVision] =
    try visionModel.map(QwenVision.open(ops, _))
    catch {
      case error: Throwable =>
        model.close()
        throw error
    }

  /** Whether it reads images. */
  def sees: Boolean = vision.isDefined

  private val sizing =
    vision.map(v => ImageSizing.qwen(v.config.patch, v.config.merge))

  private val (
    tokenizer: Tokenizer,
    template: ChatTemplate,
    specialVariables: Map[String, ujson.Value],
    ends: Set[Int]
  ) = {
    val (file, mapped) = Gguf.open(modelPath)
    try {
      val tokenizer = GgufTokenizer.read(file)
      val template = new ChatTemplate(
        file.string("tokenizer.chat_template"),
        () => LocalDateTime.now()
      )
      def token(key: String) = file.metadata.get(key).collect {
        case GgufValue.Integer(id) => id.toInt
      }
      val variables = Seq(
        "bos" -> "tokenizer.ggml.bos_token_id",
        "eos" -> "tokenizer.ggml.eos_token_id"
      )
        .flatMap((name, key) =>
          token(key)
            .map(id => s"${name}_token" -> ujson.Str(tokenizer.tokens(id)))
        )
        .toMap
      // the end of a turn: the GGUF's eos and eot, and the chat markers that close one
      val ends =
        (Seq("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id")
          .flatMap(token) ++
          Seq(
            "<|im_end|>",
            "<|endoftext|>",
            "<turn|>",
            "<end_of_turn>",
            "<|return|>"
          ).flatMap(tokenizer.id)).toSet
      (tokenizer, template, variables, ends)
    } finally mapped.close()
  }

  private val generator = new Generator(
    ops,
    model,
    tokenizer,
    context,
    pageSize = 64,
    prefillChunk = 512,
    Speculation.multiToken(ops, model, drafts, prefillChunk = 512),
    vision.map(tower =>
      image => {
        val output = tower.encode(image)
        output.deepstack.foreach(ops.release)
        output.tokens
      }
    ),
    Seq("<|im_start|>", "<start_of_turn>", "<|turn>")
      .flatMap(tokenizer.id)
      .toSet
  )

  /** The rendered prompt's ids with each image placeholder (`<|image_pad|>`)
    * repeated for its image's tokens.
    */
  private def withImages(
      ids: Array[Int],
      images: Seq[(BufferedImage, String)]
  ): Prompt =
    (sizing, tokenizer.id("<|image_pad|>")) match {
      case (Some(sizing), Some(pad)) if images.nonEmpty =>
        Prompt.withImages(
          ids,
          pad,
          images.map((picture, key) => sizing.prepare(picture, key))
        )
      case (_, _) if images.nonEmpty =>
        throw new IllegalArgumentException("this model reads text only")
      case _ => Prompt.text(ids)
    }

  def tokenize(text: String): Array[Int] =
    tokenizer.encode(text, addSpecial = false)

  /** Renders, prefills and decodes one reply; one at a time. */
  def chat(request: ChatRequest, listener: ChatListener): ChatResult =
    synchronized {
      val prompt = template.render(
        request.messages,
        request.tools,
        addGenerationPrompt = true,
        specialVariables ++ request.templateVariables
      )
      val tokens = withImages(
        tokenizer.encode(prompt, addSpecial = true),
        request.images
      )
      val ids = tokens.ids
      if (ids.length >= context)
        throw new IllegalArgumentException(
          s"the prompt is ${ids.length} tokens; the context holds $context"
        )
      val budget = math.min(
        request.maxTokens.getOrElse(Int.MaxValue),
        context - ids.length
      )
      val opened =
        prompt.lastIndexOf("<think>") > prompt.lastIndexOf("</think>")

      var stopped = false
      val stops = new StopStrings(
        request.stopStrings,
        listener.content,
        () => stopped = true
      )
      val splitter =
        new ReasoningSplitter(opened, listener.reasoning, stops.accept)
      val generated = generator.generate(
        tokens,
        budget,
        new Sampler(request.sampling),
        ends,
        piece => {
          splitter.accept(piece)
          !stopped && !listener.cancelled
        }
      )
      splitter.finish()
      stops.finish()
      val ended = generated.ids.lastOption.exists(ends) || stopped
      ChatResult(
        ids.length,
        generated.reusedTokens,
        generated.ids.size,
        if (ended || listener.cancelled) "stop" else "length",
        generated.promptSeconds * 1000,
        generated.decodeSeconds * 1000,
        generated.drafted,
        generated.accepted
      )
    }

  def close(): Unit = {
    generator.close()
    vision.foreach(_.close())
    model.close()
  }
}

/** Stop strings on the content: text is held back while it could still be the
  * start of one, and the reply ends at the first that appears (not shown).
  */
final private[decode] class StopStrings(
    stops: Seq[String],
    content: String => Unit,
    stop: () => Unit
) {
  private var pending = ""
  private var done = false

  def accept(text: String): Unit = if (!done) {
    pending += text
    stops
      .flatMap(s => Option(pending.indexOf(s)).filter(_ >= 0))
      .minOption match {
      case Some(at) =>
        emit(pending.substring(0, at))
        pending = ""
        done = true
        stop()
      case None =>
        val keep = stops
          .flatMap(s =>
            (1 until s.length).reverse.find(n => pending.endsWith(s.take(n)))
          )
          .maxOption
          .getOrElse(0)
        emit(pending.dropRight(keep))
        pending = pending.takeRight(keep)
    }
  }

  def finish(): Unit = if (!done) {
    emit(pending)
    pending = ""
  }

  private def emit(text: String): Unit = if (text.nonEmpty) content(text)
}
