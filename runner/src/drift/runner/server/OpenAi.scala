package drift.runner.server

import drift.runner.decode.*
import drift.runner.vision.ImageInput

/** OpenAI's chat-completion shapes, as llama-server writes them (drift's
  * assistant proxy reads `content` and `reasoning_content` deltas, the finish
  * reason, `usage` and llama.cpp's `timings`).
  */
object OpenAi {

  /** A request body into a `ChatRequest`, or why not. Text parts join into one
    * string (chat templates read `content` as a string), unless the message has
    * images: then its parts stay a list, each image an `{"type": "image"}` the
    * template turns into its placeholder, and the pictures go in order into the
    * request's `images`. Images need a model that sees (`vision`).
    */
  def request(
      body: ujson.Value,
      context: Int,
      vision: Boolean
  ): Either[String, ChatRequest] = {
    val messages = body.obj.get("messages").map(_.arr).getOrElse(Nil)
    def partKinds(message: ujson.Value): Set[String] =
      message.obj.get("content") match {
        case Some(ujson.Arr(parts)) =>
          parts.map(_.obj.get("type").map(_.str).getOrElse("text")).toSet
        case _ => Set("text")
      }
    val kinds = messages.flatMap(partKinds).toSet
    val refused =
      kinds -- (if (vision) Set("text", "image_url") else Set("text"))
    if (messages.isEmpty) Left("no messages")
    else if (refused.nonEmpty)
      Left(
        if (vision)
          s"this model reads text and images, not ${refused.mkString(", ")}"
        else s"this model reads text only, not ${refused.mkString(", ")}"
      )
    else {
      def isImage(part: ujson.Value) =
        part.obj.get("type").exists(_.str == "image_url")
      val decoded = messages
        .flatMap(_.obj.get("content").collect { case ujson.Arr(parts) =>
          parts
        })
        .flatten
        .filter(isImage)
        .map(part =>
          ImageInput.fromDataUrl(
            part.obj
              .get("image_url")
              .flatMap(_.objOpt)
              .flatMap(_.get("url"))
              .map(_.str)
              .getOrElse("")
          )
        )
      decoded.collectFirst { case Left(problem) => problem } match {
        case Some(problem) => Left(problem)
        case None          =>
          val images = decoded.collect { case Right(image) => image }.toSeq
          chatRequest(body, messages.map(content(_, isImage)).toSeq, images)
      }
    }
  }

  /** A message's content as the template reads it: its text parts joined, or,
    * with images, its parts in order, each image an `{"type": "image"}`.
    */
  private def content(
      message: ujson.Value,
      isImage: ujson.Value => Boolean
  ): ujson.Value = {
    def text(part: ujson.Value) = part.obj.get("text").map(_.str).getOrElse("")
    message.obj.get("content") match {
      case Some(ujson.Arr(parts)) =>
        val copy = ujson.copy(message)
        val converted: ujson.Value =
          if (parts.exists(isImage))
            ujson.Arr(parts.map { part =>
              if (isImage(part)) ujson.Obj("type" -> "image")
              else ujson.Obj("type" -> "text", "text" -> text(part))
            }.toSeq*)
          else ujson.Str(parts.map(text).mkString)
        copy("content") = converted
        copy
      case _ => message
    }
  }

  private def chatRequest(
      body: ujson.Value,
      messages: Seq[ujson.Value],
      images: Seq[(java.awt.image.BufferedImage, String)]
  ): Either[String, ChatRequest] = {
    def number(key: String) =
      body.obj.get(key).filter(_ != ujson.Null).map(_.num)
    val sampling = Sampling(
      temperature = number("temperature").getOrElse(0.8).toFloat,
      topK = number("top_k").getOrElse(40.0).toInt,
      topP = number("top_p").getOrElse(0.95).toFloat,
      minP = number("min_p").getOrElse(0.05).toFloat,
      seed = number("seed").map(_.toLong).getOrElse(System.nanoTime())
    )
    val stops = body.obj.get("stop") match {
      case Some(ujson.Str(s))   => Seq(s)
      case Some(ujson.Arr(all)) => all.map(_.str).toSeq
      case _                    => Nil
    }
    val variables = body.obj
      .get("chat_template_kwargs")
      .flatMap(_.objOpt)
      .map(_.toMap)
      .getOrElse(Map.empty)
    Right(
      ChatRequest(
        ujson.Arr(messages*),
        body.obj.get("tools").filter(_ != ujson.Null),
        number("max_tokens")
          .orElse(number("n_predict"))
          .map(_.toInt)
          .filter(_ >= 0),
        sampling,
        stops,
        variables,
        images
      )
    )
  }

  def chunk(
      id: String,
      model: String,
      delta: ujson.Obj,
      finishReason: Option[String]
  ): ujson.Obj =
    ujson.Obj(
      "id" -> id,
      "object" -> "chat.completion.chunk",
      "created" -> System.currentTimeMillis() / 1000,
      "model" -> model,
      "choices" -> ujson.Arr(
        ujson.Obj(
          "index" -> 0,
          "delta" -> delta,
          "finish_reason" -> finishReason.fold[ujson.Value](ujson.Null)(
            ujson.Str(_)
          )
        )
      )
    )

  def usage(result: ChatResult): ujson.Obj =
    ujson.Obj(
      "prompt_tokens" -> result.promptTokens,
      "completion_tokens" -> result.completionTokens,
      "total_tokens" -> (result.promptTokens + result.completionTokens)
    )

  /** llama.cpp's `timings`: each phase's tokens, milliseconds and rate. */
  def timings(result: ChatResult): ujson.Obj = {
    def rate(tokens: Int, milliseconds: Double) =
      if (milliseconds > 0) tokens / milliseconds * 1000 else 0.0
    val evaluated = result.promptTokens - result.cachedTokens
    ujson.Obj(
      "cache_n" -> result.cachedTokens,
      "prompt_n" -> evaluated,
      "prompt_ms" -> result.promptMilliseconds,
      "prompt_per_second" -> rate(evaluated, result.promptMilliseconds),
      "predicted_n" -> result.completionTokens,
      "predicted_ms" -> result.predictedMilliseconds,
      "predicted_per_second" -> rate(
        math.max(result.completionTokens - 1, 0),
        result.predictedMilliseconds
      ),
      "draft_n" -> result.drafted,
      "draft_n_accepted" -> result.accepted
    )
  }

  /** `timings` as llama-server's log prints them. */
  def timingReport(result: ChatResult): String = {
    val t = timings(result)
    def line(
        name: String,
        milliseconds: String,
        tokens: String,
        rate: String
    ) = {
      val count = t(tokens).num.toInt
      val perToken = if (count > 0) t(milliseconds).num / count else 0.0
      f"$name%16s = ${t(milliseconds).num}%10.2f ms / $count%5d tokens ($perToken%8.2f ms per token, ${t(rate).num}%8.2f tokens per second)"
    }
    val lines = Seq(
      line("prompt eval time", "prompt_ms", "prompt_n", "prompt_per_second"),
      line("eval time", "predicted_ms", "predicted_n", "predicted_per_second"),
      f"${"total time"}%16s = ${result.promptMilliseconds + result.predictedMilliseconds}%10.2f ms / ${t("prompt_n").num.toInt + result.completionTokens}%5d tokens"
    ) ++ Option.when(result.drafted > 0)(
      f"draft acceptance rate = ${result.accepted.toDouble / result.drafted}%.5f (${result.accepted}%5d accepted / ${result.drafted}%5d generated)"
    )
    lines.mkString("\n")
  }

  def error(code: Int, message: String): ujson.Obj =
    ujson.Obj(
      "error" -> ujson.Obj(
        "code" -> code,
        "message" -> message,
        "type" -> (if (code < 500) "invalid_request_error" else "server_error")
      )
    )
}
