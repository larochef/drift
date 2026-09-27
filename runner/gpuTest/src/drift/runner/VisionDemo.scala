package drift.runner

import java.nio.file.{Files, Paths}
import java.util.Base64

import drift.runner.decode.{ChatEngine, ChatListener, Sampling}
import drift.runner.ops.{HipOps, MatVecInputs}
import drift.runner.server.OpenAi

/** A question about a picture, end to end on the GPU as drift asks it: an
  * OpenAI message with the image as a data URL, the model's vision tower from
  * its mmproj, greedy decoding; then a follow-up turn, which reuses the image's
  * cached tokens.
  *
  * `./mill runner.gpuTest.runMain drift.runner.VisionDemo <model.gguf>
  * <mmproj.gguf> <image> ["<question>"] [drafts]`
  */
object VisionDemo {
  def main(arguments: Array[String]): Unit = {
    val Array(model, mmproj, image) = arguments.take(3)
    val question =
      arguments.lift(3).getOrElse("Describe this picture in three sentences.")
    val drafts = arguments.lift(4).fold(0)(_.toInt)
    val kind =
      if (image.toLowerCase.endsWith(".png")) "png" else "jpeg"
    val url = s"data:image/$kind;base64," +
      Base64.getEncoder.encodeToString(Files.readAllBytes(Paths.get(image)))
    val ops = new HipOps(Gpu.hip, MatVecInputs.Float)
    val started = System.nanoTime()
    val engine = new ChatEngine(
      ops,
      Paths.get(model),
      None,
      Some(Paths.get(mmproj)),
      context = 16384,
      drafts
    )
    println(f"loaded in ${(System.nanoTime() - started) / 1e9}%.1f s")
    var messages = Vector[ujson.Value](
      ujson.Obj(
        "role" -> "user",
        "content" -> ujson.Arr(
          ujson
            .Obj("type" -> "image_url", "image_url" -> ujson.Obj("url" -> url)),
          ujson.Obj("type" -> "text", "text" -> question)
        )
      )
    )
    for (followUp <- Seq("What is the dominant colour?", "")) {
      val body = ujson.Obj(
        "messages" -> ujson.Arr(messages*),
        "max_tokens" -> 300,
        "temperature" -> 0,
        "chat_template_kwargs" -> ujson.Obj("enable_thinking" -> false)
      )
      val request = OpenAi
        .request(body, 16384, vision = true)
        .fold(
          problem => throw new IllegalArgumentException(problem),
          _.copy(sampling = Sampling.Greedy)
        )
      val reply = new StringBuilder
      val result = engine.chat(
        request,
        new ChatListener {
          def reasoning(text: String): Unit = print(text)
          def content(text: String): Unit = {
            print(text); reply ++= text; System.out.flush()
          }
          def cancelled: Boolean = false
        }
      )
      println(
        f"\n--- prompt ${result.promptTokens} tokens (${result.cachedTokens} cached) in ${result.promptMilliseconds / 1000}%.2f s; ${result.completionTokens} tokens in ${result.predictedMilliseconds / 1000}%.2f s; ${result.accepted} of ${result.drafted} drafts accepted"
      )
      if (followUp.nonEmpty)
        messages = messages ++ Seq(
          ujson.Obj("role" -> "assistant", "content" -> reply.toString),
          ujson.Obj("role" -> "user", "content" -> followUp)
        )
    }
    engine.close()
    ops.close()
  }
}
