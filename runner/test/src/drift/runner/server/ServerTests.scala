package drift.runner.server

import java.nio.file.Paths

import utest.*

/** llama-server's flags and OpenAI's request shapes, as drift sends them. */
object ServerTests extends TestSuite {

  val tests = Tests {
    test("drift's launch line parses; speed-only flags are accepted") {
      val options = ServerOptions.parse(
        Seq(
          "-m",
          "/models/q.gguf",
          "-c",
          "131072",
          "-ngl",
          "99",
          "-fa",
          "on",
          "--spec-type",
          "draft-mtp",
          "--spec-draft-n-max",
          "2",
          "--host",
          "127.0.0.1",
          "--port",
          "9123"
        )
      )
      assert(
        options.map(o =>
          (o.model, o.context, o.host, o.port, o.drafts)
        ) == Right(
          (Paths.get("/models/q.gguf"), 131072, "127.0.0.1", 9123, 2)
        )
      )
      assert(options.toOption.get.notes.size == 2)
      assert(ServerOptions.parse(Seq("-m", "q.gguf")).map(_.drafts) == Right(0))
    }
    test("a draft model's MTP layer, as the ROCmFP4 model cards pass it") {
      val options = ServerOptions.parse(
        Seq(
          "-m",
          "q.gguf",
          "--spec-type",
          "draft-mtp",
          "--model-draft",
          "mtp.gguf",
          "--spec-draft-n-max",
          "4",
          "--spec-draft-ngl",
          "99",
          "--spec-draft-p-min",
          "0.5"
        )
      )
      assert(
        options.map(o => (o.draftModel, o.drafts)) ==
          Right((Some(Paths.get("mtp.gguf")), 4))
      )
      assert(
        ServerOptions
          .parse(Seq("-m", "q.gguf", "-md", "mtp.gguf"))
          .map(_.draftModel) == Right(Some(Paths.get("mtp.gguf")))
      )
    }
    test("what the runner cannot honour yet is refused by name") {
      // --mmproj is accepted: /props reports no vision, so drift sends no images
      assert(
        ServerOptions.parse(Seq("-m", "a.gguf", "--mmproj", "b.gguf")).isRight
      )
      assert(
        ServerOptions
          .parse(Seq("-m", "a.gguf", "--lora", "l.gguf"))
          .left
          .toOption
          .get
          .contains("--lora")
      )
      assert(
        ServerOptions.parse(Seq("-m", "a.gguf", "--frobnicate")) == Left(
          "unknown flag --frobnicate"
        )
      )
      assert(
        ServerOptions.parse(Seq("-c", "10")) == Left(
          "no model: pass -m <file.gguf>"
        )
      )
    }
    test("drift's request: text parts join, sampling and stops read") {
      val body = ujson.Obj(
        "messages" -> ujson.Arr(
          ujson.Obj(
            "role" -> "user",
            "content" -> ujson.Arr(
              ujson.Obj("type" -> "text", "text" -> "Hel"),
              ujson.Obj("type" -> "text", "text" -> "lo")
            )
          )
        ),
        "stream" -> true,
        "temperature" -> 0.2,
        "max_tokens" -> 64,
        "stop" -> "END",
        "seed" -> 7,
        "chat_template_kwargs" -> ujson.Obj("enable_thinking" -> false)
      )
      val request = OpenAi.request(body, 4096, vision = false).toOption.get
      assert(request.messages(0)("content").str == "Hello")
      assert(request.maxTokens == Some(64), request.stopStrings == Seq("END"))
      assert(request.sampling.temperature == 0.2f, request.sampling.seed == 7L)
      assert(request.templateVariables == Map("enable_thinking" -> ujson.False))
    }
    test("images are refused without a vision tower") {
      assert(
        OpenAi.request(withImage("data:"), 4096, vision = false) == Left(
          "this model reads text only, not image_url"
        )
      )
    }
    test("an image part becomes the template's image, its picture decoded") {
      val picture = new java.awt.image.BufferedImage(
        3,
        2,
        java.awt.image.BufferedImage.TYPE_INT_RGB
      )
      val bytes = new java.io.ByteArrayOutputStream
      javax.imageio.ImageIO.write(picture, "png", bytes)
      val url = "data:image/png;base64," +
        java.util.Base64.getEncoder.encodeToString(bytes.toByteArray)
      val request =
        OpenAi.request(withImage(url), 4096, vision = true).toOption.get
      assert(
        request.messages(0)("content") == ujson.Arr(
          ujson.Obj("type" -> "image"),
          ujson.Obj("type" -> "text", "text" -> "What is it?")
        )
      )
      assert(request.images.map(_._1.getWidth) == Seq(3))
      assert(
        OpenAi.request(withImage("http://x/y.png"), 4096, vision = true) ==
          Left("images must come as data: URLs")
      )
    }
  }

  private def withImage(url: String) = ujson.Obj(
    "messages" -> ujson.Arr(
      ujson.Obj(
        "role" -> "user",
        "content" -> ujson.Arr(
          ujson
            .Obj("type" -> "image_url", "image_url" -> ujson.Obj("url" -> url)),
          ujson.Obj("type" -> "text", "text" -> "What is it?")
        )
      )
    )
  )
}
