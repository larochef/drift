package drift.runner

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.file.{Files, Path, Paths, StandardCopyOption}

import drift.runner.formats.Gguf
import drift.runner.text.{GgufTokenizer, Tokenizer}

/** The real tokenizers the tests run on: official `tokenizer.json` files,
  * downloaded once into `~/.cache/drift/runner-tests/tokenizers`, and the chat
  * GGUFs drift may have in the HuggingFace cache.
  */
object TokenizerFiles {

  /** Repositories whose `tokenizer.json` the tests compare against. */
  val Repositories: Seq[String] = Seq(
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.8-Flash-Next",
    "google/gemma-4-26b-a4b-it",
    "openai/gpt-oss-20b"
  )

  private val folder =
    Paths.get(
      sys.props("user.home"),
      ".cache",
      "drift",
      "runner-tests",
      "tokenizers"
    )

  /** Mistral Small's `tokenizer.json` with the pre-tokenizer's pattern replaced
    * by Tekken's own (`mistral_common`'s, which llama.cpp's `tekken` and sd-cpp
    * use): the published file carries an older one.
    */
  def tekkenJson: Path = {
    val target = folder.resolve("mistral-tekken.json")
    if (!Files.exists(target)) {
      val original = ujson.read(
        Files.readString(json("mistralai/Mistral-Small-3.1-24B-Instruct-2503"))
      )
      original("pre_tokenizer")("pretokenizers")(0)("pattern")("Regex") =
        GgufTokenizer.PreTokenizers("tekken").regex
      Files.writeString(target, ujson.write(original))
    }
    target
  }

  def json(repository: String): Path = {
    val target = folder.resolve(repository.replace('/', '_') + ".json")
    if (!Files.exists(target)) {
      Files.createDirectories(folder)
      val response = HttpClient
        .newBuilder()
        .followRedirects(HttpClient.Redirect.ALWAYS)
        .build()
        .send(
          HttpRequest
            .newBuilder(
              URI.create(
                s"https://huggingface.co/$repository/resolve/main/tokenizer.json"
              )
            )
            .build(),
          HttpResponse.BodyHandlers.ofFile(
            target.resolveSibling(target.getFileName.toString + ".part")
          )
        )
      if (response.statusCode() != 200)
        throw new IllegalStateException(
          s"downloading $repository's tokenizer.json: HTTP ${response.statusCode()}"
        )
      Files.move(response.body(), target, StandardCopyOption.REPLACE_EXISTING)
    }
    target
  }

  /** The chat GGUFs whose vocabulary matches each repository's, when drift has
    * them.
    */
  val Ggufs: Seq[(String, String)] = Seq(
    "Qwen/Qwen3-4B" -> "Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q8_0.gguf",
    "Qwen/Qwen3.6-35B-A3B" -> "Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q4_K.gguf",
    "google/gemma-4-26b-a4b-it" -> "gemma-4-26B-A4B-it-abliterated.Q4_K_M.gguf",
    "openai/gpt-oss-20b" -> "Huihui-gpt-oss-20b-BF16-abliterated-v2.i1-Q4_K_M.gguf"
  )

  /** A Mistral Small 3.2 GGUF (FLUX.2 [dev]'s text encoder), Tekken. */
  val MistralGguf =
    "Huihui-Mistral-Small-3.2-24B-Instruct-2506-abliterated-llamacppfixed.i1-Q4_K_M.gguf"

  def gguf(fileName: String): Option[Path] = {
    val hub = Paths.get(sys.props("user.home"), ".cache", "huggingface", "hub")
    if (!Files.isDirectory(hub)) None
    else {
      val found =
        Files.find(hub, 5, (path, _) => path.getFileName.toString == fileName)
      try found.findFirst().map(_.toRealPath()).map(Option(_)).orElse(None)
      finally found.close()
    }
  }

  def ggufTokenizer(path: Path): Tokenizer = {
    val (file, mapped) = Gguf.open(path)
    try GgufTokenizer.read(file)
    finally mapped.close()
  }
}
