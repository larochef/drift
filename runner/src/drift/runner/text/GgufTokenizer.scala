package drift.runner.text

import drift.runner.formats.*

/** A tokenizer from a GGUF's `tokenizer.ggml.*` metadata. GGUF names its
  * pre-tokenizer instead of spelling it, so the regex of each name the runner
  * knows is written here, as the model's own `tokenizer.json` has it (Java's
  * engine runs it as is; llama.cpp's rewrites exist for its own engine). A name
  * not in the table is refused.
  */
object GgufTokenizer {

  /** What a pre-tokenizer name means: its regex, NFC or not, ignore_merges or
    * not — each from the model family's `tokenizer.json`.
    */
  final case class PreTokenizer(
      regex: String,
      nfc: Boolean,
      ignoreMerges: Boolean
  )

  val PreTokenizers: Map[String, PreTokenizer] = Map(
    // Qwen 2, 2.5, 3
    "qwen2" -> PreTokenizer(
      """(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
      nfc = true,
      ignoreMerges = false
    ),
    // Qwen 3.5 and on: marks join letters
    "qwen35" -> PreTokenizer(
      """(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
      nfc = true,
      ignoreMerges = false
    ),
    // Mistral's Tekken (Mistral Small 3.x, Nemo): `mistral_common`'s pattern,
    // which llama.cpp and sd-cpp use too (the published tokenizer.json files
    // carry an older, wrong one); tiktoken's rule takes a whole word in the
    // vocabulary
    "tekken" -> PreTokenizer(
      """[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
      nfc = false,
      ignoreMerges = true
    ),
    // gpt-oss (o200k)
    "gpt-4o" -> PreTokenizer(
      """[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
      nfc = false,
      ignoreMerges = true
    )
  )

  /** GGUF token types: control tokens are special; user-defined ones are added
    * tokens that are not.
    */
  private val Control = 3
  private val UserDefined = 4

  def read(gguf: GgufFile): Tokenizer = {
    def fail(message: String) = throw new FormatException(
      s"${gguf.source}: $message"
    )
    val tokens = gguf.strings("tokenizer.ggml.tokens").toArray
    val types = gguf.metadata.get("tokenizer.ggml.token_type") match {
      case Some(GgufValue.Array(values)) =>
        values.map {
          case GgufValue.Integer(n) => n.toInt
          case other                => fail(s"token type $other")
        }
      case _ => fail("no tokenizer.ggml.token_type")
    }
    val merges = gguf.strings("tokenizer.ggml.merges").map { pair =>
      val space = pair.indexOf(' ', 1)
      if (space < 0) fail(s"merge '$pair' has no separator")
      (pair.substring(0, space), pair.substring(space + 1))
    }
    val model = gguf.string("tokenizer.ggml.model")
    val (pieces, nfc) = model match {
      case "gpt2" =>
        val name = gguf.string("tokenizer.ggml.pre")
        val pre = PreTokenizers.getOrElse(
          name,
          fail(s"pre-tokenizer '$name' is not known to the runner")
        )
        (
          Pieces.ByteLevelWords(
            TokenizerJson.compile(pre.regex),
            pre.ignoreMerges
          ),
          pre.nfc
        )
      case "gemma4" => (Pieces.CharactersWithByteFallback, false)
      case other    => fail(s"tokenizer model '$other' is not supported")
    }
    val added = types.zipWithIndex.collect {
      case (kind, id) if kind == Control || kind == UserDefined =>
        tokens(id) -> id
    }.toMap
    val special = types.zipWithIndex.collect { case (Control, id) => id }.toSet
    // Gemma 4's GGUFs say no, but the model is trained with a BOS: llama.cpp adds it too
    val addBos = model == "gemma4" ||
      gguf.metadata
        .get("tokenizer.ggml.add_bos_token")
        .contains(GgufValue.Bool(true))
    val beginning =
      Option.when(addBos)(gguf.long("tokenizer.ggml.bos_token_id").toInt)
    new Tokenizer(
      tokens,
      Tokenizers.bpe(tokens, merges, fail),
      pieces,
      nfc,
      added,
      special,
      beginning
    )
  }
}
