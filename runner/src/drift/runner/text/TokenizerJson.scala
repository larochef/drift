package drift.runner.text

import drift.runner.formats.FormatException

import java.nio.file.{Files, Path}
import java.util.regex.Pattern

/** Reads a HuggingFace `tokenizer.json`. Only the pipelines the runner's models
  * use are understood; anything else is refused by name, never approximated:
  *   - Qwen, gpt-oss: an optional NFC normalizer, a regex `Split` (Isolated),
  *     then `ByteLevel` without its own regex, BPE;
  *   - Gemma: a `Replace(" ", "▁")` normalizer, a `Split(" ")` that then finds
  *     no space, BPE with byte fallback. Added tokens must be plain (no strip,
  *     no single-word, not normalized).
  */
object TokenizerJson {

  def load(path: Path): Tokenizer =
    read(ujson.read(Files.readString(path)), path.toString)

  def read(json: ujson.Value, source: String): Tokenizer = {
    def fail(message: String) = throw new FormatException(s"$source: $message")
    def kind(value: ujson.Value) =
      value.objOpt.flatMap(_.get("type")).map(_.str).getOrElse("null")

    val model = json("model")
    if (model("type").str != "BPE")
      fail(s"model ${model("type").str} is not supported")
    Seq("dropout", "continuing_subword_prefix", "end_of_word_suffix").foreach {
      key =>
        model.obj
          .get(key)
          .filterNot(v => v == ujson.Null || v.strOpt.contains(""))
          .foreach(v => fail(s"BPE $key $v is not supported"))
    }
    val byteFallback = model.obj.get("byte_fallback").exists(_.bool)
    val ignoreMerges = model.obj.get("ignore_merges").exists(_.bool)

    val normalizer = json("normalizer")
    val preTokenizer = json("pre_tokenizer")
    val (pieces, nfc) = (kind(normalizer), kind(preTokenizer)) match {
      case (normal @ ("null" | "NFC"), "Sequence") =>
        val steps = preTokenizer("pretokenizers").arr
        val split = steps.headOption.filter(step =>
          kind(step) == "Split" && step("behavior").str == "Isolated" && !step(
            "invert"
          ).bool &&
            step("pattern").obj.contains("Regex")
        )
        val byteLevel = steps
          .lift(1)
          .filter(step =>
            kind(step) == "ByteLevel" && !step("use_regex").bool && !step(
              "add_prefix_space"
            ).bool
          )
        (split, byteLevel) match {
          case (Some(s), Some(_)) if steps.size == 2 && !byteFallback =>
            (
              Pieces.ByteLevelWords(
                compile(s("pattern")("Regex").str),
                ignoreMerges
              ),
              normal == "NFC"
            )
          case _ =>
            fail(
              s"pre-tokenizer ${ujson.write(preTokenizer).take(200)} is not supported"
            )
        }
      case ("Replace", "Split")
          if normalizer("pattern").obj.get("String").exists(_.str == " ") &&
            normalizer("content").str == "▁" &&
            preTokenizer("pattern").obj
              .get("String")
              .exists(_.str == " ") && byteFallback =>
        (Pieces.CharactersWithByteFallback, false)
      case (n, p) =>
        fail(s"normalizer $n with pre-tokenizer $p is not supported")
    }

    val vocabulary = model("vocab").obj.view.mapValues(_.num.toInt).toMap
    val added = json("added_tokens").arr.toSeq
    added.foreach { token =>
      // a token matched in the normalized text: the same match where the
      // normalizer (none, or NFC) leaves the token as it is written
      val content = token("content").str
      val renormalized = token("normalized").bool &&
        java.text.Normalizer.normalize(
          content,
          java.text.Normalizer.Form.NFC
        ) != content
      if (
        renormalized || Seq("lstrip", "rstrip", "single_word")
          .exists(flag => token(flag).bool)
      )
        fail(
          s"added token ${token("content").str} has flags the runner does not apply"
        )
    }
    val size = (vocabulary.values ++ added.map(_("id").num.toInt)).max + 1
    val tokens = new Array[String](size)
    vocabulary.foreach((token, id) => tokens(id) = token)
    added.foreach(token => tokens(token("id").num.toInt) = token("content").str)

    val merges = model("merges").arr.map {
      case ujson.Arr(pair) => (pair(0).str, pair(1).str)
      case ujson.Str(pair) =>
        val space = pair.indexOf(' ', 1)
        (pair.substring(0, space), pair.substring(space + 1))
      case other => fail(s"merge $other")
    }

    val beginning = json("post_processor") match {
      case post if kind(post) == "TemplateProcessing" =>
        post("single").arr.headOption
          .flatMap(_.obj.get("SpecialToken"))
          .map(special =>
            vocabulary.getOrElse(
              special("id").str,
              fail(s"no token ${special("id").str}")
            )
          )
      case post if kind(post) == "ByteLevel" || post == ujson.Null => None
      case post => fail(s"post-processor ${kind(post)} is not supported")
    }

    new Tokenizer(
      tokens,
      Tokenizers.bpe(tokens, merges.toSeq, fail),
      pieces,
      nfc,
      added.map(token => token("content").str -> token("id").num.toInt).toMap,
      added.filter(_("special").bool).map(_("id").num.toInt).toSet,
      beginning
    )
  }

  /** HuggingFace's regexes run on Oniguruma, whose `\s` and friends are
    * Unicode-aware; Java's are ASCII unless asked.
    */
  def compile(regex: String): Pattern =
    Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS)
}

/** What both loaders share. */
object Tokenizers {

  /** The merge table of `merges` (in rank order), over ids. */
  def bpe(
      tokens: Array[String],
      merges: Seq[(String, String)],
      fail: String => Nothing
  ): Bpe = {
    val ids = new java.util.HashMap[String, Integer](tokens.length * 2)
    tokens.zipWithIndex.foreach((token, id) =>
      if (token != null) ids.putIfAbsent(token, id)
    )
    val table =
      new java.util.HashMap[java.lang.Long, java.lang.Long](merges.size * 2)
    merges.zipWithIndex.foreach { case ((left, right), rank) =>
      val (l, r, m) = (ids.get(left), ids.get(right), ids.get(left + right))
      if (l == null || r == null || m == null)
        fail(s"merge '$left' '$right' names a token the vocabulary lacks")
      val key = (l.toLong << 32) | (r.toLong & 0xffffffffL)
      table.putIfAbsent(key, (rank.toLong << 32) | (m.toLong & 0xffffffffL))
    }
    new Bpe(table)
  }
}
