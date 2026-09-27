package drift.runner.text

import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets.UTF_8
import java.text.Normalizer
import java.util.regex.Pattern
import scala.collection.mutable

/** How text becomes the symbols that BPE merges. */
enum Pieces {

  /** GPT-2 style: the text is cut by `split` (every match a word, and any text
    * between matches too), each word's UTF-8 bytes are spelled in `ByteLevel`'s
    * alphabet, one symbol per byte. With `ignoreMerges`, a word that is a token
    * already is kept whole (gpt-oss).
    */
  case ByteLevelWords(split: Pattern, ignoreMerges: Boolean)

  /** SentencePiece style (Gemma): spaces become `▁`, the text is one word, one
    * symbol per character; a character not in the vocabulary falls back to its
    * UTF-8 bytes as `<0xNN>` tokens.
    */
  case CharactersWithByteFallback
}

/** A tokenizer (`specs/42`, Chat requirements): added tokens matched first,
  * then normalization, pieces and BPE. Built from a `tokenizer.json`
  * (`TokenizerJson`) or a GGUF's vocabulary (`GgufTokenizer`); both give the
  * token ids HuggingFace's `tokenizers` gives, which the tests check against
  * the library itself.
  */
final class Tokenizer(
    /** Each id's token, as the vocabulary spells it. */
    val tokens: Array[String],
    bpe: Bpe,
    pieces: Pieces,
    /** NFC normalization before splitting (Qwen). */
    nfc: Boolean,
    /** Tokens matched verbatim in the text before anything else, by id. */
    addedTokens: Map[String, Int],
    /** The added tokens that are special (control) tokens. */
    val specialIds: Set[Int],
    /** Put before every encoded text when `addSpecial` is asked (Gemma's BOS).
      */
    val beginning: Option[Int]
) {

  private val ids: java.util.HashMap[String, Integer] = {
    val map = new java.util.HashMap[String, Integer](tokens.length * 2)
    tokens.zipWithIndex.foreach((token, id) =>
      if (token != null) map.putIfAbsent(token, id)
    )
    map
  }

  def vocabularySize: Int = tokens.length

  def id(token: String): Option[Int] = Option(ids.get(token)).map(_.intValue)

  /** Longest first, so that at a given position the longest added token wins.
    */
  private val addedPattern: Option[Pattern] =
    Option.when(addedTokens.nonEmpty)(
      Pattern.compile(
        addedTokens.keys.toSeq
          .sortBy(-_.length)
          .map(Pattern.quote)
          .mkString("|")
      )
    )

  private val byteFallback: Array[Int] =
    Array.tabulate(256)(byte => id(f"<0x$byte%02X>").getOrElse(-1))

  /** Words seen before and their ids: text repeats words endlessly. */
  private val cache =
    new java.util.concurrent.ConcurrentHashMap[String, Array[Int]]()

  /** The ids of `text`. Added tokens in the text are always matched, as
    * HuggingFace does; `addSpecial` puts the beginning token first when the
    * tokenizer has one.
    */
  def encode(text: String, addSpecial: Boolean): Array[Int] = {
    val out = mutable.ArrayBuilder.make[Int]
    if (addSpecial) beginning.foreach(out += _)
    var start = 0
    addedPattern.foreach { pattern =>
      val matcher = pattern.matcher(text)
      while (matcher.find()) {
        encodeText(text.substring(start, matcher.start()), out)
        out += addedTokens(matcher.group())
        start = matcher.end()
      }
    }
    encodeText(text.substring(start), out)
    out.result()
  }

  private def encodeText(raw: String, out: mutable.ArrayBuilder[Int]): Unit =
    if (raw.nonEmpty) {
      val text =
        if (nfc) Normalizer.normalize(raw, Normalizer.Form.NFC) else raw
      pieces match {
        case Pieces.ByteLevelWords(split, ignoreMerges) =>
          val matcher = split.matcher(text)
          var start = 0
          while (matcher.find()) {
            if (matcher.start() > start)
              out ++= byteLevelWord(
                text.substring(start, matcher.start()),
                ignoreMerges
              )
            if (matcher.end() > matcher.start())
              out ++= byteLevelWord(matcher.group(), ignoreMerges)
            start = matcher.end()
          }
          if (start < text.length)
            out ++= byteLevelWord(text.substring(start), ignoreMerges)
        case Pieces.CharactersWithByteFallback =>
          out ++= characterWord(text.replace(' ', '▁'))
      }
    }

  private def byteLevelWord(word: String, ignoreMerges: Boolean): Array[Int] =
    cache.computeIfAbsent(
      word,
      _ => {
        val spelled = ByteLevel.encode(word)
        val whole = if (ignoreMerges) id(spelled) else None
        whole.fold(bpe.merge(spelled.map(c => requiredId(c.toString)).toArray))(
          Array(_)
        )
      }
    )

  private def characterWord(word: String): Array[Int] = {
    val symbols = mutable.ArrayBuilder.make[Int]
    word.codePoints().forEach { codePoint =>
      val character = new String(Character.toChars(codePoint))
      id(character) match {
        case Some(found) => symbols += found
        case None        =>
          character.getBytes(UTF_8).foreach { byte =>
            val fallback = byteFallback(byte & 0xff)
            if (fallback < 0)
              throw new IllegalStateException(
                s"no byte token for 0x${(byte & 0xff).toHexString}"
              )
            symbols += fallback
          }
      }
    }
    bpe.merge(symbols.result())
  }

  private def requiredId(token: String): Int =
    id(token).getOrElse(
      throw new IllegalStateException(
        s"the vocabulary has no token for '$token'"
      )
    )

  /** The bytes `id` stands for: an added token's own text, a byte-level token's
    * bytes, a `<0xNN>` token's byte, `▁` as a space.
    */
  def bytesOf(id: Int): Array[Byte] = {
    val token = tokens(id)
    if (addedTokens.contains(token) && addedTokens(token) == id)
      token.getBytes(UTF_8)
    else
      pieces match {
        case Pieces.ByteLevelWords(_, _) =>
          token
            .map(c =>
              ByteLevel.byteOf
                .getOrElse(
                  c,
                  throw new IllegalStateException(
                    s"token $id is not byte-level: $token"
                  )
                )
                .toByte
            )
            .toArray
        case Pieces.CharactersWithByteFallback =>
          if (
            byteFallback(0) >= 0 && token.length == 6 && token.startsWith(
              "<0x"
            ) && token.endsWith(">")
          )
            Array(Integer.parseInt(token.substring(3, 5), 16).toByte)
          else token.replace('▁', ' ').getBytes(UTF_8)
      }
  }

  /** The text of `ids`; bytes that are not valid UTF-8 become U+FFFD. */
  def decode(ids: Seq[Int], skipSpecial: Boolean): String = {
    val bytes = new ByteArrayOutputStream()
    ids.foreach(id =>
      if (!(skipSpecial && specialIds(id))) bytes.writeBytes(bytesOf(id))
    )
    new String(bytes.toByteArray, UTF_8)
  }

  /** Decodes one token at a time for streaming: text is released only once its
    * UTF-8 sequence is complete, so a character split over tokens is never
    * shown in halves.
    */
  def streamingDecoder(skipSpecial: Boolean): StreamingDecoder =
    new StreamingDecoder(this, skipSpecial)
}

final class StreamingDecoder(tokenizer: Tokenizer, skipSpecial: Boolean) {

  private val pending = new ByteArrayOutputStream()

  /** The text completed by `id`, possibly empty. */
  def next(id: Int): String =
    if (skipSpecial && tokenizer.specialIds(id)) ""
    else {
      pending.writeBytes(tokenizer.bytesOf(id))
      val bytes = pending.toByteArray
      val complete = completeLength(bytes)
      pending.reset()
      pending.write(bytes, complete, bytes.length - complete)
      new String(bytes, 0, complete, UTF_8)
    }

  /** Whatever is left, invalid bytes as U+FFFD. */
  def finish(): String = {
    val rest = new String(pending.toByteArray, UTF_8)
    pending.reset()
    rest
  }

  /** How many leading bytes end on a character boundary: an unfinished UTF-8
    * sequence at the end (at most 3 bytes) waits for the next token.
    */
  private def completeLength(bytes: Array[Byte]): Int = {
    var start = bytes.length - 1
    while (
      start >= 0 && start >= bytes.length - 4 && (bytes(start) & 0xc0) == 0x80
    ) start -= 1
    if (start < 0) return bytes.length
    val lead = bytes(start) & 0xff
    val needed =
      if (lead >= 0xf0) 4
      else if (lead >= 0xe0) 3
      else if (lead >= 0xc0) 2
      else 1
    if (bytes.length - start < needed) start else bytes.length
  }
}
