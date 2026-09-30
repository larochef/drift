package drift.runner.text

import java.nio.file.{Files, Path}

/** A SentencePiece Unigram tokenizer from a `tokenizer.json` (T5's and UMT5's,
  * the text encoders of Wan): the normalizers (regex replacements), a Metaspace
  * pre-tokenizer (spaces to `▁`, one prepended, the text split before each
  * `▁`), the most probable segmentation of each word (Viterbi over the pieces'
  * log probabilities), unknown characters as the unknown piece (runs of them
  * fused, scored 10 below the least likely piece), and the special tokens the
  * post-processor appends (T5's `</s>`).
  */
final class Unigram private (
    pieces: java.util.HashMap[String, Integer],
    scores: Array[Double],
    longest: Int,
    unknown: Int,
    replacements: Seq[(scala.util.matching.Regex, String)],
    replacement: String,
    prependSpace: Boolean,
    appended: Seq[Int]
) {

  private val unknownScore = scores.min - 10

  def encode(text: String, addSpecial: Boolean): Array[Int] = {
    val normalized = replacements.foldLeft(text) {
      case (t, (pattern, content)) =>
        pattern.replaceAllIn(
          t,
          scala.util.matching.Regex.quoteReplacement(content)
        )
    }
    val spaced = normalized.replace(" ", replacement)
    val prefixed =
      if (prependSpace && !spaced.startsWith(replacement)) replacement + spaced
      else spaced
    val out = Array.newBuilder[Int]
    // an empty text has no words, not a lone `▁`
    if (normalized.nonEmpty) words(prefixed).foreach(word => segment(word, out))
    if (addSpecial) out ++= appended
    out.result()
  }

  /** The text cut before each `▁` (Metaspace's split). */
  private def words(text: String): Seq[String] = {
    val cuts = (0 until text.length).filter(i =>
      i > 0 && text.startsWith(replacement, i)
    )
    (0 +: cuts)
      .zip(cuts :+ text.length)
      .map((from, to) => text.substring(from, to))
      .filter(_.nonEmpty)
  }

  /** The best segmentation of `word` into pieces. */
  private def segment(
      word: String,
      out: collection.mutable.Builder[Int, Array[Int]]
  ): Unit = {
    val points = word.codePoints().toArray
    val n = points.length
    // the characters' offsets in the string, to cut substrings
    val offsets = new Array[Int](n + 1)
    (0 until n).foreach(i =>
      offsets(i + 1) = offsets(i) + Character.charCount(points(i))
    )
    val best = Array.fill(n + 1)(Double.NegativeInfinity)
    val from = new Array[Int](n + 1)
    val piece = new Array[Int](n + 1)
    best(0) = 0
    (0 until n).foreach { start =>
      if (best(start) > Double.NegativeInfinity) {
        var matched = false
        var end = start + 1
        while (end <= n && end - start <= longest) {
          val id = pieces.get(word.substring(offsets(start), offsets(end)))
          if (id != null) {
            if (end == start + 1) matched = true
            val score = best(start) + scores(id)
            if (score > best(end)) {
              best(end) = score
              from(end) = start
              piece(end) = id
            }
          }
          end += 1
        }
        if (!matched) {
          val score = best(start) + unknownScore
          if (score > best(start + 1)) {
            best(start + 1) = score
            from(start + 1) = start
            piece(start + 1) = unknown
          }
        }
      }
    }
    val ids = collection.mutable.ArrayBuffer.empty[Int]
    var at = n
    while (at > 0) {
      ids += piece(at)
      at = from(at)
    }
    // fuse runs of unknown pieces into one
    var previousUnknown = false
    ids.reverseIterator.foreach { id =>
      if (!(id == unknown && previousUnknown)) out += id
      previousUnknown = id == unknown
    }
  }
}

object Unigram {

  /** Whether `json` holds a Unigram model. */
  def holds(json: ujson.Value): Boolean =
    json.obj.get("model").exists(_.obj.get("type").exists(_.str == "Unigram"))

  def load(path: Path): Unigram =
    read(ujson.read(Files.readString(path)), path.toString)

  def read(json: ujson.Value, source: String): Unigram = {
    require(holds(json), s"$source holds no Unigram model")
    val model = json("model")
    val vocab = model("vocab").arr
    val pieces = new java.util.HashMap[String, Integer](vocab.size * 2)
    val scores = new Array[Double](vocab.size)
    var longest = 1
    vocab.zipWithIndex.foreach { (entry, id) =>
      val text = entry(0).str
      if (!pieces.containsKey(text)) pieces.put(text, id)
      scores(id) = entry(1).num
      longest = math.max(longest, text.codePointCount(0, text.length))
    }
    def normalizers(
        value: ujson.Value
    ): Seq[(scala.util.matching.Regex, String)] =
      if (value.isNull) Nil
      else
        value("type").str match {
          case "Sequence" => value("normalizers").arr.toSeq.flatMap(normalizers)
          case "Replace"  =>
            val pattern = value("pattern").obj
            val regex = pattern
              .get("Regex")
              .map(_.str)
              .getOrElse(java.util.regex.Pattern.quote(pattern("String").str))
            Seq(regex.r -> value("content").str)
          case other =>
            throw new IllegalArgumentException(
              s"$source: normalizer $other is not supported"
            )
        }
    val pre = json("pre_tokenizer")
    require(
      pre("type").str == "Metaspace",
      s"$source: pre-tokenizer ${pre("type").str} is not supported with Unigram"
    )
    val prepend = pre.obj
      .get("prepend_scheme")
      .map(_.str != "never")
      .orElse(pre.obj.get("add_prefix_space").map(_.bool))
      .getOrElse(true)
    val appended = json.obj
      .get("post_processor")
      .filterNot(_.isNull)
      .toSeq
      .flatMap(processor =>
        processor.obj.get("single").toSeq.flatMap(_.arr).flatMap { part =>
          part.obj.get("SpecialToken").toSeq.flatMap { special =>
            processor("special_tokens")(special("id").str)("ids").arr
              .map(_.num.toInt)
          }
        }
      )
    new Unigram(
      pieces,
      scores,
      longest,
      model("unk_id").num.toInt,
      normalizers(json("normalizer")),
      pre("replacement").str,
      prepend,
      appended
    )
  }
}
