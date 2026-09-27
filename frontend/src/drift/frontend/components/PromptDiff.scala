package drift.frontend.components

import com.raquo.laminar.api.L.*

/** One prompt against another, word by word (`specs/20`): what the assistant
  * changed in the working prompt, or what changed between two versions.
  *
  * A requested rewrite changes nearly every word, and a word diff of that is
  * noise, so once more than half the words changed the view opens on the plain
  * new text and lists the words of the first prompt that did not make it across
  * — what a rewrite most needs checking for. Either view is a toggle away.
  */
class PromptDiff(
    before: String,
    after: String,
    /** What `before` is, for the "not carried over" line: "the working prompt",
      * "v5".
      */
    beforeLabel: String
) extends Component {
  import PromptDiff.*

  private val parts = diff(before, after)
  private val longest = words(before).length.max(words(after).length)
  private val unchanged = parts.count {
    case Part.Same(_) => true
    case _            => false
  }
  private val rewrite = longest > 0 && unchanged * 2 < longest
  private val showChanges = Var(!rewrite)

  private def rendered(part: Part): HtmlElement = part match {
    case Part.Same(word)    => span(word)
    case Part.Added(word)   => span(cls := "diff-added", word)
    case Part.Removed(word) => span(cls := "diff-removed", word)
  }

  private def notCarriedOver: HtmlElement = dropped(before, after) match {
    case Nil =>
      p(
        cls := "is-size-7 text-secondary mt-1",
        s"Every word of $beforeLabel is still there."
      )
    case gone =>
      div(
        cls := "mt-1",
        p(
          cls := "is-size-7 text-secondary mb-1",
          s"Not carried over from $beforeLabel:"
        ),
        div(
          cls := "tags",
          gone.map(word => span(cls := "tag is-danger is-light", word))
        )
      )
  }

  lazy val element: HtmlElement = div(
    child <-- showChanges.signal.map {
      case true =>
        div(
          cls := "prompt-diff",
          parts.flatMap(part => List[Mod[HtmlElement]](rendered(part), " "))
        )
      case false => div(div(cls := "prompt-diff", after), notCarriedOver)
    },
    a(
      cls := "is-size-7",
      child.text <-- showChanges.signal.map(changes =>
        if (changes) "show the new text" else "show the changes"
      ),
      onClick --> (_ => showChanges.update(!_))
    )
  )
}

object PromptDiff {
  enum Part {
    case Same(word: String)
    case Added(word: String)
    case Removed(word: String)
  }

  private[components] def words(text: String): Vector[String] =
    text.trim.split("\\s+").toVector.filter(_.nonEmpty)

  /** Past this many table cells — two prompts of about 2000 words each — the
    * diff is not worth computing, and the view treats it as a rewrite.
    */
  private val MaxCells = 4_000_000L

  /** Longest-common-subsequence over whitespace-separated words — prompts run
    * to a few hundred words, well inside what a plain table handles.
    */
  def diff(before: String, after: String): List[Part] = {
    val a = words(before)
    val b = words(after)
    val (n, m) = (a.length, b.length)
    if (n.toLong * m > MaxCells)
      a.map(Part.Removed(_)).toList ++ b.map(Part.Added(_))
    else {
      val lcs = Array.ofDim[Int](n + 1, m + 1)
      for {
        i <- n - 1 to 0 by -1
        j <- m - 1 to 0 by -1
      }
        lcs(i)(j) =
          if (a(i) == b(j)) lcs(i + 1)(j + 1) + 1
          else lcs(i + 1)(j).max(lcs(i)(j + 1))
      val parts = List.newBuilder[Part]
      var i = 0
      var j = 0
      while (i < n && j < m)
        if (a(i) == b(j)) { parts += Part.Same(a(i)); i += 1; j += 1 }
        else if (lcs(i + 1)(j) >= lcs(i)(j + 1)) {
          parts += Part.Removed(a(i)); i += 1
        } else { parts += Part.Added(b(j)); j += 1 }
      while (i < n) { parts += Part.Removed(a(i)); i += 1 }
      while (j < m) { parts += Part.Added(b(j)); j += 1 }
      parts.result()
    }
  }

  /** Words too common to say anything about what a rewrite lost. */
  private val StopWords = Set(
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "with",
    "in",
    "on",
    "at",
    "to",
    "for",
    "by",
    "from",
    "is",
    "are",
    "as",
    "its",
    "it",
    "this",
    "that",
    "very",
    "into"
  )

  private def normalized(word: String): String =
    word.toLowerCase.filter(_.isLetterOrDigit)

  /** The words of `before` that appear nowhere in `after`, ignoring case,
    * punctuation and filler — a rewrite may reorder and rephrase freely, so
    * presence is the check, not position.
    */
  def dropped(before: String, after: String): List[String] = {
    val present = words(after).map(normalized).toSet
    words(before)
      .map(normalized)
      .filter(word => word.length > 2 && !StopWords(word) && !present(word))
      .distinct
      .toList
  }
}
