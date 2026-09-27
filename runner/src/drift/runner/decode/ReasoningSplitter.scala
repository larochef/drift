package drift.runner.decode

/** Separates a reply's reasoning from its answer as the text streams: whatever
  * lies between `<think>` and `</think>` is reasoning, the rest is content. A
  * tag split over two pieces is held back until it is whole, so neither side
  * ever shows half a tag. `inside` says whether the prompt already opened a
  * block (Qwen's templates end the generation prompt with `<think>\n`). A block
  * opens only before any content, as llama.cpp's `deepseek` reasoning format
  * has it.
  */
final class ReasoningSplitter(
    inside: Boolean,
    reasoning: String => Unit,
    content: String => Unit
) {

  private val Open = "<think>"
  private val Close = "</think>"

  private var thinking = inside
  private var contentSeen = false
  private var pending = ""

  def accept(piece: String): Unit = {
    pending += piece
    var progressed = true
    while (progressed && pending.nonEmpty) {
      progressed = false
      if (thinking) {
        val end = pending.indexOf(Close)
        if (end >= 0) {
          emitReasoning(pending.substring(0, end))
          pending = pending.substring(end + Close.length).dropWhile(_ == '\n')
          thinking = false
          progressed = true
        } else emitReasoning(release(Close))
      } else if (!contentSeen) {
        val trimmed = pending.dropWhile(_.isWhitespace)
        if (trimmed.startsWith(Open)) {
          pending = trimmed.substring(Open.length)
          thinking = true
          progressed = true
        } else if (trimmed.nonEmpty && !Open.startsWith(trimmed)) {
          contentSeen = true
          emitContent(trimmed)
          pending = ""
        }
      } else {
        emitContent(pending)
        pending = ""
      }
    }
  }

  /** Whatever was held back, once the reply is over. */
  def finish(): Unit = {
    if (thinking) emitReasoning(pending)
    else emitContent(pending.dropWhile(c => !contentSeen && c.isWhitespace))
    pending = ""
  }

  /** The pending text but a tail that could still become `tag`. */
  private def release(tag: String): String = {
    val keep = (1 until tag.length).reverse
      .find(n => pending.endsWith(tag.take(n)))
      .getOrElse(0)
    val out = pending.dropRight(keep)
    pending = pending.takeRight(keep)
    out
  }

  private def emitReasoning(text: String): Unit =
    if (text.nonEmpty) reasoning(text)
  private def emitContent(text: String): Unit = if (text.nonEmpty) content(text)
}
