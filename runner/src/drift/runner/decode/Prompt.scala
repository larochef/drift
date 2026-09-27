package drift.runner.decode

import drift.runner.vision.PreparedImage

/** An image in a prompt: its tokens are `at until at + image.tokens`. */
final case class PromptImage(at: Int, image: PreparedImage) {
  def end: Int = at + image.tokens
}

/** What a generation prefills: token ids, among them the tokens of `images`
  * (whose embeddings the vision tower gives).
  *
  * Where each token turns, as transformers' `get_rope_index` for Qwen3-VL and
  * Qwen 3.5: text counts on by one, the same position on all three axes; an
  * image's tokens take the position after the text before it, plus their row
  * (height axis) and column (width axis); the text after an image resumes past
  * its longer side. A text-only prompt turns by its slots.
  */
final case class Prompt(ids: Array[Int], images: Seq[PromptImage]) {

  /** Every token's positions, `[3, ids.length]` axis-major, and the `shift` of
    * the tokens after the prompt: slot `s` turns by `s + shift`.
    */
  lazy val (positions: Array[Int], shift: Int) = {
    val n = ids.length
    val out = new Array[Int](3 * n)
    var slot = 0
    var next = 0
    def text(until: Int): Unit =
      while (slot < until) {
        (0 until 3).foreach(a => out(a * n + slot) = next)
        next += 1
        slot += 1
      }
    images.sortBy(_.at).foreach { placed =>
      text(placed.at)
      val image = placed.image
      for {
        r <- 0 until image.tokenRows
        c <- 0 until image.tokenColumns
      } {
        out(slot) = next
        out(n + slot) = next + r
        out(2 * n + slot) = next + c
        slot += 1
      }
      next += math.max(image.tokenRows, image.tokenColumns)
    }
    text(n)
    (out, next - n)
  }

  /** Positions of slots `from until ids.length`, axis-major. */
  def positionsFrom(from: Int): Array[Int] = {
    val n = ids.length
    (0 until 3).toArray.flatMap(a => positions.slice(a * n + from, (a + 1) * n))
  }

  /** How many of the prompt's first tokens `held` holds too: the tokens match,
    * and the images among them are the same pictures, none cut by the end. At
    * most all tokens but the last, which runs for the logits.
    */
  def sharedPrefix(
      held: collection.Seq[Int],
      heldImages: Seq[PromptImage]
  ): Int = {
    val limit = math.min(held.length, ids.length - 1)
    var shared = 0
    while (shared < limit && ids(shared) == held(shared)) shared += 1
    def keyed(images: Seq[PromptImage]) =
      images.map(i => (i.at, i.image.key)).toSet
    val (mine, theirs) = (keyed(images), keyed(heldImages))
    // back to before the first image cut by the end or not in both
    (images ++ heldImages)
      .filter(i => i.at < shared)
      .filter(i =>
        i.end > shared || !mine(i.at -> i.image.key) || !theirs(
          i.at -> i.image.key
        )
      )
      .map(_.at)
      .foldLeft(shared)(math.min)
  }
}

object Prompt {
  def text(ids: Array[Int]): Prompt = Prompt(ids, Nil)

  /** Rendered ids whose image placeholders (`pad`, one per image, in order)
    * become each image's tokens.
    */
  def withImages(
      ids: Array[Int],
      pad: Int,
      images: Seq[PreparedImage]
  ): Prompt = {
    val expanded = Array.newBuilder[Int]
    val placed = Vector.newBuilder[PromptImage]
    var (length, next) = (0, 0)
    ids.foreach { id =>
      if (id == pad && next < images.size) {
        val image = images(next)
        placed += PromptImage(length, image)
        expanded ++= Array.fill(image.tokens)(pad)
        length += image.tokens
        next += 1
      } else {
        expanded += id
        length += 1
      }
    }
    if (next != images.size)
      throw new IllegalArgumentException(
        s"the template placed $next of the ${images.size} images"
      )
    Prompt(expanded.result(), placed.result())
  }
}
