package drift.runner.decode

import utest.*

import drift.runner.vision.PreparedImage

/** Where a prompt's tokens turn with images among them, and when a held prefix
  * still matches.
  */
object PromptTests extends TestSuite {

  private def image(key: String, rows: Int, columns: Int) =
    PreparedImage(Array.emptyFloatArray, 2 * rows, 2 * columns, 2, key)

  val tests = Tests {
    test(
      "an image turns on its grid; the text after resumes past its longer side"
    ) {
      // 2 text, a 2 × 3 image (6 tokens), 2 text
      val prompt =
        Prompt(Array.fill(10)(1), Seq(PromptImage(2, image("a", 2, 3))))
      val Seq(t, h, w) = prompt.positions.grouped(10).toSeq.map(_.toSeq)
      assert(t == Seq(0, 1, 2, 2, 2, 2, 2, 2, 5, 6))
      assert(h == Seq(0, 1, 2, 2, 2, 3, 3, 3, 5, 6))
      assert(w == Seq(0, 1, 2, 3, 4, 2, 3, 4, 5, 6))
      // slot 10 turns at 7
      assert(prompt.shift == -3)
      assert(Prompt.text(Array(1, 2, 3)).shift == 0)
    }
    test("a held prefix matches only with the same pictures") {
      val ids = Array.fill(12)(1)
      val a = PromptImage(2, image("a", 1, 2))
      val prompt = Prompt(ids, Seq(a))
      assert(prompt.sharedPrefix(ids.take(6), Seq(a)) == 6)
      // another picture: back to before it
      assert(
        prompt.sharedPrefix(
          ids.take(6),
          Seq(PromptImage(2, image("b", 1, 2)))
        ) == 2
      )
      // the prefix would cut the image
      assert(prompt.sharedPrefix(ids.take(3), Nil) == 2)
      assert(prompt.sharedPrefix(ids.take(2), Nil) == 2)
      // tokens that differ, and all but the last at most
      assert(prompt.sharedPrefix(ids.take(1) :+ 5, Nil) == 1)
      assert(Prompt.text(ids).sharedPrefix(ids ++ ids, Nil) == 11)
    }
  }
}
