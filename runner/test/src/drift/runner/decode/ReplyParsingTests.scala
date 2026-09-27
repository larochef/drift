package drift.runner.decode

import utest.*

/** How a streamed reply is cut into reasoning and content, and where stop
  * strings end it — whatever the pieces' boundaries.
  */
object ReplyParsingTests extends TestSuite {

  /** Runs the splitter over `text` cut into pieces of `size` characters. */
  private def split(
      text: String,
      inside: Boolean,
      size: Int
  ): (String, String) = {
    val (reasoning, content) = (new StringBuilder, new StringBuilder)
    val splitter = new ReasoningSplitter(inside, reasoning ++= _, content ++= _)
    text.grouped(size).foreach(splitter.accept)
    splitter.finish()
    (reasoning.toString, content.toString)
  }

  private def stopped(
      text: String,
      stops: Seq[String],
      size: Int
  ): (String, Boolean) = {
    val out = new StringBuilder
    var ended = false
    val filter = new StopStrings(stops, out ++= _, () => ended = true)
    text.grouped(size).takeWhile(_ => !ended).foreach(filter.accept)
    filter.finish()
    (out.toString, ended)
  }

  val tests = Tests {
    test("a think block becomes reasoning, whatever the cuts") {
      (1 to 12).foreach { size =>
        assert(
          split(
            "<think>\nweighing it\n</think>\n\nThe answer.",
            inside = false,
            size
          ) == ("\nweighing it\n", "The answer.")
        )
      }
    }
    test("a block the prompt opened") {
      (1 to 12).foreach(size =>
        assert(
          split("still thinking</think>\n\nDone.", inside = true, size) == (
            "still thinking",
            "Done."
          )
        )
      )
    }
    test("no block: all content; a tag after content stays content") {
      assert(split("Plain answer.", inside = false, 3) == ("", "Plain answer."))
      assert(
        split("A <think> in text", inside = false, 2) == (
          "",
          "A <think> in text"
        )
      )
    }
    test("an unfinished block is reasoning") {
      assert(split("<think>cut short", inside = false, 4) == ("cut short", ""))
    }
    test("stop strings end the reply and are not shown, whatever the cuts") {
      (1 to 8).foreach { size =>
        assert(
          stopped("one two STOP three", Seq("STOP"), size) == ("one two ", true)
        )
      }
      assert(
        stopped("no stop here S", Seq("STOP"), 3) == ("no stop here S", false)
      )
      assert(stopped("a\nUser: b", Seq("User:", "\n\n"), 2) == ("a\n", true))
    }
  }
}
