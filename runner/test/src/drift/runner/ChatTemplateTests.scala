package drift.runner

import java.nio.file.Files
import java.time.LocalDate

import scala.util.Try

import utest.*

import drift.runner.text.ChatTemplate

/** The runner's Jinja against transformers' `apply_chat_template`, on each
  * model's real template: every conversation of `fixtures/conversations.json`
  * with each set of flags gives the same text, or fails where transformers
  * failed.
  */
object ChatTemplateTests extends TestSuite {

  private def resource(relative: String) =
    java.nio.file.Paths
      .get(getClass.getResource(s"/chat-templates/$relative").toURI)

  private val conversations =
    ujson.read(Files.readString(resource("conversations.json")))

  private def check(model: String): Unit = {
    val folder = resource(model)
    val golden = ujson.read(Files.readString(folder.resolve("renders.json")))
    val date = LocalDate.parse(golden("date").str).atTime(12, 0)
    val template = new ChatTemplate(
      Files.readString(folder.resolve("template.jinja")),
      () => date
    )
    val variables = golden("variables").obj.toMap
    val results = golden("renders").arr.map { render =>
      val conversation =
        conversations.arr.find(_("name").str == render("conversation").str).get
      val flags = render("flags").obj
      val extra = flags.view.filterKeys(_ != "add_generation_prompt").toMap
      val ours = Try(
        template.render(
          conversation("messages"),
          conversation.obj.get("tools"),
          flags("add_generation_prompt").bool,
          variables ++ extra
        )
      )
      val label =
        s"${render("conversation").str} ${ujson.write(render("flags"))}"
      render.obj.get("text") match {
        case Some(expected) =>
          val same = ours.toOption.contains(expected.str)
          if (!same) {
            println(s"  $model, $label:")
            ours.fold(
              error => println(s"    failed: ${error.getMessage}"),
              text => {
                val at = text.zip(expected.str).indexWhere(_ != _) match {
                  case -1 => math.min(text.length, expected.str.length);
                  case i  => i
                }
                println(
                  s"    first difference at $at: ours ${ujson.write(text.slice(at - 20, at + 40))}"
                )
                println(
                  s"                         theirs ${ujson.write(expected.str.slice(at - 20, at + 40))}"
                )
              }
            )
          }
          same
        case None =>
          if (ours.isSuccess)
            println(s"  $model, $label: transformers failed, ours rendered")
          ours.isFailure
      }
    }
    println(
      s"  $model: ${results.count(identity)}/${results.size} renders as transformers"
    )
    assert(results.forall(identity))
  }

  val tests = Tests {
    test(
      "a rendered conversation tokenizes its markers as single special tokens"
    ) {
      val folder = resource("qwen3.8")
      val template = new ChatTemplate(
        Files.readString(folder.resolve("template.jinja")),
        () => LocalDate.of(2026, 9, 24).atTime(12, 0)
      )
      val text = template.render(
        ujson.Arr(ujson.Obj("role" -> "user", "content" -> "Hi")),
        None,
        true,
        Map.empty
      )
      val tokenizer = drift.runner.text.TokenizerJson
        .load(TokenizerFiles.json("Qwen/Qwen3.8-Flash-Next"))
      val ids = tokenizer.encode(text, addSpecial = true)
      val start = tokenizer.id("<|im_start|>").get
      val end = tokenizer.id("<|im_end|>").get
      // Qwen 3.8's template adds a system message of its own: count the markers in the text
      def occurrences(marker: String) =
        text.sliding(marker.length).count(_ == marker)
      assert(
        ids.head == start,
        ids.count(_ == start) == occurrences("<|im_start|>"),
        ids.count(_ == end) == occurrences("<|im_end|>")
      )
      assert(Set(start, end).subsetOf(tokenizer.specialIds))
      assert(tokenizer.decode(ids.toSeq, skipSpecial = false) == text)
    }
    test("Qwen3") { check("qwen3") }
    test("Qwen 3.6") { check("qwen3.6") }
    test("Qwen 3.8 Flash Next") { check("qwen3.8") }
    test("Gemma 4") { check("gemma4") }
    test("gpt-oss") { check("gpt-oss") }
  }
}
