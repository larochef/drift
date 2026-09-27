package drift.runner

import java.nio.file.{Files, Paths}
import java.time.LocalDateTime

import utest.*

import drift.runner.text.jinja.{Template, TemplateException, Value}

/** The interpreter on focused cases (`fixtures/jinja_cases.py`, rendered by
  * Python's jinja2 with transformers' settings): whitespace control, loops,
  * scoping, macros, Python values and methods, filters and tests — beyond what
  * the models' templates happen to use.
  */
object JinjaTests extends TestSuite {

  private val clock = () => LocalDateTime.of(2026, 9, 24, 12, 0)

  private def render(
      source: String,
      variables: (String, ujson.Value)*
  ): String =
    new Template(source, clock).render(
      variables.map((k, v) => k -> Value.fromJson(v)).toMap
    )

  val tests = Tests {
    test("the same text as jinja2") {
      val cases = ujson.read(
        Files.readString(
          Paths.get(
            getClass.getResource("/chat-templates/jinja-cases.json").toURI
          )
        )
      )
      val results = cases.arr.map { c =>
        val ours = render(
          c("template").str,
          "xs" -> ujson.Arr("a", "b", "c"),
          "x" -> ujson.Num(1),
          "d" -> ujson.Obj("k" -> "v")
        )
        if (ours != c("text").str)
          println(
            s"  ${c("name").str}: ours ${ujson.write(ours)}, jinja2 ${ujson.write(c("text").str)}"
          )
        ours == c("text").str
      }
      assert(results.forall(identity))
    }
    test("errors carry the template's message") {
      val error = intercept[TemplateException](
        render("{{ raise_exception('bad role') }}")
      )
      assert(error.getMessage == "bad role")
      assert(
        intercept[TemplateException](
          render("{{ missing.attribute }}")
        ).getMessage.contains("undefined")
      )
      assert(
        intercept[TemplateException](render("{{ 'a' + [1] }}")).getMessage
          .contains("concatenate")
      )
      assert(
        intercept[TemplateException](
          render("{% if x %}never closed")
        ).getMessage.contains("never closed")
      )
    }
    test("strftime_now reads the given clock") {
      assert(
        render(
          "{{ strftime_now('%Y-%m-%d %B %A') }}"
        ) == "2026-09-24 September Thursday"
      )
    }
  }
}
