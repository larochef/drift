package drift.runner.text

import drift.runner.text.jinja.{Template, Value}

import java.time.LocalDateTime

/** A model's chat template (`tokenizer.chat_template` in a GGUF,
  * `chat_template` in `tokenizer_config.json`), rendered as transformers'
  * `apply_chat_template` renders it: the conversation, its tools, the
  * generation prompt flag, the special tokens as variables, and any extra flags
  * the template reads (`enable_thinking`, …).
  */
final class ChatTemplate(source: String, now: () => LocalDateTime) {

  private val template = new Template(source, now)

  def render(
      messages: ujson.Value,
      tools: Option[ujson.Value],
      addGenerationPrompt: Boolean,
      variables: Map[String, ujson.Value]
  ): String =
    template.render(
      variables.view.mapValues(Value.fromJson).toMap ++ Map(
        "messages" -> Value.fromJson(messages),
        "tools" -> tools.fold[Value](Value.NoneValue)(Value.fromJson),
        "add_generation_prompt" -> Value.Bool(addGenerationPrompt)
      )
    )
}
