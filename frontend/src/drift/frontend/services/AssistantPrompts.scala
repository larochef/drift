package drift.frontend.services

import drift.shared.*

import scala.scalajs.js
import scala.util.Try

/** What the assistant is told (`specs/20`, `specs/21`, `specs/32`): the chosen
  * system template and what drift adds after it, how an attached generation is
  * described, and how a proposal is read back out of a reply.
  */
object AssistantPrompts {

  /** Appended after the system template when its `appends.rules` is on
    * (`specs/20`): the user's prompt is the intent and an image only how one
    * model read it, so a proposal edits the prompt — or rewrites it, when asked
    * — and never trades it for a description of the picture.
    */
  val PromptRules: String =
    """How to use the user's prompt:
      |- The user's prompt says what they want. An image only shows how one model rendered it, and models drop and distort details.
      |- By default, edit: start from the user's prompt, change only what the request needs, and keep everything else exactly as written.
      |- When the user asks for a rewrite (more precise, less repetitive, or described another way), rewrite the whole prompt freely, but keep every detail of what they want: change the wording, not the content.
      |- If an image is missing something the prompt asks for, strengthen or rephrase that part; never drop it.
      |- Never replace the prompt with a description of the image.""".stripMargin

  /** For a target whose architecture carries no prompting notes (`specs/20`).
    */
  val GenericPromptingNote: String =
    "There are no notes on how this model reads prompts: write clear, " +
      "specific descriptions of what should be in the image."

  // Forgiving about the fence the model actually writes (François,
  // 2026-09-10: a proposal that went unrecognised had to be copied by hand):
  // three or more backticks or tildes, the tag in any case, `positive` as a
  // synonym, and whatever else it puts on that line ("```prompt (positive)").
  // The block may also run to the end of the text - a reply cut off before its
  // closing fence still proposed something, and `parseProposal` only ever runs
  // on a finished turn, so there is no half-streamed block to catch.
  private val PromptBlock =
    """(?is)(?:`{3,}|~{3,})[ \t]*(?:prompt|positive)\b[^\n]*\n(.*?)(?:`{3,}|~{3,}|\z)""".r
  private val NegativeBlock =
    """(?is)(?:`{3,}|~{3,})[ \t]*negative\b[^\n]*\n(.*?)(?:`{3,}|~{3,}|\z)""".r

  /** A proposal read the way the template says (`specs/32`). A reply without
    * one is discussion, not a proposal.
    */
  def parseProposal(
      text: String,
      format: ProposalFormat
  ): Option[PromptProposal] =
    format match {
      case ProposalFormat.Fenced => parseFenced(text)
      case ProposalFormat.Json   => parseJson(text)
    }

  /** The last fenced `prompt` block, with the `negative` block beside it. */
  private def parseFenced(text: String): Option[PromptProposal] =
    PromptBlock.findAllMatchIn(text).toList.lastOption.map { m =>
      PromptProposal(
        m.group(1).trim,
        NegativeBlock
          .findAllMatchIn(text)
          .toList
          .lastOption
          .map(_.group(1).trim)
          .getOrElse("")
      )
    }

  /** The last JSON object in the reply — fenced or bare — minified into the
    * prompt, with a top-level `aspect_ratio` taken out and carried beside it.
    * Key order is kept as written: the model it is for was trained on one
    * order. Nothing that parses means no proposal, not a broken one.
    */
  private def parseJson(text: String): Option[PromptProposal] = {
    val end = text.lastIndexOf('}')
    if (end < 0) None
    else {
      val starts =
        text.zipWithIndex.collect { case ('{', i) if i < end => i }
      starts.iterator
        .map(start =>
          Try(js.JSON.parse(text.substring(start, end + 1))).toOption
            .filter(v => js.typeOf(v) == "object" && v != null)
        )
        .collectFirst { case Some(parsed) => parsed }
        .map { parsed =>
          val obj = parsed.asInstanceOf[js.Dictionary[js.Any]]
          val ratio = obj
            .get("aspect_ratio")
            .map(_.toString.trim)
            .filter(_.matches("""\d+:\d+"""))
          obj.remove("aspect_ratio")
          PromptProposal(js.JSON.stringify(obj), "", ratio)
        }
    }
  }

  /** What an attached generation tells the model in words: the configuration
    * and the parameters that made the image, so the picture is never sent
    * without its prompt.
    */
  def describe(generation: Generation, configurationLabel: String): String =
    generation.imageParameters match {
      case Some(p) =>
        val sampler =
          p.sampleParams.sampleMethod.map(s => s", sampler $s").getOrElse("")
        val negative =
          if (p.negativePrompt.trim.isEmpty) "(none)" else p.negativePrompt
        // Intent and result kept apart (`specs/20`): the prompt is what was
        // asked for and the image only what one model made of it — read as a
        // caption, the prompt loses to the picture.
        s"What I asked for:\nPrompt: ${p.prompt}\nNegative prompt: $negative\n" +
          s"What came out: the attached image — configuration '$configurationLabel', " +
          s"${p.width}x${p.height}, ${p.sampleParams.sampleSteps} steps$sampler, " +
          s"seed ${p.seed}. It may have missed or changed details of the prompt."
      case None =>
        s"[Generated with configuration '$configurationLabel']"
    }

  /** `W:H` in lowest terms, for the template that wants one. */
  def aspectRatioOf(width: Int, height: Int): Option[String] =
    if (width <= 0 || height <= 0) None
    else {
      @annotation.tailrec
      def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
      val g = gcd(width, height)
      Some(s"${width / g}:${height / g}")
    }

  /** The template's text first, then what drift adds after it — the brief, the
    * target's prompting notes, the prompt being worked on, the rules for using
    * it, the CFG note, the form's aspect ratio — each only when the template's
    * `appends` keep it (`specs/32`). A template not loaded yet gets every
    * addition; no template at all is sent no system message
    * (`AssistantService`). A compaction's summary closes it, standing in for
    * the messages it replaced.
    */
  def systemMessage(
      template: Option[PromptTemplate],
      projectBrief: Option[String],
      targetArchitecture: Option[Architecture],
      workingCfg: Option[Double],
      workingSize: Option[(Int, Int)],
      base: Option[PromptProposal],
      summary: Option[String]
  ): String = {
    val text = template.map(_.text).getOrElse("")
    val appends = template.map(_.appends).getOrElse(AssistantAppends())
    val brief = projectBrief
      .filter(_ => appends.brief)
      .filter(_.trim.nonEmpty)
      .map(text => s"\n\nThe project being worked on: $text")
      .getOrElse("")
    val notes = targetArchitecture
      .filter(_ => appends.promptingNotes)
      .map(architecture =>
        s"\n\nThe prompt is for ${architecture.label}. " +
          architecture.promptingNotes.getOrElse(GenericPromptingNote)
      )
      .getOrElse("")
    val working = base
      .filter(_ => appends.workingPrompt)
      .map(prompt =>
        "\n\nThe prompt the user is working on — their intent, the text to " +
          s"edit:\nPrompt: ${prompt.prompt}\nNegative prompt: " +
          (if (prompt.negativePrompt.trim.isEmpty) "(none)"
           else prompt.negativePrompt)
      )
      .getOrElse("")
    val cfg = workingCfg
      .filter(_ => appends.cfgNote)
      .filter(_ <= 1.0)
      .map(_ =>
        "\n\nThe form runs at CFG 1, where the negative prompt has no effect: " +
          "do not spend effort on it."
      )
      .getOrElse("")
    val rules = if (appends.rules) "\n\n" + PromptRules else ""
    val ratio = workingSize
      .filter(_ => appends.aspectRatio)
      .flatMap((w, h) => aspectRatioOf(w, h).map(r => (r, w, h)))
      .map((r, w, h) =>
        s"\n\nTARGET IMAGE ASPECT RATIO: $r (width:height), for a ${w}x$h image."
      )
      .getOrElse("")
    val earlier = summary.map(text => s"\n\n${summaryText(text)}").getOrElse("")
    text + brief + notes + working + cfg + rules + ratio + earlier
  }

  /** Past turns as the model sees them: text only — llama-server re-encodes
    * every image on every request, so earlier ones travel as their description
    * alone — and without the summary, which rides in the system message.
    */
  def historyOf(turns: List[AssistantService.Turn]): List[ChatMessage] =
    turns.collect {
      case turn if turn.role == "user" => ChatMessage("user", turn.text)
      case turn if turn.role == "assistant" && turn.text.nonEmpty =>
        ChatMessage("assistant", turn.text)
    }

  private def summaryText(summary: String): String =
    "A summary of the conversation so far, standing in for everything " +
      s"before the messages below:\n$summary"

  /** A text project's only system message, once compacted
    * (`specs/41-text-projects.md`).
    */
  def summaryMessage(summary: String): ChatMessage =
    ChatMessage("system", summaryText(summary))

  def summaryOf(turns: List[AssistantService.Turn]): Option[String] =
    turns.findLast(_.role == "summary").map(_.text).filter(_.trim.nonEmpty)
}
