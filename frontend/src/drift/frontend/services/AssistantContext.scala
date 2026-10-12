package drift.frontend.services

import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the assistant is told beside the messages: the templates chosen, the
  * project's brief, and the generation form as it stands — set by the pages
  * that show them, and written into the system message that leads a request.
  */
final private[services] class AssistantContext(library: PromptTemplateService) {

  /** The system template chosen where the panel shows (`specs/32`): a
    * project's, or free play's last pick; none is the raw model, sent no system
    * message at all (`specs/21`).
    */
  val templateId: Var[Option[String]] = Var(None)

  /** The project's compaction template; none means the built-in. */
  val compactionTemplateId: Var[Option[String]] = Var(None)

  /** The form's size right now, for a template that wants the aspect ratio. Set
    * by the generation panel.
    */
  val workingSize: Var[Option[(Int, Int)]] = Var(None)

  /** The live image configuration's assistant template override (`specs/32`).
    * Set by the generation panel.
    */
  val configurationTemplateId: Var[Option[String]] = Var(None)

  /** The open project's brief, appended to the system message while a workspace
    * is showing (`specs/19-…`).
    */
  val projectBrief: Var[Option[String]] = Var(None)

  /** A text project's conversation (`specs/41-text-projects.md`): the model
    * gets the messages and nothing of drift's — no template, brief or rules,
    * only a compaction's summary once there is one — and no reply is read as a
    * proposal. Set by the workspace from the project's kind.
    */
  val rawModel: Var[Boolean] = Var(false)

  /** The prompt in the generation form right now (`specs/20`), carried by every
    * request as the text to edit — not a version's, since the user may have
    * changed it since the last run, and that change is what they want help
    * with. Set by the generation panel while one is on screen.
    */
  val workingPrompt: Var[Option[PromptProposal]] = Var(None)

  /** The architecture the generation form targets (`specs/20`), whose prompting
    * notes the assistant is given. Set by the generation panel.
    */
  val targetArchitecture: Var[Option[Architecture]] = Var(None)

  /** The form's CFG: at 1 the negative prompt does nothing, which the assistant
    * should know before proposing one. Set by the generation panel.
    */
  val workingCfg: Var[Option[Double]] = Var(None)

  /** What the live image model suggests: the configuration's override, else its
    * architecture's default, else drift's helper — and nothing while no image
    * model is up, so free play can stay raw. Declared after the vars it reads:
    * a signal built from a not-yet-initialised var is a null at startup.
    */
  val suggestedTemplateId: Signal[Option[String]] =
    configurationTemplateId.signal
      .combineWith(targetArchitecture.signal)
      .map(suggestion)
      .distinct

  def suggestedTemplateNow: Option[String] =
    suggestion(configurationTemplateId.now(), targetArchitecture.now())

  private def suggestion(
      configured: Option[String],
      architecture: Option[Architecture]
  ): Option[String] =
    if (configured.isEmpty && architecture.isEmpty) None
    else
      Some(
        configured
          .orElse(architecture.flatMap(_.assistantTemplateId))
          .getOrElse(PromptTemplate.DefaultAssistantId)
      )

  private def template: Option[PromptTemplate] =
    templateId
      .now()
      .flatMap(library.now)
      .filter(_.kind == PromptKind.AssistantSystem)

  /** How a reply's proposal is read; none, for the raw model, reads none. */
  def proposalFormat: Option[ProposalFormat] =
    if (rawModel.now()) None
    else Some(template.map(_.proposalFormat).getOrElse(ProposalFormat.Fenced))

  /** What leads every request: drift's system message, or for the raw model — a
    * text project, or free play with no template picked — the summary alone,
    * and nothing before a compaction. A template picked but not loaded yet
    * still gets drift's additions.
    */
  def systemMessages(
      base: Option[PromptProposal],
      summary: Option[String]
  ): List[ChatMessage] =
    if (rawModel.now() || templateId.now().isEmpty)
      summary.map(AssistantPrompts.summaryMessage).toList
    else
      List(
        ChatMessage(
          "system",
          AssistantPrompts.systemMessage(
            template,
            projectBrief.now(),
            targetArchitecture.now(),
            workingCfg.now(),
            workingSize.now(),
            base,
            summary
          )
        )
      )

  /** A project's choices, let go as its workspace leaves. */
  def forgetProject(): Unit = {
    templateId.set(None)
    compactionTemplateId.set(None)
    rawModel.set(false)
  }
}
