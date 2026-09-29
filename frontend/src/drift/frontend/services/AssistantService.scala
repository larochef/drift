package drift.frontend.services

import drift.shared.*

import scala.concurrent.ExecutionContext
import scala.scalajs.concurrent.JSExecutionContext
import scala.scalajs.js
import scala.util.{Failure, Success}

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.raquo.laminar.api.L.*
import org.scalajs.dom

object AssistantService {

  /** One file staged for the next message: what the backend will read for the
    * model, what the page shows meanwhile, and the text that goes with it (a
    * generated image carries its recorded parameters). No bytes live here —
    * outputs are on disk already and uploads went over HTTP once.
    */
  case class Attachment(
      reference: ChatAttachment,
      previewUrl: String,
      description: String
  )

  /** One turn of the conversation (`specs/20`, `specs/21`). `role` is `user`,
    * `assistant` or `summary` — a summary stands for what a compaction moved to
    * the archive. An assistant or summary turn streams: `text` and `reasoning`
    * grow until `streaming` drops, then the token counts and any proposal are
    * known.
    */
  case class Turn(
      id: Int,
      role: String,
      text: String,
      previews: List[String] = Nil,
      /** What a user turn attached, for a retry within the page's life. Not
        * stored: past turns travel as text, so a reload has nothing to resend.
        */
      references: List[ChatAttachment] = Nil,
      reasoning: String = "",
      streaming: Boolean = false,
      error: Option[String] = None,
      promptTokens: Option[Int] = None,
      completionTokens: Option[Int] = None,
      /** Prefill and generation timings, from the server's final chunk. */
      promptMillis: Option[Double] = None,
      completionMillis: Option[Double] = None,
      promptTokensPerSecond: Option[Double] = None,
      completionTokensPerSecond: Option[Double] = None,
      proposal: Option[PromptProposal] = None,
      /** The working prompt when this reply was asked for — what its proposal
        * is diffed against, so the card still shows what the assistant changed
        * once the proposal has been applied (`specs/20`).
        */
      promptBase: Option[PromptProposal] = None,
      createdAt: Long = 0
  )

  /** A generated output as an attachment: referenced by date and file name,
    * previewed from the URL the gallery already loads.
    */
  def outputAttachment(
      generation: Generation,
      output: GenerationOutput,
      configurationLabel: String
  ): Attachment =
    Attachment(
      OutputAttachment(output.date, output.fileName),
      output.url,
      AssistantPrompts.describe(
        generation.ofOutput(output.index),
        configurationLabel
      )
    )

  /** Consecutive messages of one role merged into one. Chat templates that
    * insist on strict alternation (Gemma's, Mistral's) refuse two user turns in
    * a row, which a failed reply or a compaction request would otherwise send.
    */
  private def alternating(messages: List[ChatMessage]): List[ChatMessage] =
    messages
      .foldLeft(List.empty[ChatMessage]) { (merged, message) =>
        merged match {
          case previous :: rest if previous.role == message.role =>
            previous.copy(
              text = s"${previous.text}\n\n${message.text}",
              attachments = previous.attachments ++ message.attachments
            ) :: rest
          case _ => message :: merged
        }
      }
      .reverse
}

/** The conversation with the live assistant session: a project's — kept,
  * compacted and restarted — while a workspace holds the service (`specs/20`),
  * and free play's scratch chat otherwise (`specs/21`, `specs/22`). Attachments
  * are staged from anywhere in the app as references; one streamed `POST` per
  * reply (`ChatStreaming`), and a project's transcript saved whole
  * (`ConversationStore`). Owns no session — the pages launch and stop those.
  */
class AssistantService(val library: PromptTemplateService) {
  import AssistantService.*
  import AssistantPrompts.{historyOf, parseProposal, summaryOf}

  private given ExecutionContext = JSExecutionContext.Implicits.queue

  private val store = ConversationStore()

  /** The pictures staged for the next message (`AssistantAttachments`). */
  private val staged = AssistantAttachments()

  private val _turns = Var(List.empty[Turn])
  private val _archive = Var(List.empty[Turn])
  private val _compactedAt = Var(Option.empty[Long])
  private val _boundProject = Var(Option.empty[String])
  private val _loading = Var(false)
  private val _conversationError = Var(Option.empty[String])

  private val _properties = Var(Option.empty[AssistantProperties])
  private val _propertiesError = Var(Option.empty[String])
  private val _streaming = Var(false)

  private var inFlight: Option[dom.AbortController] = None
  private var nextId = 0

  /** Free play's scratch turns, set aside while a workspace holds the service
    * and given back when it leaves.
    */
  private var scratch = List.empty[Turn]

  /** Bumped whenever the transcript is swapped for another (binding,
    * unbinding): a reply still streaming for the old one is dropped rather than
    * written into the new.
    */
  private var epoch = 0

  /** Whether the bound conversation loaded. Until it has, saving would
    * overwrite the stored transcript with an empty one.
    */
  private var persistable = false

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

  private def proposalFormat: Option[ProposalFormat] =
    if (rawModel.now()) None
    else Some(template.map(_.proposalFormat).getOrElse(ProposalFormat.Fenced))

  val turns: Signal[List[Turn]] = _turns.signal
  val archive: Signal[List[Turn]] = _archive.signal
  val attachments: Signal[List[Attachment]] = staged.attachments
  val properties: Signal[Option[AssistantProperties]] = _properties.signal
  val propertiesError: Signal[Option[String]] = _propertiesError.signal
  val streaming: Signal[Boolean] = _streaming.signal
  val uploadError: Signal[Option[String]] = staged.uploadError
  val loading: Signal[Boolean] = _loading.signal
  val conversationError: Signal[Option[String]] = _conversationError.signal

  /** A project's conversation — kept, compacted and restarted — rather than
    * free play's scratch chat.
    */
  val bound: Signal[Boolean] = _boundProject.signal.map(_.isDefined)

  def turnsNow: List[Turn] = _turns.now()

  private def usedBy(turns: List[Turn]): Option[Int] =
    turns.reverse
      .find(turn => turn.role == "assistant" && turn.promptTokens.isDefined)
      .map(turn =>
        turn.promptTokens.getOrElse(0) + turn.completionTokens.getOrElse(0)
      )

  /** Tokens the next request starts from: the newest reply's prompt plus what
    * the reply added to it (`specs/20`).
    */
  val contextUsed: Signal[Option[Int]] = _turns.signal.map(usedBy)

  private def isFull(
      used: Option[Int],
      applied: Option[AssistantProperties]
  ): Boolean =
    (used, applied) match {
      case (Some(tokens), Some(properties)) if properties.contextSize > 0 =>
        tokens >= properties.contextSize
      case _ => false
    }

  /** At or past the context size: llama-server would truncate and the reply
    * degrade, so sending is refused until the conversation is compacted or
    * restarted (`specs/20`).
    */
  val contextFull: Signal[Boolean] =
    contextUsed.combineWith(properties).map(isFull)

  export staged.{attach, removeAttachment, uploadFile}

  // ------------------------------------------------ a project's conversation

  /** The workspace's conversation replaces free play's scratch chat, which is
    * set aside until the workspace leaves.
    */
  def bindProject(projectId: String): Unit =
    if (!_boundProject.now().contains(projectId)) {
      stop()
      epoch += 1
      val expected = epoch
      if (_boundProject.now().isEmpty) scratch = _turns.now()
      _boundProject.set(Some(projectId))
      _turns.set(Nil)
      _archive.set(Nil)
      _compactedAt.set(None)
      _conversationError.set(None)
      _loading.set(true)
      persistable = false
      store.load(projectId).onComplete { result =>
        if (epoch == expected) {
          result match {
            case Success((200, body)) =>
              val conversation = readFromString[Conversation](body)
              _turns.set(conversation.messages.map(ConversationStore.turnOf))
              _archive.set(conversation.archive.map(ConversationStore.turnOf))
              _compactedAt.set(conversation.compactedAt)
              nextId = (conversation.messages ++ conversation.archive)
                .map(_.id + 1)
                .maxOption
                .getOrElse(0)
                .max(nextId)
              persistable = true
            case Success((status, body)) =>
              _conversationError.set(
                Some(
                  s"The conversation did not load ($status): ${body.take(200)}" +
                    " — nothing is saved until the page is reloaded."
                )
              )
            case Failure(err) =>
              _conversationError.set(
                Some(
                  s"The conversation did not load: ${err.getMessage}" +
                    " — nothing is saved until the page is reloaded."
                )
              )
          }
          _loading.set(false)
        }
      }
    }

  /** Free play's scratch chat comes back as the workspace leaves. */
  def unbindProject(): Unit =
    if (_boundProject.now().isDefined) {
      stop()
      epoch += 1
      _boundProject.set(None)
      templateId.set(None)
      compactionTemplateId.set(None)
      rawModel.set(false)
      _turns.set(scratch)
      _archive.set(Nil)
      _compactedAt.set(None)
      _conversationError.set(None)
      _loading.set(false)
      persistable = false
      scratch = Nil
    }

  /** Saves the bound conversation whole after every change that should outlive
    * the page — a sent message, a finished reply, a compaction, a restart. A
    * reply still streaming is left out until it ends.
    */
  private def persist(): Unit =
    _boundProject.now().filter(_ => persistable).foreach { projectId =>
      store.save(
        projectId,
        Conversation(
          projectId,
          messages = _turns
            .now()
            .filterNot(_.streaming)
            .map(ConversationStore.messageOf),
          archive = _archive.now().map(ConversationStore.messageOf),
          compactedAt = _compactedAt.now()
        )
      ) {
        case Right(200)    => _conversationError.set(None)
        case Right(status) =>
          _conversationError.set(
            Some(s"The conversation was not saved ($status).")
          )
        case Left(message) =>
          _conversationError.set(
            Some(s"The conversation was not saved: $message")
          )
      }
    }

  // ---------------------------------------------------------------- requests

  /** What leads every request: drift's system message, or for the raw model — a
    * text project, or free play with no template picked — the summary alone,
    * and nothing before a compaction. A template picked but not loaded yet
    * still gets drift's additions.
    */
  private def systemMessages(
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

  private def timestamp(): Long = System.currentTimeMillis()

  /** Sends the draft with the staged attachments and streams the reply into a
    * new assistant turn. Refused while the conversation loads, and once the
    * context is full.
    */
  def send(sessionId: String, text: String): Unit =
    if (
      !_streaming.now() && !_loading.now() &&
      !isFull(usedBy(_turns.now()), _properties.now())
    ) {
      val attached = staged.staged
      val userText =
        (attached.map(_.description) :+ text.trim)
          .filter(_.nonEmpty)
          .mkString("\n\n")
      if (userText.nonEmpty) {
        // A text-only model gets what is known about each image — a
        // generation's parameters and prompt — never the picture, which it
        // cannot read (`specs/20`).
        val references =
          if (_properties.now().exists(!_.vision)) Nil
          else attached.map(_.reference)
        val history = _turns.now()
        val base = workingPrompt.now().filter(_.prompt.trim.nonEmpty)
        val userTurn = Turn(
          nextId,
          "user",
          userText,
          previews = attached.map(_.previewUrl),
          references = references,
          createdAt = timestamp()
        )
        val assistantTurn = Turn(
          nextId + 1,
          "assistant",
          "",
          streaming = true,
          promptBase = base,
          createdAt = timestamp()
        )
        nextId += 2
        _turns.update(_ :+ userTurn :+ assistantTurn)
        staged.clear()
        // The question is kept even if its answer never comes.
        persist()
        stream(
          sessionId,
          systemMessages(base, summaryOf(history)) ++
            (historyOf(history) :+
              ChatMessage("user", userText, references)),
          assistantTurn.id,
          proposalFormat
        )(_ => persist())
      }
    }

  /** Asks the newest question again when its reply failed or was stopped: the
    * reply is replaced, and the question goes out as it did the first time.
    */
  def retry(sessionId: String): Unit =
    if (!_streaming.now())
      _turns.now().reverse match {
        case failed :: question :: _
            if failed.role == "assistant" && failed.error.isDefined &&
              question.role == "user" =>
          val history = _turns.now().dropRight(2)
          val replacement = Turn(
            nextId,
            "assistant",
            "",
            streaming = true,
            promptBase = failed.promptBase,
            createdAt = timestamp()
          )
          nextId += 1
          _turns.set(history :+ question :+ replacement)
          stream(
            sessionId,
            systemMessages(failed.promptBase, summaryOf(history)) ++
              (historyOf(history) :+
                ChatMessage("user", question.text, question.references)),
            replacement.id,
            proposalFormat
          )(_ => persist())
        case _ => ()
      }

  /** Replaces the transcript with a summary the model writes of it
    * (`specs/20`); what the summary stands for moves to the archive. Nothing is
    * replaced unless a summary comes back whole.
    */
  def compact(sessionId: String): Unit =
    if (!_streaming.now() && !_loading.now()) {
      val transcript = _turns.now()
      val compaction =
        library.resolve(PromptKind.Compaction, compactionTemplateId.now())
      if (compaction.isEmpty)
        _conversationError.set(
          Some(
            "No compaction prompt is loaded: check Settings → Prompts, or " +
              "reload the page."
          )
        )
      else if (transcript.exists(_.role == "user")) {
        val summaryTurn =
          Turn(nextId, "summary", "", streaming = true, createdAt = timestamp())
        nextId += 1
        _turns.update(_ :+ summaryTurn)
        val earlier = summaryOf(transcript)
          .map(text => s"\n\nAn earlier summary, which yours replaces:\n$text")
          .getOrElse("")
        stream(
          sessionId,
          ChatMessage("system", compaction.get.text + earlier) ::
            (historyOf(transcript) :+
              ChatMessage(
                "user",
                "Summarise the conversation so far, as instructed."
              )),
          summaryTurn.id,
          None
        ) { whole =>
          _turns
            .now()
            .find(_.id == summaryTurn.id)
            .filter(turn => whole && turn.text.trim.nonEmpty) match {
            case Some(summary) =>
              _archive.update(_ ++ transcript)
              // Its prompt size was the old transcript's, not what the next
              // request starts from.
              _turns.set(List(summary.copy(promptTokens = None)))
              _compactedAt.set(Some(timestamp()))
              persist()
            case None =>
              _turns.update(_.filterNot(_.id == summaryTurn.id))
          }
        }
      }
    }

  /** Moves every message to the archive and starts over (`specs/20`). Versions
    * and generations stay: they are the work, the conversation is scratch.
    */
  def restart(): Unit = {
    stop()
    _archive.update(_ ++ _turns.now().filterNot(_.streaming))
    _turns.set(Nil)
    persist()
  }

  /** The Sandbox's chat thrown away as it is left (`specs/47-sandbox.md`) — set
    * aside already if a workspace has bound its own in the meantime.
    */
  def clearScratch(): Unit =
    if (_boundProject.now().isDefined) scratch = Nil
    else {
      stop()
      _turns.set(Nil)
    }

  /** Free play's scratch chat is simply emptied; a project's is restarted. */
  def clear(): Unit =
    if (_boundProject.now().isDefined) restart()
    else {
      stop()
      _turns.set(Nil)
    }

  /** Aborting the request closes the connection, which cancels the generation
    * server-side; the reader then settles the turn.
    */
  def stop(): Unit = inFlight.foreach(_.abort())

  def loadProperties(sessionId: String): Unit =
    dom
      .fetch(s"/api/assistant/$sessionId/properties")
      .toFuture
      .flatMap(response =>
        response.text().toFuture.map(body => (response.status, body))
      )
      .onComplete {
        case Success((200, body)) =>
          _properties.set(Some(readFromString[AssistantProperties](body)))
          _propertiesError.set(None)
        case Success((_, body)) =>
          _properties.set(None)
          _propertiesError.set(Some(body))
        case Failure(err) =>
          _properties.set(None)
          _propertiesError.set(Some(err.getMessage))
      }

  def clearProperties(): Unit = {
    _properties.set(None)
    _propertiesError.set(None)
  }

  private def updateTurn(id: Int)(f: Turn => Turn): Unit =
    _turns.update(_.map(turn => if (turn.id == id) f(turn) else turn))

  /** One streamed reply into the turn `turnId` — the one reading of the
    * response every request shares: a message, a retry, a compaction.
    * `onFinished` learns whether the reply came back whole. A reply for a
    * transcript that has since been swapped out is dropped.
    */
  private def stream(
      sessionId: String,
      messages: List[ChatMessage],
      turnId: Int,
      /** How a finished reply's proposal is read; none reads no proposal. */
      format: Option[ProposalFormat]
  )(onFinished: Boolean => Unit): Unit = {
    val expected = epoch
    def current: Boolean = epoch == expected
    val controller = new dom.AbortController()
    inFlight = Some(controller)
    _streaming.set(true)

    def proposalFor(turn: Turn, text: String): Option[PromptProposal] =
      format
        .filter(_ => turn.role == "assistant")
        .flatMap(parseProposal(text, _))

    def apply(event: ChatEvent): Unit =
      if (current)
        updateTurn(turnId) { turn =>
          val grown = turn.copy(
            text = turn.text + event.content.getOrElse(""),
            reasoning = turn.reasoning + event.reasoning.getOrElse("")
          )
          if (event.error.isDefined)
            grown.copy(streaming = false, error = event.error)
          else if (event.done)
            grown.copy(
              streaming = false,
              promptTokens = event.promptTokens,
              completionTokens = event.completionTokens,
              promptMillis = event.promptMillis,
              completionMillis = event.completionMillis,
              promptTokensPerSecond = event.promptTokensPerSecond,
              completionTokensPerSecond = event.completionTokensPerSecond,
              proposal = proposalFor(turn, grown.text)
            )
          else grown
        }

    // A turn still streaming when the body ends was stopped or cut off.
    def settle(reason: String): Unit = {
      val whole = current && _turns
        .now()
        .find(_.id == turnId)
        .exists(turn => !turn.streaming && turn.error.isEmpty)
      if (current)
        updateTurn(turnId) { turn =>
          if (turn.streaming)
            turn.copy(
              streaming = false,
              error = Some(reason),
              proposal = proposalFor(turn, turn.text)
            )
          else turn
        }
      // Only this request's own flags: a newer one may already be running.
      if (inFlight.contains(controller)) {
        _streaming.set(false)
        inFlight = None
      }
      if (current) onFinished(whole)
    }

    ChatStreaming.start(
      sessionId,
      ChatRequest(alternating(messages)),
      controller
    )(apply)(settle)
  }
}
