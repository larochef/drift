package drift.frontend.services

import drift.frontend.services.AssistantPrompts.*
import drift.frontend.services.AssistantService.Turn
import drift.shared.*

import scala.concurrent.ExecutionContext
import scala.scalajs.concurrent.JSExecutionContext

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** The replies asked of the live assistant session — a message, a retry, a
  * compaction — each one streamed `POST` (`ChatStreaming`) read into a turn of
  * the transcript, and the one in flight, which can be stopped.
  */
final private[services] class AssistantReplies(
    library: PromptTemplateService,
    context: AssistantContext,
    transcript: AssistantTranscript,
    /** The pictures staged for the next message (`AssistantAttachments`). */
    staged: AssistantAttachments,
    properties: AssistantSessionProperties
) {

  private given ExecutionContext = JSExecutionContext.Implicits.queue

  private val _streaming = Var(false)
  val streaming: Signal[Boolean] = _streaming.signal

  private var inFlight: Option[dom.AbortController] = None

  private def timestamp(): Long = System.currentTimeMillis()

  /** Sends the draft with the staged attachments and streams the reply into a
    * new assistant turn. Refused while the conversation loads, and once the
    * context is full.
    */
  def send(sessionId: String, text: String): Unit =
    if (
      !_streaming.now() && !transcript.loadingNow &&
      !AssistantSessionProperties.isFull(
        transcript.contextUsedNow,
        properties.now
      )
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
          if (properties.now.exists(!_.vision)) Nil
          else attached.map(_.reference)
        val history = transcript.turnsNow
        val base = context.workingPrompt.now().filter(_.prompt.trim.nonEmpty)
        val firstId = transcript.takeIds(2)
        val userTurn = Turn(
          firstId,
          "user",
          userText,
          previews = attached.map(_.previewUrl),
          references = references,
          createdAt = timestamp()
        )
        val assistantTurn = Turn(
          firstId + 1,
          "assistant",
          "",
          streaming = true,
          promptBase = base,
          createdAt = timestamp()
        )
        transcript.updateTurns(_ :+ userTurn :+ assistantTurn)
        staged.clear()
        // The question is kept even if its answer never comes.
        transcript.persist()
        stream(
          sessionId,
          context.systemMessages(base, summaryOf(history)) ++
            (historyOf(history) :+
              ChatMessage("user", userText, references)),
          assistantTurn.id,
          context.proposalFormat
        )(_ => transcript.persist())
      }
    }

  /** Asks the newest question again when its reply failed or was stopped: the
    * reply is replaced, and the question goes out as it did the first time.
    */
  def retry(sessionId: String): Unit =
    if (!_streaming.now())
      transcript.turnsNow.reverse match {
        case failed :: question :: _
            if failed.role == "assistant" && failed.error.isDefined &&
              question.role == "user" =>
          val history = transcript.turnsNow.dropRight(2)
          val replacement = Turn(
            transcript.takeIds(1),
            "assistant",
            "",
            streaming = true,
            promptBase = failed.promptBase,
            createdAt = timestamp()
          )
          transcript.setTurns(history :+ question :+ replacement)
          stream(
            sessionId,
            context.systemMessages(failed.promptBase, summaryOf(history)) ++
              (historyOf(history) :+
                ChatMessage("user", question.text, question.references)),
            replacement.id,
            context.proposalFormat
          )(_ => transcript.persist())
        case _ => ()
      }

  /** Replaces the transcript with a summary the model writes of it
    * (`specs/20`); what the summary stands for moves to the archive. Nothing is
    * replaced unless a summary comes back whole.
    */
  def compact(sessionId: String): Unit =
    if (!_streaming.now() && !transcript.loadingNow) {
      val replaced = transcript.turnsNow
      val compaction =
        library.resolve(
          PromptKind.Compaction,
          context.compactionTemplateId.now()
        )
      if (compaction.isEmpty)
        transcript.fail(
          "No compaction prompt is loaded: check Settings → Prompts, or " +
            "reload the page."
        )
      else if (replaced.exists(_.role == "user")) {
        val summaryTurn =
          Turn(
            transcript.takeIds(1),
            "summary",
            "",
            streaming = true,
            createdAt = timestamp()
          )
        transcript.updateTurns(_ :+ summaryTurn)
        val earlier = summaryOf(replaced)
          .map(text => s"\n\nAn earlier summary, which yours replaces:\n$text")
          .getOrElse("")
        stream(
          sessionId,
          ChatMessage("system", compaction.get.text + earlier) ::
            (historyOf(replaced) :+
              ChatMessage(
                "user",
                "Summarise the conversation so far, as instructed."
              )),
          summaryTurn.id,
          None
        ) { whole =>
          transcript.turnsNow
            .find(_.id == summaryTurn.id)
            .filter(turn => whole && turn.text.trim.nonEmpty) match {
            case Some(summary) =>
              transcript.compactInto(summary, replaced, timestamp())
            case None =>
              transcript.updateTurns(_.filterNot(_.id == summaryTurn.id))
          }
        }
      }
    }

  /** Aborting the request closes the connection, which cancels the generation
    * server-side; the reader then settles the turn.
    */
  def stop(): Unit = inFlight.foreach(_.abort())

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
    val expected = transcript.epoch
    def current: Boolean = transcript.epoch == expected
    val controller = new dom.AbortController()
    inFlight = Some(controller)
    _streaming.set(true)

    def proposalFor(turn: Turn, text: String): Option[PromptProposal] =
      format
        .filter(_ => turn.role == "assistant")
        .flatMap(parseProposal(text, _))

    def apply(event: ChatEvent): Unit =
      if (current)
        transcript.updateTurn(turnId) { turn =>
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
      val whole = current && transcript.turnsNow
        .find(_.id == turnId)
        .exists(turn => !turn.streaming && turn.error.isEmpty)
      if (current)
        transcript.updateTurn(turnId) { turn =>
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
      ChatRequest(AssistantReplies.alternating(messages)),
      controller
    )(apply)(settle)
  }
}

private[services] object AssistantReplies {

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
