package drift.frontend.services

import drift.frontend.services.AssistantService.Attachment
import drift.shared.*

import com.raquo.laminar.api.L.*

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
}

/** The conversation with the live assistant session: a project's — kept,
  * compacted and restarted — while a workspace holds the service (`specs/20`),
  * and free play's scratch chat otherwise (`specs/21`, `specs/22`). Attachments
  * are staged from anywhere in the app as references; one streamed `POST` per
  * reply (`ChatStreaming`), and a project's transcript saved whole
  * (`ConversationStore`). Owns no session — the pages launch and stop those.
  *
  * The service is the one face of the parts beside it: what the assistant is
  * told (`AssistantContext`), the turns and whose they are
  * (`AssistantTranscript`), the replies asked for (`AssistantReplies`), and
  * what the session says of its model (`AssistantSessionProperties`).
  */
class AssistantService(val library: PromptTemplateService) {

  /** The pictures staged for the next message (`AssistantAttachments`). */
  private val staged = AssistantAttachments()

  private val context = AssistantContext(library)
  private val transcript = AssistantTranscript()
  private val sessionProperties = AssistantSessionProperties()
  private val replies =
    AssistantReplies(library, context, transcript, staged, sessionProperties)

  export context.{
    compactionTemplateId,
    configurationTemplateId,
    projectBrief,
    rawModel,
    suggestedTemplateId,
    suggestedTemplateNow,
    targetArchitecture,
    templateId,
    workingCfg,
    workingPrompt,
    workingSize
  }

  export transcript.{
    archive,
    bound,
    contextUsed,
    conversationError,
    loading,
    turns,
    turnsNow
  }

  export sessionProperties.{
    clearProperties,
    loadProperties,
    properties,
    propertiesError
  }

  export replies.{compact, retry, send, stop, streaming}

  val attachments: Signal[List[Attachment]] = staged.attachments
  val uploadError: Signal[Option[String]] = staged.uploadError

  /** At or past the context size: llama-server would truncate and the reply
    * degrade, so sending is refused until the conversation is compacted or
    * restarted (`specs/20`).
    */
  val contextFull: Signal[Boolean] =
    contextUsed.combineWith(properties).map(AssistantSessionProperties.isFull)

  export staged.{attach, removeAttachment, uploadFile, uploadFrom}

  // ------------------------------------------------ a project's conversation

  /** The workspace's conversation replaces free play's scratch chat, which is
    * set aside until the workspace leaves.
    */
  def bindProject(projectId: String): Unit =
    if (!transcript.isBoundTo(projectId)) {
      stop()
      transcript.bind(projectId)
    }

  /** Free play's scratch chat comes back as the workspace leaves. */
  def unbindProject(): Unit =
    if (transcript.isBound) {
      stop()
      transcript.unbind(() => context.forgetProject())
    }

  /** Moves every message to the archive and starts over (`specs/20`). Versions
    * and generations stay: they are the work, the conversation is scratch.
    */
  def restart(): Unit = {
    stop()
    transcript.archiveAll()
  }

  /** The Sandbox's chat thrown away as it is left (`specs/47-sandbox.md`) — set
    * aside already if a workspace has bound its own in the meantime.
    */
  def clearScratch(): Unit =
    if (transcript.isBound) transcript.dropScratch()
    else {
      stop()
      transcript.setTurns(Nil)
    }

  /** Free play's scratch chat is simply emptied; a project's is restarted. */
  def clear(): Unit =
    if (transcript.isBound) restart()
    else {
      stop()
      transcript.setTurns(Nil)
    }
}
