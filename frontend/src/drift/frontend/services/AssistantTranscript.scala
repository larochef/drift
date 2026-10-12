package drift.frontend.services

import drift.frontend.services.AssistantService.Turn
import drift.shared.*

import scala.concurrent.ExecutionContext
import scala.scalajs.concurrent.JSExecutionContext
import scala.util.{Failure, Success}

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.raquo.laminar.api.L.*

/** The turns on screen and whose they are: a project's conversation — loaded,
  * saved whole after each change, archived by a compaction or a restart — or
  * free play's scratch chat, set aside while a workspace holds the service.
  */
final private[services] class AssistantTranscript {

  private given ExecutionContext = JSExecutionContext.Implicits.queue

  private val store = ConversationStore()

  private val _turns = Var(List.empty[Turn])
  private val _archive = Var(List.empty[Turn])
  private val _compactedAt = Var(Option.empty[Long])
  private val _boundProject = Var(Option.empty[String])
  private val _loading = Var(false)
  private val _conversationError = Var(Option.empty[String])

  private var nextId = 0

  /** Free play's scratch turns, set aside while a workspace holds the service
    * and given back when it leaves.
    */
  private var scratch = List.empty[Turn]

  /** Bumped whenever the transcript is swapped for another (binding,
    * unbinding): a reply still streaming for the old one is dropped rather than
    * written into the new.
    */
  private var _epoch = 0

  /** Whether the bound conversation loaded. Until it has, saving would
    * overwrite the stored transcript with an empty one.
    */
  private var persistable = false

  val turns: Signal[List[Turn]] = _turns.signal
  val archive: Signal[List[Turn]] = _archive.signal
  val loading: Signal[Boolean] = _loading.signal
  val conversationError: Signal[Option[String]] = _conversationError.signal

  /** A project's conversation — kept, compacted and restarted — rather than
    * free play's scratch chat.
    */
  val bound: Signal[Boolean] = _boundProject.signal.map(_.isDefined)

  def turnsNow: List[Turn] = _turns.now()
  def loadingNow: Boolean = _loading.now()
  def isBound: Boolean = _boundProject.now().isDefined
  def isBoundTo(projectId: String): Boolean =
    _boundProject.now().contains(projectId)

  /** Which transcript this is, for a reply to tell whether it is still the one
    * it was asked in.
    */
  def epoch: Int = _epoch

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

  def contextUsedNow: Option[Int] = usedBy(_turns.now())

  // ------------------------------------------------------------ the turns

  /** Reserves `count` consecutive turn ids and gives the first. */
  def takeIds(count: Int): Int = {
    val first = nextId
    nextId += count
    first
  }

  def setTurns(turns: List[Turn]): Unit = _turns.set(turns)

  def updateTurns(f: List[Turn] => List[Turn]): Unit = _turns.update(f)

  def updateTurn(id: Int)(f: Turn => Turn): Unit =
    _turns.update(_.map(turn => if (turn.id == id) f(turn) else turn))

  def fail(message: String): Unit = _conversationError.set(Some(message))

  // ------------------------------------------------ a project's conversation

  /** The workspace's conversation replaces free play's scratch chat, which is
    * set aside until the workspace leaves.
    */
  def bind(projectId: String): Unit = {
    _epoch += 1
    val expected = _epoch
    if (_boundProject.now().isEmpty) scratch = _turns.now()
    _boundProject.set(Some(projectId))
    _turns.set(Nil)
    _archive.set(Nil)
    _compactedAt.set(None)
    _conversationError.set(None)
    _loading.set(true)
    persistable = false
    store.load(projectId).onComplete { result =>
      if (_epoch == expected) {
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

  /** Free play's scratch chat comes back as the workspace leaves;
    * `forgetProject` runs once the project is let go and before the chat is
    * back.
    */
  def unbind(forgetProject: () => Unit): Unit = {
    _epoch += 1
    _boundProject.set(None)
    forgetProject()
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
  def persist(): Unit =
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

  /** A summary that came back whole takes the place of what it stands for,
    * which moves to the archive.
    */
  def compactInto(summary: Turn, replaced: List[Turn], at: => Long): Unit = {
    _archive.update(_ ++ replaced)
    // Its prompt size was the old transcript's, not what the next
    // request starts from.
    _turns.set(List(summary.copy(promptTokens = None)))
    _compactedAt.set(Some(at))
    persist()
  }

  /** Every settled turn moves to the archive, leaving none on screen. */
  def archiveAll(): Unit = {
    _archive.update(_ ++ _turns.now().filterNot(_.streaming))
    _turns.set(Nil)
    persist()
  }

  /** The scratch chat set aside is thrown away. */
  def dropScratch(): Unit = scratch = Nil
}
