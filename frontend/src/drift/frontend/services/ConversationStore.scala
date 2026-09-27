package drift.frontend.services

import drift.shared.*

import scala.concurrent.{ExecutionContext, Future}
import scala.scalajs.js

import com.github.plokhotnyuk.jsoniter_scala.core.writeToString
import org.scalajs.dom

/** A project's conversation on the backend (`specs/20`): loaded whole, saved
  * whole, one save after another so an older snapshot never lands after a newer
  * one.
  */
final private[services] class ConversationStore(using ExecutionContext) {

  private var lastSave: Future[Unit] = Future.unit

  /** The stored conversation's status and body. */
  def load(projectId: String): Future[(Int, String)] =
    dom
      .fetch(s"/api/projects/$projectId/conversation")
      .toFuture
      .flatMap(response =>
        response.text().toFuture.map(body => (response.status, body))
      )

  /** Queues a save behind the previous one; `onResult` hears the response's
    * status, or why the request failed.
    */
  def save(projectId: String, conversation: Conversation)(
      onResult: Either[String, Int] => Unit
  ): Unit = {
    val json = writeToString(conversation)
    lastSave = lastSave
      .recover { case _ => () }
      .flatMap(_ =>
        dom
          .fetch(
            s"/api/projects/$projectId/conversation",
            new dom.RequestInit {
              method = dom.HttpMethod.PUT
              headers = js.Dictionary("Content-Type" -> "application/json")
              body = json
            }
          )
          .toFuture
          .map(response => onResult(Right(response.status)))
      )
      .recover { case err => onResult(Left(err.getMessage)) }
  }
}

private[services] object ConversationStore {

  /** A stored message back into a turn. The proposal is stored with it; a
    * transcript from before that is read as fenced blocks, the only format
    * there was.
    */
  def turnOf(message: ConversationMessage): AssistantService.Turn =
    AssistantService.Turn(
      id = message.id,
      role = message.role,
      text = message.text,
      previews = message.previews,
      reasoning = message.reasoning,
      error = message.error,
      promptTokens = message.promptTokens,
      completionTokens = message.completionTokens,
      proposal = message.proposal.orElse(
        if (message.role == "assistant")
          AssistantPrompts.parseProposal(message.text, ProposalFormat.Fenced)
        else None
      ),
      promptBase = message.promptBase,
      createdAt = message.createdAt
    )

  def messageOf(turn: AssistantService.Turn): ConversationMessage =
    ConversationMessage(
      id = turn.id,
      role = turn.role,
      text = turn.text,
      reasoning = turn.reasoning,
      previews = turn.previews,
      promptBase = turn.promptBase,
      proposal = turn.proposal,
      promptTokens = turn.promptTokens,
      completionTokens = turn.completionTokens,
      error = turn.error,
      createdAt = turn.createdAt
    )
}
