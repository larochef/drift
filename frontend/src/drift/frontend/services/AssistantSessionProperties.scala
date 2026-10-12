package drift.frontend.services

import drift.shared.*

import scala.concurrent.ExecutionContext
import scala.scalajs.concurrent.JSExecutionContext
import scala.util.{Failure, Success}

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** What the live assistant session says of its model — its context size,
  * whether it reads pictures — asked of the backend once the session is up.
  */
final private[services] class AssistantSessionProperties {

  private given ExecutionContext = JSExecutionContext.Implicits.queue

  private val _properties = Var(Option.empty[AssistantProperties])
  private val _propertiesError = Var(Option.empty[String])

  val properties: Signal[Option[AssistantProperties]] = _properties.signal
  val propertiesError: Signal[Option[String]] = _propertiesError.signal

  def now: Option[AssistantProperties] = _properties.now()

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
}

private[services] object AssistantSessionProperties {

  /** Whether that many tokens fill the context the properties report. */
  def isFull(
      used: Option[Int],
      applied: Option[AssistantProperties]
  ): Boolean =
    (used, applied) match {
      case (Some(tokens), Some(properties)) if properties.contextSize > 0 =>
        tokens >= properties.contextSize
      case _ => false
    }
}
