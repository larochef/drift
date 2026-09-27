package drift.frontend.services

import drift.frontend.services.AssistantService.Attachment
import drift.shared.*

import scala.concurrent.ExecutionContext.Implicits.global
import scala.scalajs.js
import scala.util.{Failure, Success}

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** The pictures and videos staged for the next message
  * (`specs/21-assistant-page.md`): what is waiting to be sent, and the upload
  * that puts a local file there.
  *
  * A file goes up once over HTTP and is referenced by id afterwards — the page
  * never reads tens of megabytes into memory — and what the model is given of
  * it is decided when the message is sent, not here.
  */
final class AssistantAttachments {

  private val _attachments = Var(List.empty[Attachment])

  /** Staged for the next message, in the order they were added. */
  val attachments: Signal[List[Attachment]] = _attachments.signal

  private val _uploadError = Var(Option.empty[String])
  val uploadError: Signal[Option[String]] = _uploadError.signal

  /** What the next message carries, and what it leaves behind. */
  def staged: List[Attachment] = _attachments.now()

  def clear(): Unit = _attachments.set(Nil)

  def attach(attachment: Attachment): Unit =
    _attachments.update(_ :+ attachment)

  def removeAttachment(index: Int): Unit =
    _attachments.update(_.zipWithIndex.filterNot(_._2 == index).map(_._1))

  /** Uploads a local file once over HTTP and stages the reference; the preview
    * is the backend serving it back. Videos ride the same way.
    */
  def uploadFile(file: dom.File): Unit = {
    _uploadError.set(None)
    val name = js.URIUtils.encodeURIComponent(file.name)
    val mimeType = js.URIUtils.encodeURIComponent(file.`type`)
    dom
      .fetch(
        s"/api/assistant/uploads?name=$name&type=$mimeType",
        new dom.RequestInit {
          method = dom.HttpMethod.POST
          body = file
        }
      )
      .toFuture
      .flatMap(response =>
        response.text().toFuture.map(text => (response.status, text))
      )
      .onComplete {
        case Success((200, text)) =>
          val uploaded = readFromString[UploadedAttachment](text)
          val kind =
            if (uploaded.mimeType.startsWith("video/")) "video" else "image"
          attach(
            Attachment(
              UploadAttachment(uploaded.id),
              s"/api/assistant/uploads/${uploaded.id}",
              s"[Attached $kind: ${uploaded.name}]"
            )
          )
        case Success((_, text)) => _uploadError.set(Some(text))
        case Failure(err)       => _uploadError.set(Some(err.getMessage))
      }
  }
}
