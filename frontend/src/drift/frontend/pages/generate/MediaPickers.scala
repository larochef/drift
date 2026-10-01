package drift.frontend.pages.generate

import drift.frontend.components.MediaPreview

import scala.concurrent.ExecutionContext
import scala.scalajs.concurrent.JSExecutionContext
import scala.util.{Failure, Success}

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** Local-file inputs for the generation form — images, and for the video models
  * clips and sounds too. Files are read into data URLs — exactly what the
  * native sdcpp API and the drift runner accept in their input fields — and
  * previewed as what they are. The `title` on the file input names the field
  * for tests and accessibility.
  */
private[frontend] def readAsDataUrl(
    blob: dom.Blob
)(onLoaded: String => Unit): Unit = {
  val reader = new dom.FileReader()
  reader.onload = _ => onLoaded(reader.result.asInstanceOf[String])
  reader.readAsDataURL(blob)
}

/** What a file input offers: stills only, or any medium a video model reads. */
object MediaAccept {
  val Images = "image/*"
  val ImagesAndVideos = "image/*,video/*"
  val Videos = "video/*"
  val AnyMedia = "image/*,video/*,audio/*"
}

/** Fetches persisted inputs — the `/api/outputs/...` URLs a recorded generation
  * carries — back into data URLs for "reuse these parameters"
  * (`specs/12-gallery.md`), one answer per URL in order: None where the URL
  * failed to load or is the "not saved" placeholder, which is never fetched.
  * Aligned, so a guide's medium stays beside its frame index.
  */
def loadRecordedMediaAligned(
    urls: List[String]
)(onLoaded: List[Option[String]] => Unit): Unit = {
  given ExecutionContext = JSExecutionContext.Implicits.queue
  if (urls.isEmpty) onLoaded(List.empty)
  else {
    val results = Array.fill(urls.size)(Option.empty[String])
    var remaining = urls.size
    def done(): Unit = {
      remaining -= 1
      if (remaining == 0) onLoaded(results.toList)
    }
    urls.zipWithIndex.foreach { (url, index) =>
      if (!url.startsWith("/")) done()
      else
        dom
          .fetch(url)
          .toFuture
          .flatMap(response => response.blob().toFuture)
          .onComplete {
            case Success(blob) =>
              readAsDataUrl(blob) { dataUrl =>
                results(index) = Some(dataUrl)
                done()
              }
            case Failure(_) => done()
          }
    }
  }
}

/** The loaded ones of `loadRecordedMediaAligned`, in order. */
def loadRecordedMedia(
    urls: List[String]
)(onLoaded: List[String] => Unit): Unit =
  loadRecordedMediaAligned(urls)(loaded => onLoaded(loaded.flatten))

private[generate] def fileInput(
    labelText: String,
    accepted: String,
    onLoaded: String => Unit
) =
  input(
    typ := "file",
    accept := accepted,
    cls := "is-size-7",
    title := labelText,
    onChange --> { event =>
      val element = event.target.asInstanceOf[dom.html.Input]
      val files = element.files
      if (files.length > 0) readAsDataUrl(files(0))(onLoaded)
      // Cleared so choosing the same file again re-fires the change event.
      element.value = ""
    }
  )

/** A picked input with its remove button, and whatever else the row needs
  * (reordering, a frame index).
  */
private[generate] def thumbnail(
    dataUrl: String,
    onRemove: () => Unit,
    extras: Modifier[HtmlElement]*
): HtmlElement =
  div(
    cls := "mt-1 mr-2",
    styleAttr := "display: inline-flex; align-items: flex-start; gap: 4px;",
    MediaPreview(
      dataUrl,
      "max-height: 80px; max-width: 120px; border-radius: 4px;"
    ).element,
    extras,
    button(
      cls := "delete is-small",
      title := "Remove",
      onClick --> (_ => onRemove())
    )
  )

/** One optional input — init/start/end image, a control video, its mask. */
def singleMediaPicker(
    labelText: String,
    state: Var[Option[String]],
    accepted: String
): HtmlElement =
  div(
    cls := "field",
    label(cls := "label text-primary is-small", labelText),
    div(
      cls := "control",
      child <-- state.signal.map {
        case None =>
          fileInput(labelText, accepted, dataUrl => state.set(Some(dataUrl)))
        case Some(dataUrl) =>
          thumbnail(dataUrl, () => state.set(None))
      }
    )
  )

/** Zero or more inputs — the reference images of edit-style models, or a video
  * model's references, whose order the model reads (MiniMax H3 names them by
  * position in the prompt): `ordered` numbers them and adds the buttons that
  * move one.
  */
def multiMediaPicker(
    labelText: String,
    state: Var[List[String]],
    accepted: String,
    ordered: Boolean
): HtmlElement = {
  def move(index: Int, by: Int): Unit =
    state.update { list =>
      val target = index + by
      if (target < 0 || target >= list.size) list
      else list.updated(index, list(target)).updated(target, list(index))
    }
  def orderControls(index: Int): Modifier[HtmlElement] =
    if (!ordered) emptyMod
    else
      List(
        span(cls := "is-size-7 text-secondary", s"${index + 1}"),
        button(
          cls := "button is-small is-ghost px-1",
          title := "Move earlier",
          "←",
          onClick --> (_ => move(index, -1))
        ),
        button(
          cls := "button is-small is-ghost px-1",
          title := "Move later",
          "→",
          onClick --> (_ => move(index, 1))
        )
      )
  div(
    cls := "field",
    label(cls := "label text-primary is-small", labelText),
    div(
      cls := "control",
      children <-- state.signal.map(_.zipWithIndex.map { (dataUrl, index) =>
        thumbnail(
          dataUrl,
          () => state.update(_.patch(index, Nil, 1)),
          orderControls(index)
        )
      }),
      fileInput(labelText, accepted, dataUrl => state.update(_ :+ dataUrl))
    )
  )
}
