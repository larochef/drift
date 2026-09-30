package drift.frontend.pages.generate

import scala.concurrent.ExecutionContext
import scala.scalajs.concurrent.JSExecutionContext
import scala.util.{Failure, Success}

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** Local-file image inputs for the generation form. Files are read into data
  * URLs — exactly what the native sdcpp API accepts in its image fields — and
  * previewed as thumbnails. The `title` on the file input names the field for
  * tests and accessibility.
  */
private[frontend] def readAsDataUrl(
    blob: dom.Blob
)(onLoaded: String => Unit): Unit = {
  val reader = new dom.FileReader()
  reader.onload = _ => onLoaded(reader.result.asInstanceOf[String])
  reader.readAsDataURL(blob)
}

/** Fetches persisted input images — the `/api/outputs/...` URLs a recorded
  * generation carries — back into data URLs, in order, for "reuse these
  * parameters" (`specs/12-gallery.md`). A URL that fails to load is dropped,
  * and the "not saved" placeholder is never fetched.
  */
def loadRecordedImages(
    urls: List[String]
)(onLoaded: List[String] => Unit): Unit = {
  given ExecutionContext = JSExecutionContext.Implicits.queue
  val usable = urls.filter(_.startsWith("/"))
  if (usable.isEmpty) onLoaded(List.empty)
  else {
    val results = Array.fill(usable.size)(Option.empty[String])
    var remaining = usable.size
    def done(): Unit = {
      remaining -= 1
      if (remaining == 0) onLoaded(results.toList.flatten)
    }
    usable.zipWithIndex.foreach { (url, index) =>
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

private def fileInput(labelText: String, onLoaded: String => Unit) =
  input(
    typ := "file",
    accept := "image/*",
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

private def thumbnail(dataUrl: String, onRemove: () => Unit): HtmlElement =
  div(
    cls := "mt-1",
    styleAttr := "display: inline-flex; align-items: flex-start; gap: 4px;",
    img(
      src := dataUrl,
      styleAttr := "max-height: 80px; max-width: 120px; border-radius: 4px;"
    ),
    button(
      cls := "delete is-small",
      title := "Remove image",
      onClick --> (_ => onRemove())
    )
  )

/** One optional image — init/start/end. */
def singleImagePicker(
    labelText: String,
    state: Var[Option[String]]
): HtmlElement =
  div(
    cls := "field",
    label(cls := "label text-primary is-small", labelText),
    div(
      cls := "control",
      child <-- state.signal.map {
        case None =>
          fileInput(labelText, dataUrl => state.set(Some(dataUrl)))
        case Some(dataUrl) =>
          thumbnail(dataUrl, () => state.set(None))
      }
    )
  )

/** Zero or more images — the reference images of edit-style models. */
def multiImagePicker(
    labelText: String,
    state: Var[List[String]]
): HtmlElement =
  div(
    cls := "field",
    label(cls := "label text-primary is-small", labelText),
    div(
      cls := "control",
      children <-- state.signal.map(_.zipWithIndex.map { (dataUrl, index) =>
        thumbnail(dataUrl, () => state.update(_.patch(index, Nil, 1)))
      }),
      fileInput(labelText, dataUrl => state.update(_ :+ dataUrl))
    )
  )
