package drift.frontend.pages.gallery

import drift.frontend.generate.readAsDataUrl
import drift.frontend.services.HistoryService
import drift.frontend.services.HistoryService.Command
import drift.shared.*

import org.scalajs.dom

/** Files from the visitor's computer on their way into the gallery, picked in
  * the toolbar or dropped on the page.
  */
final class GalleryImport(historyService: HistoryService) {

  /** Whether the next import to land opens in the detail view: one picked image
    * is there to be worked on; several are left in the grid.
    */
  private var openImported = false

  /** `openImported`, for the page that hears the import land. */
  def opensImported: Boolean = openImported

  /** Images and videos from outside drift, each a gallery entry of its own
    * (`specs/30`). A video is sent as the file it is; anything that does not
    * say it is one goes the images' way, where the server judges the bytes.
    */
  def importFiles(files: dom.FileList): Unit = {
    val picked = (0 until files.length).map(files(_)).toList
    openImported = picked.size == 1
    picked.foreach(file =>
      if (GalleryImport.isVideo(file))
        historyService.push(Command.ImportVideo(file))
      else
        readAsDataUrl(file)(data =>
          historyService.push(Command.Import(ImageImport(file.name, data)))
        )
    )
  }
}

object GalleryImport {
  private val videoExtensions = Set("webm", "mp4", "m4v", "mov", "mkv")

  /** Whether a picked or dropped file is a video: by its type, or by its name
    * where the browser gives none (Matroska, often).
    */
  def isVideo(file: dom.File): Boolean =
    file.`type`.startsWith("video/") ||
      videoExtensions.contains(file.name.toLowerCase.split('.').last)
}
