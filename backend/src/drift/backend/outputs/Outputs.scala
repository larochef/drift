package drift.backend.outputs

import java.time.ZoneId
import java.time.format.DateTimeFormatter

/** How the outputs root is laid out, and the media it holds. */
object Outputs {

  /** Where free play's outputs live under the outputs root. Not a date, which
    * is exactly why `GenerationHistory` — which only counts date directories —
    * never sees them (`specs/22-free-play-and-scratch-generations.md`).
    */
  val ScratchDirectory: String = "scratch"

  private[backend] val DateFormat =
    DateTimeFormatter.ofPattern("yyyy-MM-dd").withZone(ZoneId.systemDefault())

  /** The media drift writes and serves under the outputs root, extension ↔ MIME
    * type: its own outputs, and the inputs a generation carried — video models
    * take clips and sounds as well as images (`specs/42`, step 14). One table
    * both ways, so an input persisted from a data URL is served back with the
    * type it came with, and "reuse these parameters" rebuilds the same data URL
    * from it. The first extension of a type is the one written.
    */
  private val MediaTypes: List[(String, String)] = List(
    "png" -> "image/png",
    "jpeg" -> "image/jpeg",
    "jpg" -> "image/jpeg",
    "webp" -> "image/webp",
    "gif" -> "image/gif",
    "webm" -> "video/webm",
    "mp4" -> "video/mp4",
    "mov" -> "video/quicktime",
    "mkv" -> "video/x-matroska",
    "avi" -> "video/x-msvideo",
    "wav" -> "audio/wav",
    "mp3" -> "audio/mpeg",
    "ogg" -> "audio/ogg",
    "flac" -> "audio/flac",
    "m4a" -> "audio/mp4"
  )

  def mimeTypeFor(format: String): String =
    MediaTypes
      .collectFirst { case (`format`, mimeType) => mimeType }
      .getOrElse("application/octet-stream")

  /** The extension a file of this MIME type is written with. Browsers name some
    * types two ways (`audio/x-wav`, `audio/wave`, `audio/mp3`), so the aliases
    * are folded in first.
    */
  def extensionFor(mimeType: String): Option[String] = {
    val canonical = mimeType match {
      case "audio/x-wav" | "audio/wave" | "audio/vnd.wave" => "audio/wav"
      case "audio/mp3"                                     => "audio/mpeg"
      case "audio/x-flac"                                  => "audio/flac"
      case "audio/x-m4a"                                   => "audio/mp4"
      case other                                           => other
    }
    MediaTypes.collectFirst { case (extension, `canonical`) => extension }
  }
}
