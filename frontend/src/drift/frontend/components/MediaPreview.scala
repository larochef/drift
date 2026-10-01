package drift.frontend.components

import com.raquo.laminar.api.L.*

/** An input shown as what it is — a still, a player for a clip, a player for a
  * sound — from either a picked file's data URL (read off its MIME type) or a
  * recorded `/api/outputs/...` URL (read off its extension). The video models'
  * references, guides and control inputs mix all three (`specs/42`).
  */
class MediaPreview(source: String, style: String, description: String = "") {

  lazy val element: HtmlElement = MediaPreview.kindOf(source) match {
    case MediaPreview.Kind.Video =>
      videoTag(
        src := source,
        VideoAttrs.controls,
        VideoAttrs.preloadMetadata,
        VideoAttrs.muted,
        title := description,
        styleAttr := style
      )
    case MediaPreview.Kind.Audio =>
      audioTag(
        src := source,
        VideoAttrs.controls,
        VideoAttrs.preloadMetadata,
        title := description,
        styleAttr := "max-width: 240px;"
      )
    case MediaPreview.Kind.Image =>
      img(
        src := source,
        alt := description,
        title := description,
        styleAttr := style
      )
  }
}

object MediaPreview {
  enum Kind derives CanEqual {
    case Image, Video, Audio
  }

  private val videoExtensions = Set("webm", "mp4", "mov", "mkv", "avi")
  private val audioExtensions = Set("wav", "mp3", "ogg", "flac", "m4a")

  def kindOf(source: String): Kind =
    if (source.startsWith("data:video/")) Kind.Video
    else if (source.startsWith("data:audio/")) Kind.Audio
    else if (source.startsWith("data:")) Kind.Image
    else {
      val extension =
        source.split('?').head.split('.').lastOption.getOrElse("").toLowerCase
      if (videoExtensions.contains(extension)) Kind.Video
      else if (audioExtensions.contains(extension)) Kind.Audio
      else Kind.Image
    }
}
