package drift.frontend.components

object BrowserUtils {

  val modelExts = Set(".safetensors", ".gguf", ".pt", ".pth", ".bin")

  def formatSize(bytes: Long): String = {
    if (bytes < 1024) s"$bytes B"
    else if (bytes < 1024L * 1024) f"${bytes / 1024.0}%.1f KB"
    else if (bytes < 1024L * 1024 * 1024) f"${bytes / (1024.0 * 1024)}%.1f MB"
    else f"${bytes / (1024.0 * 1024 * 1024)}%.1f GB"
  }

  val isModelFile: String => Boolean = ext => modelExts.contains(ext)

  /** Compact stat counts for the browser cards: 12345 → "12.3k". */
  def formatCount(count: Long): String =
    if (count >= 1000000) f"${count / 1000000.0}%.1fM"
    else if (count >= 1000) f"${count / 1000.0}%.1fk"
    else count.toString

  /** A Civitai image or video URL with CDN options in place of its
    * `original=true` segment: the CDN resizes, re-encodes and extracts frames
    * on that segment (probed 2026-09-11, `specs/24`). A URL without it stays
    * whole. One rendition, one URL, everywhere — the CDN allows an hour of
    * browser caching, so the tile's copy is free to show again in the viewer.
    */
  private def civitaiVariant(url: String, options: String): String =
    url.replace("/original=true/", s"/$options/")

  /** Whether the site serving this URL makes other renditions out of it —
    * Civitai's CDN does, on that segment; HuggingFace and ModelScope serve the
    * file they hold. A tile asks for a small copy where there is one, which is
    * all `BrowserMedia` needs to know about the site it shows.
    */
  def resizesOnUrl(url: String): Boolean = url.contains("/original=true/")

  /** A tile-sized rendition: WebP for an image (a third of the JPEG's bytes, a
    * twentieth of the original's), a 450-wide mp4 for a video.
    */
  def civitaiThumbnail(url: String, video: Boolean): String =
    civitaiVariant(url, if (video) "width=450" else "width=450,optimized=true")

  /** A video's still, as JPEG. Only `anim=false` on the video's own name gives
    * one — the same options on a `.jpeg` name answered the whole original mp4 —
    * and the CDN does not resize it: 150–750 KB, against 1–9 MB for the tile's
    * video. There is no lighter video for the viewer: at `width=1080` the CDN
    * answered 44 MB for a 16 MB original.
    */
  def civitaiStill(url: String): String =
    civitaiVariant(url, "anim=false,width=450")
}
