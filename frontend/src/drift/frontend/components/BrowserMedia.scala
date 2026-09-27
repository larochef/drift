package drift.frontend.components

import drift.shared.ModelExample

import scala.scalajs.js

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.*
import org.scalajs.dom

object VideoAttrs {

  /** The frame shown until the video plays: Civitai's still of it (`specs/24`).
    */
  def poster(url: String): Modifier[HtmlElement] =
    htmlAttr[String]("poster", StringAsIsCodec) := url

  /** The first frame is ready: the video can take its still's place. */
  val onLoadedData = eventProp[dom.Event]("loadeddata")

  val autoplay: Modifier[HtmlElement] =
    htmlAttr[Boolean]("autoplay", BooleanAsTrueFalseStringCodec) := true

  /** Enough of the file for its first frame, and no more: what a tile asks of a
    * site that serves no smaller copy.
    */
  val preloadMetadata: Modifier[HtmlElement] =
    htmlAttr[String]("preload", StringAsIsCodec) := "metadata"

  /** The DOM *property*, not the content attribute: `setAttribute("muted")`
    * only sets `defaultMuted` and browsers ignore it on an element that already
    * exists, so the previews played with sound.
    */
  val muted: Modifier[HtmlElement] =
    htmlProp[Boolean, Boolean]("muted", BooleanAsIsCodec) := true
  val loop: Modifier[HtmlElement] =
    htmlAttr[Boolean]("loop", BooleanAsTrueFalseStringCodec) := true
  val playsInline: Modifier[HtmlElement] =
    htmlAttr[Boolean]("playsInline", BooleanAsTrueFalseStringCodec) := true
  val controls: Modifier[HtmlElement] =
    htmlAttr[Boolean]("controls", BooleanAsTrueFalseStringCodec) := true
}

/** The preview filling a tile, image or video, in all three browsers and in the
  * galleries (`specs/24`, `36`, `37`). A spinner stands in until something is
  * on screen; the top-right corner says image or video, and a video's button
  * there pauses or plays it.
  *
  * Where the site makes renditions out of the URL (Civitai's CDN), a tile loads
  * a 450px copy and a video's still, so it shows a frame before any video byte
  * is spent; where it serves the file it holds (HuggingFace, ModelScope), the
  * video is mounted with its metadata only, which shows its first frame, and
  * plays from there. Either way a candidate that does not load hands over to
  * the next — a card can name a file its repository does not hold — and when
  * none is left the tile says so.
  */
class BrowserMedia(
    candidates: List[ModelExample],
    description: String = "",
    onExhausted: () => Unit = () => ()
) extends Component {
  import BrowserMedia.{lazyLoading, playVideos}

  private val index = Var(0)

  private def next(): Unit = {
    index.update(_ + 1)
    if (index.now() >= candidates.size) onExhausted()
  }

  lazy val element: HtmlElement = div(
    cls := "civitai-media",
    child <-- index.signal.map(candidates.lift).map {
      case Some(example) => show(example)
      case None          =>
        div(cls := "civitai-card-media-missing is-size-7", "example not found")
    }
  )

  /** One candidate, with the loading state of that one: the next starts its
    * own.
    */
  private def show(example: ModelExample): HtmlElement = {
    val video = example.video
    val resized = BrowserUtils.resizesOnUrl(example.url)
    val source =
      if (resized) BrowserUtils.civitaiThumbnail(example.url, video)
      else example.url

    /** A video's still, where the site can make one out of the URL. */
    val stillSource =
      Option.when(video && resized)(BrowserUtils.civitaiStill(example.url))

    /** Something is on screen: the image, the still, or the video's frame. */
    val shown = Var(false)

    /** The still answered, loaded or not: the video may start. Without a still
      * to wait for, the video is mounted from the first.
      */
    val stillSettled = Var(stillSource.isEmpty)
    val playing = Var(playVideos.now())

    /** Latched once the video is wanted: pausing keeps its frame on screen
      * rather than unmounting it.
      */
    val videoMounted = Var(video && stillSource.isEmpty)

    lazy val player: HtmlElement = videoTag(
      cls := "civitai-card-media",
      src := source,
      stillSource.map(VideoAttrs.poster),
      if (resized) VideoAttrs.autoplay else VideoAttrs.preloadMetadata,
      VideoAttrs.muted,
      VideoAttrs.loop,
      VideoAttrs.playsInline,
      VideoAttrs.onLoadedData --> (_ => shown.set(true)),
      onError --> (_ => next()),
      inContext(thisNode =>
        playing.signal --> { play =>
          if (play) {
            // A muted video may always play; the promise rejects only when a
            // pause or an unmount interrupts it, which is no failure.
            val started = thisNode.ref.asInstanceOf[js.Dynamic].play()
            if (!js.isUndefined(started))
              started.applyDynamic("catch")((_: js.Any) => ())
          } else thisNode.ref.pause()
        }
      )
    )

    // A video's still, or the picture itself; a video without a still shows
    // its own first frame instead.
    val pictureSource = if (video) stillSource else Some(source)

    div(
      pictureSource.map(source =>
        img(
          cls := "civitai-card-media",
          src := source,
          alt := example.prompt.getOrElse(description),
          lazyLoading,
          onLoad --> { _ =>
            shown.set(true)
            stillSettled.set(true)
          },
          // A missing still does not keep its video from loading; a picture
          // that does not load hands over to the next candidate.
          onError --> { _ =>
            if (video) stillSettled.set(true) else next()
          }
        )
      ),
      Option.when(video)(
        Seq(
          playVideos.signal.changes --> playing,
          stillSettled.signal.combineWith(playing.signal) --> Observer[
            (Boolean, Boolean)
          ] { case (settled, play) =>
            if (settled && play) videoMounted.set(true)
          },
          child.maybe <-- videoMounted.signal.map(Option.when(_)(player))
        )
      ),
      child.maybe <-- shown.signal.map(done =>
        Option.when(!done)(div(cls := "media-loader"))
      ),
      div(
        cls := "media-badges",
        Option.when(video)(
          button(
            cls := "media-badge media-toggle",
            title <-- playing.signal.map(on =>
              if (on) "Pause this video" else "Play this video"
            ),
            child.text <-- playing.signal.map(on => if (on) "⏸" else "▶"),
            onClick.stopPropagation --> (_ => playing.update(!_))
          )
        ),
        span(
          cls := "media-badge",
          title := (if (video) "Video" else "Image"),
          if (video) "🎬" else "🖼️"
        )
      )
    )
  }
}

object BrowserMedia {

  /** Whether video tiles play (`specs/24`): one switch for the three browsers
    * and every gallery, kept while the page lives, like the generation form's
    * folded sections. Off, a video tile shows a frame and nothing more.
    */
  val playVideos: Var[Boolean] = Var(true)

  private val lazyLoading: Modifier[HtmlElement] =
    htmlAttr[String]("loading", StringAsIsCodec) := "lazy"
}
