package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** The Examples tab of an opened repository, HuggingFace or ModelScope
  * (`specs/36-huggingface-examples.md`, `specs/37-modelscope.md`): what its
  * model makes, with the prompts when known, as tiles like Civitai's; a tile
  * opens a viewer with the prompt and the way to the repository's page.
  */
class ModelExampleGallery(
    /** The repository's page on its site, and the link's words. */
    pageUrl: String,
    pageLabel: String,
    examples: Signal[List[ModelExample]],
    loading: Signal[Boolean]
) extends Component {
  private val viewing = Var(Option.empty[ModelExample])

  lazy val element: HtmlElement = div(
    documentEvents(_.onKeyDown) --> { event =>
      if (event.key == "Escape") viewing.set(None)
    },
    ResultsPlaceholder(
      busy = loading,
      isEmpty = examples.map(_.isEmpty),
      busyText = "Loading examples...",
      emptyText = "No examples: the repository shows no gallery and holds no " +
        "image or video files. The model card may still say what it does."
    ),
    div(
      cls := "civitai-results-grid civitai-gallery-grid",
      children <-- examples.split(_.url) { (_, example, _) =>
        // An example that does not load is left out rather than shown broken.
        val broken = Var(false)
        div(
          cls := "civitai-card cursor-pointer",
          display <-- broken.signal.map(if (_) "none" else ""),
          title := example.prompt.getOrElse(""),
          onClick --> (_ => viewing.set(Some(example))),
          BrowserMedia(
            List(example),
            description = example.prompt.getOrElse(""),
            onExhausted = () => broken.set(true)
          ).element
        )
      }
    ),
    child.maybe <-- viewing.signal.map(_.map(viewer))
  )

  /** The example as large as the window allows, above the browser modal; a
    * click outside the media or Esc closes it.
    */
  private def viewer(example: ModelExample): HtmlElement =
    div(
      cls := "media-lightbox",
      ScrollLock.whileMounted,
      onClick --> (_ => viewing.set(None)),
      div(
        cls := "media-lightbox-frame",
        if (example.video)
          videoTag(
            cls := "media-lightbox-media",
            src := example.url,
            VideoAttrs.controls,
            VideoAttrs.autoplay,
            VideoAttrs.muted,
            VideoAttrs.loop,
            VideoAttrs.playsInline,
            onClick.stopPropagation --> (_ => ())
          )
        else
          img(
            cls := "media-lightbox-media",
            src := example.url,
            alt := example.prompt.getOrElse(""),
            onClick.stopPropagation --> (_ => ())
          )
      ),
      div(
        cls := "media-lightbox-caption",
        example.prompt.map(prompt =>
          span(
            styleAttr := "max-width: 60rem; overflow-wrap: anywhere;",
            onClick.stopPropagation --> (_ => ()),
            prompt
          )
        ),
        a(
          href := pageUrl,
          target := "_blank",
          rel := "noopener noreferrer",
          styleAttr := "white-space: nowrap;",
          onClick.stopPropagation --> (_ => ()),
          pageLabel
        )
      )
    )
}
