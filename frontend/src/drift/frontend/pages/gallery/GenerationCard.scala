package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.{BooleanAsAttrPresenceCodec, StringAsIsCodec}

/** One output in the grid: a lazily loaded thumbnail, the prompt and where it
  * came from. A batch's images are tiles of their own (François, 2026-09-14),
  * each marked with its place in the batch so they read as made together.
  *
  * While the gallery is selecting, the card is a checkbox instead of a link: a
  * click ticks it rather than opening the detail, so a sweep of the grid never
  * costs a modal.
  */
class GenerationCard(
    generation: Generation,
    configurationLabel: String,
    onOpen: () => Unit,
    /** Which of the generation's outputs this tile shows. */
    outputIndex: Int = 0,
    /** The project this generation belongs to, when it does — a signal, since
      * the projects may load after the day did.
      */
    projectLabel: Signal[Option[String]] = Val(None),
    /** Whether the gallery is in selection mode, and whether this card is in
      * the selection — signals, because a card is built once per generation id
      * and outlives both.
      */
    selecting: Signal[Boolean] = Val(false),
    selected: Signal[Boolean] = Val(false),
    onToggleSelected: () => Unit = () => ()
) extends Component {

  /** A video shows its still: only the detail's player loads it (bug 37). */
  private def thumbnail(output: GenerationOutput): HtmlElement =
    img(
      src := (
        if (GenerationMediaViewer.isVideo(output))
          GenerationMediaViewer.stillUrl(output)
        else output.url
      ),
      GenerationCard.loadingAttr := "lazy",
      alt := RecordedParameters.promptOf(generation).take(80)
    )

  lazy val element: HtmlElement = div(
    cls := "card bg-card gallery-card cursor-pointer" +
      (if (generation.outputs.size > 1) " is-batch" else ""),
    cls("is-selected") <-- selected,
    dataAttr("generation-id") := generation.id,
    onClick.compose(_.sample(selecting)) --> Observer[Boolean] { selecting =>
      if (selecting) onToggleSelected() else onOpen()
    },
    div(
      cls := "gallery-thumb",
      generation.outputs.lift(outputIndex).map(thumbnail).getOrElse(emptyNode),
      child <-- selecting.map {
        case false => emptyNode
        case true  =>
          span(
            cls := "gallery-select-badge",
            input(
              typ := "checkbox",
              checked <-- selected,
              // The card's own handler does the toggling; the box would
              // otherwise toggle twice on its way up.
              onClick.stopPropagation --> (_ => onToggleSelected())
            )
          )
      },
      if (generation.outputs.size > 1)
        span(
          cls := "tag is-dark gallery-batch-badge",
          title := s"Image ${outputIndex + 1} of a batch of " +
            s"${generation.outputs.size}, generated together",
          s"▦ ${outputIndex + 1}/${generation.outputs.size}"
        )
      else emptyNode,
      child <-- projectLabel.map {
        case Some(label) =>
          span(
            cls := "tag is-link gallery-project-badge",
            title := s"Project: $label",
            s"\uD83D\uDCC1 $label"
          )
        case None => emptyNode
      },
      if (generation.kind == "vid_gen")
        span(cls := "tag is-info gallery-kind-badge", "▶ video")
      else
        generation.derivation
          .map(d =>
            span(
              cls := "tag is-primary gallery-kind-badge",
              d.operation match {
                case "upscale" => "⬆ upscaled"
                case "pid"     => "⬆ PiD"
                case "redraw"  => "✨ redrawn"
                case "edit"    => "✎ edited"
                case _         => "⇲ resized"
              }
            )
          )
          .orElse(
            Option.when(generation.kind == "import")(
              span(cls := "tag is-primary gallery-kind-badge", "⤓ imported")
            )
          )
          .getOrElse(emptyNode)
    ),
    div(
      cls := "gallery-card-body",
      p(
        cls := "text-primary is-size-7 gallery-prompt",
        title := RecordedParameters.titleOf(generation),
        RecordedParameters.titleOf(generation)
      ),
      p(
        cls := "text-secondary is-size-7 gallery-prompt",
        s"$configurationLabel · ${RecordedParameters.timeOf(generation.submittedAt)}"
      )
    )
  )
}

object GenerationCard {
  // Not in Laminar's default bundle, so defined here once for the gallery.
  val loadingAttr = htmlAttr("loading", StringAsIsCodec)
  val preloadAttr = htmlAttr("preload", StringAsIsCodec)
  val mutedAttr = htmlAttr("muted", BooleanAsAttrPresenceCodec)
  val playsInlineAttr = htmlAttr("playsinline", BooleanAsAttrPresenceCodec)
  val controlsAttr = htmlAttr("controls", BooleanAsAttrPresenceCodec)
  val loopAttr = htmlAttr("loop", BooleanAsAttrPresenceCodec)
  val autoPlayAttr = htmlAttr("autoplay", BooleanAsAttrPresenceCodec)
}
