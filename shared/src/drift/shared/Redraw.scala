package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** A redraw (`specs/27-redraw.md`): an img2img pass over the source at
  * `strength`, tile by tile (`tileSize`) on one sd-server, through a run
  * configuration of an `image` architecture, with a reference image as context
  * and a built-in restoration prompt that `instructions` extend — the source's
  * prompt is not sent. Absent `steps` come from the configuration. A negative
  * seed means drift draws one.
  */
case class RedrawRequest(
    runConfigurationId: String,
    strength: Double = 0.4,
    /** Extra words appended to the built-in restoration prompt; redraw sends no
      * source prompt.
      */
    instructions: String = "",
    /** The restoration template to use (`specs/32-prompt-library.md`); none
      * means the built-in.
      */
    templateId: Option[String] = None,
    negativePrompt: String = "",
    steps: Option[Int] = None,
    seed: Long = -1,
    runtimeId: Option[String] = None,
    /** The longest side, in px, of the reference image sent as context. */
    contextSide: Int = 768,
    /** The largest tile, in px — a multiple of 16. Bigger tiles mean fewer of
      * them and more of the picture in each, but wider seams to hide.
      */
    tileSize: Int = 1280,
    /** How far the tile grid is shifted, in px on each axis
      * (`specs/27-redraw.md`): the same tiles, cut in another place, so a face
      * can sit inside one tile rather than across the seam between two. Taken
      * modulo the tile advance, and a multiple of the model's size multiple.
      * Zero is the even spread, which takes the fewest tiles.
      */
    gridOffsetX: Int = 0,
    gridOffsetY: Int = 0,
    /** The part of the image to repaint, in image pixels — none redraws the
      * whole image. The tiles cover a window grown around it, and only this
      * region is painted back over the source.
      */
    region: Option[ImageRegion] = None,
    /** The least side, in px, of the window a selection is redrawn through.
      * Given far fewer pixels than it was trained on a model sends back mush,
      * and a selection is usually small; 1024 is safe for the models drift
      * runs, which is as much as drift knows — no model states its own size.
      */
    minimumWindowSide: Int = 1024,
    /** Pixels kept around the selection inside that window: context for the
      * model, and the width of the ramp the repainted pixels are feathered back
      * over, so a partial redraw leaves no box edge.
      */
    selectionMargin: Int = 64,
    /** Blur radius, in px, applied to each tile before the model sees it — zero
      * leaves the tile as it is. An upscaler's leftovers (waxy skin, doubled
      * pores, ringing) are fine structure, and a low-strength pass preserves
      * structure, so the model sharpens them instead of replacing them;
      * softening first leaves it nothing to build on (`specs/27-redraw.md`).
      */
    softenRadius: Double = 0.0,
    /** Whether to send the whole image beside each tile as a reference. None
      * follows the architecture (`Architecture.referenceImages`): on for the
      * families sd-cpp has a reference preset for, off for the rest, where it
      * would be latency for nothing. Set it to weigh the trade by hand — a
      * reference holds composition and identity, costs roughly three times the
      * time per tile, and leaves less new texture behind.
      */
    useReference: Option[Boolean] = None,
    /** Px of the picture shown around each tile and kept as it is — zero sends
      * the tile alone. With it, the model is handed a window of the tile and
      * that much of the picture as redrawn so far on every side, with a mask
      * that has it repaint the tile only, and the tiles run in order. A model
      * shown only a close-up tile can take skin for another body part; seeing
      * its surroundings, it keeps the shapes (`specs/27-redraw.md`). The window
      * is what the model paints, so the time per tile grows with it.
      */
    contextMargin: Int = 0,
    /** Keeps every tile's input and output beside the job log. */
    keepTiles: Boolean = false,
    /** Each tile's own prompt and strength, as the assistant read them off the
      * picture (`specs/52-auto-redraw.md`) and the form may have changed them:
      * the prompt follows the restoration template, the strength replaces
      * `strength`, and a tile at zero is left as it is. Empty paints every tile
      * alike. The tiles are named by their place, so settings made for another
      * tile size or grid offset are refused rather than misapplied.
      */
    tiles: List[TileSettings] = Nil,
    /** Whether each tile takes the source's colour back (`specs/45`). Off for a
      * repair whose point is the colour — eyes that glow: the source's low
      * frequencies would paint the glow back around them (`specs/52`).
      */
    matchColour: Boolean = true,
    /** With a selection, whether the model repaints the selection alone, the
      * rest of its window kept as it is under a mask — a repair at a high
      * strength otherwise moves and relights everything the window shows, and
      * the selection no longer joins what is around it (`specs/52`). Needs a
      * server that takes a mask; ignored on one that does not.
      */
    maskSelection: Boolean = false
)
object RedrawRequest {
  given JsonValueCodec[RedrawRequest] = JsonCodecMaker.make
  given Schema[RedrawRequest] = Schema.derived
}

/** Redraws one persisted output through an image run configuration. Answers
  * with the running job, or a `Failed` one naming why it was refused.
  */
val redrawOutput: PublicEndpoint[
  (String, String, RedrawRequest),
  Unit,
  PostProcessJob,
  Any
] =
  base.post
    .in("outputs" / path[String]("date") / path[String]("file") / "redraw")
    .in(jsonBody[RedrawRequest])
    .out(jsonBody[PostProcessJob])
