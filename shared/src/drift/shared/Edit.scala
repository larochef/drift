package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** An edit (`specs/39-seamless-edit.md`): the instruction says what should be
  * different, over the whole image or `region`. Each tile goes to a run
  * configuration of an architecture tagged `edit` as the image to edit — its
  * only reference, no init image — with the edit template and the instruction;
  * tiles run in order, each cut from the picture as edited so far, and drift
  * keeps the source's own pixels wherever the model changed nothing. The tile,
  * window and margin fields mean what a redraw's do; absent `steps` come from
  * the configuration, a negative seed means drift draws one.
  */
case class EditRequest(
    runConfigurationId: String,
    /** What should be different — required: an edit without one has nothing to
      * do.
      */
    instructions: String,
    /** The edit template (`specs/32-prompt-library.md`); none means the
      * built-in.
      */
    templateId: Option[String] = None,
    steps: Option[Int] = None,
    seed: Long = -1,
    runtimeId: Option[String] = None,
    tileSize: Int = 1280,
    gridOffsetX: Int = 0,
    gridOffsetY: Int = 0,
    region: Option[ImageRegion] = None,
    minimumWindowSide: Int = 1024,
    selectionMargin: Int = 64,
    /** Keeps every tile's input, the model's raw edit, the change mask and the
      * composite in the job's tiles directory.
      */
    keepTiles: Boolean = false,
    /** The upscaler that carries the edit up (`specs/39-seamless-edit.md`): a
      * SeedVR2 run configuration. With one, the part to edit is edited in a
      * single pass, reduced to what the model takes at once
      * (`EditRequest.passOf`), and only what changed is brought back to the
      * picture's size by this upscaler. None runs the edit tile by tile.
      */
    upscaleConfigurationId: Option[String] = None,
    /** What a selection keeps of an edit carried up: every changed part that
      * touches it, as far as it goes — hair cut short is cut down to its ends,
      * outside the box too — or, when false, only what lies within the
      * selection and its margin, the rest of the picture untouched whatever the
      * model did there.
      */
    beyondSelection: Boolean = true
)
object EditRequest {
  given JsonValueCodec[EditRequest] = JsonCodecMaker.make
  given Schema[EditRequest] = Schema.derived

  /** The longest side an edit model is handed in one pass: Flux.2 Klein and
    * Qwen Image 2.1 edit a 1024 × 1536 picture whole.
    */
  val PassSide: Int = 1536

  /** The least side of the window a carried edit first sees around `selection`.
    * A selection that fits one pass with its margin is seen through a window of
    * `PassSide`: edited at the picture's own size, with nothing to carry up —
    * what is kept is the picture's own pixels and the model's, no upscaler's
    * texture between them. A larger one through four times its longest side, at
    * most twice `PassSide`. The job looks wider when that was not enough
    * (`EditCarry`).
    */
  def contextSide(selection: ImageRegion, margin: Int): Int = {
    val longest = math.max(selection.width, selection.height)
    if (longest + 2 * margin <= PassSide) PassSide
    else (4 * longest).max(PassSide).min(2 * PassSide)
  }

  /** The single pass an edit carried up runs: the size the part to edit is
    * reduced to, and the scale the upscaler brings it back by.
    */
  case class Pass(width: Int, height: Int, scale: Int)

  /** How far the upscaler's ×2 is stretched before ×4 is asked instead: a part
    * up to this many times `PassSide` is edited at `PassSide`, upscaled ×2 and
    * enlarged the rest of the way — a third at most. Measured 2026-10-07 on a
    * 4096 px picture: a face edited at 1536 and brought back this way is a
    * little soft, the same face edited at 1024 and upscaled ×4 is harsh — the
    * upscaler redraws it from too little.
    */
  val StretchedBy: Double = 8.0 / 3

  /** The pass for a part of `width` × `height` px. As it is when it fits
    * `PassSide`; halved when that is enough; else brought within `PassSide` and
    * upscaled ×2 when the rest is a small stretch (`StretchedBy`), ×4 beyond —
    * and what the upscaler leaves short or over is resized to the part. Sides
    * are multiples of `multiple`, as the model and the upscaler take them.
    */
  def passOf(width: Int, height: Int, multiple: Int): Pass = {
    val longest = math.max(width, height)
    val step = Tiling.roundUp(16, multiple)
    def side(length: Int, by: Double): Int =
      Tiling.roundUp(math.ceil(length / by).toInt, step)
    if (longest <= PassSide) Pass(side(width, 1), side(height, 1), 1)
    else if (longest <= 2 * PassSide) Pass(side(width, 2), side(height, 2), 2)
    else {
      val by = longest.toDouble / PassSide
      Pass(side(width, by), side(height, by), if (by <= StretchedBy) 2 else 4)
    }
  }
}

/** Edits one persisted output through a run configuration of an `edit`
  * architecture (`specs/39-seamless-edit.md`). Answers with the running job, or
  * a `Failed` one naming why it was refused.
  */
val editOutput: PublicEndpoint[
  (String, String, EditRequest),
  Unit,
  PostProcessJob,
  Any
] =
  base.post
    .in("outputs" / path[String]("date") / path[String]("file") / "edit")
    .in(jsonBody[EditRequest])
    .out(jsonBody[PostProcessJob])
