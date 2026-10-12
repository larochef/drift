package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** `upscalerId` names an entry of the upscaler store; `repeats` runs the model
  * that many times (×4 twice is ×16); `tileSize` overrides sd-cli's ESRGAN tile
  * (default 128); `runtimeId` pins a runtime, else the default runs it.
  */
case class UpscaleRequest(
    upscalerId: String,
    repeats: Int = 1,
    tileSize: Option[Int] = None,
    runtimeId: Option[String] = None
)
object UpscaleRequest {
  given JsonValueCodec[UpscaleRequest] = JsonCodecMaker.make
  given Schema[UpscaleRequest] = Schema.derived
}

/** Upscale by pixel diffusion (`docs/pid.md` in stable-diffusion.cpp): a run
  * configuration of a PiD architecture (its diffusion model, Gemma text encoder
  * and matching VAE) decodes the target tile by tile on one sd-server, each
  * tile's part of the source as its reference image. `prompt` conditions the
  * decoder — the source's own prompt is the natural default. A negative seed
  * means drift draws one. `width` and `height` name the target — both, or
  * neither for ×4 of the source (`PidUpscaleRequest.target`).
  */
case class PidUpscaleRequest(
    runConfigurationId: String,
    width: Option[Int] = None,
    height: Option[Int] = None,
    prompt: String = "",
    negativePrompt: String = "",
    steps: Int = 4,
    cfgScale: Double = 1.0,
    seed: Long = -1,
    runtimeId: Option[String] = None,
    /** The largest tile, in target px — none for the largest the job's runtime
      * decodes (`UpscaleTiling.withTile`).
      */
    tileSize: Option[Int] = None,
    /** How far the grid is shifted, in target px (`Tiling.axis`). */
    gridOffsetX: Int = 0,
    gridOffsetY: Int = 0
)
object PidUpscaleRequest {
  given JsonValueCodec[PidUpscaleRequest] = JsonCodecMaker.make
  given Schema[PidUpscaleRequest] = Schema.derived

  /** The longest side a PiD target may have. Bigger than this and the result
    * stops being a picture and becomes a memory problem: the blend holds the
    * whole target as an int raster and again as an image, 8 bytes a pixel — 2
    * GB at this cap, 8 GB at twice it — and 16384² is already ~120 tiles — and
    * redrawing an 8k picture already shows the limits of it.
    */
  val MaxSide: Int = 16384

  /** The target for a `source`-sized image (`specs/26-tiled-pid.md`): the size
    * asked for — sides multiples of 4 from 256 to `MaxSide`, any ratio, another
    * than the source's cropping it — or, asked for none, ×4 of the source with
    * its ratio kept, the longest side capped at `MaxSide`.
    *
    * A target no larger than the source is refused rather than run: PiD decodes
    * a *quarter* of the target, so such a job would shrink the source to that
    * quarter and hand back the same size, looking like the original and an hour
    * of GPU poorer (seen on a 8192² source, where the cap of the day made ×4
    * the source's own size).
    */
  def target(
      source: (Int, Int),
      width: Option[Int],
      height: Option[Int]
  ): Either[String, (Int, Int)] =
    (width, height) match {
      case (None, None) =>
        val scale =
          math.min(4.0, MaxSide.toDouble / math.max(source._1, source._2))
        def side(length: Int) =
          (math.floor(length * scale / 4).toInt * 4).max(4)
        grown(source, (side(source._1), side(source._2)))
      case (Some(w), Some(h))
          if List(w, h)
            .forall(side => side >= 256 && side <= MaxSide && side % 4 == 0) =>
        grown(source, (w, h))
      case (Some(_), Some(_)) =>
        Left(s"width and height must be multiples of 4 from 256 to $MaxSide")
      case _ =>
        Left("give both width and height, or neither for ×4 of the source")
    }

  /** The largest PiD tile, in px — a multiple of 64. PiD runs without flash
    * attention, which speckles its decode on ROCm and blacks it out on CPU;
    * without it ROCm decodes up to 1792² and aborts at 2048² in the attention
    * scale's kernel launch (`specs/26-tiled-pid.md`).
    */
  val MaxTile: Int = 1536

  /** The largest PiD tile on drift's own runner, in px: it decodes 1024 → 4096
    * in one pass (its attention needs no memory per score, and survives PiD's
    * large activations), so a 4096² target is a single tile. The overlap on top
    * lets n tiles cover n × 4096 — an 8192² target is 2 × 2, not 3 × 3; a 1088
    * → 4352 pass decodes as well as the 1024 → 4096 one it is trained for
    * (`specs/26-tiled-pid.md`).
    */
  val RunnerMaxTile: Int = 4096 + Tiling.Overlap

  /** The largest tile `runtime` decodes. */
  def maxTileFor(runtime: Runtime): Int = maxTileFor(runtime.engine)

  /** The largest tile an engine decodes. */
  def maxTileFor(engine: RuntimeEngine): Int =
    if (engine == RuntimeEngine.DriftRunner) RunnerMaxTile else MaxTile

  /** The reference a target is decoded from: a quarter of it, padded to
    * multiples of 16 — a PiD input must be.
    */
  def referenceSizeOf(target: (Int, Int)): (Int, Int) =
    (
      Tiling.roundUp(target._1 / 4, 16),
      Tiling.roundUp(target._2 / 4, 16)
    )

  /** How a PiD job cuts its target: the padded reference ×4, in tiles of
    * `maxTile` at most (`maxTileFor` the job's runtime), every side a multiple
    * of 64 and every start on a multiple of 4, so each tile is an exact crop of
    * the reference. The browser lays out the very tiles the backend runs, and
    * can say how many passes a job is before it starts.
    */
  def tilingFor(
      target: (Int, Int),
      maxTile: Int = MaxTile
  ): UpscaleTiling = {
    val (width, height) = referenceSizeOf(target)
    UpscaleTiling(width * 4, height * 4, maxTile, multiple = 64, align = 4)
  }

  /** The tiles a PiD job decodes, in target px, as the request asks for them:
    * its tile size, its grid shifted.
    */
  def tilesFor(
      target: (Int, Int),
      maxTile: Int = MaxTile,
      tileSize: Option[Int] = None,
      offsetX: Int = 0,
      offsetY: Int = 0
  ): List[List[Tiling.Tile]] =
    tilingFor(target, maxTile).withTile(tileSize).shifted(offsetX, offsetY).rows

  /** `target` when it is larger than `source` on at least one side, else why
    * this job would only shrink the image.
    */
  private def grown(
      source: (Int, Int),
      target: (Int, Int)
  ): Either[String, (Int, Int)] =
    Either.cond(
      target._1 > source._1 || target._2 > source._2,
      target,
      s"a PiD target of ${target._1}×${target._2} is no larger than the " +
        s"${source._1}×${source._2} source: PiD decodes a quarter of the " +
        s"target, so this would shrink the image to ${target._1 / 4}×" +
        s"${target._2 / 4} and hand back the same size" +
        (if (math.max(source._1, source._2) >= MaxSide)
           s". A PiD target stops at $MaxSide px on the longest side, which " +
             "this image already has — there is nothing left to upscale here"
         else "")
    )
}

/** A SeedVR2 upscale (`specs/51-seedvr2-upscaling.md`): the source — a picture
  * or a video — restored at `scale` times its size by a SeedVR2 run
  * configuration on the drift runner, in one diffusion step. No prompt. A
  * negative seed means drift draws one.
  */
case class SeedVr2UpscaleRequest(
    runConfigurationId: String,
    scale: Int = SeedVr2UpscaleRequest.DefaultScale,
    seed: Long = -1,
    runtimeId: Option[String] = None,
    /** The largest tile of a picture's job, in target px — none for `MaxTile`
      * (`UpscaleTiling.withTile`).
      */
    tileSize: Option[Int] = None,
    /** How far the grid is shifted, in target px (`Tiling.axis`). */
    gridOffsetX: Int = 0,
    gridOffsetY: Int = 0
)
object SeedVr2UpscaleRequest {
  given JsonValueCodec[SeedVr2UpscaleRequest] = JsonCodecMaker.make
  given Schema[SeedVr2UpscaleRequest] = Schema.derived

  /** The job's kind and the derived entry's operation. */
  val Kind = "seedvr2"

  /** The model kind of the architectures that run it. */
  val ModelKind = "seedvr2"

  val Scales: List[Int] = List(2, 4)
  val DefaultScale = 4

  /** The built-in configuration a panel opens on: the 7B. */
  val DefaultConfiguration = "starter-seedvr2-7b"

  def runs(architecture: Architecture): Boolean =
    architecture.modelKind.contains(ModelKind)

  /** The largest tile a job restores in one pass, in target px: what a PiD
    * decodes on the runner. A picture larger than that is cut in tiles, each
    * its own pass (`tilesFor`).
    */
  val MaxTile: Int = PidUpscaleRequest.RunnerMaxTile

  /** The size a `source` picture comes out at — refused past
    * `PidUpscaleRequest.MaxSide` on its longest side.
    */
  def target(source: (Int, Int), scale: Int): Either[String, (Int, Int)] = {
    val (width, height) = (source._1 * scale, source._2 * scale)
    Either.cond(
      math.max(width, height) <= PidUpscaleRequest.MaxSide,
      (width, height),
      s"×$scale of ${source._1}×${source._2} is $width×$height: an upscale " +
        s"stops at ${PidUpscaleRequest.MaxSide} px on the longest side"
    )
  }

  /** The picture the tiles are cut from: the source padded to multiples of 16,
    * so every tile's crop is a size the model takes as it is.
    */
  def referenceSizeOf(source: (Int, Int)): (Int, Int) =
    (Tiling.roundUp(source._1, 16), Tiling.roundUp(source._2, 16))

  /** How a picture's job cuts its target: the padded source times `scale`, in
    * tiles of `MaxTile` at most, every side a multiple of `16 × scale` and
    * every start on a multiple of `scale`, so each tile is an exact crop of the
    * source. One tile for a target within `MaxTile`.
    */
  def tilingFor(source: (Int, Int), scale: Int): UpscaleTiling = {
    val (width, height) = referenceSizeOf(source)
    UpscaleTiling(
      width * scale,
      height * scale,
      MaxTile,
      multiple = 16 * scale,
      align = scale
    )
  }

  /** The tiles a picture's job restores, in target px, as the request asks for
    * them: its tile size, its grid shifted.
    */
  def tilesFor(
      source: (Int, Int),
      scale: Int,
      tileSize: Option[Int] = None,
      offsetX: Int = 0,
      offsetY: Int = 0
  ): List[List[Tiling.Tile]] =
    tilingFor(source, scale).withTile(tileSize).shifted(offsetX, offsetY).rows
}

/** How an upscale cuts its target into tiles (`specs/26-tiled-pid.md`,
  * `specs/51-seedvr2-upscaling.md`), in target px: the padded target, the
  * largest tile, the multiple every side keeps and the one every start keeps,
  * and how far the grid is shifted. Neighbours overlap by `Tiling.Overlap`. The
  * browser and the backend lay the tiles out from the same value, so the grid
  * drawn over the picture is the grid of the job.
  */
case class UpscaleTiling(
    width: Int,
    height: Int,
    tile: Int,
    multiple: Int,
    align: Int,
    offsetX: Int = 0,
    offsetY: Int = 0
) {

  /** The same cut with tiles of `asked` px at most, brought to the multiple and
    * kept between `UpscaleTiling.MinimumTile` and the tile it has — the largest
    * its runtime takes. None leaves it as it is.
    */
  def withTile(asked: Option[Int]): UpscaleTiling =
    asked.fold(this)(size =>
      copy(tile =
        Tiling
          .roundUp(size, multiple)
          .max(Tiling.roundUp(UpscaleTiling.MinimumTile, multiple))
          .min(tile)
      )
    )

  /** The same cut on a grid shifted by that much, snapped to the multiple a
    * shift must keep (`Tiling.axis`).
    */
  def shifted(x: Int, y: Int): UpscaleTiling = {
    def snapped(offset: Int) =
      math.round(offset.toDouble / multiple).toInt * multiple
    copy(offsetX = snapped(x), offsetY = snapped(y))
  }

  /** The tiles, row by row. */
  def rows: List[List[Tiling.Tile]] =
    Tiling.layout(
      width,
      height,
      tile,
      Tiling.Overlap,
      multiple,
      align,
      offsetX,
      offsetY
    )

  /** How far apart the tiles start on an axis of `length` target px. */
  def strideFor(length: Int): Int =
    Tiling.strideFor(
      Tiling.roundUp(length, multiple),
      tile,
      Tiling.Overlap,
      multiple
    )
}
object UpscaleTiling {

  /** The smallest tile an upscale is cut in, in target px: under it the
    * overlaps are most of every tile.
    */
  val MinimumTile: Int = 1024
}

/** Upscales one persisted output with a model from the upscaler store. Answers
  * with the running job, or a `Failed` one naming why it was refused.
  */
val upscaleOutput: PublicEndpoint[
  (String, String, UpscaleRequest),
  Unit,
  PostProcessJob,
  Any
] =
  base.post
    .in("outputs" / path[String]("date") / path[String]("file") / "upscale")
    .in(jsonBody[UpscaleRequest])
    .out(jsonBody[PostProcessJob])

/** Upscales one persisted output through a PiD run configuration. Answers with
  * the running job, or a `Failed` one naming why it was refused.
  */
val pidUpscaleOutput: PublicEndpoint[
  (String, String, PidUpscaleRequest),
  Unit,
  PostProcessJob,
  Any
] =
  base.post
    .in("outputs" / path[String]("date") / path[String]("file") / "pid")
    .in(jsonBody[PidUpscaleRequest])
    .out(jsonBody[PostProcessJob])

/** Upscales one persisted output — a picture or a video — through a SeedVR2 run
  * configuration. Answers with the running job, or a `Failed` one naming why it
  * was refused.
  */
val seedVr2UpscaleOutput: PublicEndpoint[
  (String, String, SeedVr2UpscaleRequest),
  Unit,
  PostProcessJob,
  Any
] =
  base.post
    .in("outputs" / path[String]("date") / path[String]("file") / "seedvr2")
    .in(jsonBody[SeedVr2UpscaleRequest])
    .out(jsonBody[PostProcessJob])
