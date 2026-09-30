package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.model.StatusCode
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Post-hoc upscale of a gallery image (`specs/15-post-hoc-resize.md`,
  * `specs/26-tiled-pid.md`): a one-shot job that writes a derived entry beside
  * the original, both kept. An ESRGAN upscale is a fresh
  * `sd-cli --mode upscale` spawn on a runtime; PiD and redraw run their tiles
  * on one sd-server per job — a ready session of the configuration, or a server
  * of the job's own (`specs/27-redraw.md`).
  */

/** Running → Completed | Failed | Cancelled, or Paused and back
  * (`specs/40-pause-and-resume.md`); the download-job pattern, not the session
  * pattern: no port, no supervision, an exit and an output file.
  */
enum PostProcessState derives CanEqual {
  case Running, Completed, Failed, Cancelled, Paused

  /** Whether the job is doing something right now — a paused one is not, and
    * holds neither a process nor a server.
    */
  def isActive: Boolean = this == Running
}
object PostProcessState {
  given Schema[PostProcessState] =
    Schema.derivedEnumeration[PostProcessState].defaultStringBased
}

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
    runtimeId: Option[String] = None
)
object PidUpscaleRequest {
  given JsonValueCodec[PidUpscaleRequest] = JsonCodecMaker.make
  given Schema[PidUpscaleRequest] = Schema.derived

  /** The longest side a PiD target may have. Bigger than this and the result
    * stops being a picture and becomes a memory problem: the blend holds the
    * whole target as an int raster and again as an image, 8 bytes a pixel — 2
    * GB at this cap, 8 GB at twice it — and 16384² is already ~120 tiles
    * (François, 2026-09-22: "redrawing 8k images already starts to show the
    * limits of it, so 16k seems like a good limit").
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
    * of GPU poorer (François, 2026-09-22, a 8192² source, where the cap of the
    * day made ×4 the source's own size).
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

  /** The largest tile an engine decodes (bug 30 asks which is best). */
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

  /** The tiles a PiD job decodes, in target px: the padded reference ×4, cut
    * into tiles of `maxTile` at most (`maxTileFor` the job's runtime)
    * overlapping by `Tiling.Overlap`, every side a multiple of 64 and every
    * start on a multiple of 4, so each tile is an exact crop of the reference.
    * The browser lays out the very tiles the backend runs, and can say how many
    * passes a job is before it starts.
    */
  def tilesFor(
      target: (Int, Int),
      maxTile: Int = MaxTile
  ): List[List[Tiling.Tile]] = {
    val (width, height) = referenceSizeOf(target)
    Tiling.layout(
      width * 4,
      height * 4,
      maxTile,
      Tiling.Overlap,
      multiple = 64,
      align = 4,
      // Nothing to line up with: a PiD decode has no picture on screen to cut
      // around, so the grid stays where it falls.
      offsetX = 0,
      offsetY = 0
    )
  }

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
    keepTiles: Boolean = false
)
object RedrawRequest {
  given JsonValueCodec[RedrawRequest] = JsonCodecMaker.make
  given Schema[RedrawRequest] = Schema.derived
}

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
    keepTiles: Boolean = false
)
object EditRequest {
  given JsonValueCodec[EditRequest] = JsonCodecMaker.make
  given Schema[EditRequest] = Schema.derived
}

/** What a paused tiled job needs to be picked up again
  * (`specs/40-pause-and-resume.md`): the request as it was resolved, its drawn
  * seed filled in, so the tiles it lays out on resume are the tiles it has.
  */
enum PausedWork derives CanEqual {
  case Pid(request: PidUpscaleRequest)
  case Redraw(request: RedrawRequest)
  case Edit(request: EditRequest)
}
object PausedWork {
  given JsonValueCodec[PausedWork] = JsonCodecMaker.make
  given Schema[PausedWork] = Schema.derived
}

/** A paused job on disk, under `post-process-jobs`: enough to run the rest of
  * it after a drift restart, the tiles it already has being beside its log.
  */
case class PausedJob(
    id: String,
    /** "pid", "redraw" or "edit". */
    kind: String,
    sourceDate: String,
    sourceFileName: String,
    work: PausedWork,
    /** How many tiles the whole job is, to say what is left and to refuse a
      * resume whose layout no longer matches.
      */
    tiles: Int,
    startedAt: Long
)
object PausedJob {
  given JsonValueCodec[PausedJob] = JsonCodecMaker.make
  given JsonValueCodec[List[PausedJob]] = JsonCodecMaker.make
  given Schema[PausedJob] = Schema.derived
}

/** How far a job made of several runs is: `completed` of `total` done. */
case class PostProcessProgress(completed: Int, total: Int)
object PostProcessProgress {
  given Schema[PostProcessProgress] = Schema.derived
}

/** One post-processing job. `result` is the derived gallery entry once the job
  * completed; `outputTail` is the end of the process output on failure.
  */
case class PostProcessJob(
    id: String,
    /** "upscale", "pid", "redraw" or "edit". */
    kind: String,
    sourceDate: String,
    sourceFileName: String,
    sourceGenerationId: String,
    state: PostProcessState,
    startedAt: Long,
    completedAt: Option[Long] = None,
    error: Option[String] = None,
    /** PiD's tiles decoded so far, set from the start of the job. */
    progress: Option[PostProcessProgress] = None,
    /** Asked to pause and still running its tile
      * (`specs/40-pause-and-resume.md`): the card says so the moment the ask
      * lands, since the tile it waits on can be minutes, and offers to stop
      * that tile instead.
      */
    pauseRequested: Boolean = false,
    /** What a tile has cost this run, in seconds, once one has finished: the
      * mean of the tiles done since this run started — a resumed job times its
      * own tiles, not the wait between the two runs. What is left is
      * `tiles remaining × this` (`specs/15-post-hoc-resize.md`).
      */
    secondsPerTile: Option[Double] = None,
    /** The tiles this job runs, in its source picture's own pixels, in the
      * order they run (`specs/15-post-hoc-resize.md`): the gallery draws them
      * over the picture and colours them by `progress` — done, running, still
      * to come. Empty for a job that is not tiled.
      */
    tiles: List[ImageRegion] = List.empty,
    /** How many finished tiles the picture this job has made so far holds —
      * `getPostProcessPicture`, the job's result as it would be if it ended now
      * (`specs/15-post-hoc-resize.md`). It trails `progress` by the moment a
      * tile takes to be painted in, and it is what a screen asks the picture
      * again on. None for a job that is not tiled, or before its first tile.
      */
    paintedTiles: Option[Int] = None,
    /** Where the run inside the current tile has got to, read out of the log
      * exactly as a session's is (`specs/13-log-streaming.md`): the model
      * loading, then the sampling steps. The tile bar above it says how far the
      * job is; this one says whether anything is happening right now.
      */
    logProgress: Option[SessionProgress] = None,
    /** The last line of the log that was not a bar, shortened — what the job is
      * doing when no bar is in flight.
      */
    activity: Option[String] = None,
    outputTail: List[String] = List.empty,
    result: Option[Generation] = None
)
object PostProcessJob {
  given JsonValueCodec[PostProcessJob] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[PostProcessJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given Schema[PostProcessJob] = Schema.derived
}

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val postProcessBase = endpoint.in("api")

/** Upscales one persisted output with a model from the upscaler store. Answers
  * with the running job, or a `Failed` one naming why it was refused.
  */
val upscaleOutput: PublicEndpoint[
  (String, String, UpscaleRequest),
  Unit,
  PostProcessJob,
  Any
] =
  postProcessBase.post
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
  postProcessBase.post
    .in("outputs" / path[String]("date") / path[String]("file") / "pid")
    .in(jsonBody[PidUpscaleRequest])
    .out(jsonBody[PostProcessJob])

/** Redraws one persisted output through an image run configuration. Answers
  * with the running job, or a `Failed` one naming why it was refused.
  */
val redrawOutput: PublicEndpoint[
  (String, String, RedrawRequest),
  Unit,
  PostProcessJob,
  Any
] =
  postProcessBase.post
    .in("outputs" / path[String]("date") / path[String]("file") / "redraw")
    .in(jsonBody[RedrawRequest])
    .out(jsonBody[PostProcessJob])

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
  postProcessBase.post
    .in("outputs" / path[String]("date") / path[String]("file") / "edit")
    .in(jsonBody[EditRequest])
    .out(jsonBody[PostProcessJob])

/** Pauses a running tiled job after the tile in flight, or — forced — as soon
  * as that tile can be dropped (`specs/40-pause-and-resume.md`): its finished
  * tiles are kept either way, and its server stopped. Answers whether a running
  * job of that id was found.
  */
val pausePostProcessJob: PublicEndpoint[(String, Boolean), Unit, Boolean, Any] =
  postProcessBase.post
    .in("post-process-jobs" / path[String] / "pause")
    // `force` does not wait for the tile in flight: it drops it the way a
    // cancel does, and the tile is run again on resume.
    .in(query[Boolean]("force").default(false))
    .out(jsonBody[Boolean])

/** Carries a paused job on at the first tile it does not have. Answers with the
  * running job, or a `Failed` one naming why it could not be resumed.
  */
val resumePostProcessJob: PublicEndpoint[String, Unit, PostProcessJob, Any] =
  postProcessBase.post
    .in("post-process-jobs" / path[String] / "resume")
    .out(jsonBody[PostProcessJob])

object PostProcessPicture {

  /** The longest side a tiled job's picture is shown at on the screen — the
    * outputs' own previews' size. The backend keeps a copy this size up to date
    * as the tiles land, so fetching it costs no pass over the full picture.
    */
  val ScreenSide: Int = 2048
}

/** The picture a running or paused tiled job has made so far — its result as it
  * would be if the job ended now, the source under the tiles still to come — at
  * most `side` px on its longest edge (`PostProcessPicture.ScreenSide` at the
  * most), or at full size without it (`specs/15-post-hoc-resize.md`). A screen
  * adds `v`, the job's `paintedTiles`, so a new tile is a new address. 404
  * while there is none.
  */
val getPostProcessPicture: PublicEndpoint[
  (String, Option[Int], Option[Int]),
  Unit,
  (Array[Byte], String),
  Any
] =
  postProcessBase.get
    .in("post-process-jobs" / path[String]("job") / "picture")
    .in(query[Option[Int]]("side"))
    .in(query[Option[Int]]("v"))
    .errorOut(statusCode(StatusCode.NotFound))
    .out(byteArrayBody)
    .out(header[String]("Content-Type"))

/** Cancels a running job: the sd-cli it waits on is killed, and the img_gen job
  * a tile is waiting for is cancelled on its server. Answers whether a running
  * job of that id was found.
  */
val cancelPostProcessJob: PublicEndpoint[String, Unit, Boolean, Any] =
  postProcessBase.post
    .in("post-process-jobs" / path[String] / "cancel")
    .out(jsonBody[Boolean])

/** Every post-processing job of this drift run, newest first. */
val listPostProcessJobs: PublicEndpoint[Unit, Unit, List[PostProcessJob], Any] =
  postProcessBase.get
    .in("post-process-jobs")
    .out(jsonBody[List[PostProcessJob]])
