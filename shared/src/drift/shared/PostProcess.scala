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

/** What a paused tiled job needs to be picked up again
  * (`specs/40-pause-and-resume.md`): the request as it was resolved, its drawn
  * seed filled in, so the tiles it lays out on resume are the tiles it has.
  */
enum PausedWork derives CanEqual {
  case Pid(request: PidUpscaleRequest)
  case Redraw(request: RedrawRequest)
  case Edit(request: EditRequest)
  case SeedVr2(request: SeedVr2UpscaleRequest)
}
object PausedWork {
  given JsonValueCodec[PausedWork] = JsonCodecMaker.make(storedRecord)
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
  given JsonValueCodec[PausedJob] = JsonCodecMaker.make(storedRecord)
  given JsonValueCodec[List[PausedJob]] = JsonCodecMaker.make(storedRecord)
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

/** Pauses a running tiled job after the tile in flight, or — forced — as soon
  * as that tile can be dropped (`specs/40-pause-and-resume.md`): its finished
  * tiles are kept either way, and its server stopped. Answers whether a running
  * job of that id was found.
  */
val pausePostProcessJob: PublicEndpoint[(String, Boolean), Unit, Boolean, Any] =
  base.post
    .in("post-process-jobs" / path[String] / "pause")
    // `force` does not wait for the tile in flight: it drops it the way a
    // cancel does, and the tile is run again on resume.
    .in(query[Boolean]("force").default(false))
    .out(jsonBody[Boolean])

/** Carries a paused job on at the first tile it does not have. Answers with the
  * running job, or a `Failed` one naming why it could not be resumed.
  */
val resumePostProcessJob: PublicEndpoint[String, Unit, PostProcessJob, Any] =
  base.post
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
  base.get
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
  base.post
    .in("post-process-jobs" / path[String] / "cancel")
    .out(jsonBody[Boolean])

/** Every post-processing job of this drift run, newest first. */
val listPostProcessJobs: PublicEndpoint[Unit, Unit, List[PostProcessJob], Any] =
  base.get
    .in("post-process-jobs")
    .out(jsonBody[List[PostProcessJob]])
