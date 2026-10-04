package drift.backend.postprocess

import drift.backend.lora.LoraManager
import drift.backend.runtime.{LaunchRuntime, RuntimeManager}
import drift.backend.sdserver.NativeUpscale
import drift.backend.session.SessionManager
import drift.shared.*

import java.awt.image.BufferedImage
import java.nio.file.*

/** What a tiled job — PiD, redraw — goes through (`specs/26-tiled-pid.md`,
  * `specs/27-redraw.md`): the checks before it is recorded, the configuration's
  * LoRAs, and the tiles made one by one, judged and blended.
  */
final private[postprocess] class TiledJobs(
    jobs: PostProcessJobs,
    runtimeManager: RuntimeManager,
    sessionManager: SessionManager,
    loraManager: LoraManager
) {

  /** Checks what a tiled job needs before it is recorded — the runtime exists,
    * the run configuration resolves on it (weights cached, nothing blocking
    * it), `accepts` names no reason against its architecture — and refuses the
    * job otherwise; `start` builds and starts it from the configuration, the
    * runtime and its LoRAs as request entries
    * (`specs/28-configuration-loras.md`), or names why it cannot. What each
    * tile asks for is the job's business (`runTiles`).
    */
  def tiledJob(
      kind: String,
      date: String,
      fileName: String,
      runConfigurationId: String,
      runtimeId: Option[String],
      resuming: Option[String],
      accepts: Architecture => Option[String],
      videos: Boolean = false
  )(
      start: (
          PostProcessSource,
          RunConfiguration,
          Architecture,
          LaunchRuntime,
          ConfiguredLoras
      ) => Either[String, PostProcessJob]
  ): PostProcessJob =
    jobs
      .busyWith(date, fileName, resuming)
      .toLeft(())
      .flatMap(_ => jobs.source(date, fileName, videos))
      .flatMap { src =>
        sessionManager
          .runnerOf(runConfigurationId)
          .flatMap(runner =>
            runtimeManager
              .resolveForLaunch(RuntimeTool.SdCpp, runner, runtimeId)
          )
          .flatMap { launch =>
            sessionManager
              .resolveArguments(runConfigurationId, launch)
              .flatMap { (configuration, architecture, _) =>
                accepts(architecture)
                  .toLeft(())
                  .flatMap(_ => configuredLoras(configuration))
                  .flatMap(loras =>
                    start(
                      src,
                      configuration,
                      architecture,
                      launch,
                      loras
                    )
                  )
              }
          }
      } match {
      case Left(reason) => jobs.refused(kind, date, fileName, reason)
      case Right(job)   => job
    }

  /** The configuration's default LoRAs as request entries
    * (`specs/28-configuration-loras.md`) — refused, naming the LoRA, when one
    * is no longer installed or its files are not all downloaded: a checkpoint
    * that needs its turbo LoRA makes silent garbage without it.
    */
  private def configuredLoras(
      configuration: RunConfiguration
  ): Either[String, ConfiguredLoras] = {
    val installed = loraManager.list.map(lora => lora.id -> lora).toMap
    configuration.loras.foldLeft(
      Right(ConfiguredLoras(List.empty, LoraSampling())): Either[
        String,
        ConfiguredLoras
      ]
    ) { (result, configured) =>
      result.flatMap(loras =>
        installed.get(configured.loraId) match {
          case None =>
            Left(
              s"LoRA '${configured.loraId}', a default of '${configuration.label}', is not installed"
            )
          case Some(lora)
              if !lora.files.forall(file =>
                Files.isRegularFile(
                  loraManager.lorasRoot.resolve(lora.storagePathOf(file))
                )
              ) =>
            Left(
              s"LoRA '${lora.label}', a default of '${configuration.label}', is not fully downloaded"
            )
          case Some(lora) =>
            Right(
              ConfiguredLoras(
                loras.selections ++ lora.selections(configured.strength),
                lora.sampling.over(loras.sampling)
              )
            )
        }
      )
    }
  }

  /** Records a running job over `rows` of tiles and runs it on its own thread —
    * a new one, or the paused job `resuming` names, which keeps its id, its log
    * and the tiles it has (`specs/40-pause-and-resume.md`). `work` is what it
    * takes to finish the job later, written when it pauses.
    */
  def startTiles(
      kind: String,
      src: PostProcessSource,
      rows: List[List[Tiling.Tile]],
      work: PausedWork,
      resuming: Option[String],
      /** The same tiles in the source picture's own pixels, for the grid drawn
        * over it: a redraw's window offset added, a PiD's target scaled back.
        */
      onPicture: List[Tiling.Tile]
  )(run: PostProcessJob => Unit): PostProcessJob = {
    val tiles = rows.flatten.size
    // The tiles a resumed job has are the tiles it laid out then: a
    // configuration changed since would cut the picture elsewhere.
    resuming.flatMap(jobs.pausedWork).filter(_.tiles != tiles) match {
      case Some(paused) =>
        jobs.refused(
          kind,
          src.date,
          src.fileName,
          s"this job was paused at ${paused.tiles} tiles and would now run " +
            s"$tiles: what lays them out changed, so its tiles no longer fit " +
            "— start a new job, or cancel this one to drop them"
        )
      case None =>
        jobs.start(
          kind,
          src,
          Some(
            PostProcessProgress(
              resuming.fold(0)(jobs.files.tilesDone(_, tiles)),
              tiles
            )
          ),
          resuming,
          onPicture
        ) { job =>
          jobs.remember(job.id, work, tiles)
          run(job)
        }
    }
  }

  /** Runs this job's tiles: one `TileRun`, which says what that involves, and
    * another one later if the job is paused and carried on.
    */
  def runTiles(
      job: PostProcessJob,
      src: PostProcessSource,
      launch: LaunchRuntime,
      reference: BufferedImage,
      scale: Int,
      runConfigurationId: String,
      request: TileRequests,
      target: (Int, Int),
      rows: List[List[Tiling.Tile]],
      notes: List[String],
      derivation: Derivation,
      /** What each tile's crop goes through before the model sees it — where a
        * redraw softens away the artifacts it is meant to remove.
        */
      prepareTile: BufferedImage => BufferedImage = identity,
      /** What each returned tile goes through before it is kept, given the
        * source's crop of the tile as it was before `prepareTile` — where a
        * redraw takes the source's colour back (`specs/45`).
        */
      correctTile: Option[(BufferedImage, BufferedImage) => BufferedImage] =
        None,
      finish: PictureFinish = PictureFinish.AsPainted,
      keepTiles: Boolean = false,
      finishTile: Option[TileWindow.FinishTile] = None,
      context: Option[TileWindow.TileContext] = None,
      /** Whether this run carries a paused job on, which changes what the log
        * says and nothing else: the tiles it has are found on disk
        * (`specs/40-pause-and-resume.md`).
        */
      resumed: Boolean = false
  ): Unit =
    TileRun(
      jobs,
      sessionManager,
      job,
      src,
      launch,
      reference,
      scale,
      runConfigurationId,
      request,
      target,
      rows,
      notes,
      derivation,
      prepareTile,
      correctTile,
      finish,
      keepTiles,
      finishTile,
      context,
      resumed
    ).run()
}

/** What a tiled job asks its server for, tile by tile: the one thing that
  * differs between the models that can fill a tile — the tiles, their order,
  * the blend, the pause and the picture on screen are the same for all.
  */
private[postprocess] enum TileRequests {

  /** An `img_gen` job a tile, built from the server's defaults and features
    * (PiD, redraw, edit).
    */
  case Images(
      make: (
          GenerationDefaults,
          Map[String, Boolean],
          TileWindow.TileInput
      ) => Either[String, ImageGenerationParameters]
  )

  /** The drift runner's `upscale` job a tile (SeedVR2, `specs/51`). */
  case Upscales(make: TileWindow.TileInput => NativeUpscale)
}

/** A run configuration's default LoRAs as a job without a form takes them
  * (`specs/28`, `specs/49`): the request entries, and the sampling they were
  * made for, the later LoRA's over the earlier ones'.
  */
final private[postprocess] case class ConfiguredLoras(
    selections: List[LoraSelection],
    sampling: LoraSampling
)

private[postprocess] object TiledJobs {

  /** What a job's tile loop ends with when it was asked to pause: not a
    * failure, and not a reason anybody reads (`specs/40-pause-and-resume.md`).
    */
  val Paused: String = "paused"

  /** The least overlap between neighbouring tiles, feather-blended — PiD's and
    * a redraw's, as the browser estimates them too.
    */
  val TileOverlap: Int = Tiling.Overlap

  /** The job log's line naming the LoRAs a job applies, if any. */
  def loraNote(loras: List[LoraSelection]): List[String] =
    Option
      .when(loras.nonEmpty)(
        "LoRAs: " + loras
          .map(selection => s"${selection.path} at ${selection.multiplier}")
          .mkString(", ")
      )
      .toList

  /** A random seed for a request that names none — the same draw as
    * `GenerationManager.randomSeed`, without depending on it.
    */
  def drawSeed(): Long =
    java.util.concurrent.ThreadLocalRandom
      .current()
      .nextInt(Int.MaxValue)
      .toLong
}
