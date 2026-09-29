package drift.backend.postprocess

import drift.backend.lora.LoraManager
import drift.backend.runtime.RuntimeManager
import drift.backend.sdserver.GenerationHistory
import drift.backend.session.SessionManager
import drift.backend.storage.StorageService
import drift.backend.upscale.UpscalerManager
import drift.shared.*

import java.nio.file.Path

/** Post-hoc processing of gallery images (`specs/15-post-hoc-resize.md`,
  * `specs/26-tiled-pid.md`, `specs/27-redraw.md`, `specs/39-seamless-edit.md`).
  * Every job writes a *derived* gallery entry beside its source: a new output
  * file and sidecar in the source's date directory, the sidecar carrying a
  * [[Derivation]] naming the parent and the operation. The source is never
  * touched.
  *
  * The kinds live beside this file — `EsrganUpscale`, and `PidUpscale`,
  * `Redraw` and `Edit` through `TiledJobs` — on what every job shares,
  * `PostProcessJobs`, and the image helpers of `PostProcessImages`
  * (`specs/29-split-oversized-files.md`).
  */
final class PostProcessManager(
    storage: StorageService,
    outputsRoot: Path,
    logsRoot: Path,
    history: GenerationHistory,
    upscalerManager: UpscalerManager,
    loraManager: LoraManager,
    runtimeManager: RuntimeManager,
    sessionManager: SessionManager
) {
  private val jobs = PostProcessJobs(outputsRoot, logsRoot, history, storage)
  jobs.loadPaused()
  private val tiles =
    TiledJobs(jobs, runtimeManager, sessionManager, loraManager)
  private val esrgan = EsrganUpscale(jobs, upscalerManager, runtimeManager)
  private val pid = PidUpscale(tiles)
  private val redraws = Redraw(jobs, tiles, storage)
  private val edits = Edit(tiles, storage)

  def listJobs: List[PostProcessJob] = jobs.list

  def upscale(
      date: String,
      fileName: String,
      request: UpscaleRequest
  ): PostProcessJob = esrgan.start(date, fileName, request)

  def pidUpscale(
      date: String,
      fileName: String,
      request: PidUpscaleRequest
  ): PostProcessJob = pid.start(date, fileName, request)

  def redraw(
      date: String,
      fileName: String,
      request: RedrawRequest
  ): PostProcessJob = redraws.start(date, fileName, request)

  def edit(
      date: String,
      fileName: String,
      request: EditRequest
  ): PostProcessJob = edits.start(date, fileName, request)

  /** Cancels a running or paused job (`PostProcessJobs.cancel`); false when no
    * such job has that id.
    */
  def cancelJob(id: String): Boolean = jobs.cancel(id)

  /** Pauses a running tiled job after the tile in flight
    * (`specs/40-pause-and-resume.md`); false when no running tiled job has that
    * id.
    */
  def pauseJob(id: String, force: Boolean): Boolean = jobs.pause(id, force)

  /** The picture a tiled job has made so far, drawn over its source. */
  def picture(id: String, side: Option[Int]): Option[Array[Byte]] =
    jobs.picture(
      id,
      side.filter(_ > 0).map(_.min(PostProcessPicture.ScreenSide))
    )

  /** Carries a paused job on where it stopped, from what was stored when it
    * paused — the request as it ran, its drawn seed included.
    */
  def resumeJob(id: String): PostProcessJob =
    jobs.pausedWork(id) match {
      case None =>
        jobs.refused("resume", "", "", s"'$id' is not a paused job")
      case Some(paused) =>
        paused.work match {
          case PausedWork.Pid(request) =>
            pid.start(
              paused.sourceDate,
              paused.sourceFileName,
              request,
              Some(id)
            )
          case PausedWork.Redraw(request) =>
            redraws.start(
              paused.sourceDate,
              paused.sourceFileName,
              request,
              Some(id)
            )
          case PausedWork.Edit(request) =>
            edits.start(
              paused.sourceDate,
              paused.sourceFileName,
              request,
              Some(id)
            )
        }
    }
}
