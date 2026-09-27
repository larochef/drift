package drift.backend.postprocess

import drift.backend.runtime.{LaunchRuntime, RuntimeManager}
import drift.backend.upscale.UpscalerManager
import drift.shared.*

import java.nio.file.*
import scala.util.control.NonFatal

/** Upscale with a model of the upscaler store (`specs/15-post-hoc-resize.md`):
  * one `sd-cli --mode upscale` spawn on a runtime — no session, no diffusion
  * weights, seconds to start. Its exit code is worthless: when the model fails
  * to load, sd-cli still exits 0 and "saves" the input unchanged (verified with
  * an x2plus model this build cannot read). Success is therefore judged on the
  * log having no `[ERROR]` and the output being larger than the input.
  */
final private[postprocess] class EsrganUpscale(
    jobs: PostProcessJobs,
    upscalerManager: UpscalerManager,
    runtimeManager: RuntimeManager
) {

  def start(
      date: String,
      fileName: String,
      request: UpscaleRequest
  ): PostProcessJob =
    jobs
      .busyWith(date, fileName)
      .toLeft(())
      .flatMap(_ => jobs.source(date, fileName)) match {
      case Left(reason) => jobs.refused("upscale", date, fileName, reason)
      case Right(src)   =>
        val upscaler = upscalerManager.list.find(_.id == request.upscalerId)
        val modelFile =
          upscaler.map(u => upscalerManager.upscaleRoot.resolve(u.fileName))
        val runtime =
          runtimeManager.resolveForLaunch(
            RuntimeTool.SdCpp,
            // sd-cli upscales; only sd-cpp ships one
            RuntimeEngine.SdCpp,
            request.runtimeId
          )
        (upscaler, modelFile.filter(Files.isRegularFile(_)), runtime) match {
          case (None, _, _) =>
            jobs.refused(
              "upscale",
              date,
              fileName,
              s"upscaler '${request.upscalerId}' is not installed"
            )
          case (Some(u), None, _) =>
            jobs.refused(
              "upscale",
              date,
              fileName,
              s"upscaler '${u.label}' has no file on disk (${u.fileName})"
            )
          case (_, _, Left(reason)) =>
            jobs.refused("upscale", date, fileName, reason)
          case (Some(u), Some(model), Right(launch)) =>
            RuntimeManager.sdCliOf(launch) match {
              case Left(reason) =>
                jobs.refused("upscale", date, fileName, reason)
              case Right(_) if request.repeats < 1 || request.repeats > 4 =>
                jobs.refused(
                  "upscale",
                  date,
                  fileName,
                  "repeats must be 1 to 4"
                )
              case Right(cli) =>
                jobs.start("upscale", src)(job =>
                  run(job, src, u, model, cli, launch, request)
                )
            }
        }
    }

  private def run(
      job: PostProcessJob,
      src: PostProcessSource,
      upscaler: Upscaler,
      model: Path,
      cli: Path,
      launch: LaunchRuntime,
      request: UpscaleRequest
  ): Unit = {
    val outputFile = jobs.files.outputFileOf(job, src)
    val command =
      List(
        cli.toString,
        "--mode",
        "upscale",
        "--upscale-model",
        model.toString,
        "--init-img",
        src.file.toString,
        "--output",
        outputFile.toString,
        "--upscale-repeats",
        request.repeats.toString
      ) ++ request.tileSize.toList.flatMap(t =>
        List("--upscale-tile-size", t.toString)
      )
    runCommand(
      job,
      src,
      command,
      launch,
      verifySize = (sourceSize, outputSize) =>
        if (outputSize._1 <= sourceSize._1 && outputSize._2 <= sourceSize._2)
          Some(
            s"the output (${outputSize._1}×${outputSize._2}) is no larger than the input (${sourceSize._1}×${sourceSize._2}) — the model did not run"
          )
        else None,
      derivation = (w, h) =>
        Derivation(
          parentId = src.parent.id,
          parentDate = src.date,
          parentFileName = src.fileName,
          operation = "upscale",
          upscalerId = Some(upscaler.id),
          repeats = Some(request.repeats),
          width = Some(w),
          height = Some(h)
        )
    )
  }

  /** Runs the command, which names its output `<job id>-0.png` in the source's
    * day, and judges it: a failed run, a missing output or a size `verifySize`
    * rejects fails the job, deletes the output and attaches the log tail.
    */
  private def runCommand(
      job: PostProcessJob,
      src: PostProcessSource,
      command: List[String],
      launch: LaunchRuntime,
      verifySize: ((Int, Int), (Int, Int)) => Option[String],
      derivation: (Int, Int) => Derivation
  ): Unit = {
    val outputFile = jobs.files.outputFileOf(job, src)
    try {
      jobs.startLog(job, launch, Nil)
      jobs
        .spawn(job, command, launch)
        .toLeft(())
        .flatMap(_ =>
          Option
            .when(Files.isRegularFile(outputFile))(outputFile)
            .flatMap(PostProcessImages.imageSize)
            .toRight("sd-cli wrote no output image")
            .flatMap(outputSize =>
              PostProcessImages
                .imageSize(src.file)
                .flatMap(verifySize(_, outputSize))
                .toLeft(outputSize)
            )
        ) match {
        case Left(reason) =>
          Files.deleteIfExists(outputFile)
          jobs.fail(job, reason)
        case Right((width, height)) =>
          jobs.complete(job, src, outputFile, derivation(width, height))
      }
    } catch {
      case NonFatal(err) =>
        Files.deleteIfExists(outputFile)
        jobs.fail(job, s"running sd-cli failed: ${err.getMessage}")
    }
  }
}
