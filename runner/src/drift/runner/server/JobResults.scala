package drift.runner.server

import drift.protocol.*
import drift.runner.diffusion.*
import drift.runner.ops.Ops

import java.io.ByteArrayOutputStream
import java.util.Base64
import javax.imageio.ImageIO

/** What a job makes once the worker runs it: an `img_gen` job's images, a
  * `vid_gen` job's video, and while they sample, the step bar in the log and
  * the previews in the document of the job `queue` is generating.
  */
final private[server] class JobResults(queue: JobQueue, ops: Ops) {

  /** A step's progress line, as sd-cpp prints it: step 0 is the empty bar a
    * pass starts with, which says the sampling has begun (and how many steps it
    * is) while the first step, minutes long on a large video, is under way.
    */
  def stepPrinter(): (Int, Int) => Unit = (new StepBar).step

  /** A video's last stage, after the pipeline's own. */
  private val VideoFileStage = "writing the file"

  /** A pass's bar: a line a step, and while a step lasts, the same line redrawn
    * with the fraction of it done (`3.42/8`, at the last step's rate, or during
    * the first step at the rate it is heading for). The fraction is the
    * pipeline's own when it tells one (`within`), else the work the backend was
    * given since the step began over what the last whole step took (`work`):
    * any model's, from its second step on.
    */
  final private class StepBar {
    private var last = System.nanoTime()
    private var drawn = last
    private var at = (0, 0)
    private var rate = 0.0
    // the backend's work since the step began, and the last whole step's
    private var worked = 0L
    private var perStep = 0L
    private var told = false
    private var shown = 0.0

    def step(step: Int, steps: Int): Unit = {
      // Between two steps is where a generation can be left: every pipeline
      // reports them here, and none holds a kernel across the call.
      if (queue.generating.exists(_.cancelRequested)) throw new JobCancelled
      if (steps > 0) {
        val now = System.nanoTime()
        // a call may count several steps at once (a pass on more pixels)
        rate =
          if (step == 0) 0.0
          else (now - last) / 1e9 / math.max(1, step - at._1)
        ProgressReport.bar(JobBar.Sampling, step, steps, rate, "s/it")
        last = now
        drawn = now
        at = (step, steps)
        if (step > 0) perStep = worked
        worked = 0
        shown = 0.0
      }
    }

    /** The fraction done of the step under way, as the pipeline tells it. */
    def within(fraction: Double): Unit = {
      told = true
      draw(fraction)
    }

    /** A piece of work given to the backend (`Ops.onWork`). */
    def work(): Unit = {
      worked += 1
      if (!told && perStep > 0 && (worked & 15) == 0)
        draw(math.min(0.99, worked.toDouble / perStep))
    }

    /** Redraws the bar at `fraction` of the step under way, a second apart. */
    private def draw(fraction: Double): Unit = {
      val (step, steps) = at
      val now = System.nanoTime()
      if (
        step < steps && fraction > shown && fraction < 1 && now - drawn >= 1e9
      ) {
        shown = fraction
        val done = step + fraction
        val seconds = if (rate > 0) rate else (now - last) / 1e9 / fraction
        ProgressReport.bar(JobBar.Sampling, done, steps, seconds, "s/it")
        drawn = now
      }
    }
  }

  /** An `img_gen` job's result: `count` images of `request`, seeds in sequence.
    */
  def images(
      pipeline: ImagePipeline,
      request: ImageRequest,
      count: Int
  ): NativeJobResult = {
    ProgressReport.pass(None)
    val startedAt = System.nanoTime()
    val pngs = (0 until count).map { index =>
      val seeded = request.copy(seed = request.seed + index)
      ProgressReport.image(index + 1, count, s"(seed ${seeded.seed})")
      val bar = new StepBar
      ops.onWork = bar.work
      // the picture being made has no estimate yet, whatever the last one had
      queue.generating.foreach(_.preview = None)
      val publish = previews()
      pipeline.preview = (step, steps, picture) =>
        publish(
          index + 1,
          step,
          steps,
          () => {
            val jpeg = new ByteArrayOutputStream()
            ImageIO.write(picture(), "jpg", jpeg)
            jpeg.toByteArray
          }
        )
      val image =
        try pipeline.generate(seeded, bar.step)
        finally {
          ops.onWork = () => ()
          pipeline.preview = (_, _, _) => ()
          queue.generating.foreach(_.preview = None)
        }
      val png = new ByteArrayOutputStream()
      ImageIO.write(image, "png", png)
      png.toByteArray
    }
    // sd-cpp's closing line, which is what drift ends the batch count on.
    val seconds = (System.nanoTime() - startedAt) / 1e9
    println(f"generate_image completed in $seconds%.2fs")
    NativeJobResult(
      outputFormat = "png",
      images = pngs.zipWithIndex.map { (png, index) =>
        NativeJobImage(index, Base64.getEncoder.encodeToString(png))
      }.toList
    )
  }

  /** The least time between two previews of a picture. */
  private val PreviewIntervalNanos = 1_000_000_000L

  /** What a pipeline's `preview` hook hands a step's preview to, for one
    * picture or video: `publish(pass, step, steps, picture)` puts it in the
    * document of the job under way when its request asked, as sd-server gives
    * one — the sampling pass from 1 (an image of the batch, a video's expert),
    * the step it is the estimate of, the pass' steps and the picture's file, a
    * JPEG or a GIF here, a PNG there. One a second at most, and a pass' last
    * step always: a poll comes once a second, and steps shorter than that would
    * pay for pictures nobody sees.
    */
  private def previews(): (Int, Int, Int, () => Array[Byte]) => Unit = {
    var drawnAt = 0L
    (pass, step, steps, picture) =>
      queue.generating.filter(_.previewWanted).foreach { job =>
        val now = System.nanoTime()
        if (step == steps || now - drawnAt >= PreviewIntervalNanos) {
          val file = picture()
          job.preview = Some(
            NativeJobPreview(
              pass,
              step,
              steps,
              Base64.getEncoder.encodeToString(file)
            )
          )
          drawnAt = now
        }
      }
  }

  /** A `vid_gen` job's result: one webm. */
  def video(
      pipeline: VideoPipeline,
      request: VideoRequest
  ): NativeJobResult = {
    ProgressReport.pass(None)
    println(
      s"generating ${pipeline.alignedFrames(request.frames)} frames of ${request.width} × ${request.height} (seed ${request.seed})"
    )
    val startedAt = System.nanoTime()
    // the pipeline's stages and the file's, each announced as it starts
    val stages = pipeline.stages(request) :+ VideoFileStage
    val enter = (name: String) => {
      val index = stages.indexOf(name)
      if (index >= 0) ProgressReport.stage(index + 1, stages.size, name)
      // the estimate stays from a sampling pass to the next, and leaves when
      // the sampling is over: the decode shows none
      if (!VideoPipeline.samples(name))
        queue.generating.foreach(_.preview = None)
    }
    val bar = new StepBar
    val publish = previews()
    pipeline.stage = enter
    pipeline.withinStep = bar.within
    pipeline.preview = (pass, step, steps, clip) =>
      publish(pass, step, steps, () => PreviewGif.encode(clip()))
    ops.onWork = bar.work
    val video =
      try pipeline.generate(request, bar.step)
      finally {
        pipeline.stage = _ => ()
        pipeline.withinStep = _ => ()
        pipeline.preview = (_, _, _, _) => ()
        ops.onWork = () => ()
        queue.generating.foreach(_.preview = None)
      }
    enter(VideoFileStage)
    val webm = VideoFiles.webm(video)
    println(
      f"generate_video completed in ${(System.nanoTime() - startedAt) / 1e9}%.2fs"
    )
    NativeJobResult(
      outputFormat = "webm",
      mimeType = Some("video/webm"),
      fps = Some(video.fps),
      frameCount = Some(video.frames.size),
      b64Json = Some(Base64.getEncoder.encodeToString(webm))
    )
  }
}
