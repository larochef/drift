package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import sttp.tapir.*
import sttp.tapir.generic.auto.*

/** A session's output, and the progress drift reads out of it
  * (`specs/13-log-streaming.md`).
  *
  * The log is not only narrative: it is the **only** progress signal there is.
  * sd-server's job document carries `status` and `queue_position` and nothing
  * else - no step count, no percentage - so both bars the app wants are parsed
  * from stdout. That makes the parsing best-effort by construction: a pattern
  * that stops matching after an sd-cpp release must degrade to "no percentage",
  * never to a wrong one or a bar stuck at 40% while the work finishes.
  */

/** One captured line, with the wall-clock time drift read it. */
case class LogLine(at: Long, text: String)
object LogLine {
  given Schema[LogLine] = Schema.derived
  given JsonValueCodec[LogLine] = JsonCodecMaker.make
  given JsonValueCodec[List[LogLine]] = JsonCodecMaker.make
}

/** Which progress bar this is. They are told apart by their unit and nothing
  * else: tensors are counted at a transfer rate, sampling steps at seconds per
  * iteration, a decode in hundredths a second, and the inputs the drift runner
  * reads before the first step (a reference clip through the VAE) in tiles a
  * second.
  */
enum ProgressKind derives CanEqual {
  case Loading, Preparing, Sampling, Decoding

  def label: String = this match {
    case Loading   => "loading weights"
    case Preparing => "reading the inputs"
    case Sampling  => "sampling"
    case Decoding  => "decoding"
  }
}
object ProgressKind {
  given Schema[ProgressKind] =
    Schema.derivedEnumeration[ProgressKind].defaultStringBased
}

/** Where a session's current phase has got to. `detail` is sd-cpp's own tail of
  * the line ("6.77s/it", "16.95GB/s"), shown as written rather than
  * reformatted.
  */
case class SessionProgress(
    kind: ProgressKind,
    done: Int,
    total: Int,
    detail: String,
    /** The fraction done of the step under way, when the engine says: the drift
      * runner redraws a long step's bar as `3.42/8`.
      */
    within: Double = 0.0,
    /** Which pass this bar belongs to, when the model runs more than one -
      * "high noise" for wan 2.2's first expert. Two identical 20-step bars back
      * to back are otherwise indistinguishable (François, 2026-09-09).
      */
    note: Option[String] = None
) {
  def percent: Int =
    if (total <= 0) 0
    else (((done + within) / total) * 100).round.toInt.min(100)

  /** The seconds this bar still needs at the rate it shows, when it shows one
    * that says (a transfer rate does not: its total is counted in tensors).
    */
  def remainingSeconds: Option[Double] = {
    val left = total - done - within
    SessionProgress.Rate.findFirstMatchIn(detail).flatMap { found =>
      found.group(1).toDoubleOption.filter(_ > 0).map { rate =>
        if (found.group(2) == "s/it") left * rate else left / rate
      }
    }
  }

  /** `remainingSeconds` as the bar's caption says it: "about 4 min left for
    * this step" — the bar's own, not the generation's.
    */
  def remaining: Option[String] = remainingSeconds.map { seconds =>
    val time =
      if (seconds < 90) s"${math.max(1, seconds.round)} s"
      else s"${(seconds / 60).round} min"
    s"about $time left for this step"
  }
}
object SessionProgress {
  private val Rate = """([\d.]+)\s*(s/it|it/s|tiles/s|%/s)""".r

  given Schema[SessionProgress] = Schema.derived
}

/** Which image of a batch a session is on: `image` of `total`, counted from 1.
  * The bar in flight ([[SessionProgress]]) is that image's; this is the batch
  * around it, the way a tiled job counts its tiles around the tile's own bar.
  */
case class BatchProgress(image: Int, total: Int) {

  /** The images finished before the one in flight. */
  def completed: Int = (image - 1).max(0)
}
object BatchProgress {
  given Schema[BatchProgress] = Schema.derived
}

/** The picture a generation is sampling, as the model's last step estimated it
  * (`specs/54-step-preview.md`): small, never stored, gone with the result.
  * `image` is the sampling pass it belongs to, from 0 (the image of a batch,
  * the expert of a two-expert video model); `data` is the picture in base64, of
  * `mimeType` — a video's is a GIF that plays its estimate.
  */
case class StepPreview(
    image: Int,
    step: Int,
    steps: Int,
    mimeType: String,
    data: String
) {

  /** The picture as an `<img>` source. */
  def url: String = s"data:$mimeType;base64,$data"
}
object StepPreview {

  /** How a server is asked to draw what it is sampling (a request's `preview`):
    * sd-server's latent projection, the cheapest of its modes and the drift
    * runner's only one.
    */
  val Mode = "proj"
}

/** Which stage of a video's generation a session is on: `stage` of `total`,
  * counted from 1, and its `name` ("reading the references", "sampling",
  * "decoding the video"). The bar in flight ([[SessionProgress]]) is that
  * stage's; this is the generation around it, as [[BatchProgress]] is the batch
  * around an image.
  */
case class StageProgress(stage: Int, total: Int, name: String) {

  /** The stages finished before the one in flight. */
  def completed: Int = (stage - 1).max(0)
}
object StageProgress {
  given Schema[StageProgress] = Schema.derived
}

/** Reads sd-cpp's progress bars, which it redraws in place with a carriage
  * return and an ANSI erase - so a line-oriented reader sees each redraw as its
  * own line once a carriage return is treated as a terminator.
  *
  * The shapes, quoted from real runs (flux2, krea2 and minimax H3 on ROCm,
  * 2026-09-09), with the trailing escape written here as ESC-bracket-K:
  *
  * {{{
  *   |###                        | 15/298 - 16.95GB/s
  *   |============>              | 1/4 - 6.77s/it
  *   |=====>                     | 3/20 - 1.42it/s
  *   |=========                  | 34/100 - 1.87%/s
  * }}}
  *
  * The last is the drift runner's, for a video's VAE decode after the last
  * step: hundredths of the decode, at so many a second.
  *
  * A session prints **many** of these, not one: a load is several bars (one per
  * checkpoint), generation adds a sampling bar, and more tensors load lazily
  * between passes. Some models interleave two loads with different totals -
  * minimax H3 alternates a 624-tensor bar with a 416-tensor one. So the current
  * bar is whatever most recently redrew, and it carries its own totals; nothing
  * here tries to stitch a session's bars into one number, which could only be a
  * guess.
  */
object LogProgress {
  private val Ansi = "\\u001b\\[[0-9;]*[A-Za-z]".r

  /** `| done/total - detail`, the part of the bar that carries meaning. The
    * fill characters differ between the two bars (`#` and `=>`) and are not
    * matched on: only the counts and the unit are load-bearing.
    */
  /** The units are matched by name rather than "everything up to a space",
    * because sd-cpp does not always follow a redraw with a newline: minimax H3
    * writes its ANSI erase and then the next `[INFO   ] ...` line straight
    * after, so a greedy match reads the unit as `5.4s/it[INFO` and the bar
    * silently never appears (François, 2026-09-09). Stopping at the unit also
    * leaves the narrative behind, which [[parseWithRest]] hands back.
    */
  private val Bar =
    """\|\s*(\d+)(\.\d+)?\s*/\s*(\d+)\s+-\s+([\d.]+\s*(?:s/it|it/s|tiles/s|[KMGTP]?B/s|%/s))""".r

  /** The escape codes out, so a line can be displayed and matched. */
  def clean(text: String): String = Ansi.replaceAllIn(text, "").trim

  /** The progress a line reports, if it reports any. Anything whose unit is not
    * one of the two known ones is narrative, not progress - guessing would be
    * how a wrong percentage gets on screen.
    */
  def parse(text: String): Option[SessionProgress] =
    parseWithRest(text).map(_._1)

  /** The progress a line reports and whatever followed it on the same line -
    * the narrative sd-cpp sometimes glues onto the end of a redraw, which
    * belongs in the log as a line of its own.
    */
  def parseWithRest(text: String): Option[(SessionProgress, String)] =
    Bar.findFirstMatchIn(clean(text)).flatMap { found =>
      val done = found.group(1).toIntOption
      val within =
        Option(found.group(2)).flatMap(_.toDoubleOption).getOrElse(0.0)
      val total = found.group(3).toIntOption
      val detail = found.group(4)
      val rest = clean(text).substring(found.end).trim
      (done, total) match {
        case (Some(d), Some(t)) if t > 0 && d >= 0 =>
          // sd-cpp formats the sampling rate adaptively: `s/it` while a step
          // takes over a second, `it/s` once it is quicker - and it switches
          // mid-run, redraw to redraw. Reading only one of them means ignoring
          // about half the updates of a fast sampler, which looks like a bar
          // that sticks and then jumps.
          if (detail.endsWith("s/it") || detail.endsWith("it/s"))
            Some(
              (
                SessionProgress(ProgressKind.Sampling, d, t, detail, within),
                rest
              )
            )
          // The transfer rate is formatted adaptively too (MB/s, then GB/s),
          // which is why the unit is matched by suffix rather than in full.
          else if (detail.endsWith("tiles/s"))
            Some((SessionProgress(ProgressKind.Preparing, d, t, detail), rest))
          else if (detail.endsWith("%/s"))
            Some((SessionProgress(ProgressKind.Decoding, d, t, detail), rest))
          else if (detail.endsWith("B/s"))
            Some((SessionProgress(ProgressKind.Loading, d, t, detail), rest))
          else None
        case _ => None
      }
    }

  /** The pass a `sampling(...)` line announces - "high noise" for wan 2.2's
    * first expert - or `None` for a plain `sampling using ...`, which is the
    * ordinary single-pass case.
    */
  private val SamplingPass = """sampling\(([^)]+)\)""".r

  def samplingPassOf(text: String): Option[Option[String]] = {
    val line = clean(text)
    if (!line.contains("sampling")) None
    else
      SamplingPass.findFirstMatchIn(line) match {
        case Some(found) => Some(Some(found.group(1).trim))
        case None if line.contains("sampling using") => Some(None)
        case None                                    => None
      }
  }

  /** The line that opens each image of a batch, as both engines write it
    * (session logs, 2026-10-03):
    *
    * {{{
    *   generating image: 2/4 - seed 43      sd-cpp
    *   generating image 2/4 (seed 43)       the drift runner
    * }}}
    */
  private val BatchImage = """generating image:?\s*(\d+)\s*/\s*(\d+)""".r

  def batchOf(text: String): Option[BatchProgress] =
    BatchImage.findFirstMatchIn(clean(text)).flatMap { found =>
      (found.group(1).toIntOption, found.group(2).toIntOption) match {
        case (Some(image), Some(total)) if image >= 1 && image <= total =>
          Some(BatchProgress(image, total))
        case _ => None
      }
    }

  /** The line that opens each stage of a video, as the drift runner writes it:
    * `stage 2/5: sampling`.
    */
  private val Stage = """^stage\s+(\d+)\s*/\s*(\d+):\s*(.+)$""".r

  def stageOf(text: String): Option[StageProgress] =
    Stage.findFirstMatchIn(clean(text)).flatMap { found =>
      (found.group(1).toIntOption, found.group(2).toIntOption) match {
        case (Some(stage), Some(total)) if stage >= 1 && stage <= total =>
          Some(StageProgress(stage, total, found.group(3).trim))
        case _ => None
      }
    }

  /** Whether a line opens or closes a job: its stages are over with it, or not
    * begun.
    */
  def endsStages(text: String): Boolean = {
    val line = clean(text)
    line.startsWith("generating ") || line.contains(
      "generate_video completed"
    ) ||
    line.contains("generate_image completed") ||
    line.contains("generation failed") || line.contains("generation cancelled")
  }

  /** Whether a line closes an `img_gen` job - sd-cpp's own closing line, which
    * the drift runner prints in the same shape. The batch is over with it.
    */
  def endsBatch(text: String): Boolean =
    clean(text).contains("generate_image completed")

  /** Whether a line reads like something went wrong - what a failed session
    * surfaces instead of making the user scroll. sd-cpp tags its own levels;
    * the bare forms catch what the runtime's libraries print.
    */
  def looksLikeError(text: String): Boolean = {
    val line = clean(text)
    line.startsWith("[ERROR") || line.contains("error:") ||
    line.contains("Error:") || line.contains("failed to") ||
    line.contains("out of memory") || line.contains("HIP error")
  }
}
