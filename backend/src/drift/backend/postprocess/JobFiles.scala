package drift.backend.postprocess

import drift.shared.*

import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** Where a post-processing job's files are, and what can be read back from them
  * (`specs/15-post-hoc-resize.md`): its result beside its source, its log, its
  * tiles — kept beside the log, or in a directory of its own when the job keeps
  * them — and one tile scaled for the gallery to draw over the picture.
  *
  * Only paths and bytes: nothing here knows what state a job is in, which is
  * what makes a paused job's tiles readable from outside the job that made them
  * (`specs/40-pause-and-resume.md`).
  */
final private[postprocess] class JobFiles(
    outputsRoot: Path,
    val logsRoot: Path
) {

  /** Where a job writes its result: `<job id>-0.png` in its source's day. */
  def outputFileOf(job: PostProcessJob, src: PostProcessSource): Path =
    outputsRoot.resolve(src.date).resolve(s"${job.id}-0.png")

  def logFileOf(job: PostProcessJob): Path =
    logsRoot.resolve(s"postprocess-${job.id}.log")

  /** Where a job keeps its tiles when asked to. */
  def tilesDirOf(job: PostProcessJob): Path =
    logsRoot.resolve(s"postprocess-${job.id}-tiles")

  /** How many of a job's tiles are already on disk — where a resume starts. */
  def tilesDone(id: String, tiles: Int): Int =
    (0 until tiles).count(index =>
      tileOutputsOf(id, index).exists(Files.isRegularFile(_))
    )

  /** Where a tile's result waits between runs: beside the job log, or in the
    * job's tiles directory when it keeps its tiles — both, since only the job
    * itself knows which, and a paused job is read from outside it.
    */
  def tileOutputsOf(id: String, index: Int): List[Path] =
    List(
      logsRoot.resolve(s"postprocess-$id-tile-$index-output.png"),
      logsRoot
        .resolve(s"postprocess-$id-tiles")
        .resolve(f"tile-${index + 1}%02d-output.png")
    )

  /** One finished tile of a job, scaled to `side` px on its longest edge, as
    * PNG bytes — what the gallery paints over the picture while the job runs
    * (`specs/15-post-hoc-resize.md`). None while that tile is not done.
    */
  def tilePreview(
      id: String,
      index: Int,
      side: Int
  ): Option[(Array[Byte], String)] =
    tileOutputsOf(id, index)
      .find(Files.isRegularFile(_))
      .flatMap(file =>
        try
          Option(ImageIO.read(file.toFile)).map { tile =>
            val scaled = PostProcessImages.fitWithin(tile, side.max(64))
            val bytes = java.io.ByteArrayOutputStream()
            ImageIO.write(scaled, "png", bytes)
            (bytes.toByteArray, "image/png")
          }
        catch { case NonFatal(_) => None }
      )
}
