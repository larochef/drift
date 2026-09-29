package drift.backend.postprocess

import drift.shared.*

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, StandardCopyOption}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal

/** Where a post-processing job's files are, and what can be read back from them
  * (`specs/15-post-hoc-resize.md`): its result beside its source, its log, its
  * tiles — kept beside the log, or in a directory of its own when the job keeps
  * them — and the picture a paused job had made, kept for the gallery.
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

  /** Where a paused job keeps the picture it had made when it paused
    * (`LivePicture`), and the copies of it scaled for the screen.
    */
  def pictureFileOf(id: String): Path =
    logsRoot.resolve(s"postprocess-$id-picture.png")

  private def scaledPictureOf(id: String, side: Int): Path =
    logsRoot.resolve(s"postprocess-$id-picture-$side.png")

  /** Where a running job's picture is written at full size when it is asked
    * for (`LivePicture.fullSize`).
    */
  def livePictureFileOf(id: String): Path =
    logsRoot.resolve(s"postprocess-$id-picture-live.png")

  /** Keeps a paused job's picture, and the copy of it `screen` is for the
    * screen, replacing the ones an earlier pause kept.
    */
  def storePicture(
      id: String,
      picture: BufferedImage,
      screen: BufferedImage
  ): Unit = {
    deletePicture(id)
    write(screen, scaledPictureOf(id, PostProcessPicture.ScreenSide))
    write(picture, pictureFileOf(id))
  }

  private def write(image: BufferedImage, file: Path): Unit = {
    val partial = file.resolveSibling(file.getFileName.toString + ".part")
    ImageIO.write(image, "png", partial.toFile)
    Files.move(partial, file, StandardCopyOption.REPLACE_EXISTING)
  }

  /** Drops a job's kept picture and its scaled copies — it resumed, ended, or
    * was cancelled.
    */
  def deletePicture(id: String): Unit =
    if (Files.isDirectory(logsRoot))
      Using.resource(Files.list(logsRoot))(
        _.iterator.asScala
          .filter(
            _.getFileName.toString.startsWith(s"postprocess-$id-picture")
          )
          .foreach(Files.deleteIfExists)
      )

  /** A paused job's kept picture as PNG bytes: at most `side` px on its longest
    * edge — scaled once, then read back — or at full size. None when the job
    * kept none.
    */
  def storedPicture(id: String, side: Option[Int]): Option[Array[Byte]] = {
    val full = pictureFileOf(id)
    if (!Files.isRegularFile(full)) None
    else
      try
        side match {
          case None          => Some(Files.readAllBytes(full))
          case Some(longest) =>
            val scaled = scaledPictureOf(id, longest)
            if (!Files.isRegularFile(scaled))
              Option(ImageIO.read(full.toFile)).foreach { picture =>
                val partial =
                  scaled.resolveSibling(scaled.getFileName.toString + ".part")
                ImageIO.write(
                  PostProcessImages.fitWithin(picture, longest),
                  "png",
                  partial.toFile
                )
                Files.move(partial, scaled, StandardCopyOption.REPLACE_EXISTING)
              }
            Option.when(Files.isRegularFile(scaled))(Files.readAllBytes(scaled))
        }
      catch { case NonFatal(_) => None }
  }
}
