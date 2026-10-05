package drift.backend.postprocess

import drift.backend.assistant.AssistantProxy
import drift.backend.postprocess.PostProcessImages.*
import drift.backend.storage.StorageService
import drift.shared.*

import java.awt.{List as _, *}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** Auto redraw's reading of a picture (`specs/52-auto-redraw.md`): one call to
  * a live assistant that reads images, on the whole picture scaled down with
  * the redraw's own tiles drawn and named on it. The answer says, tile by tile,
  * what materials it shows and how hard to repaint it, and lists what looks
  * broken — never per tile, whatever the picture's size.
  */
final private[postprocess] class RedrawPlanner(
    jobs: PostProcessJobs,
    storage: StorageService,
    assistant: AssistantProxy
) {

  def plan(
      date: String,
      fileName: String,
      request: RedrawPlanRequest
  ): Either[String, RedrawPlan] =
    for {
      src <- jobs.source(date, fileName)
      _ <- TiledArea.checkArea(request.tileSize, 1024, 64)
      system <- TiledArea.templateTextOf(
        storage,
        request.templateId,
        PromptKind.RedrawDiagnosis,
        PromptTemplate.DefaultRedrawDiagnosisId
      )
      image <-
        try
          Option(ImageIO.read(src.file.toFile))
            .toRight("cannot decode the source image")
        catch {
          case NonFatal(e) =>
            Left(s"cannot read the source image: ${e.getMessage}")
        }
      sessionId <- request.sessionId.fold(assistant.visionSession)(Right(_))
      rows = RedrawPlanner.tilesOf(
        image,
        request,
        sizeMultipleOf(request.runConfigurationId)
      )
      startedAt = System.nanoTime()
      reply <- assistant.ask(
        sessionId,
        system,
        RedrawPlanner.question(src.parent),
        dataUrl(RedrawPlanner.gridded(image, rows)),
        // a sentence a tile, and the repairs after them
        maxTokens = 1000 + 120 * rows.map(_.size).sum,
        temperature = 0.2
      )
      plan <- RedrawPlan.parse(
        reply,
        RedrawPlan.named(rows),
        image.getWidth,
        image.getHeight,
        (System.nanoTime() - startedAt) / 1e9
      )
    } yield plan

  /** The size multiple of the configuration's architecture, which the tiles sit
    * on — 16, drift's own least, when either is not found: the redraw itself
    * then refuses the configuration by name.
    */
  private def sizeMultipleOf(runConfigurationId: String): Int =
    storage
      .get[RunConfiguration]("run-configurations", runConfigurationId)
      .flatMap(configuration =>
        storage.get[Architecture]("architectures", configuration.architectureId)
      )
      .fold(16)(_.sizeMultiple)
}

private[postprocess] object RedrawPlanner {

  /** The tiles a whole-picture redraw of `image` cuts for these fields, row by
    * row — `TiledArea.of`, as `Redraw` calls it.
    */
  def tilesOf(
      image: BufferedImage,
      request: RedrawPlanRequest,
      sizeMultiple: Int
  ): List[List[ImageRegion]] =
    TiledArea
      .of(
        image,
        None,
        request.tileSize,
        1024,
        64,
        request.gridOffsetX,
        request.gridOffsetY,
        sizeMultiple
      )
      .rows
      .map(_.map(tile => ImageRegion(tile.x, tile.y, tile.width, tile.height)))

  /** What follows the template: the request, and the prompt the picture came
    * from when drift made it — what that asks for is not a defect.
    */
  def question(parent: Generation): String = {
    val prompt = parent.imageParameters.map(_.prompt.trim).filter(_.nonEmpty)
    "Inspect every cell of this picture." + prompt.fold("")(text =>
      s"\n\nThe prompt the picture was generated from: $text"
    )
  }

  /** `image` within `RedrawPlan.PictureSide`, the tiles drawn on it: a thin
    * line down the middle of each overlap between neighbours, and each tile's
    * name on a dark ground in its top left corner.
    */
  def gridded(
      image: BufferedImage,
      rows: List[List[ImageRegion]]
  ): BufferedImage = {
    val picture = copyOf(fitWithin(image, RedrawPlan.PictureSide))
    val scale = picture.getWidth.toDouble / image.getWidth
    // where one tile's cell ends and the next begins: the overlap's middle
    def boundaries(spans: List[(Int, Int)]): List[Int] =
      spans.zip(spans.drop(1)).map { case ((start, length), (next, _)) =>
        (start + length + next) / 2
      }
    val columns = boundaries(
      rows.headOption.toList.flatten.map(t => t.x -> t.width)
    )
    val lines = boundaries(rows.flatMap(_.headOption).map(t => t.y -> t.height))
    val lefts = 0 :: columns
    val tops = 0 :: lines
    val graphics = picture.createGraphics()
    try {
      graphics.setRenderingHint(
        RenderingHints.KEY_TEXT_ANTIALIASING,
        RenderingHints.VALUE_TEXT_ANTIALIAS_ON
      )
      graphics.setColor(Color.WHITE)
      graphics.setStroke(BasicStroke(1f))
      columns.foreach { x =>
        val at = math.round(x * scale).toInt
        graphics.drawLine(at, 0, at, picture.getHeight)
      }
      lines.foreach { y =>
        val at = math.round(y * scale).toInt
        graphics.drawLine(0, at, picture.getWidth, at)
      }
      val size = (RedrawPlan.PictureSide / 64).max(14)
      graphics.setFont(Font(Font.SANS_SERIF, Font.BOLD, size))
      val metrics = graphics.getFontMetrics
      for {
        (top, row) <- tops.zipWithIndex
        (left, column) <- lefts.zipWithIndex
      } {
        val name = RedrawPlan.nameOf(row, column)
        val x = math.round(left * scale).toInt + 3
        val y = math.round(top * scale).toInt + 3
        graphics.setColor(Color.BLACK)
        graphics.fillRect(x - 1, y - 1, metrics.stringWidth(name) + 4, size + 4)
        graphics.setColor(Color.WHITE)
        graphics.drawString(name, x + 1, y + metrics.getAscent)
      }
    } finally graphics.dispose()
    picture
  }
}
