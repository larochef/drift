package drift.backend.postprocess

import drift.backend.storage.StorageService
import drift.shared.*

import java.awt.image.BufferedImage

/** What a tiled job works on, before any tile runs (`specs/27-redraw.md`,
  * `specs/39-seamless-edit.md`): the fields a request has to get right, the
  * selection checked against the picture, the prompt template it is told its
  * job by, and the window and tiles laid over it.
  *
  * A redraw and an edit ask these of the same code so they cut a picture the
  * same way, and the browser lays out the very same tiles to price a job before
  * it runs (`shared/.../Tiling`).
  */
private[postprocess] object TiledArea {

  /** The tile, window and margin fields a redraw and an edit share, checked. */
  def checkArea(
      tileSize: Int,
      minimumWindowSide: Int,
      selectionMargin: Int
  ): Either[String, Unit] =
    for {
      _ <- Either.cond(
        tileSize >= 512 && tileSize <= 2048 && tileSize % 16 == 0,
        (),
        "the tile size must be a multiple of 16 from 512 to 2048 px"
      )
      _ <- Either.cond(
        minimumWindowSide >= 256 && minimumWindowSide <= 4096,
        (),
        "the smallest window must be from 256 to 4096 px"
      )
      _ <- Either.cond(
        selectionMargin >= 0 && selectionMargin <= 512,
        (),
        "the margin around a selection must be from 0 to 512 px"
      )
    } yield ()

  /** The shortest side a selection may have: a stray click on the image is not
    * a request to repaint four pixels.
    */
  val MinimumRegionSide: Int = 16

  /** The part of the source to work on, checked against it — none is the whole
    * image.
    */
  def regionOf(
      region: Option[ImageRegion],
      width: Int,
      height: Int
  ): Either[String, Option[ImageRegion]] =
    region match {
      case None => Right(None)
      case Some(selection)
          if selection.width < MinimumRegionSide ||
            selection.height < MinimumRegionSide =>
        Left(s"the selection must be at least $MinimumRegionSide px on a side")
      case Some(selection)
          if selection.x < 0 || selection.y < 0 ||
            selection.x + selection.width > width ||
            selection.y + selection.height > height =>
        Left("the selection must lie inside the image")
      case Some(selection) => Right(Some(selection))
    }

  /** The text of a prompt template of `kind` (`specs/32-prompt-library.md`):
    * the request's, else the built-in `defaultId`. A missing one, or one of
    * another kind, fails the job rather than sending nothing — a redraw is told
    * its job by a restoration template, an edit by an edit template, and the
    * model sees the picture either way, so the source's prompt is not sent.
    */
  def templateTextOf(
      storage: StorageService,
      templateId: Option[String],
      kind: PromptKind,
      defaultId: String
  ): Either[String, String] = {
    val id = templateId.getOrElse(defaultId)
    storage.get[PromptTemplate]("prompt-templates", id) match {
      case Some(template) if template.kind == kind => Right(template.text)
      case Some(template)                          =>
        Left(s"prompt template '${template.label}' is not a $kind template")
      case None => Left(s"unknown prompt template '$id'")
    }
  }

  /** What a redraw or an edit works on: `area`, the whole image or the window a
    * selection is worked through; `reference`, that part of the source padded
    * to the model's multiple; and the tiles laid over it.
    */
  case class Area(
      area: ImageRegion,
      reference: BufferedImage,
      rows: List[List[Tiling.Tile]]
  )

  /** The `Area` of `image` for `region`. Every side drift asks for — the
    * window, the padding, each tile — is a multiple of `sizeMultiple`, because
    * sd-cpp aligns the request up to it and answers with the bigger image,
    * which the tile check then refuses. The window is wide enough for the
    * model, with the margin the paste feathers over; the grid is the one the
    * browser drew over the picture, moved the same way.
    */
  def of(
      image: BufferedImage,
      region: Option[ImageRegion],
      tileSize: Int,
      minimumWindowSide: Int,
      selectionMargin: Int,
      gridOffsetX: Int,
      gridOffsetY: Int,
      sizeMultiple: Int
  ): Area = {
    val area =
      region.fold(ImageRegion(0, 0, image.getWidth, image.getHeight))(
        selection =>
          Tiling.window(
            selection,
            image.getWidth,
            image.getHeight,
            minimumWindowSide,
            selectionMargin,
            multiple = sizeMultiple
          )
      )
    val reference = PostProcessImages.padded(
      image.getSubimage(area.x, area.y, area.width, area.height),
      Tiling.roundUp(area.width, sizeMultiple),
      Tiling.roundUp(area.height, sizeMultiple)
    )
    val rows = Tiling.layout(
      reference.getWidth,
      reference.getHeight,
      // `layout` wants a tile size on the multiple too, and the field is only
      // held to 16.
      Tiling.roundUp(tileSize, sizeMultiple),
      TiledJobs.TileOverlap,
      multiple = sizeMultiple,
      align = 1,
      // The shift is already on the multiple; the axis takes it modulo the
      // stride.
      offsetX = Tiling.roundUp(gridOffsetX, sizeMultiple),
      offsetY = Tiling.roundUp(gridOffsetY, sizeMultiple)
    )
    Area(area, reference, rows)
  }
}
