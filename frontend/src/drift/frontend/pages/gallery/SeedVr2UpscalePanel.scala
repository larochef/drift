package drift.frontend.pages.gallery

import drift.frontend.components.{Component, LaunchOrDownload}
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the upscale task shows for a SeedVR2 model
  * (`specs/51-seedvr2-upscaling.md`): the scale, a seed, and the button — for a
  * picture and for a video alike. One diffusion step on the drift runner, no
  * prompt; it restores harder than the other upscalers and says so. A picture
  * past one pass is cut in tiles (`SeedVr2UpscaleRequest.tilesFor`), which the
  * line before the button counts and the viewer can draw.
  */
class SeedVr2UpscalePanel(
    output: Signal[Option[GenerationOutput]],
    /** The chosen configuration: the task's model select writes it. */
    configurationVar: Var[String],
    /** The picture on screen, for the size it really has. */
    viewed: Signal[Option[ViewedImage]],
    /** Where this panel puts the tiles its job would run, in the picture's own
      * pixels.
      */
    gridTiles: Var[List[ImageRegion]],
    showTileGrid: Var[Boolean],
    prerequisites: LaunchPrerequisites,
    onSeedVr2: (GenerationOutput, SeedVr2UpscaleRequest) => Unit
) extends Component {

  private val scaleVar = Var(SeedVr2UpscaleRequest.DefaultScale)
  private val seedVar = Var("")
  private val advancedVar = Var(false)

  private val isVideo: Signal[Boolean] =
    output.map(_.exists(GenerationMediaViewer.isVideo)).distinct

  /** The size a picture would come out at, or why the job is refused. */
  private val planned: Signal[Option[Either[String, (Int, Int)]]] =
    viewed
      .combineWith(scaleVar.signal, isVideo)
      .map((shown, scale, video) =>
        shown
          .filterNot(_ => video)
          .map(picture =>
            SeedVr2UpscaleRequest
              .target((picture.width, picture.height), scale)
          )
      )
      .distinct

  /** The tiles a picture's job would run, in the picture's own pixels. */
  private val tiles: Signal[List[ImageRegion]] =
    viewed
      .combineWith(scaleVar.signal, planned)
      .map { (shown, scale, outcome) =>
        (shown, outcome) match {
          case (Some(picture), Some(Right(_))) =>
            SeedVr2UpscaleRequest
              .tilesFor((picture.width, picture.height), scale)
              .flatten
              .map(tile =>
                ImageRegion(
                  tile.x / scale,
                  tile.y / scale,
                  tile.width / scale,
                  tile.height / scale
                )
              )
          case _ => List.empty
        }
      }
      .distinct

  private val summary: Signal[String] =
    isVideo
      .combineWith(scaleVar.signal, planned, tiles)
      .map { (video, scale, outcome, laid) =>
        if (video) s"→ ×$scale · the whole video, batch by batch: minutes"
        else
          outcome match {
            case Some(Right((width, height))) =>
              s"→ ×$scale of the source · $width × $height" +
                (if (laid.size == 1) " · 1 tile" else s" · ${laid.size} tiles")
            case Some(Left(reason)) => s"→ ×$scale — $reason"
            case None               => s"→ ×$scale of the source"
          }
      }
      .distinct

  private val refused: Signal[Boolean] =
    planned.map(_.exists(_.isLeft)).distinct

  private def scaleSelect: HtmlElement =
    div(
      cls := "select is-small",
      select(
        onChange.mapToValue.map(_.toInt) --> scaleVar,
        SeedVr2UpscaleRequest.Scales.map(scale =>
          option(
            value := scale.toString,
            selected <-- scaleVar.signal.map(_ == scale),
            s"×$scale"
          )
        )
      )
    )

  lazy val element: HtmlElement = div(
    tiles.changes --> gridTiles,
    intro(
      "SeedVR2 restores and enlarges a picture or a video in one diffusion " +
        "step, on the drift runner — a video keeps its frame rate and its " +
        "soundtrack. The original stays in the gallery."
    ),
    group("size", plainField(scaleSelect)),
    advanced(advancedVar, group("pass", seedField(seedVar))),
    foot(
      div(
        cls := "post-foot-lines",
        div(child.text <-- summary),
        div(
          cls := "post-foot-controls",
          cls("is-hidden") <-- isVideo,
          checkField(
            showTileGrid,
            "show the grid",
            "draw the tiles this upscale would run over the picture — one " +
              "pass of the model each"
          )
        )
      ),
      LaunchOrDownload(
        prerequisites.of(configurationVar.signal),
        prerequisites,
        button(
          cls := "button is-small is-link",
          disabled <-- refused,
          "⬆ Upscale",
          onClick.compose(_.sample(output)) --> (_.foreach(selected =>
            onSeedVr2(
              selected,
              SeedVr2UpscaleRequest(
                runConfigurationId = configurationVar.now(),
                scale = scaleVar.now(),
                seed = seedOf(seedVar)
              )
            )
          ))
        )
      ).element
    )
  )
}
