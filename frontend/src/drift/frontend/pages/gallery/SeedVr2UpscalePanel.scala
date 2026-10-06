package drift.frontend.pages.gallery

import drift.frontend.components.{Component, LaunchOrDownload}
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.pages.gallery.TileAreaFields.Cut
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the upscale task shows for a SeedVR2 model
  * (`specs/51-seedvr2-upscaling.md`): the scale, a seed, and the button — for a
  * picture and for a video alike. One diffusion step on the drift runner, no
  * prompt; it restores harder than the other upscalers and says so. A picture
  * past one pass is cut in tiles (`SeedVr2UpscaleRequest.tilingFor`): a tiled
  * task like the others, with their tile field and their grid
  * (`TileAreaFields`).
  */
class SeedVr2UpscalePanel(
    output: Signal[Option[GenerationOutput]],
    /** The chosen configuration: the task's model select writes it. */
    configurationVar: Var[String],
    /** The picture on screen, for the size it really has. */
    viewed: Var[Option[ViewedImage]],
    /** Where this panel's tiles go, for the picture to draw them. */
    geometry: Var[TileGeometry],
    showTileGrid: Var[Boolean],
    gridOffset: Var[TileOffset],
    /** Whether this is the model the upscale task on screen runs. */
    active: Signal[Boolean],
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
    viewed.signal
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

  /** How a picture's job would cut it — nothing for a video or a refusal. */
  private val cut: Signal[Option[Cut]] =
    viewed.signal
      .combineWith(scaleVar.signal, planned)
      .map { (shown, scale, outcome) =>
        (shown, outcome) match {
          case (Some(picture), Some(Right(_))) =>
            Some(
              Cut.Whole(
                SeedVr2UpscaleRequest
                  .tilingFor((picture.width, picture.height), scale),
                scale.toDouble
              )
            )
          case _ => None
        }
      }
      .distinct

  private val area = TileAreaFields(
    cut,
    takesSelection = false,
    viewed,
    geometry,
    showTileGrid,
    gridOffset,
    active
  )

  private val summary: Signal[String] =
    isVideo
      .combineWith(scaleVar.signal, planned, area.plan)
      .map { (video, scale, outcome, plan) =>
        if (video) s"→ ×$scale · the whole video, batch by batch: minutes"
        else
          outcome match {
            case Some(Right((width, height))) =>
              s"→ ×$scale of the source · $width × $height" +
                plan.fold("")(laid =>
                  if (laid.tiles == 1) " · 1 tile"
                  else s" · ${laid.tiles} tiles"
                )
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
    area.publishGeometry,
    intro(
      "SeedVR2 restores and enlarges a picture or a video in one diffusion " +
        "step, on the drift runner — a video keeps its frame rate and its " +
        "soundtrack. The original stays in the gallery."
    ),
    group("size", plainField(scaleSelect)),
    advanced(
      advancedVar,
      div(cls("is-hidden") <-- isVideo, area.areaGroup),
      group("pass", seedField(seedVar))
    ),
    foot(
      div(
        cls := "post-foot-lines",
        div(child.text <-- summary),
        area.gridControls.amend(cls("is-hidden") <-- isVideo)
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
                seed = seedOf(seedVar),
                tileSize = area.askedTileSize,
                gridOffsetX = area.targetGridOffset._1,
                gridOffsetY = area.targetGridOffset._2
              )
            )
          ))
        )
      ).element
    )
  )
}
