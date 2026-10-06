package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.pages.gallery.UpscaleTaskPanel.*
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The upscale task (`specs/15-post-hoc-resize.md`, `specs/26-tiled-pid.md`,
  * `specs/51-seedvr2-upscaling.md`): one task whatever enlarges the picture
  * (François, 2026-10-04) — the model is its first parameter, and the select
  * lists them all: the SeedVR2 and PiD run configurations and the ESRGAN models
  * of the upscaler store. What follows the select is what the chosen model
  * takes, each kind's own panel built once and hidden while another is chosen.
  * A video has the SeedVR2 models alone, the only ones that take it.
  */
class UpscaleTaskPanel(
    /** The selected output, a picture or a video. */
    output: Signal[Option[GenerationOutput]],
    seedVr2Configurations: Signal[List[ConfigurationOption]],
    pidConfigurations: Signal[List[ConfigurationOption]],
    upscalers: Signal[List[Upscaler]],
    /** The picture on screen, for the size it really has. */
    viewed: Var[Option[ViewedImage]],
    /** Where the chosen model's tiles go, for the picture to draw them. */
    geometry: Var[TileGeometry],
    showTileGrid: Var[Boolean],
    gridOffset: Var[TileOffset],
    /** Whether the upscale task is the one on screen. */
    active: Signal[Boolean],
    sourcePrompt: String,
    prerequisites: LaunchPrerequisites,
    onSeedVr2: (GenerationOutput, SeedVr2UpscaleRequest) => Unit,
    onPid: (GenerationOutput, PidUpscaleRequest) => Unit,
    onUpscale: (GenerationOutput, UpscaleRequest) => Unit
) extends Component {

  private val image: Signal[Option[GenerationOutput]] =
    output.map(_.filterNot(GenerationMediaViewer.isVideo))

  private val isVideo: Signal[Boolean] =
    output.map(_.exists(GenerationMediaViewer.isVideo)).distinct

  /** The select's value, a `Choice.key`; empty until one is picked. */
  private val choiceVar = Var("")
  private val seedVr2Var = Var("")
  private val pidVar = Var("")
  private val esrganVar = Var("")

  /** Every model that can upscale the selected output: SeedVR2 first, its
    * built-in default at the head, then PiD, then ESRGAN.
    */
  private val choices: Signal[List[Choice]] =
    seedVr2Configurations
      .combineWith(pidConfigurations, upscalers, isVideo)
      .map { (seedVr2, pid, esrgan, video) =>
        val (preferred, others) =
          seedVr2.partition(_.id == SeedVr2UpscaleRequest.DefaultConfiguration)
        (preferred ++ others).map(c => Choice(SeedVr2Kind, c.id, c.label)) ++
          (if (video) List.empty
           else
             pid.map(c => Choice(PidKind, c.id, c.label)) ++
               esrgan.map(u => Choice(EsrganKind, u.id, u.label)))
      }
      .distinct

  /** The model the task runs with: the one picked, else the first. */
  private val chosen: Signal[Option[Choice]] =
    choices
      .combineWith(choiceVar.signal)
      .map((list, key) => list.find(_.key == key).orElse(list.headOption))
      .distinct

  private val kind: Signal[String] = chosen.map(_.fold("")(_.kind)).distinct

  private def modelSelect: HtmlElement =
    div(
      cls := "select is-small",
      select(
        onChange.mapToValue --> choiceVar,
        children <-- choices.map(
          _.map(choice =>
            option(
              value := choice.key,
              selected <-- chosen.map(_.exists(_.key == choice.key)),
              choice.label
            )
          )
        )
      )
    )

  private def body(of: String, panel: HtmlElement): HtmlElement =
    div(cls("is-hidden") <-- kind.map(_ != of), panel)

  lazy val element: HtmlElement = div(
    // each kind's panel reads its own model from the one select
    chosen --> Observer[Option[Choice]](_.foreach { choice =>
      choice.kind match {
        case SeedVr2Kind => seedVr2Var.set(choice.id)
        case PidKind     => pidVar.set(choice.id)
        case _           => esrganVar.set(choice.id)
      }
    }),
    // the grid on the picture is the chosen model's: each tiled kind's panel
    // publishes its own while it is the one chosen, and a model that cuts
    // nothing leaves no grid behind
    active.combineWith(kind) --> Observer[(Boolean, String)] { (shown, of) =>
      if (shown && of != SeedVr2Kind && of != PidKind) geometry.set(NoTiles)
    },
    p(
      cls := "post-intro text-secondary",
      cls("is-hidden") <-- choices.map(_.nonEmpty),
      child.text <-- isVideo.map(video =>
        if (video)
          "No video upscaler yet: use the \"SeedVR2 7B upscale\" starter " +
            "configuration, or create a run configuration on the " +
            "\"SeedVR2 upscaler\" architecture, on the drift runner."
        else
          "No upscaler yet: use a SeedVR2 or PiD starter configuration, or " +
            "install an ESRGAN model in the model cache."
      )
    ),
    div(
      cls("is-hidden") <-- choices.map(_.isEmpty),
      group("model", plainField(modelSelect))
    ),
    body(
      SeedVr2Kind,
      SeedVr2UpscalePanel(
        output,
        seedVr2Var,
        viewed,
        geometry,
        showTileGrid,
        gridOffset,
        active.combineWith(kind).map(_ && _ == SeedVr2Kind).distinct,
        prerequisites,
        onSeedVr2
      ).element
    ),
    body(
      PidKind,
      PidUpscalePanel(
        image,
        pidConfigurations,
        pidVar,
        viewed,
        geometry,
        showTileGrid,
        gridOffset,
        active.combineWith(kind).map(_ && _ == PidKind).distinct,
        sourcePrompt,
        prerequisites,
        onPid
      ).element
    ),
    body(
      EsrganKind,
      UpscalePanel(image, upscalers, esrganVar, onUpscale).element
    )
  )
}

object UpscaleTaskPanel {

  val SeedVr2Kind = "seedvr2"
  val PidKind = "pid"
  val EsrganKind = "esrgan"

  /** One model of the select: what it is, and its id among its kind. */
  final case class Choice(kind: String, id: String, label: String) {
    def key: String = s"$kind:$id"
  }
}
