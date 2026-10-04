package drift.frontend.pages.gallery

import drift.frontend.components.{Component, LaunchOrDownload}
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the upscale task shows for a PiD model (`specs/26-tiled-pid.md`): a
  * diffusion upscale through a PiD run configuration, the model itself chosen
  * in the task's select (`UpscaleTaskPanel`). Empty size fields ask the backend for ×4 of the source with its
  * ratio kept (`PidUpscaleRequest.target`), and the line before the button says
  * what that comes to in pixels — or why the job would not be an upscale at
  * all, which the backend refuses.
  *
  * The prompt it sends is a field of this panel, where it belongs: it used to
  * be handed to the section and rendered under the redraw row, a long way from
  * the button that sends it (François, 2026-09-20).
  */
class PidUpscalePanel(
    image: Signal[Option[GenerationOutput]],
    pidConfigurations: Signal[List[ConfigurationOption]],
    /** The chosen configuration: the task's model select writes it. */
    configurationVar: Var[String],
    /** The picture on screen, for the size it really has. */
    viewed: Signal[Option[ViewedImage]],
    /** Where this panel puts the tiles its job would decode, in the picture's
      * own pixels, for the viewer to draw them.
      */
    gridTiles: Var[List[ImageRegion]],
    /** Whether the picture draws them — the same switch the redraw panel has,
      * since one grid is on screen at a time.
      */
    showTileGrid: Var[Boolean],
    /** The source's prompt, inherited — what the prompt field starts with. */
    sourcePrompt: String,
    /** What each configuration still has to download: the job's button becomes
      * the download while the chosen model cannot start (`specs/46`).
      */
    prerequisites: LaunchPrerequisites,
    onPid: (GenerationOutput, PidUpscaleRequest) => Unit
) extends Component {

  private val widthVar = Var("")
  private val heightVar = Var("")
  private val stepsVar = Var("4")
  private val seedVar = Var("")
  private val promptVar = Var(sourcePrompt)
  private val advancedVar = Var(false)

  /** The largest tile the chosen configuration's runner decodes: one tile for a
    * 4096² target on drift's runner.
    */
  private val maxTile: Signal[Int] =
    pidConfigurations
      .combineWith(configurationVar.signal)
      .map((options, id) =>
        options
          .find(_.id == id)
          .fold(PidUpscaleRequest.MaxTile)(o =>
            PidUpscaleRequest.maxTileFor(o.runner)
          )
      )
      .distinct

  private val hasConfigurations: Signal[Boolean] =
    pidConfigurations.map(_.nonEmpty).distinct

  /** What the size fields as they stand ask for, and what it comes to on the
    * picture on screen — its refusal included, since a target that does not
    * raise the resolution is not an upscale and the job says so.
    */
  private val planned: Signal[(String, Option[Either[String, (Int, Int)]])] =
    widthVar.signal
      .combineWith(heightVar.signal, viewed)
      .map { (w, h, shown) =>
        val asked =
          if (w.trim.isEmpty && h.trim.isEmpty) "→ ×4 of the source"
          else s"→ $w × $h"
        (
          asked,
          shown.map(picture =>
            PidUpscaleRequest.target(
              (picture.width, picture.height),
              w.trim.toIntOption,
              h.trim.toIntOption
            )
          )
        )
      }
      .distinct

  /** The tiles the job would decode, in the picture's own pixels: laid out over
    * the target (`PidUpscaleRequest.tilesFor`) and brought back to the source's
    * scale, since the picture on screen is the source.
    */
  private val tiles: Signal[List[ImageRegion]] =
    planned
      .combineWith(viewed, maxTile)
      .map { (_, outcome, shown, largest) =>
        (outcome, shown) match {
          case (Some(Right(target)), Some(picture)) if target._1 > 0 =>
            val scale = picture.width.toDouble / target._1
            PidUpscaleRequest
              .tilesFor(target, largest)
              .flatten
              .map(tile =>
                ImageRegion(
                  math.round(tile.x * scale).toInt,
                  math.round(tile.y * scale).toInt,
                  math.round(tile.width * scale).toInt,
                  math.round(tile.height * scale).toInt
                )
              )
          case _ => List.empty
        }
      }
      .distinct

  private val publishTiles: Modifier[HtmlElement] =
    tiles.changes --> gridTiles

  private val target: Signal[String] =
    planned
      .combineWith(tiles)
      .map { (asked, outcome, laid) =>
        val passes = Option
          .when(laid.nonEmpty)(
            if (laid.size == 1) " · 1 tile" else s" · ${laid.size} tiles"
          )
          .getOrElse("")
        outcome match {
          case Some(Right((width, height))) if asked.endsWith("source") =>
            s"$asked · $width × $height$passes"
          case Some(Left(reason)) => s"$asked — $reason"
          case _                  => s"$asked$passes"
        }
      }
      .distinct

  /** Whether this job would not be an upscale at all — the backend refuses it,
    * so the button does not offer it.
    */
  private val refused: Signal[Boolean] =
    planned.map(_._2.exists(_.isLeft)).distinct

  private def form: HtmlElement = div(
    cls("is-hidden") <-- hasConfigurations.map(!_),
    intro(
      "A pixel-diffusion decoder re-renders the image larger, ×4 by default. " +
        "Minutes and a loaded model, and it adds detail the upscaler cannot " +
        "— the original stays in the gallery."
    ),
    advanced(
      advancedVar,
      group(
        "size",
        field(
          "width",
          numberField(widthVar, "5.5rem").amend(
            placeholder := "×4",
            title := "target size, multiples of 4 — leave both empty for ×4 " +
              "of the source with its ratio kept; another ratio crops the " +
              "source"
          )
        ),
        field(
          "height",
          numberField(heightVar, "5.5rem").amend(placeholder := "×4")
        )
      ),
      group(
        "pass",
        field(
          "steps",
          numberField(stepsVar, "4.5rem").amend(
            minAttr := "1",
            title := "4 for the \"4step\" distilled decoders, which is what " +
              "the built-in architectures assume"
          )
        ),
        seedField(seedVar)
      ),
      group(
        "prompt",
        wideField(
          promptField(
            promptVar,
            lines = 3,
            hint = "prompt for PiD (the source's by default)"
          )
        )
      )
    ),
    foot(
      div(
        cls := "post-foot-lines",
        div(child.text <-- target),
        div(
          cls := "post-foot-controls",
          checkField(
            showTileGrid,
            "show the grid",
            "draw the tiles this decode would run over the picture — one " +
              "pass of the model each, shown where they fall on the source"
          )
        )
      ),
      LaunchOrDownload(
        prerequisites.of(configurationVar.signal),
        prerequisites,
        button(
          cls := "button is-small is-link",
          disabled <-- refused,
          "✨ PiD upscale",
          onClick.compose(_.sample(image)) --> (_.foreach(output =>
            onPid(
              output,
              PidUpscaleRequest(
                runConfigurationId = configurationVar.now(),
                width = widthVar.now().trim.toIntOption,
                height = heightVar.now().trim.toIntOption,
                prompt = promptVar.now(),
                steps = stepsVar.now().trim.toIntOption.getOrElse(4),
                seed = seedOf(seedVar)
              )
            )
          ))
        )
      ).element
    )
  )

  lazy val element: HtmlElement = div(
    publishTiles,
    pidConfigurations --> Observer[List[ConfigurationOption]](list =>
      if (configurationVar.now().isEmpty)
        list.headOption.foreach(option => configurationVar.set(option.id))
    ),
    p(
      cls := "post-intro text-secondary",
      cls("is-hidden") <-- hasConfigurations,
      "No PiD upscaler yet: create a run configuration on a \"PiD\" " +
        "architecture — its decoder, the Gemma 2 2B text encoder and the " +
        "matching VAE — and it appears here."
    ),
    form
  )
}
