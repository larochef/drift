package drift.frontend.pages.generate

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** VAE tiling (`specs/10-generation-time-upscaling.md`), shown wherever the
  * capabilities report the feature — both modes have it. Decoding the VAE in
  * tiles is what lets a large hires result finish instead of aborting sd-server
  * on ROCm, so the note points there. Temporal tiling and the extra args are
  * video-VAE only.
  */
class VaeTilingSection(
    state: GenerationFormState,
    capabilities: SessionCapabilities,
    currentMode: String
) extends Component {
  import state.*
  import FormFields.{field, numberField}

  private val summary: Signal[Option[String]] =
    vaeTilingEnabledVar.signal.map(Option.when(_)("on"))

  /** The relative VAE tile size to default to — chosen so each tile decodes at
    * roughly the base-generation resolution, which is the size the model
    * already handles, whatever the upscale factor. `rel = base ÷ target`, so a
    * 2× hires gives 0.5 and a 4× gives 0.25: the tile stays base-sized and only
    * the tile *count* grows with the target. Working in fractions keeps it
    * model-agnostic — no need to know the VAE's latent-to-pixel ratio.
    *
    * Falls back to 0.5 when there is nothing to divide by (hires off, or the
    * size fields are blank). Clamped to (0, 1]: 1 means one tile, i.e. no
    * tiling, which is right when target equals base.
    */
  private def adaptiveVaeRel(): (Double, Double) =
    if (!hiresEnabledVar.now()) (0.5, 0.5)
    else {
      def targetOf(base: Int, targetVar: Var[String]): Double =
        targetVar
          .now()
          .trim
          .toIntOption
          .filter(_ > 0)
          .map(_.toDouble)
          .getOrElse {
            val scale = hiresScaleVar
              .now()
              .trim
              .toDoubleOption
              .filter(_ > 0)
              .getOrElse(2.0)
            base * scale
          }
      def rel(base: Int, target: Double): Double = {
        val raw = (base / target).min(1.0).max(0.05)
        math.round(raw * 100) / 100.0
      }
      (
        widthVar.now().trim.toIntOption.filter(_ > 0),
        heightVar.now().trim.toIntOption.filter(_ > 0)
      ) match {
        case (Some(bw), Some(bh)) =>
          (
            rel(bw, targetOf(bw, hiresTargetWidthVar)),
            rel(bh, targetOf(bh, hiresTargetHeightVar))
          )
        case _ => (0.5, 0.5)
      }
    }

  /** Turning tiling on fills a fast, safe tile size when nothing is set yet:
    * the server default (32 latent ≈ tiny tiles) tiles so finely that a large
    * VAE pass runs hundreds of tiles — correct, but very slow. The default
    * instead sizes each tile to about the base-generation resolution (see
    * [[adaptiveVaeRel]]), a handful of tiles that stay safe at any upscale. The
    * user can still override, and an explicit size already typed is left alone.
    */
  private def toggleVaeTiling(on: Boolean): Unit = {
    vaeTilingEnabledVar.set(on)
    val noTileSizeSet =
      vaeTileSizeXVar.now().trim.isEmpty && vaeTileSizeYVar
        .now()
        .trim
        .isEmpty &&
        vaeRelSizeXVar.now().trim.isEmpty && vaeRelSizeYVar.now().trim.isEmpty
    if (on && noTileSizeSet) {
      val (relX, relY) = adaptiveVaeRel()
      vaeRelSizeXVar.set(relX.toString)
      vaeRelSizeYVar.set(relY.toString)
      vaeTargetOverlapVar.set("0.25")
    }
  }

  lazy val element: HtmlElement = {
    val features =
      capabilities.featuresByMode.getOrElse(currentMode, Map.empty)
    if (!features.getOrElse("vae_tiling", false)) div()
    else
      CollapsibleSection(
        "VAE tiling",
        summary,
        div(
          label(
            cls := "checkbox label text-primary is-small mb-1",
            input(
              typ := "checkbox",
              cls := "mr-1",
              checked <-- vaeTilingEnabledVar.signal,
              onChange.mapToChecked --> (on => toggleVaeTiling(on))
            ),
            "VAE tiling (fixes high-res crashes)"
          ),
          child <-- vaeTilingEnabledVar.signal.map {
            case false => emptyNode
            case true  =>
              div(
                div(
                  cls := "columns is-mobile",
                  div(
                    cls := "column",
                    numberField(
                      "Tile width (latent, 0 = default)",
                      vaeTileSizeXVar,
                      Some(0)
                    )
                  ),
                  div(
                    cls := "column",
                    numberField(
                      "Tile height (latent)",
                      vaeTileSizeYVar,
                      Some(0)
                    )
                  )
                ),
                numberField(
                  "Tile overlap (fraction of tile)",
                  vaeTargetOverlapVar
                ),
                div(
                  cls := "columns is-mobile",
                  div(
                    cls := "column",
                    numberField("Relative tile width", vaeRelSizeXVar)
                  ),
                  div(
                    cls := "column",
                    numberField("Relative tile height", vaeRelSizeYVar)
                  )
                ),
                if (currentMode == "vid_gen")
                  div(
                    label(
                      cls := "checkbox is-size-7 text-secondary mb-1",
                      input(
                        typ := "checkbox",
                        cls := "mr-1",
                        checked <-- vaeTemporalTilingVar.signal,
                        onChange.mapToChecked --> vaeTemporalTilingVar
                      ),
                      "Temporal tiling (video VAE)"
                    ),
                    field(
                      "Extra tiling args (key=value)",
                      input(
                        cls := "input is-small",
                        placeholder := "temporal_tile_frames=4, temporal_tile_overlap=1",
                        value <-- vaeExtraTilingArgsVar.signal,
                        onInput.mapToValue --> vaeExtraTilingArgsVar
                      )
                    )
                  )
                else emptyNode,
                p(
                  cls := "is-size-7 text-secondary",
                  "Defaults each tile to about your base-generation size " +
                    "(relative = base ÷ target), so it stays safe whatever the " +
                    "upscale — a few large tiles, not the server's hundreds of " +
                    "tiny ones. Tile size is in latent units (the pixel ratio " +
                    "depends on the model's VAE); relative size overrides it — " +
                    "below 1 is a fraction of the image, 1 or more a tile count " +
                    "per side. Overlap caps at 0.5. Smaller tiles use less VRAM " +
                    "but blend more slowly."
                )
              )
          }
        )
      ).element
  }
}
