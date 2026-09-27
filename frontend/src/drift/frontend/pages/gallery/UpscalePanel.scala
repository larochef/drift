package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Upscale with a model of the upscaler store (ESRGAN,
  * `specs/15-post-hoc-resize.md`): the model, how many times it runs, and the
  * button. The cheapest of the three tasks and the only one that invents
  * nothing — no session, no prompt, no seed.
  */
class UpscalePanel(
    image: Signal[Option[GenerationOutput]],
    upscalers: Signal[List[Upscaler]],
    onUpscale: (GenerationOutput, UpscaleRequest) => Unit
) extends Component {

  private val upscalerVar = Var("")
  private val repeatsVar = Var("1")
  private val advancedVar = Var(false)

  private def modelSelect: HtmlElement =
    div(
      cls := "select is-small",
      select(
        onChange.mapToValue --> upscalerVar,
        children <-- upscalers.map { list =>
          if (list.isEmpty)
            List(option(value := "", "no upscaler installed"))
          else
            list.map(u =>
              option(
                value := u.id,
                selected <-- upscalerVar.signal.map(v =>
                  v == u.id || (v.isEmpty && list.headOption
                    .exists(_.id == u.id))
                ),
                u.label
              )
            )
        }
      )
    )

  lazy val element: HtmlElement = div(
    // The select shows the first installed model; the request must name it
    // even when the user never touched the select.
    upscalers --> Observer[List[Upscaler]](list =>
      if (upscalerVar.now().isEmpty)
        list.headOption.foreach(u => upscalerVar.set(u.id))
    ),
    intro(
      "An ESRGAN model enlarges the image, ×2 or ×4 a pass depending on the " +
        "model. Seconds, no model loaded, no new detail — it sharpens what " +
        "is already there, and the original stays in the gallery."
    ),
    group("model", plainField(modelSelect)),
    advanced(
      advancedVar,
      group(
        "passes",
        plainField(
          numberField(repeatsVar, "4.5rem").amend(
            minAttr := "1",
            maxAttr := "4",
            title := "how many times the model runs, each pass on the last " +
              "one's result"
          )
        )
      )
    ),
    foot(
      child.text <-- upscalers.map(list =>
        if (list.isEmpty) "install an upscaler in the model cache first" else ""
      ),
      button(
        cls := "button is-small is-link",
        "⬆ Upscale",
        disabled <-- upscalers.map(_.isEmpty),
        onClick.compose(_.sample(image)) --> (_.foreach { output =>
          val chosen = Some(upscalerVar.now()).filter(_.nonEmpty)
          onUpscale(
            output,
            UpscaleRequest(
              upscalerId = chosen.getOrElse(""),
              repeats = repeatsVar.now().trim.toIntOption.getOrElse(1)
            )
          )
        })
      )
    )
  )
}
