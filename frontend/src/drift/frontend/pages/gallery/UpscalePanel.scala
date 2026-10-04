package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What the upscale task shows for a model of the upscaler store (ESRGAN,
  * `specs/15-post-hoc-resize.md`), chosen in the task's select
  * (`UpscaleTaskPanel`): how many times it runs, and the button. The cheapest
  * of the three tasks and the only one that invents nothing — no session, no
  * prompt, no seed.
  */
class UpscalePanel(
    image: Signal[Option[GenerationOutput]],
    upscalers: Signal[List[Upscaler]],
    /** The chosen model: the task's model select writes it. */
    upscalerVar: Var[String],
    onUpscale: (GenerationOutput, UpscaleRequest) => Unit
) extends Component {

  private val repeatsVar = Var("1")
  private val advancedVar = Var(false)

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
