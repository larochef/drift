package drift.frontend.components

import com.raquo.laminar.api.L.*

/** The controls a browser puts between its search field and its results
  * (`specs/24`, `36`, `37`): what to sort by, which base model to look under,
  * what to leave out. The sites choose different ones; they look the same.
  */
object BrowserFilters {

  /** One row of them. */
  def row(controls: Mod[HtmlElement]*): HtmlElement =
    div(cls := "field mb-3", div(cls := "control", controls))

  /** A small select and, beside it, what it chooses ("sort order", "trained
    * on"). `chosen` is the option selected when the row is built.
    */
  def choice(
      options: Seq[(String, String)],
      chosen: String,
      about: String,
      onChoose: String => Unit
  ): Seq[HtmlElement] = Seq(
    div(
      cls := "select is-small",
      select(
        options.map((key, words) =>
          option(value := key, words, selected := (key == chosen))
        ),
        onChange.mapToValue --> (picked => onChoose(picked))
      )
    ),
    span(cls := "ml-2 is-size-7 text-secondary", about)
  )

  /** A tick box, with the reason for it in its tooltip. `onTicked` runs after
    * the state changes — a filter the site applies searches again, one applied
    * to the page in hand does nothing more.
    */
  def checkbox(
      text: String,
      tooltip: String,
      state: Var[Boolean],
      onTicked: () => Unit = () => ()
  ): HtmlElement =
    label(
      cls := "checkbox is-size-7 text-secondary ml-4",
      title := tooltip,
      input(
        typ := "checkbox",
        checked <-- state.signal,
        onChange.mapToChecked --> { ticked =>
          state.set(ticked)
          onTicked()
        }
      ),
      s" $text"
    )
}
