package drift.frontend.pages.generate

import com.raquo.laminar.api.L.*

/** The labelled controls the generation form's sections share. */
object FormFields {

  def field(labelText: String, control: HtmlElement): HtmlElement =
    div(
      cls := "field",
      label(cls := "label text-primary is-small", labelText),
      div(cls := "control", control)
    )

  def numberField(
      labelText: String,
      state: Var[String],
      minimum: Option[Int] = None,
      maximum: Option[Int] = None
  ): HtmlElement =
    field(
      labelText,
      input(
        cls := "input is-small",
        typ := "number",
        minimum.map(v => minAttr := v.toString),
        maximum.map(v => maxAttr := v.toString),
        value <-- state.signal,
        onInput.mapToValue --> state
      )
    )

  def selectField(
      labelText: String,
      state: Var[String],
      options: List[String]
  ): HtmlElement =
    field(
      labelText,
      div(
        cls := "select is-small is-fullwidth",
        select(
          onChange.mapToValue --> state,
          option(
            value := "",
            selected <-- state.signal.map(_.isEmpty),
            "(model default)"
          ),
          options.map(name =>
            option(
              value := name,
              selected <-- state.signal.map(_ == name),
              name
            )
          )
        )
      )
    )
}
