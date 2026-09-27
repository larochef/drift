package drift.frontend.components

import com.raquo.laminar.api.L.*

/** The modal-card chrome shared by the three browsers: background, head with a
  * title, scrollable body, and a footer split left/right.
  *
  * This is deliberately not a type-parameterized base class. The browsers
  * differ in the data they show (`HuggingFaceModelInfo` vs
  * `CivitaiModelListInfo` vs `FileEntry`) but agree on the layout, so the thing
  * worth sharing takes values, not type parameters.
  */
class BrowserModal(
    title: Signal[String],
    body: Seq[Mod[HtmlElement]],
    onCancel: () => Unit,
    headerAction: Mod[HtmlElement] = emptyNode,
    footerLeft: Mod[HtmlElement] = emptyNode,
    footerRight: Mod[HtmlElement] = emptyNode,
    modalMods: Seq[Mod[HtmlElement]] = Nil,
    cardMods: Seq[Mod[HtmlElement]] = Nil
) extends Component {
  lazy val element: HtmlElement =
    div(
      cls := "modal is-active",
      ScrollLock.whileMounted,
      modalMods,
      div(cls := "modal-background", onClick --> (_ => onCancel())),
      div(
        cls := "modal-card",
        cardMods,
        div(
          cls := "modal-card-head",
          p(cls := "modal-card-title", child.text <-- title),
          headerAction
        ),
        div(cls := "modal-card-body", body),
        div(
          cls := "modal-card-foot",
          div(
            cls := "level is-mobile mb-0",
            div(cls := "level-left", footerLeft),
            div(cls := "level-right", footerRight)
          )
        )
      )
    )
}

object BrowserModal {

  /** The plain "Cancel" button every browser puts in its footer. */
  def cancelButton(onCancel: () => Unit): HtmlElement =
    button(cls := "button", "Cancel", onClick --> (_ => onCancel()))

  /** The "back to the result list" button of the two-phase browsers. */
  def backButton(onBack: () => Unit): HtmlElement =
    button(cls := "button", "← Back", onClick --> (_ => onBack()))
}
