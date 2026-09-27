package drift.frontend.components

import com.raquo.laminar.api.L.*

/** A part of a form folded away behind its own title: a chevron, a title that
  * clicks, whatever else the title line carries, and the body under them.
  *
  * The shape is the architecture page's LoRA collection, which had it first
  * (`specs/09-lora-management.md`); the post-processing panels' advanced
  * parameters asked for the same one rather than a button of their own
  * (François, 2026-09-21), so it lives here and both use it.
  *
  * `keepBody` says what folding does to the body. `false` removes it — a list
  * rebuilt from a signal loses nothing and a card stays light. `true` only
  * hides it, which is what a form wants: what has been typed into it survives
  * every close and open.
  */
class FoldedSection(
    title: Signal[String],
    open: Var[Boolean],
    /** Beside the title, folded or not — a count, a download in progress. */
    note: Modifier[HtmlElement] = emptyMod,
    /** At the right of the title line: what belongs to the section as a whole
      * rather than to its contents.
      */
    action: Modifier[HtmlElement] = emptyMod,
    keepBody: Boolean = false,
    body: => HtmlElement
) extends Component {

  private val head: HtmlElement = div(
    cls := "level is-mobile mb-2",
    div(
      cls := "level-left",
      div(
        cls := "is-flex is-align-items-baseline cursor-pointer",
        onClick --> (_ => open.update(!_)),
        span(
          cls := "text-secondary is-size-7 mr-2",
          child.text <-- open.signal.map(if (_) "▾" else "▸")
        ),
        label(
          cls := "label text-primary mb-0 cursor-pointer",
          child.text <-- title
        ),
        note
      )
    ),
    div(cls := "level-right", action)
  )

  private val contents: Modifier[HtmlElement] =
    if (keepBody)
      div(
        cls := "folded-body",
        cls("is-hidden") <-- open.signal.map(!_),
        body
      )
    else child <-- open.signal.map(shown => if (shown) body else emptyNode)

  lazy val element: HtmlElement = div(cls := "folded-section", head, contents)
}
