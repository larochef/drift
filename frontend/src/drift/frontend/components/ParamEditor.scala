package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

/** One flag of one layer: either the value it is given, or — `removed` — the
  * instruction to keep it off the command line altogether
  * (`specs/16-parameter-resolution.md`). The typed value survives the toggle,
  * so turning a removal back into an override does not lose it.
  */
case class ParamRow(
    id: Int,
    key: String,
    value: String,
    removed: Boolean = false
)

class ParamEditor(
    initialParameters: List[(String, String)] = Nil,
    initialRemoved: List[String] = Nil,
    /** Whether a row may be a removal. False for the bottom layer: an
      * architecture default that is "not passed" is a default that is simply
      * not written.
      */
    allowsRemoval: Boolean = true,
    /** What the layers below this one set, offered as chips: the flag, the
      * value it would pass, and the layer that asked for it. The flag name is
      * the hard part of overriding or removing one — it has to match to the
      * character, and a misspelling removes nothing while looking like it did —
      * and the layer is what says whether a value is still the architecture's
      * or already a model's, which two identical numbers cannot (François,
      * 2026-09-20).
      */
    inherited: Signal[List[ResolvedParameter]] = Val(Nil)
) extends Component {
  private var nextId = 0

  private def rowsOf(
      parameters: List[(String, String)],
      removed: List[String]
  ): List[ParamRow] =
    parameters.map((key, value) => row(key, value, removed = false)) ++
      removed.map(key => row(key, "", removed = true))

  private def row(key: String, value: String, removed: Boolean): ParamRow = {
    val id = nextId
    nextId += 1
    ParamRow(id, key, value, removed)
  }

  private val rows: Var[List[ParamRow]] =
    Var(rowsOf(initialParameters, initialRemoved))

  def reset(
      newParameters: List[(String, String)] = Nil,
      newRemoved: List[String] = Nil
  ): Unit = {
    nextId = 0
    rows.set(rowsOf(newParameters, newRemoved))
  }

  /** The flags this layer sets. */
  def snapshot(): Map[String, String] =
    rows
      .now()
      .filter(r => r.key.nonEmpty && !r.removed)
      .map(r => (r.key, r.value))
      .toMap

  /** The flags this layer takes off the command line. */
  def removedSnapshot(): List[String] =
    rows.now().filter(r => r.key.nonEmpty && r.removed).map(_.key).distinct

  /** The flags set below that this layer says nothing about yet — what the
    * chips offer, so a flag is picked rather than retyped.
    */
  private val untouched: Signal[List[ResolvedParameter]] =
    inherited.combineWith(rows.signal).map { (below, current) =>
      val taken = current.map(_.key).toSet
      below.filterNot(p => taken(p.flag)).distinctBy(_.flag).sortBy(_.flag)
    }

  private def update(id: Int)(change: ParamRow => ParamRow): Unit =
    rows.update(_.map(r => if (r.id == id) change(r) else r))

  lazy val element: HtmlElement = {
    div(
      children <-- rows.signal.split(_.id) { (id, initial, rowSignal) =>
        div(
          cls := "columns is-mobile is-vcentered mb-1",
          div(
            cls := "column",
            input(
              cls := "input is-small",
              placeholder := "Key",
              value <-- rowSignal.map(_.key),
              onInput.mapToValue --> (key => update(id)(_.copy(key = key)))
            )
          ),
          div(
            cls := "column",
            // Rebuilt only when the row is toggled: a removed row has no value
            // to type, and saying so is clearer than a disabled box.
            child <-- rowSignal.map(_.removed).distinct.map {
              case false =>
                input(
                  cls := "input is-small",
                  placeholder := "Value",
                  value <-- rowSignal.map(_.value),
                  onInput.mapToValue --> (value =>
                    update(id)(_.copy(value = value))
                  )
                )
              case true =>
                span(
                  cls := "tag is-warning is-light",
                  title := "This flag is kept off the command line",
                  "not passed"
                )
            }
          ),
          if (allowsRemoval)
            div(
              cls := "column is-narrow",
              button(
                cls <-- rowSignal.map(r =>
                  if (r.removed) "button is-small is-warning"
                  else "button is-small"
                ),
                title <-- rowSignal.map(r =>
                  if (r.removed) "Give this flag a value again"
                  else "Do not pass this flag at all, whatever is set below"
                ),
                child.text <-- rowSignal.map(r => if (r.removed) "↩" else "🚫"),
                onClick --> (_ => update(id)(r => r.copy(removed = !r.removed)))
              )
            )
          else emptyNode,
          div(
            cls := "column is-narrow",
            button(
              cls := "button is-danger is-small",
              title := "Drop this row — the layers below decide again",
              "🗑️",
              onClick --> (_ => rows.update(_.filterNot(_.id == id)))
            )
          )
        )
      },
      // Each chip carries the layer that set it, and both things that can be
      // done about it: overriding it here needs its value, removing it needs
      // one click and not a hand-typed flag.
      child <-- untouched.map {
        case Nil   => emptyNode
        case flags =>
          div(
            cls := "mb-2",
            p(
              cls := "help text-secondary mb-1",
              if (allowsRemoval)
                "Already set below \u2014 \u270E overrides one here, " +
                  "\uD83D\uDEAB keeps it off the command line:"
              else "Already set below:"
            ),
            div(
              cls := "tags-editor",
              flags.map(parameter =>
                div(
                  cls := "tags has-addons mb-0",
                  span(
                    cls := "tag is-small",
                    title := s"set by the ${parameter.layer.name}",
                    if (parameter.value.isEmpty) parameter.flag
                    else s"${parameter.flag} ${parameter.value}",
                    span(
                      cls := "is-size-7 text-secondary ml-1",
                      s"\u00B7 ${parameter.layer.name}"
                    )
                  ),
                  span(
                    cls := "tag is-small is-info cursor-pointer",
                    title := "Override it here",
                    "\u270E",
                    onClick --> (_ =>
                      rows.update(
                        _ :+ row(
                          parameter.flag,
                          parameter.value,
                          removed = false
                        )
                      )
                    )
                  ),
                  if (allowsRemoval)
                    span(
                      cls := "tag is-small is-warning cursor-pointer",
                      title := "Keep it off the command line",
                      "\uD83D\uDEAB",
                      onClick --> (_ =>
                        rows.update(
                          _ :+ row(parameter.flag, "", removed = true)
                        )
                      )
                    )
                  else emptyNode
                )
              )
            )
          )
      },
      button(
        cls := "button is-small is-info mb-3 add-row-button",
        span(cls := "plus-icon", "+"),
        " Add parameter",
        onClick --> (_ => rows.update(_ :+ row("", "", removed = false)))
      )
    )
  }
}
