package drift.frontend.components

import drift.shared.*

import com.raquo.laminar.api.L.*

case class CheckpointRow(
    id: Int,
    name: String,
    flag: String,
    familyId: String,
    required: Boolean
)

class CheckpointEditor(init: List[CheckpointRef] = Nil) extends Component {
  private var nextId = 0
  private val rows: Var[List[CheckpointRow]] = Var(init.map(toRow))

  private def toRow(cp: CheckpointRef): CheckpointRow = {
    val id = nextId
    nextId += 1
    CheckpointRow(id, cp.name, cp.flag, cp.familyId, cp.required)
  }

  def reset(newCheckpoints: List[CheckpointRef] = Nil): Unit = {
    nextId = 0
    rows.set(newCheckpoints.map(toRow))
  }

  def snapshot(): List[CheckpointRef] =
    rows
      .now()
      .map(r => CheckpointRef(r.name, r.familyId, r.flag, r.required))

  lazy val element: HtmlElement =
    div(
      children <-- rows.signal.split(_.id) { (id, initial, rowSignal) =>
        div(
          cls := "columns is-mobile is-vcentered mb-1",
          div(
            cls := "column",
            input(
              cls := "input is-small",
              placeholder := "Name",
              value <-- rowSignal.map(_.name),
              onInput.mapToValue --> (v =>
                rows.update(
                  _.map(r => if (r.id == id) r.copy(name = v) else r)
                )
              )
            )
          ),
          div(
            cls := "column",
            input(
              cls := "input is-small",
              placeholder := "sd-cpp flag",
              value <-- rowSignal.map(_.flag),
              onInput.mapToValue --> (v =>
                rows.update(
                  _.map(r => if (r.id == id) r.copy(flag = v) else r)
                )
              )
            )
          ),
          div(
            cls := "column",
            input(
              cls := "input is-small",
              placeholder := "Family ID",
              value <-- rowSignal.map(_.familyId),
              onInput.mapToValue --> (v =>
                rows.update(
                  _.map(r => if (r.id == id) r.copy(familyId = v) else r)
                )
              )
            )
          ),
          div(
            cls := "column is-narrow",
            label(
              cls := "checkbox text-secondary is-size-7",
              input(
                cls := "mr-1",
                `type` := "checkbox",
                checked <-- rowSignal.map(_.required),
                onInput.mapToChecked --> (v =>
                  rows.update(
                    _.map(r => if (r.id == id) r.copy(required = v) else r)
                  )
                )
              ),
              "required"
            )
          ),
          div(
            cls := "column is-narrow",
            button(
              cls := "button is-danger is-small",
              "\uD83D\uDDD1\uFE0F",
              onClick --> (_ => rows.update(_.filterNot(_.id == id)))
            )
          )
        )
      },
      button(
        cls := "button is-small is-info mb-3 add-row-button",
        span(cls := "plus-icon", "+"),
        " Add checkpoint",
        onClick --> { _ =>
          val id = nextId;
          nextId += 1
          rows.update(_ :+ CheckpointRow(id, "", "", "", required = true))
        }
      )
    )
}
