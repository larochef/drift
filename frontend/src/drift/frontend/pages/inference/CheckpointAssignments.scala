package drift.frontend.pages.inference

import drift.frontend.components.Component
import drift.shared.{CheckpointRef, Model}

import com.raquo.laminar.api.L.*

class CheckpointAssignments(
    checkpoints: List[CheckpointRef],
    allModels: Signal[List[Model]],
    assignments: Var[List[(String, String)]]
) extends Component {
  lazy val element: HtmlElement =
    if (checkpoints.isEmpty) {
      div(
        p(
          cls := "text-secondary is-size-7",
          "Select an architecture to assign models to checkpoints."
        )
      )
    } else {
      div(
        cls := "field",
        label(cls := "label text-primary", "Checkpoint assignments"),
        children <-- allModels.map { models =>
          // Read the assignments synchronously instead of combining the two
          // signals. This list is rebuilt whenever `models` arrives, and the
          // rebuild has to carry the current selection over; re-rendering on
          // every assignment change as well would replace the select out from
          // under the user the instant they pick an option.
          val current = assignments.now().toMap
          checkpoints.map { cp =>
            val familyModels = models.filter(m => m.familyId == cp.familyId)
            val assigned = current.getOrElse(cp.name, "")
            div(
              cls := "columns is-mobile is-vcentered mb-1",
              div(
                cls := "column is-3",
                p(
                  cls := "text-secondary is-size-7",
                  s"${cp.name} (${cp.flag})"
                )
              ),
              div(
                cls := "column",
                select(
                  cls := "select",
                  onChange.mapToValue --> (v =>
                    assignments.update { assigns =>
                      val filtered = assigns.filter(_._1 != cp.name)
                      if (v.nonEmpty) filtered :+ (cp.name -> v) else filtered
                    }
                  ),
                  option(
                    value := "",
                    if (cp.required) "Select model (required)" else "None",
                    selected := assigned.isEmpty
                  ),
                  familyModels.map(m =>
                    option(
                      value := m.id,
                      m.label,
                      selected := assigned == m.id
                    )
                  )
                )
              )
            )
          }
        }
      )
    }
}
