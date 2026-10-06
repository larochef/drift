package drift.frontend.pages.inference

import drift.frontend.components.{Component, ErrorBanner}
import drift.frontend.pages.architectures.ModelForm
import drift.frontend.services.{BrowserServices, ModelService}
import drift.shared.*

import com.raquo.laminar.api.L.*

/** A model per checkpoint slot of the architecture, and a way to register one
  * the slot lacks without going through the architectures page: **+ Add model**
  * opens the architecture page's own `ModelForm` — its browser on the slot's
  * architecture — and the model it saves is assigned to the slot. Only the
  * slots of the chosen runner show (`bugs/36`): one that is another runner's is
  * neither asked for nor passed.
  */
class CheckpointAssignments(
    architecture: Option[Architecture],
    modelService: ModelService,
    browsers: BrowserServices,
    assignments: Var[List[(String, String)]],
    runner: Signal[RuntimeEngine]
) extends Component {
  private val checkpoints = architecture.map(_.checkpoints).getOrElse(Nil)

  /** The model being added, in its own modals. Outside the rows, which the
    * model list rebuilds, so a pick survives the rebuild.
    */
  private val adding = Var(Option.empty[ModelForm])

  /** Saved but not yet stored: the slot is assigned once the server has the
    * model, so a refused one never leaves the slot pointing at nothing.
    */
  private var pending = Map.empty[String, String]

  private def assign(slot: String, modelId: String): Unit =
    assignments.update { assigns =>
      val filtered = assigns.filter(_._1 != slot)
      if (modelId.nonEmpty) filtered :+ (slot -> modelId) else filtered
    }

  private def save(slot: String, model: Model): Unit = {
    pending += model.id -> slot
    modelService.push(ModelService.Command.Create(model))
    adding.set(None)
  }

  /** The slot's current model, for its select to show. */
  private def assigned(slot: String): Signal[String] =
    assignments.signal.map(_.toMap.getOrElse(slot, "")).distinct

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
        ErrorBanner(modelService),
        modelService.events --> Observer[ModelService.Event] {
          case ModelService.Event.Created(model) =>
            pending.get(model.id).foreach { slot =>
              pending -= model.id
              assign(slot, model.id)
            }
          case _ => ()
        },
        child <-- adding.signal.map(_.map(_.element).getOrElse(emptyNode)),
        children <-- modelService.allModels.combineWith(runner).map {
          (models, engine) =>
            checkpoints.filter(_.appliesTo(engine)).map { cp =>
              val familyModels = models.filter(m => m.familyId == cp.familyId)
              div(
                cls := "checkpoint-slot",
                p(
                  cls := "checkpoint-slot-name text-secondary is-size-7",
                  s"${cp.name} (${cp.flag})"
                ),
                // Controlled by the assignments rather than rebuilt with them,
                // so a pick is never replaced under the user, and a model
                // added from here shows as picked once it is stored.
                select(
                  cls := "select checkpoint-slot-select",
                  onChange.mapToValue --> (v => assign(cp.name, v)),
                  option(
                    value := "",
                    if (cp.required) "Select model (required)" else "None"
                  ),
                  familyModels.map(m => option(value := m.id, m.label)),
                  value <-- assigned(cp.name)
                ),
                architecture.map(arch =>
                  button(
                    cls := "button is-small is-info checkpoint-slot-add",
                    span(cls := "plus-icon", "+"),
                    " Add model",
                    title := s"Find a ${cp.name} model and register it " +
                      s"with ${arch.label}",
                    onClick --> (_ =>
                      adding.set(
                        Some(
                          ModelForm(
                            browsers,
                            arch,
                            cp.familyId,
                            modelService.allModels,
                            onCancel = () => adding.set(None),
                            onSave = model => save(cp.name, model),
                            slot = Some(cp)
                          )
                        )
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
