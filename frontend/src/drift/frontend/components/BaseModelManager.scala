package drift.frontend.components

import com.raquo.laminar.api.L.*

/** A small list of words with an add box: an architecture's Civitai base
  * models, and its tags.
  */
class BaseModelManager(initialModels: List[String] = Nil) {
  private val currentModels = Var(initialModels)
  private[frontend] val newInput = Var("")

  val models: Signal[List[String]] = currentModels.signal

  def addBaseModel(): Unit = {
    val input = newInput.now().trim
    if (input.nonEmpty) {
      val current = currentModels.now()
      if (!current.contains(input)) {
        currentModels.set(current :+ input)
        newInput.set("")
      }
    }
  }

  def removeBaseModel(name: String): Unit = {
    currentModels.update(_.filter(_ != name))
  }

  def snapshot(): List[String] = currentModels.now()

  def reset(models: List[String] = Nil): Unit = {
    currentModels.set(models)
    newInput.set("")
  }
}

/** An architecture's Civitai base models — what its model and LoRA browsers
  * filter Civitai by: each configured one removable, followed in the same row
  * by a "+" tag of the same look — or a "Configure Civitai base models" button
  * while there are none — that opens a box where Enter adds the name typed and
  * Escape closes it (François, 2026-09-15).
  */
class CivitaiBaseModelsEditor(
    manager: BaseModelManager,
    onRemove: String => Unit,
    /** Called once a name is added, for a caller that saves at once. */
    onSave: () => Unit = () => (),
    /** Whether to show its own title; a form labels its fields itself. */
    heading: Boolean = true
) extends Component {

  private val adding = Var(false)

  /** The look every chip of the row shares, the "+" included. */
  private val chipClass = "tag is-info is-small m-1"
  private val chipStyle =
    "display: inline-flex; align-items: center; width: fit-content; padding: 0.5em 0.6em 0.5em 0.6em; margin: 0.25em; line-height: 1;"

  private def close(): Unit = {
    manager.newInput.set("")
    adding.set(false)
  }

  private def add(): Unit = {
    val before = manager.snapshot()
    manager.addBaseModel()
    if (manager.snapshot() != before) {
      onSave()
      adding.set(false)
    }
  }

  private def chip(bm: String): HtmlElement =
    span(
      cls := chipClass,
      styleAttr := chipStyle,
      span(styleAttr := "display: inline-block;", bm),
      span(
        styleAttr := "display: inline-flex; align-items: center; line-height: 1; font-size: 1em; cursor: pointer; padding: 0 0.4em; margin-left: 0.4em;",
        title := s"Remove $bm",
        onClick --> (_ => onRemove(bm)),
        "x"
      )
    )

  private def plusChip: HtmlElement =
    span(
      cls := s"$chipClass cursor-pointer",
      styleAttr := chipStyle,
      title := "Add a Civitai base model",
      "+",
      onClick --> (_ => adding.set(true))
    )

  private def addBox: HtmlElement =
    div(
      cls := "field has-addons mt-1",
      div(
        cls := "control is-expanded",
        input(
          cls := "input is-small",
          placeholder := "e.g. Flux.2 Klein 9B — Enter to add",
          value <-- manager.newInput.signal,
          onInput.mapToValue --> manager.newInput,
          onKeyDown.filter(_.key == "Enter") --> (_ => add()),
          onKeyDown.filter(_.key == "Escape") --> (_ => close()),
          onMountCallback(context => context.thisNode.ref.focus())
        )
      ),
      div(
        cls := "control",
        button(cls := "button is-small", "Cancel", onClick --> (_ => close()))
      )
    )

  lazy val element: HtmlElement = div(
    cls := "mt-3",
    if (heading)
      p(
        cls := "has-text-weight-bold text-secondary is-size-7",
        "Civitai base models"
      )
    else emptyNode,
    div(
      cls := "tags mb-0",
      // The "+" sits at the end of the row, beside the names it adds to.
      children <-- manager.models
        .combineWith(adding.signal)
        .map { (models, open) =>
          models.map(chip) ++
            Option.when(models.nonEmpty && !open)(plusChip).toList
        }
    ),
    child <-- adding.signal
      .combineWith(manager.models.map(_.isEmpty).distinct)
      .map {
        case (true, _)     => addBox
        case (false, true) =>
          button(
            cls := "button is-small is-info is-light",
            "+ Configure Civitai base models",
            onClick --> (_ => adding.set(true))
          )
        case (false, false) => emptyNode
      }
  )
}
