package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The chain: the original a generation was made from and the gallery entries
  * its inputs were picked from (`specs/50-inputs-from-the-gallery.md`), then
  * what was made from it — a post-processing of it, or a generation it was an
  * input of. All resolved in the loaded generations, so they follow the
  * history.
  */
class GenerationLineage(
    generation: Generation,
    parent: Signal[Option[Generation]],
    /** The loaded entries among those its inputs came from. */
    inputs: Signal[List[Generation]],
    derivatives: Signal[List[Generation]],
    onOpen: String => Unit
) extends Component {

  private def gone(id: String): HtmlElement =
    span(
      cls := "text-primary is-size-7",
      s"$id — not in the loaded days, or deleted"
    )

  private def entryButton(label: String, entry: Generation): HtmlElement =
    button(
      cls := "button is-small",
      s"$label (${RecordedParameters.titleOf(entry).take(40)})",
      onClick --> (_ => onOpen(entry.id))
    )

  lazy val element: HtmlElement = div(
    child <-- parent.combineWith(inputs, derivatives).map {
      (parentNow, inputsNow, derived) =>
        if (
          generation.derivation.isEmpty && generation.inputSources.isEmpty &&
          derived.isEmpty
        ) emptyNode
        else
          div(
            cls := "gallery-lineage mt-3",
            generation.derivation.map { d =>
              div(
                cls := "mb-2",
                span(cls := "text-secondary is-size-7 mr-2", "Made from"),
                parentNow match {
                  case Some(p) => entryButton("Open original", p)
                  case None    => gone(d.parentId)
                }
              )
            },
            if (generation.inputSources.isEmpty) emptyNode
            else
              div(
                cls := "mb-2",
                span(cls := "text-secondary is-size-7 mr-2", "Inputs from"),
                div(
                  cls := "buttons mb-0",
                  generation.inputSources.map(source =>
                    inputsNow.find(_.id == source.generationId) match {
                      case Some(entry) =>
                        entryButton(GenerationLineage.slotLabel(source), entry)
                      case None => gone(source.generationId)
                    }
                  )
                )
              ),
            if (derived.isEmpty) emptyNode
            else
              div(
                span(cls := "text-secondary is-size-7 mr-2", "Made from this"),
                div(
                  cls := "buttons",
                  derived.map(entry =>
                    button(
                      cls := "button is-small",
                      entry.derivation
                        .filter(_.parentId == generation.id)
                        .map(RecordedParameters.operationOf)
                        .orElse(
                          entry.inputSources
                            .find(_.generationId == generation.id)
                            .map(source =>
                              s"${GenerationLineage.slotLabel(source)} of " +
                                RecordedParameters.titleOf(entry).take(40)
                            )
                        )
                        .getOrElse(entry.id),
                      onClick --> (_ => onOpen(entry.id))
                    )
                  )
                )
              )
          )
    }
  )
}

object GenerationLineage {

  /** A slot as the form names it: "init" is the init image, "ref2" the third
    * reference.
    */
  def slotLabel(source: InputSource): String = {
    val name = source.slot.takeWhile(!_.isDigit)
    val position = source.slot
      .drop(name.length)
      .toIntOption
      .fold("")(index => s" ${index + 1}")
    (name match {
      case "init"         => "Init image"
      case "end"          => "End image"
      case "mask"         => "Mask"
      case "ref"          => "Reference"
      case "reference"    => "Reference"
      case "frame"        => "Control frame"
      case "guide"        => "Guide"
      case "control"      => "Control video"
      case "control-mask" => "Control mask"
      case "source"       => "Source video"
      case other          => other
    }) + position
  }
}
