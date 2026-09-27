package drift.frontend.pages.gallery

import drift.frontend.components.{Component, ScrollLock}
import drift.shared.*

import com.raquo.laminar.api.L.*

/** What a generation recorded: every parameter and where it ran (`element`),
  * and the input images it was given (`inputs`), which the detail shows below
  * the chain.
  */
class GenerationParameters(
    generation: Generation,
    configurationLabel: Signal[String]
) extends Component {

  private def row(name: String, value: String): HtmlElement =
    tr(
      td(cls := "text-secondary", name),
      td(cls := "text-primary text-break", value)
    )

  /** The fields that are prose rather than a value. In the table they were
    * squeezed into the half a column the labels left, for the one field the
    * page most needs to read (François, 2026-09-19), so they get the full width
    * instead, shortened, with the whole text a click away.
    */
  private val proseFields = Set("Prompt", "Negative prompt", "Instructions")

  /** The prose field opened over the page, if any. */
  private val opened = Var(Option.empty[(String, String)])

  private def proseBlock(name: String, text: String): HtmlElement =
    div(
      cls := "gallery-prose",
      div(
        cls := "gallery-prose-head",
        span(cls := "text-secondary is-size-7", name),
        button(
          cls := "button is-small gallery-prose-open",
          title := s"read the whole ${name.toLowerCase}",
          "⤢",
          onClick --> (_ => opened.set(Some((name, text))))
        )
      ),
      p(
        cls := "text-primary is-size-7 gallery-prose-text",
        title := "click to read it all",
        text,
        onClick --> (_ => opened.set(Some((name, text))))
      )
    )

  /** The whole text, over everything, until it is closed. */
  private def proseModal(name: String, text: String): HtmlElement =
    div(
      cls := "modal is-active gallery-prose-modal",
      ScrollLock.whileMounted,
      documentEvents(_.onKeyDown).filter(_.key == "Escape") --> (_ =>
        opened.set(None)
      ),
      div(cls := "modal-background", onClick --> (_ => opened.set(None))),
      div(
        cls := "modal-card",
        headerTag(
          cls := "modal-card-head",
          p(cls := "modal-card-title", name),
          button(
            cls := "delete",
            aria.label := "close",
            onClick --> (_ => opened.set(None))
          )
        ),
        sectionTag(
          cls := "modal-card-body",
          p(cls := "gallery-prose-full", text)
        )
      )
    )

  lazy val element: HtmlElement = {
    val metadata =
      Option(generation.sessionId)
        .filter(_.nonEmpty)
        .map("Session" -> _)
        .toList ++
        List(
          "Submitted" -> RecordedParameters.dateTimeOf(generation.submittedAt)
        ) ++
        RecordedParameters.durationOf(generation).map("Duration" -> _) ++
        List("Generation id" -> generation.id)
    val (prose, values) =
      RecordedParameters
        .rows(generation)
        .partition((name, _) => proseFields.contains(name))
    div(
      prose.map(proseBlock),
      child.maybe <-- opened.signal.map(_.map(proseModal)),
      table(
        cls := "table is-narrow is-fullwidth gallery-parameters",
        tbody(
          values.map((name, value) => row(name, value)),
          tr(
            td(cls := "text-secondary", "Configuration"),
            td(
              cls := "text-primary text-break",
              child.text <-- configurationLabel
            )
          ),
          metadata.map((name, value) => row(name, value))
        )
      )
    )
  }

  lazy val inputs: Node = {
    val images = RecordedParameters.inputImages(generation)
    if (images.isEmpty) emptyNode
    else
      div(
        cls := "gallery-inputs mt-3",
        p(cls := "text-secondary is-size-7 mb-1", "Input images"),
        div(
          styleAttr := "display: flex; flex-wrap: wrap; gap: 0.5rem;",
          images.map { (name, url) =>
            figure(
              cls := "m-0",
              img(src := url, alt := name, title := name),
              p(cls := "text-secondary is-size-7 has-text-centered", name)
            )
          }
        )
      )
  }
}
