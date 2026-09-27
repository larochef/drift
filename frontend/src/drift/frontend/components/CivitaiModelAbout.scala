package drift.frontend.components

import drift.frontend.services.CivitaiService
import drift.shared.*

import com.raquo.laminar.api.L.*

/** An opened Civitai model's About tab
  * (`specs/24-model-details-in-browsers.md`): its description, tags, and each
  * version's notes and trigger words, as the detail endpoint cleaned them — the
  * search payload carries none.
  */
class CivitaiModelAbout(
    service: CivitaiService,
    modelId: String,
    /** The base models of the architecture being browsed, for the version
      * chips.
      */
    civitaiBaseModels: List[String]
) extends Component {

  private def versionNotes(version: CivitaiModelVersion): HtmlElement =
    div(
      cls := "version-section mt-5",
      div(
        cls := "is-flex is-align-items-center mb-2",
        h3(
          cls := "is-size-6 has-text-weight-bold has-text-primary",
          version.name
        ),
        CivitaiTags.baseModel(version, civitaiBaseModels)
      ),
      Option.when(version.trainedWords.nonEmpty)(
        div(
          cls := "tags mb-2",
          span(cls := "is-size-7 text-secondary mr-2", "Trigger words"),
          version.trainedWords.map(word =>
            span(cls := "tag is-info is-small", word)
          )
        )
      ),
      version.description.map(html => RichText(html).element)
    )

  lazy val element: HtmlElement =
    div(
      p(
        cls := "is-size-7 mb-3",
        a(
          href := s"https://civitai.com/models/$modelId",
          target := "_blank",
          rel := "noopener noreferrer",
          "Open on civitai.com ↗"
        )
      ),
      child <-- service.detail.combineWith(service.searching).map {
        case (None, true) =>
          p(cls := "text-secondary", "Loading the description...")
        // A failed load: the error banner above says why.
        case (None, false)     => emptyNode
        case (Some(detail), _) =>
          div(
            Option.when(detail.tags.nonEmpty)(
              div(
                cls := "tags mb-3",
                detail.tags.map(tag => span(cls := "tag is-small", tag))
              )
            ),
            detail.description match {
              case Some(html) => RichText(html).element
              case None       => p(cls := "text-secondary", "No description.")
            },
            detail.modelVersions
              .filter(version =>
                version.description.nonEmpty || version.trainedWords.nonEmpty
              )
              .map(versionNotes)
          )
      }
    )
}
