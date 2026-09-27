package drift.frontend.pages.projects

import drift.frontend.components.Component
import drift.frontend.pages.gallery.{GenerationCard, GenerationDetailHost}
import drift.frontend.services.{AssistantService, HistoryService}
import drift.shared.*

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.StringAsIsCodec

/** The project's results under the generation panel, grouped by the version
  * that made them, newest version first.
  */
class ProjectResults(
    project: Signal[Option[Project]],
    generations: Signal[List[Generation]],
    assistantService: AssistantService,
    historyService: HistoryService,
    onSelectVersion: PromptVersion => Unit,
    /** Opens the detail on one output of a generation. */
    onOpen: (String, Int) => Unit,
    /** Makes an output the project's cover, or with `None` goes back to the
      * newest result.
      */
    onCover: Option[GenerationOutput] => Unit
) extends Component {

  private val loadingAttr = htmlAttr("loading", StringAsIsCodec)

  private val tileStyle =
    "max-height: 160px; max-width: 220px; border-radius: 4px; display: block;"

  /** One result: the image or video opens the detail on itself, and the buttons
    * under it stage it for the assistant, make it the project's cover, or throw
    * the whole generation away — a misfire (too few steps, a cat with two
    * tails) is worth deleting on the spot, files and all (François,
    * 2026-09-09).
    */
  private def resultTile(
      generation: Generation,
      output: GenerationOutput,
      index: Int,
      project: Option[Project]
  ): HtmlElement = {
    val isVideo = output.mimeType.startsWith("video/")
    val isCover = project.flatMap(_.cover).exists(_.isOf(output))
    val seed = generation.imageParameters
      .map(_.seed)
      .orElse(generation.videoParameters.map(_.seed))
      .getOrElse(-1L)
    val media: Seq[Modifier[HtmlElement]] = Seq(
      src := output.url,
      styleAttr := tileStyle,
      cls := "cursor-pointer",
      title := s"seed $seed — click for the details",
      onClick --> (_ => onOpen(generation.id, index))
    )
    div(
      // A video shows its first frame, as on a gallery card.
      if (isVideo)
        videoTag(
          GenerationCard.preloadAttr := "metadata",
          GenerationCard.mutedAttr := true,
          GenerationCard.playsInlineAttr := true,
          media
        )
      else img(loadingAttr := "lazy", media),
      div(
        cls := "buttons are-small mt-1",
        styleAttr := "gap: 0.25rem;",
        // The assistant is sent images, as from the gallery's detail.
        Option.when(!isVideo)(
          button(
            cls := "button is-small",
            "🤖",
            title := "Send this image and its parameters to the assistant",
            onClick --> { _ =>
              assistantService.attach(
                AssistantService.outputAttachment(
                  generation,
                  output,
                  generation.runConfigurationId
                )
              )
            }
          )
        ),
        // Only what the project's card can show: the cover endpoint ignores a
        // chosen output of the other kind.
        Option.when(
          project.exists(p => output.mimeType.startsWith(s"${p.kind.noun}/"))
        )(
          button(
            cls := (if (isCover) "button is-small is-info"
                    else "button is-small"),
            "🖼",
            title := (
              if (isCover)
                "The project's cover — click to show the newest result instead"
              else "Use as the project's cover, shown on its card"
            ),
            onClick --> (_ => onCover(Option.when(!isCover)(output)))
          )
        ),
        button(
          cls := "button is-small is-danger is-outlined",
          "🗑",
          title :=
            (if (generation.outputs.size > 1)
               s"Delete this generation — all ${generation.outputs.size} outputs of the batch"
             else "Delete this result and its files"),
          onClick --> (_ =>
            GenerationDetailHost.confirmAndDelete(generation, historyService)
          )
        )
      )
    )
  }

  lazy val element: HtmlElement = div(
    cls := "mt-4",
    h2(cls := "is-size-6 text-primary", "Results"),
    children <-- generations.combineWith(project).map {
      (generations, project) =>
        val versions = project.map(_.versions).getOrElse(Nil)
        val completed =
          generations.filter(_.status == GenerationStatus.Completed)
        if (completed.isEmpty)
          List(p(cls := "text-secondary is-size-7", "Nothing generated yet."))
        else
          completed
            .groupBy(_.promptVersionId)
            .toList
            .sortBy((id, _) =>
              -versions.find(v => id.contains(v.id)).map(_.number).getOrElse(0)
            )
            .map { (versionId, list) =>
              val version = versions.find(v => versionId.contains(v.id))
              div(
                cls := "mb-3",
                p(
                  cls := "text-secondary is-size-7 mb-1",
                  version
                    .map(v => s"v${v.number} · ${v.note}")
                    .getOrElse("untagged"),
                  version.map(v =>
                    a(
                      cls := "ml-2",
                      "select",
                      onClick --> (_ => onSelectVersion(v))
                    )
                  )
                ),
                div(
                  cls := "is-flex is-flex-wrap-wrap",
                  styleAttr := "gap: 0.5rem;",
                  list.flatMap(generation =>
                    // Indexed before filtering: the detail's strip counts
                    // every output.
                    generation.outputs.zipWithIndex
                      .filter((output, _) =>
                        output.mimeType.startsWith("image/") ||
                          output.mimeType.startsWith("video/")
                      )
                      .map((output, index) =>
                        resultTile(generation, output, index, project)
                      )
                  )
                )
              )
            }
    }
  )
}
