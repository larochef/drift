package drift.frontend.pages.projects

import drift.frontend.components.*
import drift.frontend.pages.gallery.{
  GenerationDetailHost,
  GenerationMediaViewer
}
import drift.frontend.services.{AssistantService, HistoryService}
import drift.shared.*

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.StringAsIsCodec

/** The project's versions (`specs/19-projects-and-prompt-versions.md`), newest
  * first, each with what it made; and the comparison of two versions' prompts.
  * One component at two sizes: the page itself while no model is loaded, and
  * under the result once one is (François, 2026-09-29).
  */
class VersionHistory(
    size: VersionHistory.Size,
    project: Signal[Option[Project]],
    generations: Signal[List[Generation]],
    selectedVersion: Signal[Option[PromptVersion]],
    /** The selected version's id as it stands, for the comparison to open on.
      */
    selectedVersionIdNow: () => Option[String],
    currentProject: () => Option[Project],
    labelOf: String => Signal[String],
    assistantService: AssistantService,
    historyService: HistoryService,
    onSelect: PromptVersion => Unit,
    /** Opens the detail on one output of a generation. */
    onOpen: (String, Int) => Unit,
    /** Makes an output the project's cover, or with `None` goes back to the
      * newest result.
      */
    onCover: Option[GenerationOutput] => Unit
) extends Component {

  private val loadingAttr = htmlAttr("loading", StringAsIsCodec)

  /** Two versions' prompts as a diff (`specs/20`, François 2026-09-11) — "what
    * did I change between the one I liked and this one", v5 against v8.
    */
  private val compareOpen = Var(false)
  private val compareFrom = Var("")
  private val compareTo = Var("")

  /** Opens on the selected version against the one before it. */
  private def openCompare(): Unit =
    currentProject().filter(_.versions.size >= 2).foreach { current =>
      val versions = current.versions.sortBy(_.number)
      val to = ProjectWorkspacePage
        .resolveVersion(current, selectedVersionIdNow())
        .getOrElse(versions.last)
      val from = versions
        .takeWhile(_.number < to.number)
        .lastOption
        .getOrElse(versions.head)
      compareFrom.set(from.id)
      compareTo.set(to.id)
      compareOpen.set(true)
    }

  private def versionSelect(
      state: Var[String],
      versions: List[PromptVersion]
  ): HtmlElement =
    div(
      cls := "select is-small",
      select(
        onChange.mapToValue --> state,
        versions
          .sortBy(-_.number)
          .map(version =>
            option(
              value := version.id,
              selected := state.now() == version.id,
              s"v${version.number} · ${version.note}".take(70)
            )
          )
      )
    )

  private def compareModal: HtmlElement =
    BrowserModal(
      title = Val("Compare versions"),
      body = Seq(
        child <-- project
          .combineWith(compareFrom.signal, compareTo.signal)
          .map { (current, fromId, toId) =>
            val versions = current.map(_.versions).getOrElse(Nil)
            (versions.find(_.id == fromId), versions.find(_.id == toId)) match {
              case (Some(from), Some(to)) =>
                div(
                  div(
                    cls := "is-flex is-align-items-center mb-3",
                    styleAttr := "gap: 0.5rem;",
                    versionSelect(compareFrom, versions),
                    span("→"),
                    versionSelect(compareTo, versions)
                  ),
                  p(cls := "has-text-weight-bold is-size-7 mb-1", "Prompt"),
                  PromptDiff(
                    from.prompt,
                    to.prompt,
                    s"v${from.number}"
                  ).element,
                  if (
                    from.negativePrompt.nonEmpty ||
                    to.negativePrompt.nonEmpty
                  )
                    div(
                      p(
                        cls := "has-text-weight-bold is-size-7 mb-1 mt-3",
                        "Negative prompt"
                      ),
                      PromptDiff(
                        from.negativePrompt,
                        to.negativePrompt,
                        s"v${from.number}"
                      ).element
                    )
                  else emptyNode
                )
              case _ => p(cls := "text-secondary", "Pick two versions.")
            }
          }
      ),
      onCancel = () => compareOpen.set(false),
      footerRight = button(
        cls := "button",
        "Close",
        onClick --> (_ => compareOpen.set(false))
      )
    ).element

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
    val seed = generation
      .seedOf(output.index)
      .orElse(generation.videoParameters.map(_.seed))
      .getOrElse(-1L)
    val media: Seq[Modifier[HtmlElement]] = Seq(
      cls := "version-history-media cursor-pointer",
      title := s"seed $seed — click for the details",
      onClick --> (_ => onOpen(generation.id, index))
    )
    div(
      // A video shows its still, as on a gallery card: only the detail's
      // player loads it (bug 37).
      img(
        loadingAttr := "lazy",
        src := (
          if (isVideo) GenerationMediaViewer.stillUrl(output) else output.url
        ),
        media
      ),
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

  private def tiles(
      made: List[Generation],
      project: Option[Project]
  ): HtmlElement = div(
    cls := "version-history-tiles",
    made.flatMap(generation =>
      // Indexed before filtering: the detail's strip counts every output.
      generation.outputs.zipWithIndex
        .filter((output, _) =>
          output.mimeType.startsWith("image/") ||
            output.mimeType.startsWith("video/")
        )
        .map((output, index) => resultTile(generation, output, index, project))
    )
  )

  /** One version: what changed and on which configuration, the way to take its
    * recipe, and what it made. The prompt is the tooltip.
    */
  private def versionRow(
      version: PromptVersion,
      selected: Boolean,
      made: List[Generation],
      project: Option[Project]
  ): HtmlElement = div(
    cls := (if (selected) "version-history-row is-selected"
            else "version-history-row"),
    div(
      cls := "version-history-head",
      title := s"Prompt: ${version.prompt}" +
        (if (version.negativePrompt.trim.isEmpty) ""
         else s"\nNegative: ${version.negativePrompt}"),
      strong(cls := "text-primary", s"v${version.number}"),
      span(
        cls := "text-secondary",
        s" · ${version.note} · ",
        child.text <-- labelOf(version.runConfigurationId)
      ),
      if (selected)
        span(
          cls := "tag is-primary is-light is-small ml-2",
          title := "What the form follows and the next run compares against",
          "selected"
        )
      else
        a(
          cls := "ml-2",
          "use this recipe",
          title := "Seed the form with this version's recipe",
          onClick --> (_ => onSelect(version))
        )
    ),
    if (made.isEmpty)
      p(cls := "text-secondary is-size-7", "Nothing kept from this version.")
    else tiles(made, project)
  )

  lazy val element: HtmlElement = div(
    cls := (size match {
      case VersionHistory.Size.Large => "version-history is-large"
      case VersionHistory.Size.Small => "version-history is-small"
    }),
    child <-- compareOpen.signal.map {
      case true  => compareModal
      case false => emptyNode
    },
    div(
      cls := "is-flex is-align-items-baseline mb-2",
      styleAttr := "gap: 1rem;",
      h2(cls := "is-size-6 text-primary mb-0", "Versions"),
      child <-- project.map(_.exists(_.versions.size >= 2)).distinct.map {
        case true =>
          a(
            cls := "is-size-7",
            "compare",
            title := "Compare two versions' prompts",
            onClick --> (_ => openCompare())
          )
        case false => emptyNode
      }
    ),
    children <-- project
      .combineWith(selectedVersion, generations)
      .map { (project, selected, generations) =>
        val made = generations
          .filter(_.status == GenerationStatus.Completed)
          .groupBy(_.promptVersionId)
        val rows = project
          .map(_.versions.reverse)
          .getOrElse(Nil)
          .map(version =>
            versionRow(
              version,
              selected.exists(_.id == version.id),
              made.getOrElse(Some(version.id), Nil),
              project
            )
          )
        // Results of no version — made before versions existed.
        val untagged = made
          .get(None)
          .map(list =>
            div(
              cls := "version-history-row",
              div(cls := "version-history-head text-secondary", "untagged"),
              tiles(list, project)
            )
          )
        if (rows.isEmpty && untagged.isEmpty)
          List(
            p(
              cls := "text-secondary is-size-7",
              "No version yet. The first generation creates v1."
            )
          )
        else rows ++ untagged
      }
  )
}

object VersionHistory {

  /** How much room the history has: the page's width (large tiles), or the
    * column under the result (small ones).
    */
  enum Size {
    case Large, Small
  }
}
