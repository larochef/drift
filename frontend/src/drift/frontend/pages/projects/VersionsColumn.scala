package drift.frontend.pages.projects

import drift.frontend.components.*
import drift.frontend.pages.gallery.GenerationCard
import drift.shared.*

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.StringAsIsCodec

/** The workspace's versions (`specs/19-projects-and-prompt-versions.md`),
  * newest first, each card its number and the thumbnails of what it made; and
  * the comparison of two versions' prompts.
  */
class VersionsColumn(
    project: Signal[Option[Project]],
    generations: Signal[List[Generation]],
    selectedVersion: Signal[Option[PromptVersion]],
    /** The selected version's id as it stands, for the comparison to open on.
      */
    selectedVersionIdNow: () => Option[String],
    currentProject: () => Option[Project],
    labelOf: String => Signal[String],
    onSelect: PromptVersion => Unit
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

  /** One version: its number and the thumbnails of what it made, nothing else
    * on the card (François, 2026-09-08); origin, note, configuration, count and
    * prompt are the tooltip.
    */
  private def versionRow(
      version: PromptVersion,
      selected: Boolean,
      made: List[Generation]
  ): HtmlElement = div(
    cls := (if (selected) "box bg-card mb-2 workspace-version is-selected"
            else "box bg-card mb-2 workspace-version cursor-pointer"),
    title <-- labelOf(version.runConfigurationId).map(configuration =>
      s"v${version.number} · ${version.origin.toString.toLowerCase} · ${version.note}\n" +
        s"$configuration · ${made.size} ${
            if (version.kind == "vid_gen") "video" else "image"
          }${if (made.size == 1) "" else "s"}\n" +
        s"Prompt: ${version.prompt}" +
        (if (version.negativePrompt.trim.isEmpty) ""
         else s"\nNegative: ${version.negativePrompt}")
    ),
    onClick --> (_ => onSelect(version)),
    p(cls := "text-primary is-size-7 mb-1", strong(s"v${version.number}")),
    div(
      cls := "workspace-version-thumbs",
      made
        .flatMap(_.outputs)
        .take(6)
        .map(output =>
          // A video's thumbnail is its first frame, as on a gallery card.
          if (output.mimeType.startsWith("video/"))
            videoTag(
              src := output.url,
              GenerationCard.preloadAttr := "metadata",
              GenerationCard.mutedAttr := true,
              GenerationCard.playsInlineAttr := true,
              cls := "workspace-version-thumb"
            )
          else
            img(
              src := output.url,
              loadingAttr := "lazy",
              cls := "workspace-version-thumb"
            )
        )
    )
  )

  lazy val element: HtmlElement = div(
    cls := "workspace-versions",
    child <-- compareOpen.signal.map {
      case true  => compareModal
      case false => emptyNode
    },
    div(
      cls := "is-flex is-align-items-baseline is-justify-content-space-between workspace-versions-head",
      h2(cls := "is-size-6 text-primary", "Versions"),
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
        project.map(_.versions.reverse).getOrElse(Nil) match {
          case Nil =>
            List(
              p(
                cls := "text-secondary is-size-7",
                "No version yet. The first generation creates v1."
              )
            )
          case versions =>
            versions.map(version =>
              versionRow(
                version,
                selected.exists(_.id == version.id),
                made.getOrElse(Some(version.id), Nil)
              )
            )
        }
      }
  )
}
