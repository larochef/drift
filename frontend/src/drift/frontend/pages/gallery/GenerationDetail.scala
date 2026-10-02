package drift.frontend.pages.gallery

import drift.frontend.components.*
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*

/** One way of reusing the shown generation, decided by the page from what is
  * live: the same configuration takes every parameter, another one the task,
  * nothing running offers a launch first.
  */
/** `run` takes the batch index of the output on screen: reusing one image of a
  * batch reproduces that image (`Generation.ofOutput`).
  */
case class ReuseAction(label: String, buttonClass: String, run: Int => Unit)

/** `otherConfigurations` (id → label) are the models this generation's task can
  * be launched on to compare them on the same job — offered only while nothing
  * is live, since sessions run one at a time.
  */
case class ReuseOffer(
    actions: List[ReuseAction],
    note: Option[String],
    otherConfigurations: List[(String, String)] = List.empty,
    /** The configuration picked, then the batch index shown, as `run`. */
    launchOther: (String, Int) => Unit = (_, _) => (),
    /** A configuration that cannot launch yet: its missing runtime and weights
      * are offered in the actions' place (`specs/46`).
      */
    preparing: Option[String] = None
)

/** A run configuration a post-processing panel offers: a PiD one for PiD, an
  * image one for redraw.
  */
case class ConfigurationOption(
    id: String,
    label: String,
    /** Whether a redraw here sends the whole image beside each tile by default
      * — true where the architecture takes a reference as *context*
      * (`ReferenceImageUse.Context`). A model whose preset edits the reference
      * instead is given none, however capable it looks.
      */
    referenceImages: Boolean,
    /** The multiple every side of this configuration's tiles has to be on
      * (`Architecture.sizeMultiple`), so the panel counts the tiles the backend
      * will actually lay out. Read off the architecture, like the rest of this:
      * no default to fall back to.
      */
    sizeMultiple: Int,
    /** The engine it runs on (`specs/43`), which decides a PiD job's tiles. */
    runner: RuntimeEngine
)

/** The full-size view of one generation: the output (a player for video), a
  * strip to pick through a batch, the input images where present, every
  * recorded parameter, and the actions — reuse, post-hoc upscale
  * (`specs/15-post-hoc-resize.md`, `specs/26-tiled-pid.md`) and delete. A
  * derived entry can be compared with its original at matched zoom; a parent
  * lists what was made from it.
  *
  * Built once per opened generation (`bugs/24`): what changes while it is open
  * — sessions, configurations, the loaded generations, the jobs — comes in as
  * signals, so the compare toggle, the batch strip and typed fields survive a
  * status tick.
  */
class GenerationDetail(
    generation: Generation,
    configurationLabel: Signal[String],
    offer: Signal[ReuseOffer],
    upscalers: Signal[List[Upscaler]],
    /** Run configurations of pixel-diffusion-decoder architectures — the
      * diffusion upscalers.
      */
    pidConfigurations: Signal[List[ConfigurationOption]],
    /** Run configurations of `image` architectures — what can redraw. */
    redrawConfigurations: Signal[List[ConfigurationOption]],
    restorationTemplates: Signal[List[PromptTemplate]],
    /** Run configurations of architectures tagged `edit` — what can edit by
      * instruction (`specs/39-seamless-edit.md`).
      */
    editConfigurations: Signal[List[ConfigurationOption]],
    editTemplates: Signal[List[PromptTemplate]],
    /** Live sessions, id → label: each holds a model in memory beside any
      * upscale job, so the section offers to stop them (never on its own).
      */
    liveSessions: Signal[List[(String, String)]],
    jobs: Signal[List[PostProcessJob]],
    /** The original this entry was made from, once the loaded generations hold
      * it.
      */
    parent: Signal[Option[Generation]],
    /** The loaded gallery entries its inputs were picked from (`specs/50`). */
    inputs: Signal[List[Generation]],
    derivatives: Signal[List[Generation]],
    /** The prompt the PiD panel prefills — inherited through the derivation
      * chain (`RecordedParameters.sourcePromptOf`).
      */
    sourcePrompt: String,
    onOpen: String => Unit,
    /** Stages an image output for the assistant (`specs/21-assistant-page.md`),
      * with the configuration's label as it reads now.
      */
    onAskAssistant: (GenerationOutput, String) => Unit,
    onUpscale: (GenerationOutput, UpscaleRequest) => Unit,
    onPid: (GenerationOutput, PidUpscaleRequest) => Unit,
    onRedraw: (GenerationOutput, RedrawRequest) => Unit,
    onEdit: (GenerationOutput, EditRequest) => Unit,
    onCancelJob: String => Unit,
    /** Pauses a tiled job after its current tile, and carries a paused one on
      * (`specs/40-pause-and-resume.md`).
      */
    onPauseJob: (String, Boolean) => Unit,
    onResumeJob: String => Unit,
    /** Closes a finished job's card, leaving the job and its result alone. */
    onDismissJob: String => Unit,
    onStopSession: String => Unit,
    onClose: () => Unit,
    onDelete: () => Unit,
    /** Which section of the right column is open. The host holds it, so the
      * choice survives opening the next image.
      */
    openSection: Var[String],
    /** Which post-processing task is on screen — redraw, PiD upscale or the
      * upscaler. The host holds it like the open section, and the selection
      * tool on the picture follows it.
      */
    openTask: Var[String],
    /** Whether the right column is folded away, leaving the whole card to the
      * picture — what a landscape image wants, where width is the limit and
      * height is not.
      */
    panelHidden: Var[Boolean],
    /** What each configuration still has to download: a model that cannot start
      * yet is offered its download instead (`specs/46`).
      */
    prerequisites: LaunchPrerequisites,
    /** Which output of a batch the strip starts on — the one that was clicked
      * where the caller shows a batch as separate tiles.
      */
    initialOutputIndex: Int = 0
) extends Component {

  /** What is on screen and what is drawn over it — one chain, in one place
    * (`DetailPicture`).
    */
  private val picture = DetailPicture(
    generation,
    parent,
    jobs,
    openSection,
    panelHidden,
    openTask,
    initialOutputIndex
  )

  /** Which section of the right column is open; the host holds it, so the
    * choice survives opening the next image.
    */
  /** The images that went *into* this generation — an init image, a mask, the
    * references — which a plain text-to-image run does not have. The section is
    * offered only when there is something in it: an empty box invites the
    * question of what it was for (François, 2026-09-19).
    */
  private val inputImages: List[(String, String)] =
    RecordedParameters.inputMedia(generation)

  private def sections: List[DetailSection] = List(
    DetailSection(
      "parameters",
      "Parameters",
      Val(None),
      div(parameters.element)
    ),
    DetailSection(
      "post",
      "Redraw & upscale",
      jobs
        .map(_.count(job => job.state.isActive))
        .distinct
        .map(running => Option.when(running > 0)(s"$running running")),
      postProcessSection
    )
  ) ++ Option
    .when(inputImages.nonEmpty)(
      DetailSection("inputs", "Inputs", Val(None), div(parameters.inputs))
    )
    .toList ++ List(
    DetailSection(
      "history",
      "History",
      parent
        .combineWith(derivatives)
        .map((from, made) =>
          from.size + generation.inputSources.size + made.size
        )
        .distinct
        .map(count => Option.when(count > 0)(count.toString)),
      GenerationLineage(generation, parent, inputs, derivatives, onOpen).element
    )
  )

  /** Built once: the panels keep what is typed into them while the tabs move
    * from section to section.
    */
  private lazy val postProcessSection: HtmlElement =
    PostProcessSection(
      picture.selectedIndex.signal.map(generation.outputs.lift),
      upscalers,
      pidConfigurations,
      redrawConfigurations,
      restorationTemplates,
      editConfigurations,
      editTemplates,
      liveSessions,
      jobs,
      prerequisites,
      picture.viewed,
      picture.redrawGeometry,
      picture.pidTiles,
      picture.showTileGrid,
      picture.gridOffset,
      sourcePrompt,
      openTask,
      onOpen,
      onUpscale,
      onPid,
      onRedraw,
      onEdit,
      onCancelJob,
      onPauseJob,
      onResumeJob,
      onDismissJob,
      onStopSession
    ).element

  /** Result / Original, where there is an original to compare with. */
  private def compareToggle: Node =
    if (generation.derivation.isEmpty) emptyNode
    else
      div(
        cls := "buttons has-addons gallery-compare",
        button(
          cls <-- picture.showOriginal.signal.map(o =>
            s"button is-small ${if (o) "" else "is-primary is-selected"}"
          ),
          "Result",
          onClick --> (_ => picture.showOriginal.set(false))
        ),
        button(
          cls <-- picture.showOriginal.signal.map(o =>
            s"button is-small ${if (o) "is-primary is-selected" else ""}"
          ),
          "Original",
          onClick --> (_ => picture.showOriginal.set(true))
        )
      )

  /** The file itself, in a tab of its own: the page shows a copy scaled to the
    * screen and offers no zoom, so this is where full resolution is looked at.
    */
  private def fullSizeLink: Signal[Option[HtmlElement]] =
    picture.shownOutput
      .map(_.filterNot(GenerationMediaViewer.isVideo).map { output =>
        a(
          cls := "button is-small gallery-full-size",
          href := output.url,
          // `rel="external"` is what frontroute's LinkHandler reads; without
          // it a same-origin link is swallowed into the router, which has no
          // route for an API path (François, 2026-09-19).
          rel := "external",
          target := "_blank",
          title := "the file as it is, at full resolution, in a new tab",
          "⤢ Full size"
        )
      })
      .distinct

  /** The picture a job on this output has made so far, at full resolution in a
    * tab of its own — what the finished tiles look like before the job ends.
    */
  private def jobPictureLink: Signal[Option[HtmlElement]] =
    picture.jobPicture
      .map(_.map(_.fullSize))
      .distinct
      .map(_.map { url =>
        a(
          cls := "button is-small gallery-full-size",
          href := url,
          rel := "external",
          target := "_blank",
          title := "the picture as the job has made it so far, at full " +
            "resolution, in a new tab",
          "⤢ Full size so far"
        )
      })

  private val parameters = GenerationParameters(
    generation,
    configurationLabel,
    picture.selectedIndex.signal.map(
      generation.outputs.lift(_).fold(0)(_.index)
    )
  )

  /** The batch index of the output on screen — not its place in the strip,
    * which differs once a batch has lost an image.
    */
  private def shownOutputIndex(): Int =
    generation.outputs.lift(picture.selectedIndex.now()).fold(0)(_.index)

  /** The configuration picked to try the task on; empty, or one no longer
    * offered, means the first offered.
    */
  private val selectedOther = Var("")

  private def tryOnAnotherModel(offer: ReuseOffer): Node =
    if (offer.otherConfigurations.isEmpty) emptyNode
    else {
      val ids = offer.otherConfigurations.map(_._1)
      val chosen =
        selectedOther.signal.map(id => if (ids.contains(id)) id else ids.head)
      div(
        cls := "field has-addons mb-0 gallery-try-on",
        div(
          cls := "control",
          div(
            cls := "select",
            select(
              onChange.mapToValue --> selectedOther,
              offer.otherConfigurations.map { (id, label) =>
                option(
                  value := id,
                  selected <-- chosen.map(_ == id),
                  label
                )
              }
            )
          )
        ),
        div(
          cls := "control",
          LaunchOrDownload(
            prerequisites.of(chosen),
            prerequisites,
            button(
              cls := "button is-link",
              "▶ Try this task on that model",
              onClick.compose(_.sample(chosen)) --> (id =>
                offer.launchOther(id, shownOutputIndex())
              )
            )
          ).element
        )
      )
    }

  lazy val element: HtmlElement = div(
    cls := "modal is-active gallery-detail",
    ScrollLock.whileMounted,
    documentEvents(_.onKeyDown).filter(_.key == "Escape") --> (_ => onClose()),
    div(cls := "modal-background", onClick --> (_ => onClose())),
    div(
      cls := "modal-card",
      headerTag(
        cls := "modal-card-head",
        p(
          cls := "modal-card-title",
          s"${RecordedParameters.kindLabel(generation)} · " +
            RecordedParameters.dateTimeOf(generation.submittedAt)
        ),
        // What is being looked at belongs in the header, where nothing has to
        // be scrolled to reach it; what is done with it stays in the footer.
        Option.when(generation.outputs.size > 1)(
          span(
            cls := "tag is-small gallery-batch-count",
            child.text <-- picture.selectedIndex.signal.map(index =>
              s"${index + 1} of ${generation.outputs.size}"
            )
          )
        ),
        compareToggle,
        child.maybe <-- fullSizeLink,
        child.maybe <-- jobPictureLink,
        button(
          cls := "button is-small gallery-panel-toggle",
          child.text <-- panelHidden.signal.map(hidden =>
            if (hidden) "◧ Details" else "◨ Hide details"
          ),
          title := "fold the panel away and give the whole card to the picture",
          onClick --> (_ => panelHidden.update(!_))
        ),
        button(
          cls := "delete",
          aria.label := "close",
          onClick --> (_ => onClose())
        )
      ),
      sectionTag(
        cls := "modal-card-body",
        div(
          cls := "columns",
          div(
            cls <-- panelHidden.signal.map(hidden =>
              if (hidden) "column is-12" else "column is-7"
            ),
            GenerationMediaViewer(
              generation,
              picture.selectedIndex,
              picture.shownOutput,
              picture.selectable,
              picture.viewed,
              picture.redrawGeometry.signal,
              picture.drawnTiles,
              picture.jobPicture.map(_.map(_.screen)).distinct,
              picture.showTileGrid.signal,
              picture.gridOffset
            ).element
          ),
          div(
            cls := "column is-5 gallery-detail-side",
            cls("is-hidden") <-- panelHidden.signal,
            DetailTabs(sections, openSection).element
          )
        )
      ),
      footerTag(
        cls := "modal-card-foot gallery-detail-foot",
        children <-- offer.map(
          _.actions.map(action =>
            button(
              cls := s"button ${action.buttonClass}",
              action.label,
              onClick --> (_ => action.run(shownOutputIndex()))
            )
          )
        ),
        child <-- offer.map(
          _.preparing.fold(emptyNode)(id =>
            LaunchOrDownload(
              prerequisites.of(Val(id)),
              prerequisites,
              span()
            ).element
          )
        ),
        child <-- offer.map(tryOnAnotherModel),
        generation.outputs
          .find(_.mimeType.startsWith("image/"))
          .map(output =>
            button(
              cls := "button",
              "🤖 Ask the assistant",
              title := "Send this image and its parameters to the assistant",
              onClick.compose(_.sample(configurationLabel)) --> (label =>
                onAskAssistant(
                  generation.outputs
                    .lift(picture.selectedIndex.now())
                    .getOrElse(output),
                  label
                )
              )
            )
          ),
        child <-- offer.map(
          _.note.fold(emptyNode: Node)(note =>
            p(cls := "text-secondary is-size-7", note)
          )
        ),
        button(
          cls := "button is-danger",
          "🗑 Delete",
          onClick --> (_ => onDelete())
        ),
        button(cls := "button", "Close", onClick --> (_ => onClose()))
      )
    )
  )
}
