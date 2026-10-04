package drift.frontend.pages.gallery

import drift.frontend.components.{Component, FoldedSection}
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Post-processing of the selected output (`specs/15-post-hoc-resize.md`,
  * `specs/26-tiled-pid.md`, `specs/27-redraw.md`, `specs/39-seamless-edit.md`,
  * `specs/51-seedvr2-upscaling.md`): three tasks — redraw, edit, upscale — one
  * on screen at a time behind a picker (a video has the upscale alone, the
  * only one that takes it), and beneath them what they all share, the live
  * sessions holding memory and the jobs on this output.
  *
  * One at a time rather than three rows stacked (François, 2026-09-20): only
  * one is ever run, redraw has the fields of a small form and wants the height,
  * and the box dragged on the picture means something to redraw alone — so the
  * picker is also what tells the viewer whether to offer the selection tool at
  * all.
  *
  * Every panel is built **once** and hidden rather than removed, like the tabs
  * above them (`DetailTabs`): what has been typed into a panel survives a
  * status tick, a change of output, and a change of task.
  */
class PostProcessSection(
    selectedOutput: Signal[Option[GenerationOutput]],
    upscalers: Signal[List[Upscaler]],
    pidConfigurations: Signal[List[ConfigurationOption]],
    seedVr2Configurations: Signal[List[ConfigurationOption]],
    redrawConfigurations: Signal[List[ConfigurationOption]],
    restorationTemplates: Signal[List[PromptTemplate]],
    editConfigurations: Signal[List[ConfigurationOption]],
    editTemplates: Signal[List[PromptTemplate]],
    liveSessions: Signal[List[(String, String)]],
    jobs: Signal[List[PostProcessJob]],
    /** What each configuration still has to download: a panel whose model
      * cannot start yet offers the download instead of its job.
      */
    prerequisites: LaunchPrerequisites,
    /** The image on screen and the box drawn on it, for the redraw panel. */
    viewed: Var[Option[ViewedImage]],
    /** Where the redraw panel publishes what a selection would cost. */
    geometry: Var[RedrawGeometry],
    /** Where the upscale task publishes the tiles its job would run, in the
      * picture's own pixels — its grid is not a redraw's.
      */
    pidTiles: Var[List[ImageRegion]],
    /** Whether the picture shows the tiles a redraw would run — the redraw
      * panel's checkbox, the viewer's to draw.
      */
    showTileGrid: Var[Boolean],
    /** Where that grid is cut: the viewer moves it, the redraw panel sends it
      * and resets it.
      */
    gridOffset: Var[TileOffset],
    sourcePrompt: String,
    /** Which task is on screen. The host holds it, so the choice survives
      * opening the next image, and the viewer reads it to decide whether a box
      * may be drawn.
      */
    openTask: Var[String],
    onOpen: String => Unit,
    onUpscale: (GenerationOutput, UpscaleRequest) => Unit,
    onPid: (GenerationOutput, PidUpscaleRequest) => Unit,
    onSeedVr2: (GenerationOutput, SeedVr2UpscaleRequest) => Unit,
    onRedraw: (GenerationOutput, RedrawRequest) => Unit,
    onEdit: (GenerationOutput, EditRequest) => Unit,
    onCancelJob: String => Unit,
    onPauseJob: (String, Boolean) => Unit,
    onResumeJob: String => Unit,
    onDismissJob: String => Unit,
    onStopSession: String => Unit
) extends Component {

  /** The output the panels act on — none for a video. */
  private val image: Signal[Option[GenerationOutput]] =
    selectedOutput.map(_.filterNot(GenerationMediaViewer.isVideo))

  private val isVideo: Signal[Boolean] =
    selectedOutput.map(_.exists(GenerationMediaViewer.isVideo)).distinct

  /** The task on screen: the open one, or the only one a video has. */
  private val shownTask: Signal[String] =
    openTask.signal
      .combineWith(isVideo)
      .map((task, video) => if (video) PostProcessSection.UpscaleTask else task)
      .distinct

  /** Redraw first: it is the one that reads the picture, and the one most often
    * wanted on an image that is already the right size; edit beside it, since
    * it reads the picture the same way.
    */
  private val tasks: List[(String, String, HtmlElement)] = List(
    (
      PostProcessSection.RedrawTask,
      "Redraw",
      RedrawPanel(
        image,
        redrawConfigurations,
        restorationTemplates,
        viewed,
        geometry,
        showTileGrid,
        gridOffset,
        openTask.signal.map(_ == PostProcessSection.RedrawTask),
        prerequisites,
        onRedraw
      ).element
    ),
    (
      PostProcessSection.EditTask,
      "Edit",
      EditPanel(
        image,
        editConfigurations,
        editTemplates,
        viewed,
        geometry,
        showTileGrid,
        gridOffset,
        openTask.signal.map(_ == PostProcessSection.EditTask),
        prerequisites,
        onEdit
      ).element
    ),
    (
      PostProcessSection.UpscaleTask,
      "Upscale",
      UpscaleTaskPanel(
        selectedOutput,
        seedVr2Configurations,
        pidConfigurations,
        upscalers,
        viewed.signal,
        pidTiles,
        showTileGrid,
        sourcePrompt,
        prerequisites,
        onSeedVr2,
        onPid,
        onUpscale
      ).element
    )
  )

  private def picker: HtmlElement =
    div(
      cls := "buttons has-addons post-tasks mb-2",
      tasks.map { case (id, name, _) =>
        button(
          cls := "button is-small",
          cls("is-link") <-- shownTask.map(_ == id),
          cls("is-selected") <-- shownTask.map(_ == id),
          cls("is-hidden") <-- isVideo.map(
            _ && id != PostProcessSection.UpscaleTask
          ),
          name,
          onClick --> (_ => openTask.set(id))
        )
      }
    )

  lazy val element: HtmlElement = div(
    cls := "gallery-scale mt-2",
    cls("is-hidden") <-- selectedOutput.map(_.isEmpty),
    onMountCallback(_ =>
      if (!tasks.map(_._1).contains(openTask.now()))
        openTask.set(PostProcessSection.RedrawTask)
    ),
    picker,
    tasks.map { case (id, _, body) =>
      div(
        cls := "post-panel",
        cls("is-hidden") <-- shownTask.map(_ != id),
        body
      )
    },
    LiveSessionNotice(liveSessions, onStopSession).element,
    PostProcessJobList(
      selectedOutput,
      jobs,
      onOpen,
      onCancelJob,
      onPauseJob,
      onResumeJob,
      onDismissJob
    ).element
  )
}

/** The task names, and the controls the three panels share. */
object PostProcessSection {

  val RedrawTask = "redraw"
  val EditTask = "edit"

  /** The tasks that read the picture: a box may be drawn and the tiles are
    * shown while one of them is open.
    */
  val TiledTasks: Set[String] = Set(RedrawTask, EditTask)
  val UpscaleTask = "upscale"

  /** A panel's opening line: what this task does to the image, and what it
    * costs. Each panel says it for itself — one caption over all three could
    * only be true of the part of them it was written for.
    */
  def intro(text: String): HtmlElement =
    p(cls := "post-intro is-size-7 text-secondary", text)

  /** One labelled row of a panel: the name of the group at the left, its fields
    * beside it, wrapping under one another when the column is narrow. Every
    * field carries its own name, so a wrapped row still reads — which a single
    * joined strip of a dozen controls never did.
    */
  def group(name: String, fields: Modifier[HtmlElement]*): HtmlElement =
    div(
      cls := "post-group",
      span(cls := "post-group-label is-size-7 text-secondary", name),
      div(cls := "post-group-fields", fields)
    )

  /** One field of a group: its own name, then the control, joined. */
  def field(name: String, control: HtmlElement): HtmlElement =
    div(
      cls := "field has-addons post-field",
      div(cls := "control", span(cls := "button is-small is-static", name)),
      div(cls := "control", control)
    )

  /** A field the group's own name is enough for. */
  def plainField(control: HtmlElement): HtmlElement =
    div(cls := "field post-field", div(cls := "control", control))

  /** A field that takes the whole width of its group, and a line of its own — a
    * prompt, where the room is the point.
    */
  def wideField(control: HtmlElement): HtmlElement =
    div(
      cls := "field post-field is-wide",
      div(cls := "control is-expanded", control)
    )

  /** Everything of a task but the model, folded away behind its title: the
    * three panels each show a model and a button, and nothing else until it is
    * asked for (François, 2026-09-21). A panel that fits on two lines can be
    * read at a glance, and these parameters are set once and left alone.
    *
    * `FoldedSection` is the architecture page's LoRA fold, body kept rather
    * than removed — what has been typed in survives closing it, as it survives
    * a change of task.
    */
  def advanced(open: Var[Boolean], rows: Modifier[HtmlElement]*): HtmlElement =
    FoldedSection(
      title = Val("Advanced"),
      open = open,
      keepBody = true,
      body = div(rows)
    ).element.amend(cls := "post-advanced")

  /** A panel's last line: what the job would be, then the button that starts
    * it. Nothing of a task is placed after its own button.
    */
  def foot(summary: Modifier[HtmlElement], action: HtmlElement): HtmlElement =
    div(
      cls := "post-foot",
      div(cls := "post-foot-summary is-size-7 text-secondary", summary),
      action
    )

  def numberField(state: Var[String], width: String): HtmlElement =
    input(
      cls := "input is-small",
      typ := "number",
      styleAttr := s"width: $width;",
      value <-- state.signal,
      onInput.mapToValue --> state
    )

  /** A prompt: prose, several lines of it, in the width the panel has. A
    * restoration prompt and a negative are read and edited, not glanced at — a
    * one-line input showed a sliver of either (François, 2026-09-20).
    */
  def promptField(
      state: Var[String],
      lines: Int,
      hint: String
  ): HtmlElement =
    textArea(
      cls := "textarea is-small",
      rows := lines,
      placeholder := hint,
      value <-- state.signal,
      onInput.mapToValue --> state
    )

  /** A checkbox of a panel, sized and aligned like the fields beside it. */
  def checkField(
      state: Var[Boolean],
      name: String,
      explanation: String
  ): HtmlElement =
    label(
      cls := "checkbox is-size-7 post-check",
      title := explanation,
      input(
        typ := "checkbox",
        checked <-- state.signal,
        onClick.mapToChecked --> state
      ),
      s" $name"
    )

  /** A tiled job's seed field: empty draws one when the job starts, and every
    * tile of the job runs with that one seed.
    */
  def seedField(state: Var[String]): HtmlElement =
    field(
      "seed",
      numberField(state, "8rem").amend(
        minAttr := "0",
        placeholder := "random",
        title := "empty: a seed drawn when the job starts — every tile " +
          "runs with the same one, recorded on the result"
      )
    )

  /** The request's seed: -1, which draws one, unless a non-negative one is set.
    */
  def seedOf(state: Var[String]): Long =
    state.now().trim.toLongOption.filter(_ >= 0).getOrElse(-1L)
}
