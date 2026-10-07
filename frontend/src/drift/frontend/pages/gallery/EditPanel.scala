package drift.frontend.pages.gallery

import drift.frontend.components.*
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.pages.gallery.TileAreaFields.TilePlan
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Edit through a run configuration of an `edit` architecture
  * (`specs/39-seamless-edit.md`): the instruction says what should be
  * different, and the rest of the picture stays what it was — drift keeps the
  * source's own pixels wherever the model changed nothing.
  *
  * The instruction is the one field always on screen: an edit is nothing
  * without it. A box dragged on the picture narrows the change to that part,
  * through the same tiles, window and margin a redraw uses.
  *
  * With a SeedVR2 upscaler the edit is made once, on the part reduced to what
  * the model takes in one pass, and what changed is carried up by the upscaler:
  * the default when one is installed, since an edit made tile by tile cannot
  * agree with itself across tiles.
  */
class EditPanel(
    image: Signal[Option[GenerationOutput]],
    /** Run configurations of architectures tagged `edit`. */
    editConfigurations: Signal[List[ConfigurationOption]],
    /** The edit templates to choose from (`specs/32`). */
    editTemplates: Signal[List[PromptTemplate]],
    /** The SeedVR2 configurations an edit can be carried up by. */
    upscaleConfigurations: Signal[List[ConfigurationOption]],
    viewed: Var[Option[ViewedImage]],
    geometry: Var[TileGeometry],
    showTileGrid: Var[Boolean],
    gridOffset: Var[TileOffset],
    /** Whether edit is the task on screen — the one whose tiles the picture
      * draws.
      */
    active: Signal[Boolean],
    /** What each configuration still has to download: the job's button becomes
      * the download while the chosen model cannot start (`specs/46`).
      */
    prerequisites: LaunchPrerequisites,
    onEdit: (GenerationOutput, EditRequest) => Unit
) extends Component {

  private val configurationVar = Var("")
  private val instructionVar = Var("")
  private val templateVar =
    Var[Option[String]](Some(PromptTemplate.DefaultEditId))
  private val stepsVar = Var("")
  private val seedVar = Var("")
  private val keepTilesVar = Var(false)
  private val beyondSelectionVar = Var(true)
  private val advancedVar = Var(false)
  // The upscaler picked — the empty one is "tile by tile" — or nothing picked
  // yet: the first one installed.
  private val upscalerVar = Var[Option[String]](None)

  /** The upscaler the edit is carried up by, if any. */
  private val upscaler: Signal[Option[ConfigurationOption]] =
    upscaleConfigurations
      .combineWith(upscalerVar.signal)
      .map((list, picked) =>
        // the built-in default at the head, as the upscale task has it
        picked.fold(
          list
            .find(_.id == SeedVr2UpscaleRequest.DefaultConfiguration)
            .orElse(list.headOption)
        )(id => list.find(_.id == id))
      )
      .distinct
  private val upscalerNow = Var[Option[ConfigurationOption]](None)

  private val hasConfigurations: Signal[Boolean] =
    editConfigurations.map(_.nonEmpty).distinct

  private val area = TileAreaFields(
    TileAreaFields.selectionCut(editConfigurations, configurationVar.signal),
    takesSelection = true,
    viewed,
    geometry,
    showTileGrid,
    gridOffset,
    active
  )

  private val hasInstruction: Signal[Boolean] =
    instructionVar.signal.map(_.trim.nonEmpty).distinct

  private def request: EditRequest =
    EditRequest(
      runConfigurationId = configurationVar.now(),
      instructions = instructionVar.now().trim,
      templateId = templateVar.now(),
      steps = stepsVar.now().trim.toIntOption,
      seed = seedOf(seedVar),
      runtimeId = None,
      tileSize = area.tileSize,
      gridOffsetX = area.gridOffsetX,
      gridOffsetY = area.gridOffsetY,
      region = area.region,
      minimumWindowSide = area.minimumWindowSide,
      selectionMargin = area.selectionMargin,
      keepTiles = keepTilesVar.now(),
      upscaleConfigurationId = upscalerNow.now().map(_.id),
      beyondSelection = beyondSelectionVar.now()
    )

  private def configurationSelect: HtmlElement =
    div(
      cls := "select is-small",
      select(
        onChange.mapToValue --> configurationVar,
        children <-- editConfigurations.map(
          _.map(c =>
            option(
              value := c.id,
              selected <-- configurationVar.signal.map(_ == c.id),
              c.label
            )
          )
        )
      )
    )

  /** How the edit reaches the picture's size: carried up by an upscaler, or
    * made tile by tile.
    */
  private def upscalerSelect: HtmlElement =
    div(
      cls := "select is-small",
      select(
        title := "the edit is made once, on the picture reduced to what the " +
          "model takes in one pass, and what changed is brought back to the " +
          "picture's size by this upscaler; tile by tile, each tile is " +
          "edited on its own",
        onChange.mapToValue.map(Some(_)) --> upscalerVar,
        children <-- upscaleConfigurations.map(
          _.map(c =>
            option(
              value := c.id,
              selected <-- upscaler.map(_.exists(_.id == c.id)),
              c.label
            )
          ) :+ option(
            value := "",
            selected <-- upscaler.map(_.isEmpty),
            "none — tile by tile"
          )
        )
      )
    )

  /** What should be different: the reason for the job, so not folded away. */
  private def instructionGroup: HtmlElement = group(
    "change",
    wideField(
      promptField(
        instructionVar,
        lines = 3,
        hint = "what should be different — \"make the bikini top red\", " +
          "\"she wears a thin gold necklace\""
      )
    )
  )

  /** What the model is told before the instruction. */
  private def templateGroup: HtmlElement = group(
    "prompt",
    plainField(PromptTemplatePicker(editTemplates, templateVar).element)
  )

  /** The pass: no strength — the model is given the tile to edit, not a picture
    * to repaint from part of the way.
    */
  private def passGroup: HtmlElement = group(
    "pass",
    field(
      "steps",
      numberField(stepsVar, "4.5rem").amend(
        minAttr := "1",
        placeholder := "own",
        title := "empty: the configuration's own steps"
      )
    ),
    seedField(seedVar)
  )

  /** What a selection keeps of the change, and what is left on disk when the
    * job is done.
    */
  private def optionsGroup: HtmlElement = group(
    "options",
    checkField(
      beyondSelectionVar,
      "follow the change past the box",
      "an edit carried up keeps every changed part that touches the box, " +
        "as far as it goes — hair cut short is cut down to its ends, outside " +
        "the box too. Unticked, nothing outside the box and its margin changes"
    ),
    checkField(
      keepTilesVar,
      "keep the tiles",
      "keep every tile's input, the model's raw edit, the change mask — " +
        "white where drift took the edit, black where it kept the original " +
        "— and the composite, in the job's tiles directory"
    )
  )

  /** The last line: what this job would change and what it costs. */
  private val costLine: Signal[Node] =
    area.costLine(
      stepsVar.signal
        .combineWith(upscaler)
        .map((steps, carrier) =>
          carrier.fold(EditPanel.costOf(_, steps))(by =>
            EditPanel.carriedCostOf(_, steps, by.label, area.selectionMargin)
          )
        )
    )

  private def form: HtmlElement = div(
    cls("is-hidden") <-- hasConfigurations.map(!_),
    intro(
      "Say what should be different; the model makes the change and drift " +
        "keeps the original wherever nothing changed, so the rest of the " +
        "picture stays as it was. Drag a box to change that part alone; " +
        "the original stays in the gallery."
    ),
    group("model", plainField(configurationSelect)),
    group("carried up by", plainField(upscalerSelect)),
    instructionGroup,
    advanced(
      advancedVar,
      templateGroup,
      passGroup,
      area.areaGroup,
      optionsGroup
    ),
    foot(
      div(
        cls := "post-foot-lines",
        div(child <-- costLine),
        area.gridControls
      ),
      LaunchOrDownload(
        prerequisites.of(configurationVar.signal),
        prerequisites,
        button(
          cls := "button is-small is-link",
          disabled <-- hasInstruction.map(!_),
          title <-- hasInstruction.map(present =>
            if (present) "" else "say what should be different first"
          ),
          child.text <-- area.hasSelection.map(selected =>
            if (selected) "✨ Edit selection" else "✨ Edit"
          ),
          onClick.compose(_.sample(image)) --> (_.foreach(output =>
            onEdit(output, request)
          ))
        )
      ).element
    )
  )

  lazy val element: HtmlElement = div(
    area.publishGeometry,
    upscaler --> upscalerNow,
    editConfigurations --> Observer[List[ConfigurationOption]](list =>
      if (configurationVar.now().isEmpty)
        list.headOption.foreach(option => configurationVar.set(option.id))
    ),
    p(
      cls := "post-intro text-secondary",
      cls("is-hidden") <-- hasConfigurations,
      "Nothing to edit with yet: an edit needs a run configuration on an " +
        "architecture tagged edit — a model that changes an image by " +
        "instruction, such as Flux.2 Klein."
    ),
    form
  )
}

object EditPanel {

  /** The line before the button: what is changed, and what it costs. The model
    * starts every tile from noise, so each samples all of its steps.
    */
  def costOf(plan: TilePlan, steps: String): String = {
    val sampling = steps.trim.toIntOption match {
      case None        => "the configuration's steps"
      case Some(asked) => s"$asked steps each"
    }
    s"${TileAreaFields.extentOf(plan)} · $sampling"
  }

  /** The same line for an edit carried up: one pass of the model, at the size
    * the part is reduced to, then the upscaler over what changed.
    */
  def carriedCostOf(
      plan: TilePlan,
      steps: String,
      upscaler: String,
      margin: Int
  ): String = {
    val sampling = steps.trim.toIntOption match {
      case None        => "the configuration's steps"
      case Some(asked) => s"$asked steps"
    }
    plan.selection match {
      case None =>
        val pass =
          EditRequest.passOf(plan.window.width, plan.window.height, 16)
        val carried =
          if (pass.scale == 1) "at its own size"
          else s"what changed carried up ×${pass.scale} by $upscaler"
        s"whole image ${plan.window.width}×${plan.window.height} · one pass " +
          s"at ${pass.width}×${pass.height}, $sampling · $carried"
      case Some(selection) =>
        // The window is the backend's to lay out against the picture's edges;
        // its side says what the pass costs.
        val side = EditRequest.contextSide(selection, margin)
        val pass = EditRequest.passOf(side, side, 16)
        val carried =
          if (pass.scale == 1)
            s"at its own size, nothing to carry up — seen wider, and carried " +
              s"up by $upscaler, if the change runs out of that window"
          else s"carried up ×${pass.scale} by $upscaler"
        s"selection ${selection.width}×${selection.height}, seen in a window " +
          s"about $side px wide · one pass, $sampling · $carried"
    }
  }
}
