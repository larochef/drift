package drift.frontend.pages.gallery

import drift.frontend.components.*
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.pages.gallery.TileAreaFields.TilePlan
import drift.frontend.services.LaunchPrerequisites
import drift.shared.*

import com.raquo.laminar.api.L.*

/** Redraw through an image run configuration (`specs/27-redraw.md`): a
  * low-strength img2img pass, tile by tile, same size in and out. Strength 0.4,
  * tiles of 1280; empty steps mean the configuration's.
  *
  * A box dragged on the picture narrows the pass to that part: the same tiles
  * over a window grown around the selection, feathered back into the rest.
  *
  * The fields are grouped by what they decide — the prompt, the pass, the
  * reference, the area — and every one of them is on screen: they used to sit
  * in a single wrapping strip behind an "Advanced" button, where a number was
  * rarely beside the thing it belonged to (François, 2026-09-20). What the job
  * will cost, selection or not, is the last line before the button; the tiles
  * are laid out here with the very code the backend runs.
  */
class RedrawPanel(
    image: Signal[Option[GenerationOutput]],
    redrawConfigurations: Signal[List[ConfigurationOption]],
    /** The restoration templates to choose from (`specs/32`). */
    restorationTemplates: Signal[List[PromptTemplate]],
    /** The image on screen and the box drawn on it, from the media viewer. */
    viewed: Var[Option[ViewedImage]],
    /** Where this panel's tile size, margin and window go, for the box on the
      * image to count its tiles and stick to their boundaries.
      */
    geometry: Var[RedrawGeometry],
    /** Whether the picture draws the tiles this panel's fields describe. */
    showTileGrid: Var[Boolean],
    /** How far that grid is shifted: dragged on the picture, sent with the job,
      * and reset from this panel.
      */
    gridOffset: Var[TileOffset],
    /** Whether redraw is the task on screen — the one whose tiles the picture
      * draws.
      */
    active: Signal[Boolean],
    /** What each configuration still has to download: the job's button becomes
      * the download while the chosen model cannot start (`specs/46`).
      */
    prerequisites: LaunchPrerequisites,
    onRedraw: (GenerationOutput, RedrawRequest) => Unit
) extends Component {

  private val configurationVar = Var("")
  private val strengthVar = Var("0.4")
  private val instructionsVar = Var("")
  private val templateVar =
    Var[Option[String]](Some(PromptTemplate.DefaultRedrawId))
  private val negativeVar = Var("")
  private val stepsVar = Var("")
  private val seedVar = Var("")
  private val keepTilesVar = Var(false)
  private val contextSideVar = Var("768")
  private val softenVar = Var("0")
  private val contextVar = Var("0")
  private val advancedVar = Var(false)

  /** Whether the whole image goes along as a reference: empty follows the
    * architecture, which is what most jobs want.
    */
  private val referenceVar = Var[Option[Boolean]](None)

  private val hasConfigurations: Signal[Boolean] =
    redrawConfigurations.map(_.nonEmpty).distinct

  /** What the chosen configuration's architecture does with a reference. */
  private val architectureReference: Signal[Boolean] =
    redrawConfigurations
      .combineWith(configurationVar.signal)
      .map((options, chosen) =>
        options.find(_.id == chosen).exists(_.referenceImages)
      )
      .distinct

  /** Whether this job will send one, the override included — the reference size
    * only means anything then.
    */
  private val sendsReference: Signal[Boolean] =
    referenceVar.signal
      .combineWith(architectureReference)
      .map((forced, byArchitecture) => forced.getOrElse(byArchitecture))
      .distinct

  private def number(state: Var[String], fallback: Int): Int =
    state.now().trim.toIntOption.getOrElse(fallback)

  private val area = TileAreaFields(
    redrawConfigurations,
    configurationVar.signal,
    viewed,
    geometry,
    showTileGrid,
    gridOffset,
    active
  )

  private def request: RedrawRequest =
    RedrawRequest(
      runConfigurationId = configurationVar.now(),
      strength = strengthVar.now().trim.toDoubleOption.getOrElse(0.4),
      instructions = instructionsVar.now(),
      templateId = templateVar.now(),
      negativePrompt = negativeVar.now(),
      steps = stepsVar.now().trim.toIntOption,
      seed = seedOf(seedVar),
      keepTiles = keepTilesVar.now(),
      contextSide = number(contextSideVar, 768),
      tileSize = area.tileSize,
      region = area.region,
      useReference = referenceVar.now(),
      softenRadius = softenVar.now().trim.toDoubleOption.getOrElse(0.0),
      contextMargin = number(contextVar, 0),
      minimumWindowSide = area.minimumWindowSide,
      selectionMargin = area.selectionMargin,
      gridOffsetX = area.gridOffsetX,
      gridOffsetY = area.gridOffsetY
    )

  private def configurationSelect: HtmlElement =
    div(
      cls := "select is-small",
      select(
        onChange.mapToValue --> configurationVar,
        children <-- redrawConfigurations.map(
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

  /** The prompt the tiles are painted with: a template of the library, and the
    * words for this image alone under it.
    */
  private def promptGroups: List[HtmlElement] = List(
    group(
      "prompt",
      plainField(
        PromptTemplatePicker(restorationTemplates, templateVar).element
      ),
      wideField(
        promptField(
          instructionsVar,
          lines = 3,
          hint = "extra instructions, optional — describe what should be " +
            "seen, not what to do"
        )
      )
    ),
    group(
      "negative",
      wideField(
        promptField(
          negativeVar,
          lines = 2,
          hint = "acts only when the configuration's cfg is above 1"
        )
      )
    )
  )

  /** How hard the model is allowed to work on each tile. */
  private def passGroup: HtmlElement = group(
    "pass",
    field(
      "strength",
      numberField(strengthVar, "4.5rem").amend(
        stepAttr := "0.05",
        minAttr := "0.05",
        maxAttr := "1",
        title := "how much the model may change — around 0.35–0.45 repaints " +
          "detail while the whole image, sent beside each tile, holds the " +
          "anatomy"
      )
    ),
    field(
      "steps",
      numberField(stepsVar, "4.5rem").amend(
        minAttr := "1",
        placeholder := "own",
        title := "empty: the configuration's own steps"
      )
    ),
    seedField(seedVar),
    field(
      "soften",
      numberField(softenVar, "4.5rem").amend(
        minAttr := "0",
        maxAttr := "8",
        stepAttr := "1",
        title := "blur each tile by this many px before the model sees it. " +
          "With an eager model such as Krea 2, 1 px takes away the " +
          "upscaler's bumpy edges it would turn into stray hairs; at " +
          "strength 0.6 or more with 16-20 steps, 1-2 px keeps it from " +
          "sharpening waxy leftovers back. On a quiet model (Flux.2 Klein, " +
          "ERNIE) it only flattens the texture. 0 keeps the tile as it is"
      )
    )
  )

  /** What else the model sees: the whole image beside each tile and how big it
    * goes, and the context shown around each tile.
    */
  private def referenceGroup: HtmlElement = group(
    "reference",
    plainField(
      div(
        cls := "select is-small reference-choice",
        select(
          title := "the whole image sent beside each tile: it holds " +
            "composition and identity, costs about three times the time per " +
            "tile, and leaves less new texture. 'model default' sends one " +
            "only where the preset treats it as context: a model with no " +
            "preset ignores it, and an editing model would paint the whole " +
            "picture into every tile.",
          onChange.mapToValue --> Observer[String] {
            case "on"  => referenceVar.set(Some(true))
            case "off" => referenceVar.set(Some(false))
            case _     => referenceVar.set(None)
          },
          option(
            value := "default",
            selected <-- referenceVar.signal.map(_.isEmpty),
            child.text <-- architectureReference.map(on =>
              if (on) "model default (sent)" else "model default (none)"
            )
          ),
          option(
            value := "on",
            selected <-- referenceVar.signal.map(_.contains(true)),
            "always send"
          ),
          option(
            value := "off",
            selected <-- referenceVar.signal.map(_.contains(false)),
            "never send"
          )
        )
      )
    ),
    field(
      "at",
      numberField(contextSideVar, "5rem").amend(
        minAttr := "256",
        maxAttr := "2048",
        stepAttr := "64",
        title := "longest side of the reference image, in px — smaller is " +
          "faster, larger shows the model more"
      )
    ).amend(cls("is-hidden") <-- sendsReference.map(!_)),
    field(
      "context",
      numberField(contextVar, "4.5rem").amend(
        minAttr := "0",
        maxAttr := "512",
        stepAttr := "16",
        title := "px of the picture shown around each tile and kept as it is: " +
          "the model is handed the tile inside that much of its " +
          "surroundings, as redrawn so far, and repaints the tile alone — " +
          "a close-up tile seen alone can be taken for another body part. " +
          "The model paints the whole window, so a 1280 tile with 128 of " +
          "context costs about what a 1536 tile does. 0 sends the tile alone"
      )
    )
  )

  /** What is left on disk when the job is done. */
  private def optionsGroup: HtmlElement = group(
    "options",
    checkField(
      keepTilesVar,
      "keep the tiles",
      "keep every tile's input and output, and the reference image, beside " +
        "the job log"
    )
  )

  /** The last line: what this job would repaint and what it costs. */
  private val costLine: Signal[Node] =
    area.costLine(
      stepsVar.signal
        .combineWith(strengthVar.signal, contextVar.signal)
        .map((steps, strength, context) =>
          RedrawPanel.costOf(_, steps, strength, context)
        )
    )

  private def form: HtmlElement = div(
    cls("is-hidden") <-- hasConfigurations.map(!_),
    intro(
      "The model repaints the image at the size it is, tile by tile — detail " +
        "and texture, not a bigger picture. Drag a box on it to repaint that " +
        "part alone; the original stays in the gallery."
    ),
    group("model", plainField(configurationSelect)),
    advanced(
      advancedVar,
      promptGroups,
      passGroup,
      referenceGroup,
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
          child.text <-- viewed.signal.map(shown =>
            if (shown.exists(_.selection.isDefined)) "✨ Redraw selection"
            else "✨ Redraw"
          ),
          onClick.compose(_.sample(image)) --> (_.foreach(output =>
            onRedraw(output, request)
          ))
        )
      ).element
    )
  )

  lazy val element: HtmlElement = div(
    area.publishGeometry,
    redrawConfigurations --> Observer[List[ConfigurationOption]](list =>
      if (configurationVar.now().isEmpty)
        list.headOption.foreach(option => configurationVar.set(option.id))
    ),
    p(
      cls := "post-intro text-secondary",
      cls("is-hidden") <-- hasConfigurations,
      "Nothing to redraw with yet: create a run configuration on an image " +
        "architecture and it appears here."
    ),
    form
  )
}

object RedrawPanel {

  /** The line before the button: what is repainted, and what it costs. An
    * img2img pass walks its schedule from `strength`, so the steps sampled are
    * a fraction of the steps asked for.
    */
  def costOf(
      plan: TilePlan,
      steps: String,
      strength: String,
      context: String
  ): String = {
    val sampling = steps.trim.toIntOption match {
      case None        => "the configuration's steps"
      case Some(asked) =>
        val sampled = strength.trim.toDoubleOption
          .filter(value => value > 0 && value <= 1)
          .map(value => s", about ${math.round(asked * value)} sampled")
          .getOrElse("")
        s"$asked steps each$sampled"
    }
    val surroundings = context.trim.toIntOption
      .filter(_ > 0)
      .map(px => s" · each seen with $px px around it")
      .getOrElse("")
    s"${TileAreaFields.extentOf(plan)} · $sampling$surroundings"
  }
}
