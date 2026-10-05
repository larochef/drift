package drift.frontend.pages.gallery

import drift.frontend.components.*
import drift.frontend.pages.gallery.PostProcessSection.*
import drift.frontend.services.ApiClient
import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

/** Auto redraw (`specs/52-auto-redraw.md`): the assistant reads the picture
  * once, scaled down with the job's tiles drawn on it, and answers with each
  * tile's prompt and strength and with the repairs it proposes. Nothing runs
  * unseen: the tiles are listed and stay editable, the job starts on the
  * panel's own button, and a repair — which changes what the picture shows — is
  * only ever a button here.
  */
class AutoRedrawCard(
    image: Signal[Option[GenerationOutput]],
    /** The fields a reading is asked for, as the form has them now. */
    fields: () => RedrawPlanRequest,
    /** Whether a box narrows the job: a reading is of the whole picture. */
    hasSelection: Signal[Boolean],
    /** Sets the form up to repaint one proposed repair. */
    onRepair: PlannedRepair => Unit
) extends Component {

  private val planFn = ApiClient.streamWithFailureReason(planRedraw)
  private val asks = new EventBus[(String, String, RedrawPlanRequest)]

  private val planVar = Var(Option.empty[RedrawPlan])

  /** The reading's tiles as the user left them. */
  private val tilesVar = Var(List.empty[PlannedTile])
  private val askingVar = Var(false)
  private val errorVar = Var(Option.empty[String])
  private val useVar = Var(true)
  private val listVar = Var(false)

  /** What the job carries: each tile's settings while a reading is in use,
    * nothing otherwise.
    */
  def settings: List[TileSettings] =
    if (useVar.now() && planVar.now().isDefined) tilesVar.now().map(_.settings)
    else Nil

  private def change(name: String)(edit: TileSettings => TileSettings): Unit =
    tilesVar.update(
      _.map(tile =>
        if (tile.name == name) tile.copy(settings = edit(tile.settings))
        else tile
      )
    )

  private def tileRow(
      name: String,
      tile: Signal[PlannedTile]
  ): HtmlElement =
    div(
      cls := "auto-redraw-tile",
      span(cls := "auto-redraw-name", name),
      span(
        cls := "auto-redraw-holds is-size-7 text-secondary",
        child.text <-- tile.map(_.holds.mkString(", "))
      ),
      input(
        cls := "input is-small auto-redraw-strength",
        typ := "number",
        stepAttr := "0.05",
        minAttr := "0",
        maxAttr := "1",
        title := "this tile's strength — 0 leaves it as it is",
        controlled(
          value <-- tile.map(_.settings.strength.toString),
          onInput.mapToValue.map(_.toDoubleOption) --> (_.foreach(strength =>
            change(name)(_.copy(strength = strength))
          ))
        )
      ),
      input(
        cls := "input is-small auto-redraw-prompt",
        typ := "text",
        title := "what this tile shows, added after the restoration prompt",
        controlled(
          value <-- tile.map(_.settings.prompt),
          onInput.mapToValue --> (text => change(name)(_.copy(prompt = text)))
        )
      )
    )

  private def repairRow(repair: PlannedRepair): HtmlElement =
    div(
      cls := "auto-redraw-repair",
      span(
        cls := "is-size-7",
        strong(repair.defect),
        Option.when(repair.fix.nonEmpty)(s" → ${repair.fix}")
      ),
      button(
        cls := "button is-small is-light",
        title := "selects this part and sets the form to repaint it alone, " +
          "at a high strength — nothing starts until you redraw the selection",
        "Set up this repair",
        onClick --> (_ => onRepair(repair))
      )
    )

  private val summary: Signal[String] =
    planVar.signal.combineWith(tilesVar.signal).map {
      case (None, _)           => ""
      case (Some(plan), tiles) =>
        val painted = tiles.map(_.settings.strength).filter(_ > 0)
        val untouched = tiles.size - painted.size
        s"${tiles.size} tiles read in ${math.round(plan.seconds)} s" +
          (if (painted.nonEmpty)
             s" · strength ${painted.min} to ${painted.max}"
           else "") +
          (if (untouched > 0) s" · $untouched left as they are" else "")
    }

  lazy val element: HtmlElement = div(
    cls := "auto-redraw",
    // a reading belongs to the picture it was made on
    image.map(_.map(_.fileName)).distinct --> (_ => {
      planVar.set(None)
      tilesVar.set(Nil)
      errorVar.set(None)
    }),
    asks.events.flatMapSwitch(input => planFn(input).recoverToTry) --> Observer[
      Try[RedrawPlan]
    ] {
      case Success(plan) =>
        askingVar.set(false)
        errorVar.set(None)
        tilesVar.set(plan.tiles)
        planVar.set(Some(plan))
      case Failure(err) =>
        askingVar.set(false)
        errorVar.set(Some(Option(err.getMessage).getOrElse(err.toString)))
    },
    group(
      "auto",
      plainField(
        button(
          cls := "button is-small",
          cls("is-loading") <-- askingVar.signal,
          disabled <-- askingVar.signal.combineWith(hasSelection).map(_ || _),
          title := "the running assistant reads the whole picture once, with " +
            "these tiles drawn on it, and sets each tile's prompt and " +
            "strength; it needs a model that reads images. A reading is of " +
            "the whole picture: clear the selection first",
          child.text <-- planVar.signal.map(plan =>
            if (plan.isDefined) "🤖 Ask again" else "🤖 Read the picture"
          ),
          onClick.compose(_.sample(image)) --> (_.foreach { output =>
            askingVar.set(true)
            errorVar.set(None)
            asks.writer.onNext((output.date, output.fileName, fields()))
          })
        )
      ),
      plainField(
        label(
          cls := "checkbox is-size-7",
          cls("is-hidden") <-- planVar.signal.map(_.isEmpty),
          input(
            typ := "checkbox",
            controlled(
              checked <-- useVar.signal,
              onClick.mapToChecked --> useVar
            )
          ),
          " use it for this redraw"
        )
      ),
      span(
        cls := "is-size-7 text-secondary auto-redraw-summary",
        child.text <-- summary
      )
    ),
    p(
      cls := "is-size-7 has-text-danger auto-redraw-error",
      cls("is-hidden") <-- errorVar.signal.map(_.isEmpty),
      child.text <-- errorVar.signal.map(_.getOrElse(""))
    ),
    div(
      cls("is-hidden") <-- planVar.signal.map(_.isEmpty),
      children <-- planVar.signal.map(
        _.toList
          .flatMap(_.notes)
          .map(note => p(cls := "is-size-7 text-secondary", note))
      ),
      div(
        cls := "auto-redraw-repairs",
        children <-- planVar.signal.map(
          _.toList.flatMap(_.repairs).map(repairRow)
        )
      ),
      FoldedSection(
        title = Val("Tiles"),
        open = listVar,
        body = div(
          cls := "auto-redraw-tiles",
          children <-- tilesVar.signal.split(_.name)((name, _, tile) =>
            tileRow(name, tile)
          )
        )
      ).element
    )
  )
}
