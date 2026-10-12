package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.frontend.gallery.{GenerationDetailHost, ProjectMover}
import drift.frontend.services.HistoryService
import drift.frontend.services.HistoryService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** The gallery's bulk selection: which cards are ticked, and the bar that moves
  * or deletes them.
  */
final class GallerySelection(
    /** Everything the loaded days hold, which the ticked ids are read in. */
    loaded: Signal[List[Generation]],
    historyService: HistoryService,
    projects: Signal[List[Project]]
) extends Component {

  /** Clearing out a day's worth of misfires one confirmation at a time is the
    * wrong tool: while selecting, a click ticks a card instead of opening it,
    * and one button deletes everything ticked. The ids are enough — the day
    * comes off the generation when the delete goes out.
    */
  val selecting = Var(false)
  val selection = Var(Set.empty[String])

  def toggleSelecting(): Unit = {
    selection.set(Set.empty)
    selecting.update(!_)
  }

  /** One confirmation for the whole sweep, naming what it costs on disk; the
    * deletes then go out one per generation, the same command a single delete
    * uses.
    */
  private def deleteSelected(loaded: List[Generation]): Unit = {
    val chosen = loaded.filter(generation => selection.now()(generation.id))
    if (chosen.nonEmpty) {
      val files = chosen.map(GenerationDetailHost.fileCount).sum
      val confirmed = window.confirm(
        s"Delete ${chosen.size} generation${
            if (chosen.size == 1) "" else "s"
          }" +
          s" and their $files file(s)?\n\nThis cannot be undone."
      )
      if (confirmed) {
        chosen.foreach(GenerationDetailHost.delete(_, historyService))
        selection.set(Set.empty)
      }
    }
  }

  /** The bulk bar, shown only while selecting. */
  lazy val element: HtmlElement = div(
    child <-- selecting.signal.map {
      case false => emptyNode
      case true  =>
        div(
          cls := "gallery-selection-bar mb-4",
          span(
            cls := "text-primary is-size-7",
            child.text <-- selection.signal.map(chosen =>
              s"${chosen.size} selected"
            )
          ),
          span(
            cls := "text-secondary is-size-7",
            "clicking a card ticks it instead of opening it"
          ),
          button(
            cls := "button is-small",
            "Clear",
            disabled <-- selection.signal.map(_.isEmpty),
            onClick --> (_ => selection.set(Set.empty))
          ),
          ProjectMover(
            projects,
            targets = loaded
              .combineWith(selection.signal)
              .map((list, chosen) => list.filter(g => chosen(g.id))),
            state = historyService.moveState,
            onMove = (chosen, projectId) =>
              historyService.push(Command.Move(chosen, projectId)),
            label = count => s"Move $count selected"
          ).element,
          button(
            cls := "button is-small is-danger",
            child.text <-- selection.signal.map(chosen =>
              s"🗑 Delete ${chosen.size} selected"
            ),
            disabled <-- selection.signal.map(_.isEmpty),
            onClick.compose(_.sample(loaded)) --> Observer[List[Generation]](
              deleteSelected
            )
          )
        )
    }
  )
}
