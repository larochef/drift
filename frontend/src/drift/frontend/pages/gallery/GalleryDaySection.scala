package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.frontend.gallery.{GenerationCard, GenerationDetailHost}
import drift.frontend.services.HistoryService
import drift.frontend.services.HistoryService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*

/** One day's group. Keyed rendering throughout: a card is built once per
  * generation id and survives socket pushes, filter changes and other days
  * loading, so thumbnails are not re-requested on every update.
  */
final class GalleryDaySection(
    date: String,
    daySignal: Signal[HistoryDay],
    historyService: HistoryService,
    filters: GalleryFilters,
    labels: GalleryLabels,
    selection: GallerySelection,
    /** Opens a card's output in the detail view. */
    onOpen: GenerationDetailHost.Open => Unit
) extends Component {

  private val loaded: Signal[Option[List[Generation]]] =
    historyService.generationsByDay.map(_.get(date)).distinct
  private val loading: Signal[Boolean] =
    historyService.loadingDays.map(_.contains(date)).distinct
  private val shown: Signal[List[Generation]] =
    loaded.combineWith(filters.filter).map { (loaded, matches) =>
      loaded.getOrElse(List.empty).filter(matches)
    }

  lazy val element: HtmlElement = div(
    cls := "gallery-day",
    dataAttr("date") := date,
    // Under a filter a day shows for what it has that matches: not while it
    // is still being read, and not at all with nothing to show.
    display <-- loaded.combineWith(shown, filters.filtering, loading).map {
      case (Some(_), Nil, _, _)  => "none"
      case (None, _, true, true) => "none"
      case _                     => ""
    },
    div(
      cls := "gallery-day-heading",
      h2(cls := "subtitle text-primary mb-0", date),
      span(
        cls := "tag",
        // What is shown once the day is read; what it holds until then.
        child.text <-- loaded.combineWith(shown, daySignal).map {
          case (Some(_), matching, _) => matching.size.toString
          case (None, _, day)         => day.count.toString
        }
      ),
      child <-- loaded.combineWith(loading, daySignal).map {
        case (None, true, _) =>
          span(cls := "text-secondary is-size-7", "loading…")
        case (None, false, day) =>
          button(
            cls := "button is-small",
            s"Show ${day.count} generation${if (day.count == 1) "" else "s"}",
            onClick --> (_ => historyService.push(Command.LoadDay(date)))
          )
        case (Some(_), _, _) => emptyNode
      },
      child <-- selection.selecting.signal.map {
        case false => emptyNode
        case true  =>
          a(
            cls := "is-size-7",
            "select all of this day",
            onClick.compose(_.sample(shown)) --> Observer[List[Generation]](
              generations =>
                selection.selection.update(_ ++ generations.map(_.id))
            )
          )
      }
    ),
    div(
      cls := "gallery-grid",
      children <-- shown
        .combineWith(labels.labels)
        .map((generations, known) =>
          // One tile per output: a batch's images each get theirs, side by
          // side. A generation without outputs still
          // shows as one tile.
          generations.flatMap(generation =>
            (if (generation.outputs.isEmpty) List(0)
             else generation.outputs.indices.toList)
              .map(index => (generation, index, known))
          )
        )
        .split(tile => (tile._1.id, tile._2)) { (_, initial, current) =>
          val (generation, index, known) = initial
          val id = generation.id
          GenerationCard(
            generation,
            labels.labelOf(known, generation.runConfigurationId),
            () => onOpen(GenerationDetailHost.Open(id, index)),
            outputIndex = index,
            // from the record as it now is: a move shows on the card at once
            projectLabel = current
              .map(_._1.projectId)
              .combineWith(labels.projectLabels.signal)
              .map((projectId, known) => projectId.flatMap(known.get))
              .distinct,
            selecting = selection.selecting.signal,
            selected = selection.selection.signal.map(_(id)),
            onToggleSelected = () =>
              selection.selection.update(current =>
                if (current(id)) current - id else current + id
              )
          ).element
        }
    )
  )
}
