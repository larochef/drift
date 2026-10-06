package drift.frontend.services

import drift.shared.*

import scala.concurrent.duration.DurationInt
import scala.util.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom

object HistoryService {
  enum Command {
    case LoadDays
    case LoadDay(date: String)

    /** Loads whatever day holds this generation — for a page shown one
      * generation by its URL, which may be older than the days it loads up
      * front. A generation already loaded costs nothing.
      */
    case LoadGeneration(generationId: String)
    case Delete(date: String, generationId: String)

    /** An image from outside drift, as a gallery entry of its own (`specs/30`).
      */
    case Import(image: ImageImport)

    /** A video from outside drift, likewise; the file itself is sent. */
    case ImportVideo(file: dom.File)

    /** Entries given to a project, or taken out of any (`specs/19`). */
    case Move(generations: List[Generation], projectId: Option[String])
  }
  enum Event {
    case Deleted(generationId: String)
    case Imported(generation: Generation)

    /** Every entry a move rewrote, what was derived from the moved ones
      * included.
      */
    case Moved(generations: List[Generation])
  }

  /** Where the last move stands, for whatever asked for it to say so: `ids`
    * are the entries asked for, `rewritten` counts what was derived from them
    * too.
    */
  enum MoveState {
    case Idle
    case Moving(ids: Set[String])
    case Done(ids: Set[String], projectId: Option[String], rewritten: Int)
    case Failed(ids: Set[String])
  }
}

/** The gallery's read side (`specs/12-gallery.md`): the day index and the days
  * loaded so far, keyed by date. A day is loaded on demand — the layout on disk
  * is the paging — and a generation that completes while the gallery is open is
  * folded into its day straight off the status socket instead of re-fetching
  * the listing.
  */
class HistoryService(statusSocket: StatusSocketService) extends ServiceErrors {
  import HistoryService.{Command, Event}

  private val daysFn = ApiClient.stream(listHistoryDays)
  private val dayFn = ApiClient.stream(listHistoryDay)
  private val dayOfFn = ApiClient.stream(findHistoryGenerationDay)
  private val deleteFn = ApiClient.stream(deleteHistoryGeneration)
  private val importFn = ApiClient.streamWithFailureReason(importHistoryImage)

  /** A video is as long to send as it is large. */
  private val moveFn = ApiClient.streamWithFailureReason(moveHistoryGenerations)

  private val _moveState = Var[HistoryService.MoveState](
    HistoryService.MoveState.Idle
  )
  val moveState: Signal[HistoryService.MoveState] = _moveState.signal
  private val importVideoFn =
    ApiClient.streamWithFailureReason(importHistoryVideo, within = 30.minutes)

  private val _days = Var(List.empty[HistoryDay])

  /** Every day with recorded generations, newest first. */
  val days: Signal[List[HistoryDay]] = _days.signal

  /** Whether the index has answered at least once — an empty gallery must say
    * "nothing yet", not look like it is still loading.
    */
  private val _daysLoaded = Var(false)
  val daysLoaded: Signal[Boolean] = _daysLoaded.signal

  /** The loaded days' generations, each list newest first. */
  private val _generationsByDay = Var(Map.empty[String, List[Generation]])
  val generationsByDay: Signal[Map[String, List[Generation]]] =
    _generationsByDay.signal

  private val _loadingDays = Var(Set.empty[String])
  val loadingDays: Signal[Set[String]] = _loadingDays.signal

  /** Generations already folded from the socket — every push repeats a
    * session's whole list, so the fold must be idempotent per id.
    */
  private var folded = Set.empty[String]

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private def newestFirst(list: List[Generation]): List[Generation] =
    list.sortBy(-_.submittedAt)

  private def byDateDescending(days: List[HistoryDay]): List[HistoryDay] =
    days.sortBy(_.date)(using Ordering[String].reverse)

  /** A derived entry a post-processing job just wrote — same fold as a
    * completion pushed by the socket.
    */
  def adopt(generation: Generation): Unit = fold(generation)

  /** A generation whose record changed — its project — in place, where its day
    * is loaded.
    */
  private def replace(generation: Generation): Unit =
    _generationsByDay.update(
      _.view
        .mapValues(_.map(g => if (g.id == generation.id) generation else g))
        .toMap
    )

  /** A completion pushed by the socket. A loaded day takes the generation in
    * place; a day that is not loaded (or not yet listed) only needs the index
    * re-read — one directory walk — so that its count is right when it is
    * opened, whose listing then reads the sidecar like any other.
    */
  private def fold(generation: Generation): Unit =
    generation.outputs.headOption.map(_.date).foreach { date =>
      if (!folded.contains(generation.id)) {
        folded += generation.id
        _generationsByDay.now().get(date) match {
          case Some(list) if list.exists(_.id == generation.id) => ()
          case Some(list)                                       =>
            _generationsByDay.update(
              _ + (date -> newestFirst(generation :: list))
            )
            if (_days.now().exists(_.date == date))
              _days.update(
                _.map(day =>
                  if (day.date == date) day.copy(count = day.count + 1) else day
                )
              )
            else
              _days
                .update(days => byDateDescending(HistoryDay(date, 1) :: days))
          case None => push(Command.LoadDays)
        }
      }
    }

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.LoadDays => () }
      .flatMapSwitch(_ => daysFn(()).recoverToTry) --> Observer[
      Try[List[HistoryDay]]
    ] {
      case Success(listing) =>
        clearError()
        _days.set(byDateDescending(listing))
        _daysLoaded.set(true)
        // The service outlives the page: a day loaded on an earlier visit is
        // shown at once, then refreshed in place — no flash of nothing — and
        // one that has vanished from disk is dropped.
        _generationsByDay.now().keys.foreach { date =>
          if (listing.exists(_.date == date)) push(Command.LoadDay(date))
          else _generationsByDay.update(_ - date)
        }
      case Failure(err) => reportFailure("Listing the history", err)
    },
    cmdBus.events
      .collect { case Command.LoadDay(date) => date }
      .flatMapMerge { date =>
        _loadingDays.update(_ + date)
        dayFn(date).map(listing => (date, listing)).recoverToTry
      } --> Observer[Try[(String, List[Generation])]] {
      case Success((date, listing)) =>
        clearError()
        _loadingDays.update(_ - date)
        _generationsByDay.update(_ + (date -> newestFirst(listing)))
      case Failure(err) =>
        // The date is not recoverable from the failure; a retry re-marks it.
        _loadingDays.set(Set.empty)
        reportFailure("Loading a day of history", err)
    },
    // A generation named by a URL: the day it lives in is asked for, then
    // loaded like any other — so the detail opens on arrival instead of
    // waiting for the visitor to page back far enough to reach it
    // (François, 2026-09-20).
    cmdBus.events
      .collect { case Command.LoadGeneration(generationId) => generationId }
      .filterNot(generationId =>
        _generationsByDay.now().values.exists(_.exists(_.id == generationId))
      )
      .flatMapMerge(generationId => dayOfFn(generationId).recoverToTry) -->
      Observer[Try[Option[HistoryDay]]] {
        case Success(Some(day)) =>
          clearError()
          // The index may not have answered yet, or may not list a day added
          // since it did; the listing needs it to render the day at all.
          if (!_days.now().exists(_.date == day.date))
            _days.update(days => byDateDescending(day :: days))
          if (!_generationsByDay.now().contains(day.date))
            push(Command.LoadDay(day.date))
        case Success(None) => clearError()
        case Failure(err)  =>
          reportFailure("Finding a generation in the history", err)
      },
    cmdBus.events
      .collect { case Command.Delete(date, generationId) =>
        (date, generationId)
      }
      .flatMapMerge((date, generationId) =>
        deleteFn((date, generationId))
          .map(deleted => (date, generationId, deleted))
          .recoverToTry
      ) --> Observer[Try[(String, String, Boolean)]] {
      case Success((date, generationId, true)) =>
        clearError()
        val remaining = _generationsByDay
          .now()
          .get(date)
          .map(_.filterNot(_.id == generationId))
        remaining match {
          case Some(Nil)  => _generationsByDay.update(_ - date)
          case Some(list) => _generationsByDay.update(_ + (date -> list))
          case None       => ()
        }
        _days.update(_.flatMap { day =>
          if (day.date != date) Some(day)
          else if (day.count <= 1) None
          else Some(day.copy(count = day.count - 1))
        })
        evtBus.writer.onNext(Event.Deleted(generationId))
      case Success((_, generationId, false)) =>
        reportFailure(
          "Deleting the generation",
          s"nothing on disk belongs to '$generationId'."
        )
      case Failure(err) => reportFailure("Deleting the generation", err)
    },
    cmdBus.events
      .collect { case Command.Import(image) => image }
      .flatMapMerge(image => importFn(image).recoverToTry) --> Observer[
      Try[Generation]
    ] {
      case Success(generation) =>
        clearError()
        fold(generation)
        evtBus.writer.onNext(Event.Imported(generation))
      case Failure(err) => reportFailure("Importing an image", err)
    },
    cmdBus.events
      .collect { case Command.ImportVideo(file) => file }
      .flatMapMerge(file =>
        importVideoFn((file.name, file)).recoverToTry
      ) --> Observer[Try[Generation]] {
      case Success(generation) =>
        clearError()
        fold(generation)
        evtBus.writer.onNext(Event.Imported(generation))
      case Failure(err) => reportFailure("Importing a video", err)
    },
    cmdBus.events
      .collect { case Command.Move(generations, projectId) =>
        (generations.map(_.id).toSet, projectId, generations)
      }
      .flatMapMerge { (ids, projectId, generations) =>
        _moveState.set(HistoryService.MoveState.Moving(ids))
        moveFn(
          GenerationMove(generations.flatMap(GenerationReference.of), projectId)
        ).recoverToTry.map(result => (ids, projectId, result))
      } --> Observer[(Set[String], Option[String], Try[List[Generation]])] {
      case (ids, projectId, Success(moved)) =>
        clearError()
        moved.foreach(replace)
        _moveState.set(
          HistoryService.MoveState.Done(ids, projectId, moved.size)
        )
        evtBus.writer.onNext(Event.Moved(moved))
      case (ids, _, Failure(err)) =>
        _moveState.set(HistoryService.MoveState.Failed(ids))
        reportFailure("Moving to the project", err)
    },
    // Completions arrive by themselves: the socket pushes a session's list
    // whenever any of its generations change.
    statusSocket.generations --> Observer[Map[String, List[Generation]]](
      _.values.flatten
        // A free-play result has no sidecar and lives under `outputs/scratch/`
        // (`specs/22-…`): folding it in would invent a day called "scratch".
        // Keeping it clears the flag, and then it folds like any other.
        .filter(g =>
          g.status == GenerationStatus.Completed && g.outputs.nonEmpty &&
            !g.scratch
        )
        .foreach(fold)
    )
  )
}
