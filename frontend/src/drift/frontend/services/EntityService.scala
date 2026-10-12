package drift.frontend.services

import drift.frontend.services.EntityService.{Command, Event}

import scala.util.*

import com.raquo.laminar.api.L.*

object EntityService {

  /** What a page asks of an entity service. */
  enum Command[+E] {
    case Load
    case Create(entity: E)
    case Update(id: String, entity: E)
    case Delete(id: String)
  }

  /** What a write did, once the server answered and the list holds it. */
  enum Event[+E] {
    case Created(entity: E)
    case Updated(entity: E)
    case Deleted(id: String)
  }
}

/** The words of an entity's failure messages: the action each command names,
  * and the reason given when the server answers a save or a delete with "no".
  */
final case class EntityWording(
    loading: String,
    creating: String,
    saving: String,
    deleting: String,
    /** The entity of this id was not there to save. */
    missing: String => String,
    /** The server kept the entity of this id. */
    refused: String => String
)

object EntityWording {
  val noLongerExists: String => String = id => s"'$id' no longer exists."

  val builtInsAreProtected: String => String = id =>
    s"the server refused to delete '$id' (built-in entries are protected)."
}

/** A stored entity's list and its four commands, written once: the list is
  * loaded from the server, and every write the server accepts is applied to it
  * and announced on `events`. A service says what is its own — its endpoint
  * clients, how an id is read, its wording — and overrides the placement of an
  * entity in the list when appending is not it.
  */
abstract class EntityService[E](
    list: Unit => EventStream[List[E]],
    create: E => EventStream[E],
    update: ((String, E)) => EventStream[Option[E]],
    delete: String => EventStream[Boolean],
    idOf: E => String,
    wording: EntityWording
) extends ServiceErrors {

  /** The list as loaded and as the writes left it. */
  protected val entities: Var[List[E]] = Var(List.empty[E])

  private val cmdBus = new EventBus[Command[E]]
  private val evtBus = new EventBus[Event[E]]

  val events: EventStream[Event[E]] = evtBus.events

  def push(command: Command[E]): Unit = cmdBus.writer.onNext(command)

  /** The list a load shows. */
  protected def arranged(loaded: List[E]): List[E] = loaded

  /** The list once the server created an entity. */
  protected def withCreated(all: List[E], created: E): List[E] =
    all :+ created

  /** The list once the server saved the entity of an id. */
  protected def withUpdated(all: List[E], id: String, saved: E): List[E] =
    all.map(existing => if (idOf(existing) == id) saved else existing)

  // Writes use flatMapMerge, not flatMapSwitch: switching unsubscribes from the
  // request already in flight, so two quick saves would leave the first one's
  // result unapplied even though the server stored it. Load keeps switching --
  // there the newest list is the only one worth having.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => list(()).recoverToTry)
      --> Observer[Try[List[E]]] {
        case Success(loaded) =>
          clearError()
          entities.set(arranged(loaded))
        case Failure(err) => reportFailure(wording.loading, err)
      },
    cmdBus.events
      .collect { case Command.Create(entity) => entity }
      .flatMapMerge(entity => create(entity).recoverToTry)
      --> Observer[Try[E]] {
        case Success(created) =>
          clearError()
          entities.update(withCreated(_, created))
          evtBus.writer.onNext(Event.Created(created))
        case Failure(err) => reportFailure(wording.creating, err)
      },
    cmdBus.events
      .collect { case Command.Update(id, entity) => (id, entity) }
      .flatMapMerge { case (id, entity) =>
        update((id, entity)).map(saved => (id, saved)).recoverToTry
      } --> Observer[Try[(String, Option[E])]] {
      // The saved copy, not the one sent: the server drops parameters and
      // overrides that override nothing (`specs/16-parameter-resolution.md`)
      // and answers a built-in as it is stored, and the list has to show what
      // is stored.
      case Success((id, Some(saved))) =>
        clearError()
        entities.update(withUpdated(_, id, saved))
        evtBus.writer.onNext(Event.Updated(saved))
      case Success((id, None)) =>
        reportFailure(wording.saving, wording.missing(id))
      case Failure(err) => reportFailure(wording.saving, err)
    },
    cmdBus.events
      .collect { case Command.Delete(id) => id }
      .flatMapMerge { id =>
        delete(id).map(deleted => (id, deleted)).recoverToTry
      } --> Observer[Try[(String, Boolean)]] {
      case Success((id, true)) =>
        clearError()
        entities.update(_.filterNot(idOf(_) == id))
        evtBus.writer.onNext(Event.Deleted(id))
      case Success((id, false)) =>
        reportFailure(wording.deleting, wording.refused(id))
      case Failure(err) => reportFailure(wording.deleting, err)
    }
  )
}
