package drift.backend.routes

import drift.backend.storage.StorageService

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import sttp.model.StatusCode
import sttp.shared.Identity
import sttp.tapir.*
import sttp.tapir.server.ServerEndpoint

def endpointsFor[T: JsonValueCodec](
    storage: StorageService,
    entityType: String,
    extractId: T => String,
    listEp: PublicEndpoint[Unit, Unit, List[T], Any],
    getEp: PublicEndpoint[String, Unit, Option[T], Any],
    createEp: PublicEndpoint[T, Unit, T, Any],
    updateEp: PublicEndpoint[(String, T), Unit, Option[T], Any],
    deleteEp: PublicEndpoint[String, Unit, Boolean, Any],
    /** Whether a stored entity is one drift ships. Those are rewritten from the
      * reference data at every start, so one is neither deleted nor changed: an
      * edit would hold until the next start and then silently revert.
      */
    builtIn: T => Boolean = (_: T) => false,
    onDeleted: T => Unit = (_: T) => (),
    /** How an update folds into what is already stored, when anything about the
      * entity is the backend's to own rather than the client's to send back.
      * The default replaces the document wholesale.
      */
    mergeUpdate: (T, T) => T = (_: T, incoming: T) => incoming,
    /** What a created entity becomes before it is stored, when a field is the
      * backend's to set — a built-in flag a client must not claim.
      */
    onCreate: T => T = (incoming: T) => incoming
): List[ServerEndpoint[Any, Identity]] = List(
  listEp.serverLogicSuccess[Identity](_ => storage.list[T](entityType)),
  getEp.serverLogicSuccess[Identity](id => storage.get[T](entityType, id)),
  createEp.serverLogicSuccess[Identity] { v =>
    val created = onCreate(v)
    storage.save[T](entityType, extractId(created), created)
  },
  updateEp
    .errorOut(statusCode(StatusCode.BadRequest).and(stringBody))
    .serverLogic[Identity] { (id, v) =>
      updated(storage, entityType, extractId, builtIn, mergeUpdate)(id, v)
    },
  deleteEp.serverLogicSuccess[Identity] { id =>
    storage.get[T](entityType, id) match {
      case Some(e) if !builtIn(e) =>
        val deleted = storage.delete(entityType, id)
        if (deleted) onDeleted(e)
        deleted
      case Some(_) => false
      // Missing, or on disk but unreadable: there is no `builtIn` flag left to
      // honour, so let the file go instead of failing forever. `delete` still
      // reports false when there was nothing there.
      case None => storage.delete(entityType, id)
    }
  }
)

/** What an update stores, or why it is refused: an entity is filed under its
  * own id, so a body naming another id than the address it is sent to would
  * leave a file whose name and content disagree. A built-in is answered as it
  * is stored.
  *
  * The update is gated on the file existing rather than on it decoding: an
  * entity whose JSON went stale or got corrupted would otherwise reject every
  * save for good, and the client only sees a 200 with a `null` body.
  */
private[routes] def updated[T: JsonValueCodec](
    storage: StorageService,
    entityType: String,
    extractId: T => String,
    builtIn: T => Boolean,
    mergeUpdate: (T, T) => T
)(id: String, incoming: T): Either[String, Option[T]] =
  if (extractId(incoming) != id)
    Left(
      s"An update sent to '$id' in $entityType carries the id " +
        s"'${extractId(incoming)}': an entity cannot change its id."
    )
  else if (storage.exists(entityType, id))
    storage.get[T](entityType, id) match {
      case Some(stored) if builtIn(stored) => Right(Some(stored))
      case stored                          =>
        Right(
          Some(
            storage.save[T](
              entityType,
              id,
              stored.map(mergeUpdate(_, incoming)).getOrElse(incoming)
            )
          )
        )
    }
  else Right(None)
