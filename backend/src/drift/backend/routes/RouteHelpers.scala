package drift.backend.routes

import drift.backend.storage.StorageService

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
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
    canDelete: T => Boolean = (_: T) => true,
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
  // Gate the update on the file existing rather than on it decoding: an entity
  // whose JSON went stale or got corrupted would otherwise reject every save
  // for good, and the client only sees a 200 with a `null` body.
  updateEp.serverLogicSuccess[Identity] { (id, v) =>
    if (storage.exists(entityType, id))
      Some(
        storage.save[T](
          entityType,
          id,
          storage.get[T](entityType, id).map(mergeUpdate(_, v)).getOrElse(v)
        )
      )
    else None
  },
  deleteEp.serverLogicSuccess[Identity] { id =>
    storage.get[T](entityType, id) match {
      case Some(e) if canDelete(e) =>
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
