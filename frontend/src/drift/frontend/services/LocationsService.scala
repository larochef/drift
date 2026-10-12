package drift.frontend.services

import drift.shared.{getLocations, FileLocations}

import scala.util.*

import com.raquo.laminar.api.L.*

/** Where this drift keeps its files (`FileLocations`), as its backend resolved
  * them: read once when the shell mounts, so a page names the folders this
  * drift really uses and writes none itself.
  */
class LocationsService extends ServiceErrors {

  private val locationsFn = ApiClient.stream(getLocations)

  private val _locations = Var(Option.empty[FileLocations])

  /** `None` until the backend has answered. */
  val locations: Signal[Option[FileLocations]] = _locations.signal

  def now: Option[FileLocations] = _locations.now()

  /** One of the folders as text, empty until the backend has answered. */
  def shown(folder: FileLocations => String): Signal[String] =
    locations.map(_.fold("")(folder))

  val effects: Modifier[HtmlElement] =
    EventStream
      .fromValue(())
      .flatMapSwitch(_ => locationsFn(()).recoverToTry) --> Observer[
      Try[FileLocations]
    ] {
      case Success(read) =>
        clearError()
        _locations.set(Some(read))
      case Failure(err) =>
        reportFailure("Reading where drift keeps its files", err)
    }
}
