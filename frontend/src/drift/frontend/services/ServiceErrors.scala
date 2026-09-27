package drift.frontend.services

import com.raquo.laminar.api.L.*

/** Failure reporting shared by the entity services.
  *
  * The tapir clients are built with `toClientThrowErrors`, so any non-2xx
  * response rejects the promise and Airstream turns that into a stream error.
  * An `Observer` built from a plain function has nowhere to put such an error,
  * which is why a failed save used to look exactly like a save that did
  * nothing. Every command pipeline recovers into a `Try` instead and reports
  * the failure here; pages render `lastError`.
  */
trait ServiceErrors {
  private val _lastError = Var(Option.empty[String])

  /** The most recent failed command, until it is cleared or superseded. */
  val lastError: Signal[Option[String]] = _lastError.signal

  def clearError(): Unit = _lastError.set(None)

  protected def reportFailure(action: String, reason: String): Unit =
    _lastError.set(Some(s"$action failed: $reason"))

  protected def reportFailure(action: String, err: Throwable): Unit =
    reportFailure(action, Option(err.getMessage).getOrElse(err.toString))
}
