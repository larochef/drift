package drift.runner.state

/** A copy of some state taken at one point of a sequence, to come back to;
  * `close` frees it.
  */
trait Snapshot extends AutoCloseable {

  /** Puts the state back as it was when taken. */
  def restore(): Unit
}
