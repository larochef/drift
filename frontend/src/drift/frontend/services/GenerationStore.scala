package drift.frontend.services

import drift.shared.*

import com.raquo.laminar.api.L.*

/** Every generation the page knows, each once, by its id: what the gallery
  * shows by day, a project by its id and a session's panel by its session are
  * views of it. A generation that changes — it completes, it moves to a
  * project, it is deleted — changes here, and every view follows.
  *
  * It is told by whoever learns of a generation: the status socket, which
  * pushes a session's whole list whenever one of its generations changes, and
  * the services, with what a listing or a request answered.
  */
class GenerationStore(statusSocket: StatusSocketService) {

  private val _byId = Var(Map.empty[String, Generation])

  val byId: Signal[Map[String, Generation]] = _byId.signal

  /** The current value, for event handlers. */
  def now: Map[String, Generation] = _byId.now()

  /** Takes these generations as they now are. Nothing is emitted for what the
    * store already holds as it is: the socket repeats whole lists, and a view
    * rebuilt for nothing rebuilds whatever is open on it.
    */
  def put(generations: Iterable[Generation]): Unit = {
    val held = _byId.now()
    val changed = generations.filterNot(g => held.get(g.id).contains(g))
    if (changed.nonEmpty) _byId.update(_ ++ changed.map(g => g.id -> g))
  }

  def put(generation: Generation): Unit = put(List(generation))

  /** Takes a generation the store does not hold yet, and leaves one it holds as
    * it is: a job's result is the record as the job wrote it, which a move to a
    * project may have changed since.
    */
  def adopt(generation: Generation): Unit =
    if (!_byId.now().contains(generation.id)) put(generation)

  /** Lets go of generations that are no more. */
  def forget(ids: Iterable[String]): Unit = {
    val gone = ids.toSet
    if (gone.exists(_byId.now().contains)) _byId.update(_ -- gone)
  }

  /** `listing` is all there is of `scope`: what the store held of it and the
    * listing lacks is gone, and the rest is taken as listed.
    */
  def settle(scope: Generation => Boolean, listing: List[Generation]): Unit = {
    val listed = listing.map(_.id).toSet
    val held = _byId.now()
    val gone = held.values.filter(g => scope(g) && !listed(g.id)).map(_.id)
    val changed = listing.filterNot(g => held.get(g.id).contains(g))
    if (gone.nonEmpty || changed.nonEmpty)
      _byId.update(all => (all -- gone) ++ changed.map(g => g.id -> g))
  }

  /** Mounted before the services': what they do on a push finds the store
    * already told.
    */
  val effects: Modifier[HtmlElement] =
    statusSocket.generations --> Observer[Map[String, List[Generation]]](
      bySession => put(bySession.values.flatten)
    )
}

object GenerationStore {

  /** Whether a generation is in the gallery: completed, with what it made, and
    * kept — a free-play result has no sidecar and lives under
    * `outputs/scratch/` until it is.
    */
  def recorded(generation: Generation): Boolean =
    generation.status == GenerationStatus.Completed &&
      generation.outputs.nonEmpty && !generation.scratch

  /** The day a recorded generation is filed under. */
  def dayOf(generation: Generation): Option[String] =
    generation.outputs.headOption.map(_.date)
}
