package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object ProjectService {
  enum Command {
    case Load
    case Create(project: Project)
    case Update(id: String, project: Project)

    /** Deletes the project; `withGenerations` deletes the images it made too —
      * their files and sidecars, so they leave the gallery as well.
      */
    case Delete(id: String, withGenerations: Boolean)

    /** The project's generations from the sidecars, newest first. */
    case LoadGenerations(projectId: String)

  }
  enum Event {
    case Created(project: Project)
    case Deleted(id: String)
  }
}

/** Projects and their versions (`specs/19-projects-and-prompt-versions.md`),
  * plus each project's generations: loaded from the sidecars once, then kept
  * current from the status socket — a completion tagged with a project is
  * folded in, and one naming a version the project does not know yet reloads
  * the projects, since the backend appended that version on submit.
  */
class ProjectService(statusSocket: StatusSocketService) extends ServiceErrors {
  import ProjectService.{Command, Event}

  private val listFn = ApiClient.stream(listProjects)
  private val createFn = ApiClient.stream(createProject)
  private val updateFn = ApiClient.stream(updateProject)
  private val deleteFn = ApiClient.stream(deleteProject)
  private val deleteGenerationsFn = ApiClient.stream(deleteProjectGenerations)
  private val generationsFn = ApiClient.stream(listProjectGenerations)

  private val _projects = Var(List.empty[Project])
  private val _generations = Var(Map.empty[String, List[Generation]])

  val projects: Signal[List[Project]] = _projects.signal

  /** Per project id, newest first. */
  val generations: Signal[Map[String, List[Generation]]] = _generations.signal

  def project(id: String): Signal[Option[Project]] =
    _projects.signal.map(_.find(_.id == id)).distinct

  def generationsOf(id: String): Signal[List[Generation]] =
    _generations.signal.map(_.getOrElse(id, Nil))

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private def newestFirst(list: List[Generation]): List[Generation] =
    list.sortBy(-_.submittedAt)

  /** A derived entry a post-processing job just wrote — it inherits its
    * parent's project, so it belongs beside it in the results.
    */
  def adopt(generation: Generation): Unit = fold(generation)

  /** Drops a generation deleted elsewhere (the detail view, a result's delete
    * button) from every project's list, so the results grid follows the disk.
    */
  def forget(generationId: String): Unit =
    _generations.update(
      _.view.mapValues(_.filterNot(_.id == generationId)).toMap
    )

  /** Folds a pushed generation into its project's list; a new version id means
    * the project document changed too.
    */
  private def fold(generation: Generation): Unit =
    generation.projectId.foreach { projectId =>
      val list = _generations.now().getOrElse(projectId, Nil)
      // Only when it actually says something new: the socket repeats a
      // session's whole list on every push, and a list rebuilt for nothing
      // emits, which rebuilds whatever the workspace has open on it — the
      // detail modal included, mid-edit.
      if (!list.contains(generation))
        _generations.update(
          _ + (projectId -> newestFirst(
            generation :: list.filterNot(_.id == generation.id)
          ))
        )
      val known = _projects
        .now()
        .find(_.id == projectId)
        .exists(project =>
          generation.promptVersionId
            .forall(id => project.versions.exists(_.id == id))
        )
      if (!known) push(Command.Load)
    }

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ => listFn(()).recoverToTry)
      --> Observer[Try[List[Project]]] {
        case Success(projects) =>
          clearError()
          _projects.set(projects.sortBy(-_.lastUsedAt))
        case Failure(err) => reportFailure("Loading projects", err)
      },
    cmdBus.events
      .collect { case Command.Create(project) => project }
      .flatMapMerge(project => createFn(project).recoverToTry)
      --> Observer[Try[Project]] {
        case Success(project) =>
          clearError()
          _projects.update(list =>
            (project :: list.filterNot(_.id == project.id))
              .sortBy(-_.lastUsedAt)
          )
          evtBus.writer.onNext(Event.Created(project))
        case Failure(err) => reportFailure("Creating the project", err)
      },
    cmdBus.events
      .collect { case Command.Update(id, project) => (id, project) }
      .flatMapMerge((id, project) => updateFn((id, project)).recoverToTry)
      --> Observer[Try[Option[Project]]] {
        case Success(Some(project)) =>
          clearError()
          _projects.update(_.map(p => if (p.id == project.id) project else p))
        case Success(None) =>
          reportFailure("Updating the project", "it no longer exists.")
        case Failure(err) => reportFailure("Updating the project", err)
      },
    // The images first, when asked for: a project deleted before its
    // generations would leave nothing to look them up by.
    cmdBus.events
      .collect { case Command.Delete(id, withGenerations) =>
        (id, withGenerations)
      }
      .flatMapMerge { (id, withGenerations) =>
        val images =
          if (withGenerations) deleteGenerationsFn(id)
          else EventStream.fromValue(List.empty[String])
        images
          .flatMapSwitch(_ => deleteFn(id))
          .map(ok => (id, ok))
          .recoverToTry
      }
      --> Observer[Try[(String, Boolean)]] {
        case Success((id, true)) =>
          clearError()
          _projects.update(_.filterNot(_.id == id))
          _generations.update(_ - id)
          evtBus.writer.onNext(Event.Deleted(id))
        case Success((id, false)) =>
          reportFailure("Deleting the project", s"the server refused '$id'.")
        case Failure(err) => reportFailure("Deleting the project", err)
      },
    cmdBus.events
      .collect { case Command.LoadGenerations(id) => id }
      .flatMapMerge(id => generationsFn(id).map(id -> _).recoverToTry)
      --> Observer[Try[(String, List[Generation])]] {
        case Success((id, list)) =>
          clearError()
          _generations.update(_ + (id -> newestFirst(list)))
        case Failure(err) =>
          reportFailure("Loading the project's generations", err)
      },
    // Live updates: the socket repeats each session's whole list, so folding
    // is idempotent.
    statusSocket.generations --> Observer[Map[String, List[Generation]]] {
      // Free play is not project work until it is kept (`specs/22-…`).
      bySession => bySession.values.flatten.filterNot(_.scratch).foreach(fold)
    }
  )
}
