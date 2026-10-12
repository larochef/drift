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
  * plus each project's generations: read from the sidecars once, and from then
  * on a view of the generations the page knows (`GenerationStore`) — whatever
  * is tagged with the project is in it. A generation naming a version the
  * project does not know yet reloads the projects, since the backend appended
  * that version on submit.
  */
class ProjectService(statusSocket: StatusSocketService, store: GenerationStore)
    extends ServiceErrors {
  import ProjectService.{Command, Event}

  private val listFn = ApiClient.stream(listProjects)
  private val createFn = ApiClient.stream(createProject)
  private val updateFn = ApiClient.stream(updateProject)
  private val deleteFn = ApiClient.stream(deleteProject)
  private val deleteGenerationsFn = ApiClient.stream(deleteProjectGenerations)
  private val generationsFn = ApiClient.stream(listProjectGenerations)

  private val _projects = Var(List.empty[Project])

  /** The projects whose generations have been read from the sidecars. */
  private val _loadedProjects = Var(Set.empty[String])

  val projects: Signal[List[Project]] = _projects.signal

  /** Whether the list has answered at least once — an empty page must invite a
    * first project, not before it knows there is none.
    */
  private val _projectsLoaded = Var(false)
  val projectsLoaded: Signal[Boolean] = _projectsLoaded.signal

  /** Per loaded project, newest first. Free play is not project work until it
    * is kept (`specs/22-…`).
    */
  val generations: Signal[Map[String, List[Generation]]] =
    store.byId
      .combineWith(_loadedProjects.signal)
      .map { (all, loaded) =>
        val tagged = all.values.toList
          .filterNot(_.scratch)
          .groupBy(_.projectId.getOrElse(""))
        loaded.map(id => id -> newestFirst(tagged.getOrElse(id, Nil))).toMap
      }
      .distinct

  def project(id: String): Signal[Option[Project]] =
    _projects.signal.map(_.find(_.id == id)).distinct

  def generationsOf(id: String): Signal[List[Generation]] =
    generations.map(_.getOrElse(id, Nil)).distinct

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  private def newestFirst(list: List[Generation]): List[Generation] =
    list.sortBy(-_.submittedAt)

  /** A derived entry a post-processing job just wrote — it inherits its
    * parent's project, so it belongs beside it in the results.
    */
  def adopt(generation: Generation): Unit = {
    store.adopt(generation)
    fold(generation)
  }

  /** Drops a generation deleted elsewhere (the detail view, a result's delete
    * button), so the results grid follows the disk.
    */
  def forget(generationId: String): Unit = store.forget(List(generationId))

  /** Entries that changed project: each is in the list it belongs to now; the
    * projects are re-read, a move may have appended a version or cleared a
    * cover.
    */
  def moved(generations: List[Generation]): Unit = {
    store.put(generations)
    push(Command.Load)
  }

  /** A generation tagged with a project whose version the project does not know
    * yet means the project document changed too.
    */
  private def fold(generation: Generation): Unit =
    generation.projectId.foreach { projectId =>
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
          _projectsLoaded.set(true)
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
          .flatMapSwitch(deleted => deleteFn(id).map(ok => (id, ok, deleted)))
          .recoverToTry
      }
      --> Observer[Try[(String, Boolean, List[String])]] {
        case Success((id, true, deleted)) =>
          clearError()
          _projects.update(_.filterNot(_.id == id))
          _loadedProjects.update(_ - id)
          // The images deleted with it leave the gallery as well.
          store.forget(deleted)
          evtBus.writer.onNext(Event.Deleted(id))
        case Success((id, false, _)) =>
          reportFailure("Deleting the project", s"the server refused '$id'.")
        case Failure(err) => reportFailure("Deleting the project", err)
      },
    cmdBus.events
      .collect { case Command.LoadGenerations(id) => id }
      .flatMapMerge(id => generationsFn(id).map(id -> _).recoverToTry)
      --> Observer[Try[(String, List[Generation])]] {
        case Success((id, list)) =>
          clearError()
          // The listing is the project as the disk has it; what is still
          // being made has no sidecar yet, and stays.
          store.settle(
            g => g.projectId.contains(id) && GenerationStore.recorded(g),
            list
          )
          _loadedProjects.update(_ + id)
        case Failure(err) =>
          reportFailure("Loading the project's generations", err)
      },
    // Live updates: the store takes the generations; a version it names that
    // the project lacks is what is left to notice here.
    statusSocket.generations --> Observer[Map[String, List[Generation]]] {
      bySession => bySession.values.flatten.filterNot(_.scratch).foreach(fold)
    }
  )
}
