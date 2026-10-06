package drift.frontend.pages.gallery

import drift.frontend.components.Component
import drift.frontend.services.HistoryService.MoveState
import drift.shared.{Generation, Project}

import com.raquo.laminar.api.L.*

/** Moves gallery entries to a project (`specs/19`): a select showing the
  * project they are in, a button that moves them to the one picked, and what
  * became of it — moving, moved where and with how many derived entries, or
  * not moved.
  */
class ProjectMover(
    projects: Signal[List[Project]],
    /** The entries the button moves: the one on screen, or the ticked ones. */
    targets: Signal[List[Generation]],
    state: Signal[MoveState],
    onMove: (List[Generation], Option[String]) => Unit,
    /** The button's words for that many entries. */
    label: Int => String
) extends Component {
  private val Undecided = "\u0000undecided"
  private val NoProject = ""

  private def valueOf(projectId: Option[String]): String =
    projectId.getOrElse(NoProject)

  /** The project every target is in, when they share one. */
  private val current: Signal[Option[Option[String]]] =
    targets.map(_.map(_.projectId).distinct match {
      case List(shared) => Some(shared)
      case _            => None
    })

  /** What the select was set to; none until it is touched, and again once the
    * targets change — it then shows where they are.
    */
  private val picked = Var(Option.empty[String])

  private val shown: Signal[String] =
    picked.signal.combineWith(current).map { (choice, where) =>
      choice.orElse(where.map(valueOf)).getOrElse(Undecided)
    }

  /** The entries this one last sent, so that the outcome shown is its own. */
  private val sent = Var(Set.empty[String])

  private def concerns(ids: Set[String]): Signal[Boolean] =
    sent.signal
      .combineWith(targets)
      .map((mine, list) => ids == mine || ids == list.map(_.id).toSet)

  private val moving: Signal[Boolean] = state.flatMapSwitch {
    case MoveState.Moving(ids) => concerns(ids)
    case _                     => Val(false)
  }

  private val outcome: Signal[Option[(String, String)]] =
    state.combineWith(projects).flatMapSwitch {
      case (MoveState.Done(ids, projectId, rewritten), list) =>
        val where = projectId.fold("out of any project")(id =>
          "to " + list.find(_.id == id).fold(id)(_.label)
        )
        val derived = rewritten - ids.size
        val also =
          if (derived > 0) s", with $derived made from it" else ""
        concerns(ids).map(
          Option.when(_)(("has-text-success", s"✓ Moved $where$also"))
        )
      case (MoveState.Failed(ids), _) =>
        concerns(ids).map(
          Option.when(_)(("has-text-danger", "✗ Not moved — see the message above"))
        )
      case _ => Val(None)
    }

  /** Something to move, somewhere else than where it is, and no move of it
    * under way.
    */
  private val ready: Signal[Boolean] =
    targets.combineWith(shown, current, moving).map {
      (list, choice, where, busy) =>
        list.nonEmpty && choice != Undecided && !busy &&
        !where.map(valueOf).contains(choice)
    }

  lazy val element: HtmlElement = span(
    cls := "project-mover",
    // New targets: the select goes back to showing where they are
    targets.map(_.map(_.id).toSet).distinct.changes --> (_ => picked.set(None)),
    select(
      cls := "select is-small",
      disabled <-- targets.combineWith(moving).map((list, busy) =>
        list.isEmpty || busy
      ),
      // The choice is marked on its option rather than set on the select: the
      // projects arrive after the select is built, and a value naming an
      // option that is not there yet is dropped — the entry's own project
      // then showed as "No project"
      children <-- projects.combineWith(current, shown).map {
        (list, where, choice) =>
          def entry(id: String, text: String) =
            option(value := id, selected := (id == choice), text)
          Option
            .when(where.isEmpty)(entry(Undecided, "Choose a project…"))
            .toList ++
            List(entry(NoProject, "No project")) ++
            list.map(project => entry(project.id, project.label))
      },
      onChange.mapToValue --> Observer[String](choice =>
        picked.set(Some(choice))
      )
    ),
    button(
      cls := "button is-small is-info ml-1",
      cls("is-loading") <-- moving,
      title := "Moves it to the project picked, with everything derived " +
        "from it — upscales, redraws, edits. Its recipe becomes one of the " +
        "project's versions.",
      disabled <-- ready.map(!_),
      child.text <-- targets.map(list => label(list.size)),
      onClick.compose(_.sample(targets, shown)) --> Observer[
        (List[Generation], String)
      ] { (list, choice) =>
        sent.set(list.map(_.id).toSet)
        onMove(list, Option.when(choice != NoProject)(choice))
      }
    ),
    child <-- outcome.map {
      case Some((colour, text)) =>
        span(cls := s"is-size-7 ml-2 $colour", text)
      case None => emptyNode
    }
  )
}
