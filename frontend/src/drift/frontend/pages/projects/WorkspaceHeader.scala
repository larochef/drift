package drift.frontend.pages.projects

import drift.frontend.Page
import drift.frontend.components.{Component, LaunchOrDownload}
import drift.frontend.services.{
  LaunchPrerequisites,
  ProjectService,
  SessionService
}
import drift.frontend.services.ProjectService.Command
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The workspace's header: the project's name and brief, both edited in place,
  * its kind and NSFW flag, the way back to the list, and the pickers that
  * launch or switch the image or video model and the assistant.
  */
class WorkspaceHeader(
    project: Signal[Option[Project]],
    /** The project as it stands, for the handlers. */
    currentProject: () => Option[Project],
    projectService: ProjectService,
    sessionService: SessionService,
    sessions: WorkspaceSessions,
    prerequisites: LaunchPrerequisites,
    /** The modal creating a configuration of a tool, for a project of a kind
      * (none: any), and what to do with the one created — a picker's **+ New**.
      */
    newConfiguration: (
        RuntimeTool,
        Option[ProjectKind],
        RunConfiguration => Unit,
        () => Unit
    ) => HtmlElement
) extends Component {

  // Editing state apart from the text: the signal that builds an editor
  // must not carry the draft, or every keystroke rebuilds the input and
  // drops the focus (François, 2026-09-08).
  private val labelEditing = Var(false)
  private val labelDraft = Var("")
  private val briefEditing = Var(false)
  private val briefDraft = Var("")

  private val projectKind: Signal[Option[ProjectKind]] =
    project.map(_.map(_.kind)).distinct

  private def saveLabel(): Unit = {
    val draft = labelDraft.now().trim
    if (draft.nonEmpty)
      currentProject().foreach(current =>
        projectService.push(
          Command.Update(current.id, current.copy(label = draft))
        )
      )
    labelEditing.set(false)
  }

  private def saveBrief(): Unit = {
    currentProject().foreach(current =>
      projectService.push(
        Command
          .Update(current.id, current.copy(brief = briefDraft.now().trim))
      )
    )
    briefEditing.set(false)
  }

  /** An in-place editor: the input is built once per editing session and bound
    * to its draft, so typing never rebuilds it.
    */
  private def inlineEditor(
      draft: Var[String],
      placeholderText: String,
      onSave: () => Unit,
      onCancel: () => Unit,
      small: Boolean
  ): HtmlElement = div(
    cls := "field has-addons mb-1",
    div(
      cls := "control is-expanded",
      input(
        cls := (if (small) "input is-small" else "input"),
        placeholder := placeholderText,
        controlled(value <-- draft.signal, onInput.mapToValue --> draft),
        onKeyDown.filter(_.key == "Enter") --> (_ => onSave()),
        onKeyDown.filter(_.key == "Escape") --> (_ => onCancel()),
        onMountCallback(context => context.thisNode.ref.focus())
      )
    ),
    div(
      cls := "control",
      button(
        cls := (if (small) "button is-small is-primary"
                else "button is-primary"),
        "Save",
        disabled <-- draft.signal.map(_.trim.isEmpty),
        onClick --> (_ => onSave())
      )
    ),
    div(
      cls := "control",
      button(
        cls := (if (small) "button is-small" else "button"),
        "Cancel",
        onClick --> (_ => onCancel())
      )
    )
  )

  /** A tool's model picker: its configurations, the live one marked, picking
    * another stopping the live session and launching that one — or, when its
    * weights are not on disk yet, downloading them and leaving the live one
    * running (`specs/46-starter-configurations.md`). Rebuilt only when the live
    * session, the configurations or a download's start or end change, not with
    * progress, which would reset an open select.
    */
  private def modelPicker(
      tool: RuntimeTool,
      heading: Signal[String],
      /** The project's kind, which narrows the list to the models making it. */
      kind: Signal[Option[ProjectKind]]
  ): HtmlElement = {
    // A model picked before it could launch: it stays selected while what it
    // lacks downloads, and launches once nothing is missing (François,
    // 2026-09-28). Picking again, or the empty entry, replaces it.
    val picked = Var(Option.empty[String])

    /** A configuration created from **+ New**, picked once the cache has a word
      * on each of its models — a model registered with it has none yet, and no
      * word reads as nothing missing.
      */
    val created = Var(Option.empty[String])
    val creating = Var(false)

    def launch(id: String, live: Option[(String, String)]): Unit = {
      live.foreach((sessionId, _) =>
        sessionService.push(SessionService.Command.Stop(sessionId))
      )
      sessionService.push(SessionService.Command.Launch(id, None))
    }

    /** What picking `id` in the select does: launch it, or — lacking weights or
      * a runtime — keep it picked while they come.
      */
    def choose(
        id: String,
        live: Option[(String, String)],
        missing: Map[String, LaunchPrerequisites.Missing]
    ): Unit =
      if (live.exists(_._2 == id)) picked.set(None)
      else
        missing.get(id) match {
          // Kept until it can launch; the runtime is chosen in the notice
          // beside the picker.
          case Some(lacks) =>
            picked.set(Some(id))
            lacks.weights.foreach(w => prerequisites.download(w.idle))
          case None =>
            picked.set(None)
            launch(id, live)
        }

    div(
      cls := "field is-grouped is-align-items-center mb-0",
      picked.signal
        .combineWith(prerequisites.byConfiguration, sessions.liveKey(tool))
        --> Observer[
          (
              Option[String],
              Map[String, LaunchPrerequisites.Missing],
              Option[(String, String)]
          )
        ] { (pick, missing, live) =>
          pick.filterNot(missing.contains).foreach { id =>
            picked.set(None)
            if (!live.exists(_._2 == id)) launch(id, live)
          }
        },
      created.signal
        .combineWith(
          prerequisites.unsettled,
          prerequisites.byConfiguration,
          sessions.liveKey(tool)
        )
        --> Observer[
          (
              Option[String],
              Set[String],
              Map[String, LaunchPrerequisites.Missing],
              Option[(String, String)]
          )
        ] { (fresh, unsettled, missing, live) =>
          fresh.filterNot(unsettled.contains).foreach { id =>
            created.set(None)
            choose(id, live, missing)
          }
        },
      span(cls := "text-secondary is-size-7 mr-2", child.text <-- heading),
      child <-- sessions
        .liveKey(tool)
        .combineWith(
          sessions.configurationsOf(tool, kind),
          kind,
          prerequisites.byConfiguration,
          picked.signal
        )
        .map { (live, listed, kind, missing, pick) =>
          val liveConfiguration = live.map(_._2)
          val shown = pick.orElse(liveConfiguration)
          // Sessions are not the project's: a live model of the other kind
          // stays listed, marked, so the select names what actually runs.
          val configurations =
            listed.filter((c, fits) => fits || liveConfiguration.contains(c.id))
          div(
            cls := "control",
            select(
              cls := "select is-small",
              onChange.mapToValue --> Observer[String] { id =>
                if (id.isEmpty) picked.set(None)
                else choose(id, live, missing)
              },
              option(
                value := "",
                selected := shown.isEmpty,
                if (configurations.isEmpty)
                  kind
                    .fold("no configuration")(k =>
                      s"no ${k.noun} configuration"
                    )
                else "choose a model…"
              ),
              configurations.map((c, fits) =>
                option(
                  value := c.id,
                  selected := shown.contains(c.id),
                  c.label + (
                    // the upstream engine goes without saying (`specs/43`)
                    if (c.runner == RuntimeEngine.upstream(tool)) ""
                    else s" · ${c.runner.displayName}"
                  ) + (
                    missing
                      .get(c.id)
                      .fold("")(lacks =>
                        s" · ⬇ ${LaunchPrerequisites.describe(lacks)}"
                      )
                  ) + (
                    if (!liveConfiguration.contains(c.id)) ""
                    else if (fits) " (live)"
                    else
                      kind
                        .fold(" (live)")(k => s" (live, not a ${k.noun} model)")
                  )
                )
              )
            )
          )
        },
      // No runtime to run this tool's models: said once beside the picker, with
      // the install (`specs/46`).
      child <-- sessions
        .configurationsOf(tool, kind)
        .combineWith(prerequisites.byConfiguration)
        .map((listed, missing) =>
          // Said once for the whole picker: every build its configurations can
          // take, and a pick on another engine switches all that run on it.
          // Only the ones it offers — a video model is not an image project's
          // concern.
          LaunchPrerequisites.RuntimeNeed
            .merge(
              listed
                .collect { case (c, true) => c }
                .flatMap(c => missing.get(c.id).flatMap(_.runtime))
            )
            .fold(emptyNode)(need =>
              div(
                cls := "control ml-2",
                LaunchOrDownload.runtimeNotice(need, prerequisites)
              )
            )
        ),
      div(
        cls := "control",
        button(
          cls := "button is-small is-info",
          span(cls := "plus-icon", "+"),
          " New",
          title <-- heading.map(name =>
            s"Create a configuration for the ${name.toLowerCase} and pick it"
          ),
          onClick --> (_ => creating.set(true))
        )
      ),
      child <-- creating.signal.combineWith(kind).map {
        case (false, _)   => emptyNode
        case (true, kind) =>
          newConfiguration(
            tool,
            kind,
            configuration => {
              prerequisites.refreshCache()
              created.set(Some(configuration.id))
            },
            () => creating.set(false)
          )
      },
      child <-- sessions.liveStatus(tool).map {
        case Some(status) =>
          span(
            cls := (status match {
              case SessionStatus.Ready => "tag is-success is-small ml-2"
              case _                   => "tag is-info is-small ml-2"
            }),
            status.toString.toLowerCase
          )
        case None => emptyNode
      }
    )
  }

  lazy val element: HtmlElement = div(
    cls := "mb-3",
    div(
      cls := "level is-marginless",
      div(
        cls := "level-left",
        div(
          // The name, renamed in place; only the editing flag rebuilds this.
          child <-- project
            .map(_.map(_.label))
            .distinct
            .combineWith(labelEditing.signal)
            .map {
              case (Some(label), false) =>
                h1(
                  cls := "title text-primary mb-1",
                  label,
                  a(
                    cls := "ml-2 is-size-7",
                    title := "Rename project",
                    "rename",
                    onClick --> { _ =>
                      labelDraft.set(label)
                      labelEditing.set(true)
                    }
                  )
                )
              case (Some(_), true) =>
                inlineEditor(
                  labelDraft,
                  "Project name",
                  () => saveLabel(),
                  () => labelEditing.set(false),
                  small = false
                )
              case _ => h1(cls := "title text-primary mb-1", "Project")
            },
          child <-- project
            .map(_.map(_.brief))
            .distinct
            .combineWith(briefEditing.signal)
            .map {
              case (Some(brief), false) =>
                p(
                  cls := "text-secondary is-size-7 mb-0",
                  if (brief.isEmpty) "no brief yet" else brief,
                  a(
                    cls := "ml-2",
                    "edit",
                    onClick --> { _ =>
                      briefDraft.set(brief)
                      briefEditing.set(true)
                    }
                  )
                )
              case (Some(_), true) =>
                inlineEditor(
                  briefDraft,
                  "What is being made — the assistant reads this",
                  () => saveBrief(),
                  () => briefEditing.set(false),
                  small = true
                )
              case _ => emptyNode
            }
        )
      ),
      div(
        cls := "level-right",
        select(
          cls := "select is-small mr-3",
          title := "What the project makes — the model picker offers only " +
            "the models that make it",
          onChange.mapToValue --> Observer[String] { value =>
            ProjectKind.values
              .find(_.toString == value)
              .foreach(kind =>
                currentProject()
                  .filter(_.kind != kind)
                  .foreach(current =>
                    projectService.push(
                      Command.Update(current.id, current.copy(kind = kind))
                    )
                  )
              )
          },
          ProjectKind.values.toList.map(kind =>
            option(
              value := kind.toString,
              selected <-- project.map(_.exists(_.kind == kind)),
              s"${kind.noun.capitalize} project"
            )
          )
        ),
        label(
          cls := "checkbox text-secondary is-size-7 mr-3",
          title :=
            "An NSFW project is hidden from the list unless asked for, and its LoRA picker shows the NSFW ones",
          input(
            typ := "checkbox",
            checked <-- project.map(_.exists(_.nsfw)),
            onChange.mapToChecked --> Observer[Boolean] { checked =>
              currentProject().foreach(current =>
                projectService.push(
                  Command.Update(current.id, current.copy(nsfw = checked))
                )
              )
            }
          ),
          " NSFW"
        ),
        button(
          cls := "button is-small",
          "← Projects",
          onClick --> (_ => Page.Projects.navigate())
        )
      )
    ),
    // A text project has no image model to pick, and its chat model is
    // the whole of it rather than an assistant (`specs/41-text-projects.md`).
    child <-- projectKind.map {
      case Some(ProjectKind.Text) =>
        div(
          cls := "is-flex is-flex-wrap-wrap",
          modelPicker(RuntimeTool.LlamaCpp, Val("Chat model"), Val(None))
        )
      // One picker per row, whatever the notices beside them say: a row that
      // shortened while installing pulled the assistant up beside the image
      // model, and the header jumped about (François, 2026-09-28).
      case kind =>
        div(
          cls := "is-flex is-flex-direction-column",
          styleAttr := "gap: 0.5rem;",
          modelPicker(
            RuntimeTool.SdCpp,
            Val(kind match {
              case Some(ProjectKind.Video) => "Video model"
              case _                       => "Image model"
            }),
            projectKind
          ),
          modelPicker(RuntimeTool.LlamaCpp, Val("Assistant"), Val(None))
        )
    }
  )
}
