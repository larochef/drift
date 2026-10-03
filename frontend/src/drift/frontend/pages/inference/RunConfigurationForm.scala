package drift.frontend.pages.inference

import drift.frontend.components.*
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*

class RunConfigurationForm(
    architectures: Signal[List[Architecture]],
    /** The registered models, and where a slot's missing one is created. */
    modelService: ModelService,
    /** The installed LoRAs — the default LoRAs' choice — and the way to add
      * more to the architecture.
      */
    loraService: LoraService,
    browsers: BrowserServices,
    /** The assistant system templates, for the configuration's override of its
      * architecture's default (`specs/32`).
      */
    assistantTemplates: Signal[List[PromptTemplate]] = Val(Nil),
    /** The installed runtimes, to say which runners are there (`specs/43`). */
    runtimes: Signal[List[Runtime]] = Val(Nil)
) extends Component {
  private val allModels = modelService.allModels
  private val idVar = Var("")

  /** The engine it runs on; the architecture's first runner until chosen. */
  private val runnerVar = Var[RuntimeEngine](RuntimeEngine.SdCpp)
  private val assistantTemplateVar = Var(Option.empty[String])
  private val labelVar = Var("")
  private val archIdVar = Var("")
  private val assignments = Var(List.empty[(String, String)])
  private val loraIds = Var(List.empty[String])
  private val loraStrengths = Var(Map.empty[String, String])

  def snapshot(): RunConfiguration = {
    val id = idVar.now()
    val label = labelVar.now()
    val archId = archIdVar.now()
    val assigns = assignments.now().filter(_._1.nonEmpty).toMap
    val params = paramEditor.snapshot()
    val now = System.currentTimeMillis()
    RunConfiguration(
      id,
      label,
      archId,
      assigns,
      params,
      paramEditor.removedSnapshot(),
      createdAt = now,
      lastUsedAt = now,
      loras = LoraPicker.configured(loraIds.now(), loraStrengths.now()),
      assistantTemplateId = assistantTemplateVar.now(),
      runner = runnerVar.now()
    )
  }

  def reset(): Unit = {
    idVar.set("")
    labelVar.set("")
    archIdVar.set("")
    assignments.set(Nil)
    paramEditor.reset()
    loraIds.set(Nil)
    loraStrengths.set(Map.empty)
  }

  def valid: Boolean =
    idVar.now().nonEmpty && labelVar.now().nonEmpty && archIdVar.now().nonEmpty

  private val checkpointNames: Signal[Set[String]] =
    archIdVar.signal
      .combineWith(architectures)
      .map { (archId, archs) =>
        archs
          .find(_.id == archId)
          .map(_.checkpoints.map(_.name).toSet)
          .getOrElse(Set.empty)
      }
      .distinct

  /** An image or video architecture, whose configuration can prefer an
    * assistant prompt; a chat configuration is the assistant.
    */
  private val generatesMedia: Signal[Boolean] =
    archIdVar.signal
      .combineWith(architectures)
      .map((archId, archs) =>
        archs.exists(a => a.id == archId && a.tool == RuntimeTool.SdCpp)
      )
      .distinct

  private val selectedArchitecture: Signal[Option[Architecture]] =
    archIdVar.signal
      .combineWith(architectures)
      .map((archId, archs) => archs.find(_.id == archId))
      .distinct

  // Declared after `selectedArchitecture`, which its chips are drawn from.
  private val paramEditor = ParamEditor(
    inherited = RunConfigurationForm
      .inheritedParameters(selectedArchitecture, assignments.signal, allModels)
  )

  private lazy val assistantTemplateBlock: HtmlElement = div(
    cls := "field",
    RunConfigurationForm.assistantTemplateField(
      assistantTemplates,
      assistantTemplateVar
    )
  )

  /** Every architecture takes LoRAs; a chat model applies them per reply
    * (`specs/35-assistant-loras.md`).
    */
  private lazy val loraField: HtmlElement = div(
    cls := "field",
    label(cls := "label text-primary", "Default LoRAs"),
    p(
      cls := "help text-secondary mb-2",
      "Applied wherever this configuration runs: a generation form starts " +
        "with them, redraw and PiD send them with every tile, and an " +
        "assistant applies them to every reply."
    ),
    LoraPicker(
      collection = loraService.loadedLoras,
      architectureId = archIdVar.signal.map(Option(_).filter(_.nonEmpty)),
      selectedIds = loraIds,
      strengths = loraStrengths,
      scope = "this configuration",
      // Installed for the architecture, so every configuration of it can pick
      // them.
      headerAction = child <-- selectedArchitecture.map(
        _.map(architecture =>
          LoraInstallButton(browsers, architecture, loraService).element
        ).getOrElse(emptyNode)
      )
    ).element
  )

  lazy val element: HtmlElement = div(
    cls := "card bg-card mb-4",
    checkpointNames.changes --> Observer[Set[String]] { names =>
      assignments.update(_.filter { case (name, _) => names.contains(name) })
    },
    // LoRAs belong to one architecture: another one starts without any.
    archIdVar.signal.changes --> (_ => loraIds.set(Nil)),
    div(
      cls := "card-content",
      div(
        cls := "columns",
        div(
          cls := "column",
          div(
            cls := "field",
            label(cls := "label text-primary", "ID"),
            input(
              cls := "input",
              placeholder := "e.g. my-config",
              value <-- idVar.signal,
              onInput.mapToValue --> idVar
            )
          )
        ),
        div(
          cls := "column",
          div(
            cls := "field",
            label(cls := "label text-primary", "Label"),
            input(
              cls := "input",
              placeholder := "My Config",
              value <-- labelVar.signal,
              onInput.mapToValue --> labelVar
            )
          )
        )
      ),
      child <-- architectures.combineWith(archIdVar.signal).map {
        (archs, archId) =>
          select(
            cls := "select",
            onChange.mapToValue --> archIdVar,
            option(value := "", "Select architecture"),
            archs.map(a =>
              option(
                value := a.id,
                a.label,
                if (a.id == archId) selected := true else emptyNode
              )
            )
          )
      },
      // a new architecture starts on the drift runner where it runs on it and
      // the runner is installed — the default wherever it is available —
      // else on its first runner, its upstream engine
      selectedArchitecture.changes.withCurrentValueOf(runtimes) --> {
        (selected, installed) =>
          selected.foreach(a =>
            runnerVar.set(
              if (
                a.runners.contains(RuntimeEngine.DriftRunner) &&
                installed.exists(r =>
                  r.tool == a.tool &&
                    r.engine == RuntimeEngine.DriftRunner && r.valid
                )
              ) RuntimeEngine.DriftRunner
              else a.runners.headOption.getOrElse(RuntimeEngine.upstream(a.tool))
            )
          )
      },
      RunConfigurationForm
        .runnerField(selectedArchitecture, runtimes, runnerVar),
      child <-- archIdVar.signal.combineWith(architectures).map {
        (archId, archs) =>
          CheckpointAssignments(
            archs.find(_.id == archId),
            modelService,
            browsers,
            assignments
          ).element
      },
      child <-- generatesMedia.map(
        if (_) assistantTemplateBlock else emptyNode
      ),
      child <-- selectedArchitecture
        .map(_.isDefined)
        .distinct
        .map(if (_) loraField else emptyNode),
      div(
        cls := "field",
        label(cls := "label text-primary", "Override parameters"),
        p(
          cls := "help text-secondary mb-2",
          "What this configuration wants over its architecture, its models " +
            "and the runtime. \uD83D\uDEAB keeps a flag one of them sets off " +
            "the command line entirely."
        ),
        paramEditor.element
      )
    )
  )
}

object RunConfigurationForm {

  /** What the layers below a configuration set, as a signal of the form's own
    * fields: `CommandLine.inheritedParameters`
    * (`specs/16-parameter-resolution.md`) does the resolving, so what the chips
    * offer and what a launch would pass cannot disagree.
    */
  def inheritedParameters(
      architecture: Signal[Option[Architecture]],
      assignments: Signal[List[(String, String)]],
      allModels: Signal[List[Model]]
  ): Signal[List[ResolvedParameter]] =
    architecture.combineWith(assignments, allModels).map {
      (selected, assigned, models) =>
        selected.toList.flatMap(a =>
          CommandLine.inheritedParameters(
            RunConfiguration(
              id = "",
              label = "",
              architectureId = a.id,
              assignments = assigned.filter(_._1.nonEmpty).toMap,
              overriddenParameters = Map.empty,
              createdAt = 0L,
              lastUsedAt = 0L,
              runner = RuntimeEngine.upstream(a.tool)
            ),
            a,
            models
          )
        )
    }

  /** The engine a configuration runs on (`specs/43`), among its architecture's
    * runners; one not installed says so. Shared with the edit card.
    */
  def runnerField(
      architecture: Signal[Option[Architecture]],
      runtimes: Signal[List[Runtime]],
      choice: Var[RuntimeEngine]
  ): HtmlElement = div(
    child <-- architecture.combineWith(runtimes, choice.signal).map {
      case (None, _, _)                 => emptyNode
      case (Some(a), installed, chosen) =>
        def there(engine: RuntimeEngine) =
          installed
            .exists(r => r.tool == a.tool && r.engine == engine && r.valid)
        div(
          cls := "field",
          label(cls := "label text-primary", "Runner"),
          select(
            cls := "select",
            onChange.mapToValue --> Observer[String](name =>
              RuntimeEngine.values.find(_.toString == name).foreach(choice.set)
            ),
            a.runners.map(engine =>
              option(
                value := engine.toString,
                if (engine == chosen) selected := true else emptyNode,
                engine.displayName + (if (there(engine)) ""
                                      else " (not installed)")
              )
            )
          ),
          p(
            cls := "help text-secondary",
            "The engine every launch of this configuration runs on: sessions, " +
              "projects, the assistant, PiD, redraw and edit. A launch can still " +
              "pick another runtime."
          )
        )
    }
  )

  /** The configuration's assistant template override (`specs/32`): none means
    * the architecture's default. Shared with the edit card.
    */
  def assistantTemplateField(
      templates: Signal[List[PromptTemplate]],
      choice: Var[Option[String]]
  ): HtmlElement = div(
    cls := "field",
    label(cls := "label text-primary", "Assistant prompt"),
    PromptTemplatePicker(
      templates,
      choice,
      none = Some("the architecture's default"),
      small = false
    ).element,
    p(
      cls := "help text-secondary",
      "The system prompt a project's assistant is offered when this " +
        "configuration's model starts."
    )
  )
}
