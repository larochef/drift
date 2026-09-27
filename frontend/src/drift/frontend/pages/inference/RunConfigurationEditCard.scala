package drift.frontend.pages.inference

import drift.frontend.components.*
import drift.frontend.services.{BrowserServices, LoraService}
import drift.shared.*

import com.raquo.laminar.api.L.*

class RunConfigurationEditCard(
    /** The configuration as it was when editing began. */
    val rm: RunConfiguration,
    /** Its architecture — the checkpoint slots to assign, and whether it takes
      * LoRAs, which only sd-cpp does.
      */
    architecture: Option[Architecture],
    allModels: Signal[List[Model]],
    /** The installed LoRAs — the default LoRAs' choice — and the way to add
      * more to the architecture.
      */
    loraService: LoraService,
    browsers: BrowserServices,
    assistantTemplates: Signal[List[PromptTemplate]] = Val(Nil),
    /** The installed runtimes, to say which runners are there (`specs/43`). */
    runtimes: Signal[List[Runtime]] = Val(Nil)
) extends Component {
  private val labelVar = Var(rm.label)
  private val runnerVar = Var(rm.runner)
  private val assistantTemplateVar = Var(rm.assistantTemplateId)
  private val assignments = Var(rm.assignments.toList)
  private val paramEditor = ParamEditor(
    inherited = RunConfigurationForm.inheritedParameters(
      Val(architecture),
      assignments.signal,
      allModels
    )
  )
  private val loraIds = Var(rm.loras.map(_.loraId))
  private val loraStrengths = Var(LoraPicker.strengthsOf(rm.loras))

  paramEditor.reset(rm.overriddenParameters.toList, rm.removedParameters)

  def snapshot(): RunConfiguration =
    rm.copy(
      label = labelVar.now(),
      assignments = assignments.now().filter(_._1.nonEmpty).toMap,
      overriddenParameters = paramEditor.snapshot(),
      removedParameters = paramEditor.removedSnapshot(),
      loras = LoraPicker.configured(loraIds.now(), loraStrengths.now()),
      assistantTemplateId = assistantTemplateVar.now(),
      runner = runnerVar.now()
    )

  def reset(): Unit = {
    runnerVar.set(rm.runner)
    labelVar.set(rm.label)
    assignments.set(rm.assignments.toList)
    paramEditor.reset(rm.overriddenParameters.toList, rm.removedParameters)
    loraIds.set(rm.loras.map(_.loraId))
    loraStrengths.set(LoraPicker.strengthsOf(rm.loras))
    assistantTemplateVar.set(rm.assistantTemplateId)
  }

  lazy val element: HtmlElement = div(
    // The modal it opens in carries the title.
    cls := "card bg-card mb-4",
    div(
      cls := "card-content",
      div(
        cls := "field",
        label(cls := "label text-primary", "Label"),
        input(
          cls := "input",
          placeholder := "Label",
          value <-- labelVar.signal,
          onInput.mapToValue --> labelVar
        )
      ),
      div(
        cls := "field",
        label(cls := "label text-primary", "Architecture"),
        p(cls := "text-secondary", rm.architectureId)
      ),
      RunConfigurationForm.runnerField(Val(architecture), runtimes, runnerVar),
      div(
        cls := "field",
        label(cls := "label text-primary", "Checkpoint assignments"),
        CheckpointAssignments(
          architecture.map(_.checkpoints).getOrElse(Nil),
          allModels,
          assignments
        ).element
      ),
      // The assistant prompt an image or video configuration prefers; a chat
      // configuration is the assistant.
      if (architecture.exists(_.tool == RuntimeTool.SdCpp))
        div(
          cls := "field",
          RunConfigurationForm.assistantTemplateField(
            assistantTemplates,
            assistantTemplateVar
          )
        )
      else emptyNode,
      // Chat models take LoRAs too, applied per reply
      // (`specs/35-assistant-loras.md`).
      architecture match {
        case None                   => emptyNode
        case Some(loraArchitecture) =>
          div(
            cls := "field",
            label(cls := "label text-primary", "Default LoRAs"),
            p(
              cls := "help text-secondary mb-2",
              "Applied wherever this configuration runs: a generation form " +
                "starts with them, redraw and PiD send them with every tile, " +
                "and an assistant applies them to every reply."
            ),
            LoraPicker(
              collection = loraService.loadedLoras,
              architectureId = Val(Some(rm.architectureId)),
              selectedIds = loraIds,
              strengths = loraStrengths,
              scope = "this configuration",
              // Installed for the architecture, so every configuration of it
              // can pick them.
              headerAction = LoraInstallButton(
                browsers,
                loraArchitecture,
                loraService
              ).element
            ).element
          )
      },
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
