package drift.frontend.pages.settings

import drift.frontend.components.*
import drift.frontend.services.PromptTemplateService
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** The prompt library (`specs/32-prompt-library.md`): the templates by kind,
  * built-ins read-only, user copies editable in place.
  */
class PromptsSection(service: PromptTemplateService) extends Component {
  import PromptTemplateService.Command

  /** The template being edited, if any — one at a time — and whether it is new
    * (saved with a create) or a stored one (saved with an update).
    */
  private val editing = Var(Option.empty[(PromptTemplate, Boolean)])

  private def blank(kind: PromptKind): PromptTemplate =
    PromptTemplate(
      id = s"${kind.toString.toLowerCase}-${System.currentTimeMillis()}",
      kind = kind,
      label = "",
      text = ""
    )
  private val expanded = Var(Set.empty[String])

  private def kindLabel(kind: PromptKind): String = kind match {
    case PromptKind.AssistantSystem   => "Assistant system prompts"
    case PromptKind.RedrawRestoration => "Redraw restoration prompts"
    case PromptKind.Compaction        => "Compaction prompts"
    case PromptKind.Edit              => "Edit prompts"
    case PromptKind.RedrawDiagnosis   => "Redraw diagnosis prompts"
  }

  private def kindBlurb(kind: PromptKind): String = kind match {
    case PromptKind.AssistantSystem =>
      "Chosen in the assistant panel. The template comes first; drift appends " +
        "the project brief, the model's prompting notes, the prompt being " +
        "worked on, the editing rules and the CFG note — each only when the " +
        "template keeps it."
    case PromptKind.RedrawRestoration =>
      "Chosen under Advanced on a redraw. Sent with the tile's position and " +
        "any extra instructions; the source prompt is not."
    case PromptKind.Compaction =>
      "Used when a project's conversation is compacted into a summary."
    case PromptKind.Edit =>
      "Chosen under Advanced on an edit. The instruction follows it; the " +
        "tile is the image the model changes, and nothing else is sent."
    case PromptKind.RedrawDiagnosis =>
      "What the assistant is asked when a redraw has it read the picture: " +
        "sent once, with the picture scaled down and the tiles drawn on it. " +
        "The answer must keep its JSON shape — the cells, and the repairs."
  }

  private def copyOf(template: PromptTemplate): PromptTemplate =
    template.copy(
      id = s"${template.id}-copy-${System.currentTimeMillis()}",
      label = s"${template.label} (copy)",
      builtIn = false
    )

  private def row(template: PromptTemplate): HtmlElement = {
    val isExpanded = expanded.signal.map(_.contains(template.id))
    div(
      cls := "box bg-card p-3 mb-2",
      div(
        cls := "level is-mobile is-marginless",
        div(
          cls := "level-left",
          p(
            cls := "text-primary",
            strong(template.label),
            Option.when(template.builtIn)(
              span(cls := "tag is-dark is-small ml-2", "built-in")
            )
          )
        ),
        div(
          cls := "level-right",
          div(
            cls := "buttons are-small",
            button(
              cls := "button",
              child.text <-- isExpanded.map(if (_) "Hide" else "Show"),
              onClick --> (_ =>
                expanded.update(set =>
                  if (set.contains(template.id)) set - template.id
                  else set + template.id
                )
              )
            ),
            button(
              cls := "button",
              "Duplicate",
              onClick --> (_ => service.push(Command.Create(copyOf(template))))
            ),
            Option.unless(template.builtIn)(
              button(
                cls := "button",
                "Edit",
                onClick --> (_ => editing.set(Some((template, false))))
              )
            ),
            Option.unless(template.builtIn)(
              button(
                cls := "button is-danger is-light",
                "Delete",
                onClick --> { _ =>
                  if (window.confirm(s"Delete the prompt '${template.label}'?"))
                    service.push(Command.Delete(template.id))
                }
              )
            )
          )
        )
      ),
      child <-- isExpanded.map(open =>
        if (!open) emptyNode
        else
          textArea(
            cls := "textarea is-small mt-2",
            readOnly := true,
            rows := math.min(20, template.text.count(_ == '\n') + 2),
            value := template.text
          )
      )
    )
  }

  private def editor(template: PromptTemplate, isNew: Boolean): HtmlElement = {
    val labelVar = Var(template.label)
    val textVar = Var(template.text)
    val appendsVar = Var(template.appends)
    val formatVar = Var(template.proposalFormat)
    def appendBox(
        name: String,
        hint: String,
        get: AssistantAppends => Boolean,
        set: (AssistantAppends, Boolean) => AssistantAppends
    ): HtmlElement = label(
      cls := "checkbox mr-3",
      title := hint,
      input(
        typ := "checkbox",
        checked <-- appendsVar.signal.map(get),
        onChange.mapToChecked --> Observer[Boolean](on =>
          appendsVar.update(set(_, on))
        )
      ),
      s" $name"
    )
    div(
      cls := "box bg-card p-3 mb-3",
      div(
        cls := "field",
        label(cls := "label is-small", "Label"),
        input(
          cls := "input is-small",
          typ := "text",
          controlled(
            value <-- labelVar.signal,
            onInput.mapToValue --> labelVar
          )
        )
      ),
      div(
        cls := "field",
        label(cls := "label is-small", "Text"),
        textArea(
          cls := "textarea is-small",
          rows := 14,
          controlled(
            value <-- textVar.signal,
            onInput.mapToValue --> textVar
          )
        )
      ),
      Option.when(template.kind == PromptKind.AssistantSystem)(
        div(
          div(
            cls := "field",
            label(cls := "label is-small", "drift appends after the template"),
            appendBox(
              "project brief",
              "\"The project being worked on: …\" — the brief typed on the " +
                "project, so the model knows what the images are for. Nothing " +
                "is added in the Sandbox.",
              _.brief,
              (a, on) => a.copy(brief = on)
            ),
            appendBox(
              "prompting notes",
              "\"The prompt is for <model>. …\" — the target architecture's " +
                "prompting notes (how that model reads prompts: sentences, " +
                "tags, JSON…), or a generic note when it has none.",
              _.promptingNotes,
              (a, on) => a.copy(promptingNotes = on)
            ),
            appendBox(
              "the prompt being worked on",
              "\"The prompt the user is working on — their intent, the text " +
                "to edit: …\" — the prompt and negative prompt as they stand " +
                "in the generation form right now, so a proposal can start " +
                "from them.",
              _.workingPrompt,
              (a, on) => a.copy(workingPrompt = on)
            ),
            appendBox(
              "editing rules",
              "drift's rules for using the working prompt: edit it rather than " +
                "rewrite, keep every detail the user asked for, never replace " +
                "it with a description of the image. Turn off for a template " +
                "whose job is to convert an idea rather than edit a prompt.",
              _.rules,
              (a, on) => a.copy(rules = on)
            ),
            appendBox(
              "CFG note",
              "When the form runs at CFG 1: \"the negative prompt has no " +
                "effect, do not spend effort on it.\" Nothing is added at " +
                "other CFG values.",
              _.cfgNote,
              (a, on) => a.copy(cfgNote = on)
            ),
            appendBox(
              "the form's aspect ratio",
              "\"TARGET IMAGE ASPECT RATIO: W:H (width:height), for a WxH " +
                "image.\" from the form's size — for templates that produce a " +
                "caption sized to the picture, like the Ideogram caption writer.",
              _.aspectRatio,
              (a, on) => a.copy(aspectRatio = on)
            )
          ),
          div(
            cls := "field",
            label(cls := "label is-small", "Proposals are read as"),
            div(
              cls := "select is-small",
              title := "How a proposal is found in a reply. Fenced: the last " +
                "```prompt block, with a ```negative block beside it — what " +
                "drift's own helper asks for. JSON: the last {…} object in " +
                "the reply, minified into the prompt with no negative; an " +
                "aspect_ratio key is taken out and sizes the form instead.",
              select(
                option(
                  value := "Fenced",
                  selected <-- formatVar.signal
                    .map(_ == ProposalFormat.Fenced),
                  "fenced prompt / negative blocks"
                ),
                option(
                  value := "Json",
                  selected <-- formatVar.signal.map(_ == ProposalFormat.Json),
                  "the last JSON object in the reply"
                ),
                onChange.mapToValue --> Observer[String] {
                  case "Json" => formatVar.set(ProposalFormat.Json)
                  case _      => formatVar.set(ProposalFormat.Fenced)
                }
              )
            )
          )
        )
      ),
      div(
        cls := "buttons are-small mt-3",
        button(
          cls := "button is-primary",
          "Save",
          disabled <-- labelVar.signal
            .combineWith(textVar.signal)
            .map((l, t) => l.trim.isEmpty || t.trim.isEmpty),
          onClick --> { _ =>
            val saved = template.copy(
              label = labelVar.now().trim,
              text = textVar.now(),
              appends = appendsVar.now(),
              proposalFormat = formatVar.now()
            )
            service.push(
              if (isNew) Command.Create(saved) else Command.Update(saved)
            )
            editing.set(None)
          }
        ),
        button(cls := "button", "Cancel", onClick --> (_ => editing.set(None)))
      )
    )
  }

  private def kindBlock(kind: PromptKind): HtmlElement = div(
    cls := "mb-5",
    div(
      cls := "level is-mobile is-marginless mb-1",
      div(
        cls := "level-left",
        h3(cls := "is-size-6 text-primary mb-0", kindLabel(kind))
      ),
      div(
        cls := "level-right",
        button(
          cls := "button is-small is-primary",
          "New prompt",
          onClick --> (_ => editing.set(Some((blank(kind), true))))
        )
      )
    ),
    p(cls := "text-secondary is-size-7 mb-2", kindBlurb(kind)),
    child <-- editing.signal.map {
      case Some((t, isNew)) if t.kind == kind => editor(t, isNew)
      case _                                  => emptyNode
    },
    children <-- service.ofKind(kind).map(_.map(row))
  )

  lazy val element: HtmlElement = div(
    div(
      cls := "level mt-5",
      div(
        cls := "level-left",
        h2(cls := "is-size-5 text-primary", "Prompts")
      )
    ),
    ErrorBanner(service),
    kindBlock(PromptKind.AssistantSystem),
    kindBlock(PromptKind.RedrawRestoration),
    kindBlock(PromptKind.RedrawDiagnosis),
    kindBlock(PromptKind.Edit),
    kindBlock(PromptKind.Compaction)
  )
}
