package drift.frontend.pages.assistant

import drift.frontend.components.*
import drift.frontend.pages.gallery.GalleryPicker
import drift.frontend.services.{AssistantService, GenerationService}
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** The chat with a live assistant session (`specs/21-assistant-page.md`,
  * `specs/20`): the turns with reasoning folded away, a composer with staged
  * images, a context meter, and "Apply to form" on every proposal. In a
  * workspace it is the project's conversation — kept, compacted and restarted;
  * in free play a scratch chat that just clears.
  */
class AssistantPanel(
    sessionId: String,
    sessionSignal: Signal[Option[Session]],
    service: AssistantService,
    generationService: GenerationService,
    onStop: () => Unit,
    /** Inside a project workspace the proposal goes straight to the panel
      * beside the chat; on the Assistant page it is handed to the next panel
      * and Inference opens.
      */
    proposalTarget: Option[AssistantTurns.ProposalTarget] = None,
    /** Free play (`specs/22-free-play-and-scratch-generations.md`): the
      * Assistant page is a configuration page that happens to run a model, so
      * it starts with **no** system prompt — what answers is the raw model,
      * which is the question being asked of it. A workspace starts at the
      * default plus the project's brief. The toggle stays either way.
      */
    freePlay: Boolean = false,
    /** The gallery as a source of pictures to attach (`specs/50`). */
    pickFromGallery: Option[GalleryPicker.Open] = None
) extends Component {
  private val showSystemPrompt = Var(false)
  private val showArchive = Var(false)

  private val progressTag = htmlTag("progress")

  /** The context used, as a percentage, once a reply has reported it. */
  private val contextPercent: Signal[Option[Int]] =
    service.contextUsed.combineWith(service.properties).map {
      case (Some(used), Some(properties)) if properties.contextSize > 0 =>
        Some(used * 100 / properties.contextSize)
      case _ => None
    }

  private def confirmRestart(): Unit = {
    val count = service.turnsNow.count(turn => !turn.streaming)
    if (
      dom.window.confirm(
        s"Restart the conversation? Its $count messages move to the " +
          "archive; versions and generations stay."
      )
    ) service.restart()
  }

  /** Offered when the image model that just started prefers another system
    * template than the conversation uses (`specs/32`): switch, or keep — a kept
    * choice is not asked again until the suggestion changes.
    */
  private val dismissedSuggestion = Var(Option.empty[String])

  private def switchOffer: Signal[Node] =
    service.templateId.signal
      .combineWith(
        service.suggestedTemplateId,
        dismissedSuggestion.signal,
        service.library.templates
      )
      .map { (current, suggested, dismissed, all) =>
        suggested.filter(s =>
          current.isDefined && !current.contains(s) && !dismissed.contains(s)
        ) match {
          case None    => emptyNode
          case Some(s) =>
            def labelOf(id: Option[String]) =
              id.flatMap(i => all.find(_.id == i)).map(_.label).getOrElse("—")
            div(
              cls := "notification is-info is-light py-2 px-3 is-size-7 mb-2",
              s"The model that just started is set up for the prompt " +
                s"“${labelOf(Some(s))}”; this conversation uses " +
                s"“${labelOf(current)}”. ",
              div(
                cls := "buttons are-small mt-1 mb-0",
                button(
                  cls := "button is-info is-small",
                  "Switch",
                  onClick --> (_ => service.templateId.set(Some(s)))
                ),
                button(
                  cls := "button is-small",
                  "Keep",
                  onClick --> (_ => dismissedSuggestion.set(Some(s)))
                )
              )
            )
        }
      }

  /** The system template (`specs/32`): picked here, edited in Settings →
    * Prompts. A workspace saves the pick to its project; free play keeps the
    * last one and may pick none, the raw model.
    */
  private def systemPromptEditor: HtmlElement = div(
    cls := "field",
    label(cls := "label is-small text-secondary", "System prompt"),
    PromptTemplatePicker(
      service.library.ofKind(PromptKind.AssistantSystem),
      service.templateId,
      none = Option.when(freePlay)("(none — the raw model)")
    ).element,
    child <-- service.templateId.signal
      .combineWith(service.library.templates)
      .map { (id, all) =>
        id.flatMap(i => all.find(_.id == i)) match {
          case Some(t) =>
            textArea(
              cls := "textarea is-small mt-2",
              readOnly := true,
              rows := 8,
              value := t.text
            )
          case None =>
            p(
              cls := "text-secondary is-size-7 mt-1",
              "No template: the raw model, sent no system message."
            )
        }
      }
  )

  private def contextMeter: HtmlElement = div(
    cls := "mb-2",
    child <-- service.contextUsed
      .combineWith(service.properties, service.bound)
      .map {
        case (Some(used), Some(properties), bound)
            if properties.contextSize > 0 =>
          val percent = used * 100 / properties.contextSize
          val colour =
            if (percent >= 95) "is-danger"
            else if (percent >= 75) "is-warning"
            else "is-info"
          val remedy = if (bound) "compact or restart" else "clear the chat"
          val advice =
            if (percent >= 100) s" — full: $remedy before sending"
            else if (percent >= 75) s" — $remedy before it overflows"
            else ""
          div(
            progressTag(
              cls := s"progress is-small $colour",
              value := used.min(properties.contextSize).toString,
              maxAttr := properties.contextSize.toString
            ),
            p(
              cls := "text-secondary is-size-7",
              s"context: $used / ${properties.contextSize} tokens ($percent%)$advice"
            )
          )
        case _ => emptyNode
      }
  )

  /** Loading, or why this project's conversation is not being kept. */
  private def conversationStatus: HtmlElement = div(
    child <-- service.loading.map {
      case true =>
        p(cls := "text-secondary is-size-7 mb-2", "Loading the conversation…")
      case false => emptyNode
    },
    child <-- service.conversationError.map {
      case Some(error) => p(cls := "has-text-danger is-size-7 mb-2", error)
      case None        => emptyNode
    }
  )

  private def archiveElement: HtmlElement = div(
    child <-- service.archive.map(_.size).distinct.map {
      case 0     => emptyNode
      case count =>
        p(
          cls := "mb-2",
          a(
            cls := "is-size-7",
            child.text <-- showArchive.signal.map(open =>
              if (open) "hide the archived transcript"
              else s"show the archived transcript ($count messages)"
            ),
            onClick --> (_ => showArchive.update(!_))
          )
        )
    },
    child <-- showArchive.signal.combineWith(service.archive).map {
      case (true, archived) if archived.nonEmpty =>
        div(
          cls := "mb-3",
          archived.reverse.map(turn =>
            div(
              cls := "box bg-card p-2 mb-1",
              p(
                cls := "text-secondary is-size-7 mb-0",
                AssistantTurns.roleLabel(turn.role)
              ),
              pre(cls := "assistant-text is-size-7", turn.text)
            )
          )
        )
      case _ => emptyNode
    }
  )

  lazy val element: HtmlElement = div(
    cls := "content",
    // The system template belongs to the context (`specs/32`): a workspace
    // sets its project's; free play follows the target architecture's
    // default when the target changes, none being raw, and keeps a pick
    // made after that.
    Option.when(freePlay)(
      service.suggestedTemplateId --> Observer[Option[String]](
        service.templateId.set
      )
    ),
    // What the server applied is only readable once it serves.
    sessionSignal
      .map(_.exists(_.status == SessionStatus.Ready))
      .distinct --> Observer[Boolean] { ready =>
      if (ready) service.loadProperties(sessionId)
      else service.clearProperties()
    },
    // A text project talks to the raw model (`specs/41-text-projects.md`):
    // no template to pick, none to switch to.
    child <-- showSystemPrompt.signal
      .combineWith(service.rawModel.signal)
      .map {
        case (true, false) => systemPromptEditor
        case (true, true)  =>
          p(
            cls := "text-secondary is-size-7 mb-2",
            "A text project has no system prompt: the model gets the " +
              "conversation alone, and a compaction's summary once there is one."
          )
        case _ => emptyNode
      },
    Option.unless(freePlay)(child <-- service.rawModel.signal.map {
      case true  => emptyNode
      case false => div(child <-- switchOffer)
    }),
    // The composer stays on top and the turns run newest first (François,
    // 2026-09-08): the reply being written is right under the input, and
    // nothing ever needs scrolling to the bottom.
    AssistantComposer(
      sessionId,
      service,
      sessionSignal,
      contextPercent,
      showSystemPrompt,
      () => confirmRestart(),
      onStop,
      pickFromGallery
    ).element,
    contextMeter,
    conversationStatus,
    archiveElement,
    AssistantTurns(
      sessionId,
      service,
      generationService,
      proposalTarget
    ).element
  )
}
