package drift.frontend.pages.assistant

import drift.frontend.components.*
import drift.frontend.services.AssistantService
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom

/** What the user says next (`specs/21-assistant-page.md`): the conversation's
  * controls, the pictures staged for the model, the text area that sends with
  * Ctrl+Enter, and the file picker beside it.
  *
  * The draft is its own — nothing else reads it — while the system prompt it
  * opens belongs to the panel, which decides where that editor goes
  * (`specs/29-split-oversized-files.md`).
  */
class AssistantComposer(
    sessionId: String,
    service: AssistantService,
    sessionSignal: Signal[Option[Session]],
    /** How full the context is, which colours the Compact button where it is
      * the thing to do.
      */
    contextPercent: Signal[Option[Int]],
    /** Whether the system prompt editor is open: this is where it is toggled,
      * the panel is where it is shown.
      */
    showSystemPrompt: Var[Boolean],
    /** Restarting a project's conversation, confirmed by the panel — it is the
      * one that knows what is being thrown away.
      */
    onRestart: () => Unit,
    onStop: () => Unit
) extends Component {

  private val draft = Var("")

  private def attachmentsStrip: HtmlElement = div(
    children <-- service.attachments.map(_.zipWithIndex.map {
      (attachment, index) =>
        div(
          cls := "mb-1",
          styleAttr := "display: inline-flex; align-items: flex-start; gap: 4px;",
          img(
            src := attachment.previewUrl,
            styleAttr := "max-height: 80px; max-width: 120px; border-radius: 4px;",
            title := attachment.description
          ),
          button(
            cls := "delete is-small",
            title := "Remove image",
            onClick --> (_ => service.removeAttachment(index))
          )
        )
    }),
    // A text-only model is sent what is known about each image, never the
    // picture — said here, so it is never mistaken for one that saw it
    // (`specs/20`).
    child <-- service.attachments.combineWith(service.properties).map {
      case (staged, Some(properties))
          if staged.nonEmpty && !properties.vision =>
        p(
          cls := "is-size-7 text-secondary mb-1",
          "This model has no vision: it gets what is known about each image " +
            "— a generation's parameters and prompt — but not the picture."
        )
      case _ => emptyNode
    }
  )

  /** Uploaded once over HTTP, never read into the page: a video of tens of
    * megabytes goes the same way as a picture.
    */
  private def fileAttach: HtmlElement = div(
    input(
      typ := "file",
      accept := "image/*,video/*",
      cls := "is-size-7",
      title := "Attach an image or a video",
      onChange --> { event =>
        val element = event.target.asInstanceOf[dom.html.Input]
        if (element.files.length > 0) service.uploadFile(element.files(0))
        element.value = ""
      }
    ),
    child <-- service.uploadError.map {
      case Some(error) => p(cls := "has-text-danger is-size-7", error)
      case None        => emptyNode
    }
  )

  private def sendDraft(): Unit = {
    val text = draft.now()
    service.send(sessionId, text)
    draft.set("")
  }

  /** The conversation's controls, in the composer card rather than the header
    * (François, 2026-09-14): the system prompt, and — a project's conversation
    * being kept — compact and restart rather than clear; free play's scratch
    * chat just clears (`specs/20`).
    */
  private def conversationControls: HtmlElement = div(
    cls := "buttons are-small mb-2",
    button(
      cls := "button",
      "System prompt",
      onClick --> (_ => showSystemPrompt.update(!_))
    ),
    children <-- service.bound.map {
      case true =>
        List(
          button(
            // Highlighted past three quarters of the context, where it is the
            // thing to do.
            cls <-- contextPercent.map(percent =>
              if (percent.exists(_ >= 75)) "button is-info" else "button"
            ),
            "Compact",
            title := "Summarise the conversation and archive what the " +
              "summary stands for, to free the context",
            disabled <-- service.streaming
              .combineWith(service.turns)
              .map((streaming, turns) =>
                streaming || !turns.exists(_.role == "user")
              ),
            onClick --> (_ => service.compact(sessionId))
          ),
          button(
            cls := "button",
            "Restart",
            title := "Archive the whole conversation and start over; " +
              "versions and generations stay",
            disabled <-- service.streaming,
            onClick --> (_ => onRestart())
          )
        )
      case false =>
        List(
          button(
            cls := "button",
            "Clear chat",
            onClick --> (_ => service.clear())
          )
        )
    }
  )

  private def composer: HtmlElement = div(
    cls := "box bg-card p-3",
    // Above the text area, so they never read as part of a reply.
    conversationControls,
    attachmentsStrip,
    div(
      cls := "field",
      textArea(
        cls := "textarea",
        rows := 3,
        placeholder :=
          "Ask for a change, or describe what you want… (Ctrl+Enter sends)",
        controlled(value <-- draft.signal, onInput.mapToValue --> draft),
        onKeyDown
          .filter(e => e.key == "Enter" && (e.ctrlKey || e.metaKey))
          .preventDefault --> (_ => sendDraft())
      )
    ),
    div(
      cls := "level is-mobile",
      div(
        cls := "level-left",
        div(cls := "level-item", fileAttach)
      ),
      div(
        cls := "level-right",
        child <-- service.streaming.map {
          case true =>
            button(
              cls := "button is-warning is-small",
              "⏹ Stop",
              onClick --> (_ => service.stop())
            )
          case false =>
            button(
              cls := "button is-primary is-small",
              "Send",
              // Refused while the conversation loads, and once the context is
              // full — the service refuses too, this only says so first.
              disabled <-- draft.signal
                .combineWith(
                  service.attachments,
                  sessionSignal,
                  service.loading,
                  service.contextFull
                )
                .map((text, attachments, session, loading, full) =>
                  (text.trim.isEmpty && attachments.isEmpty) ||
                    !session.exists(_.status == SessionStatus.Ready) ||
                    loading || full
                ),
              onClick --> (_ => sendDraft())
            )
        }
      )
    )
  )

  lazy val element: HtmlElement = composer
}
