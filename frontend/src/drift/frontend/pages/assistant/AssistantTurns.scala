package drift.frontend.pages.assistant

import drift.frontend.Page
import drift.frontend.components.*
import drift.frontend.services.{AssistantService, GenerationService}
import drift.frontend.services.AssistantService.Turn
import drift.shared.*

import com.raquo.laminar.api.L.*

/** The conversation itself (`specs/21-assistant-page.md`): every turn of it,
  * newest first — what was asked with its staged pictures, what came back with
  * its reasoning folded away and its timings, and each proposal on a card with
  * the button that applies it.
  *
  * Its own component because a turn is where most of the chat's markup lives,
  * and none of it has anything to say about the composer above it or the
  * session's header (`specs/29-split-oversized-files.md`).
  */
class AssistantTurns(
    sessionId: String,
    service: AssistantService,
    generationService: GenerationService,
    /** In a workspace a proposal goes to the panel beside the chat; on the
      * Assistant page it is handed on and Inference opens.
      */
    onApplyProposal: Option[PromptProposal => Unit]
) extends Component {

  private def turnElement(
      id: Int,
      initial: Turn,
      signal: Signal[Turn]
  ): HtmlElement =
    initial.role match {
      case "user" =>
        div(
          cls := "box bg-card p-3 mb-2 assistant-turn-user",
          p(cls := "text-secondary is-size-7 mb-1", "You"),
          child <-- signal.map(turn =>
            div(
              if (turn.previews.nonEmpty)
                div(
                  cls := "mb-2",
                  turn.previews.map(previewUrl =>
                    img(
                      src := previewUrl,
                      styleAttr :=
                        "max-height: 120px; max-width: 160px; border-radius: 4px; margin-right: 4px;"
                    )
                  )
                )
              else emptyNode,
              pre(cls := "assistant-text", turn.text)
            )
          )
        )
      // A compaction's summary stands for the archived messages before it
      // (`specs/20`), so it reads as a different kind of block from a reply.
      case "summary" =>
        div(
          cls := "box bg-card p-3 mb-2 assistant-turn-summary",
          p(
            cls := "text-secondary is-size-7 mb-1",
            "Summary of the earlier conversation",
            child <-- signal.map(turn =>
              if (turn.streaming) span(cls := "ml-2", "compacting…")
              else emptyNode
            )
          ),
          child <-- signal.map(turn =>
            div(
              pre(cls := "assistant-text", turn.text),
              turn.error match {
                case Some(error) =>
                  p(cls := "has-text-danger is-size-7", error)
                case None => emptyNode
              }
            )
          )
        )
      case _ =>
        div(
          cls := "box bg-card p-3 mb-2 assistant-turn-assistant",
          p(
            cls := "text-secondary is-size-7 mb-1",
            "Assistant",
            child <-- signal.map(turn =>
              if (turn.streaming) span(cls := "ml-2", "…") else emptyNode
            )
          ),
          child <-- signal.map { turn =>
            div(
              if (turn.reasoning.nonEmpty)
                detailsTag(
                  cls := "mb-2",
                  summaryTag(
                    cls := "text-secondary is-size-7",
                    s"thinking… (${turn.reasoning.length} characters)"
                  ),
                  pre(cls := "assistant-text is-size-7", turn.reasoning)
                )
              else emptyNode,
              pre(cls := "assistant-text", turn.text),
              turn.error match {
                case Some(error) =>
                  div(
                    p(cls := "has-text-danger is-size-7 mb-1", error),
                    // Only the newest reply can be asked again: anything
                    // after it has been answered already.
                    child <-- service.turns
                      .map(_.lastOption.exists(_.id == id))
                      .combineWith(service.streaming)
                      .map {
                        case (true, false) =>
                          button(
                            cls := "button is-small",
                            "↻ Retry",
                            title := "Ask again; this reply is replaced",
                            onClick --> (_ => service.retry(sessionId))
                          )
                        case _ => emptyNode
                      }
                  )
                case None => emptyNode
              },
              turn.proposal match {
                case Some(proposal) => proposalCard(proposal, turn.promptBase)
                case None           => emptyNode
              },
              timingsLine(turn)
            )
          }
        )
    }

  /** Prefill and generation, each as "n tokens in s (tok/s)": prefill is the
    * prompt with its images, the number that grows with the context and the
    * attachments; generation is what the model produced, the number that shows
    * what MTP and the quantisation buy.
    */
  private def timingsLine(turn: Turn): Node = {
    def phase(
        name: String,
        tokens: Option[Int],
        millis: Option[Double],
        rate: Option[Double]
    ): Option[String] =
      tokens.map { n =>
        val duration =
          millis.map(ms => f" in ${ms / 1000}%.1f s").getOrElse("")
        val speed = rate.map(r => f" ($r%.1f tok/s)").getOrElse("")
        s"$name $n tokens$duration$speed"
      }
    val parts = List(
      phase(
        "prefill",
        turn.promptTokens,
        turn.promptMillis,
        turn.promptTokensPerSecond
      ),
      phase(
        "generation",
        turn.completionTokens,
        turn.completionMillis,
        turn.completionTokensPerSecond
      )
    ).flatten
    if (parts.isEmpty) emptyNode
    else p(cls := "text-secondary is-size-7 mt-1", parts.mkString(" · "))
  }

  /** The parsed proposal, with the action the loop exists for: the prompt and
    * negative prompt land in the generation form's fields — and nothing more,
    * generating stays the form's own button (`specs/20`).
    */
  private def proposalCard(
      proposal: PromptProposal,
      base: Option[PromptProposal]
  ): HtmlElement = div(
    cls := "notification is-primary is-light py-2 px-3 mt-2",
    p(cls := "has-text-weight-bold is-size-7 mb-1", "Proposed prompt"),
    promptView(base.map(_.prompt), proposal.prompt, "the working prompt"),
    if (
      proposal.negativePrompt.nonEmpty || base.exists(_.negativePrompt.nonEmpty)
    )
      div(
        p(cls := "has-text-weight-bold is-size-7 mb-1 mt-1", "Negative"),
        promptView(
          base.map(_.negativePrompt),
          proposal.negativePrompt,
          "the working negative prompt"
        )
      )
    else emptyNode,
    button(
      cls := "button is-primary is-small mt-2",
      if (onApplyProposal.isDefined) "Apply to form"
      else "Apply to generation form",
      title :=
        "Fills the prompt and negative prompt of the generation form" +
          (if (onApplyProposal.isDefined) "" else " and opens Inference"),
      onClick --> { _ =>
        onApplyProposal match {
          case Some(apply) => apply(proposal)
          case None        =>
            generationService.requestPromptProposal(proposal)
            Page.Models.navigate()
        }
      }
    )
  )

  /** A proposed text as what it changed (`specs/20`): diffed against the
    * working prompt the request carried, so a detail the model quietly dropped
    * shows before Apply. With no base — an empty form, or the Assistant page
    * with no form on screen — it is the plain text.
    */
  private def promptView(
      before: Option[String],
      after: String,
      beforeLabel: String
  ): HtmlElement =
    before.filter(_.trim.nonEmpty) match {
      case Some(text) => PromptDiff(text, after, beforeLabel).element
      case None       => pre(cls := "assistant-text is-size-7", after)
    }

  /** What the next request starts from, against the context the server applied
    * (`specs/20`). Past three quarters it says what to do; at the top sending
    * is refused, since llama-server would truncate and the reply degrade.
    */

  lazy val element: HtmlElement =
    div(children <-- service.turns.map(_.reverse).split(_.id)(turnElement))
}

object AssistantTurns {

  def roleLabel(role: String): String = role match {
    case "user"    => "You"
    case "summary" => "Summary"
    case _         => "Assistant"
  }

  /** What compactions and restarts moved out of the transcript (`specs/20`),
    * newest first like the chat, folded until asked for.
    */
}
