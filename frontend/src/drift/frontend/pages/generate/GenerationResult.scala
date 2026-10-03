package drift.frontend.pages.generate

import drift.frontend.components.*
import drift.frontend.services.*
import drift.shared.*

import com.raquo.laminar.api.L.*
import com.raquo.laminar.codecs.BooleanAsAttrPresenceCodec

/** The panel's latest generation: queued or running with its elapsed time, the
  * session's progress and a cancel where the server allows one; completed with
  * its outputs — side by side for a batch —, the assistant hand-off and, in
  * free play, Keep; failed with the server's error; or cancelled.
  */
class GenerationResult(
    generation: Generation,
    capabilities: Option[SessionCapabilities],
    sessionSignal: Signal[Option[Session]],
    service: GenerationService,
    assistantService: AssistantService,
    projectService: ProjectService,
    /** The configuration's label as it reads now, for the assistant hand-off.
      */
    configurationLabel: () => String,
    /** Free play (`specs/22-free-play-and-scratch-generations.md`): a result
      * can be kept, and nothing else is.
      */
    scratch: Boolean,
    /** Where a Keep puts the result — "" is the gallery alone. The panel's, so
      * the choice carries over to the next result.
      */
    keepProjectVar: Var[String],
    /** Opens one of this generation's outputs full screen, on pages that have a
      * detail view to open it in — the workspace's.
      */
    onOpenOutput: Option[(Generation, Int) => Unit]
) extends Component {

  // Not in Laminar's default bundle, so defined here once.
  private val controlsAttr = htmlAttr("controls", BooleanAsAttrPresenceCodec)
  private val loopAttr = htmlAttr("loop", BooleanAsAttrPresenceCodec)

  private val tick: Signal[Int] =
    EventStream.periodic(1000).startWith(0)

  /** The seed the request carried — always concrete now, so a result can be
    * reproduced from what is on screen. A batch spans one seed per image.
    */
  private def seedText: String = {
    val count = generation.imageParameters.map(_.batchCount).getOrElse(1)
    generation
      .seedOf(0)
      .orElse(generation.videoParameters.map(_.seed).filter(_ >= 0))
      .map(seed =>
        if (count > 1) s" · seeds $seed–${seed + count - 1}"
        else s" · seed $seed"
      )
      .getOrElse("")
  }

  private def elapsedSeconds(from: Long, until: Option[Long]): Long =
    ((until.getOrElse(System.currentTimeMillis()) - from) / 1000).max(0)

  private def outputElement(output: GenerationOutput, index: Int): HtmlElement =
    if (output.mimeType.startsWith("video/"))
      videoTag(
        VideoRelease.onUnmount,
        controlsAttr := true,
        loopAttr := true,
        src := output.url,
        styleAttr := "max-width: 100%;"
      )
    else
      // Animated WebP included: it renders through img, not video.
      onOpenOutput match {
        case Some(open) =>
          img(
            src := output.url,
            styleAttr := "max-width: 100%;",
            cls := "cursor-pointer",
            title := "Click to open it full screen",
            onClick --> (_ => open(generation, index))
          )
        case None => img(src := output.url, styleAttr := "max-width: 100%;")
      }

  /** Stages the output, scaled down, with its recorded parameters for the
    * assistant (`specs/21-assistant-page.md`). Images only: nothing sends a
    * video to a vision model yet.
    *
    * The assistant is the drawer beside this panel, in a workspace and in the
    * Sandbox alike, so staging the image is the whole action.
    */
  private def askAssistantButton(output: GenerationOutput): Node =
    if (!output.mimeType.startsWith("image/")) emptyNode
    else
      button(
        cls := "button is-small mt-1",
        "🤖 Ask the assistant",
        title := "Send this image and its parameters to the assistant",
        onClick --> { _ =>
          assistantService.attach(
            AssistantService
              .outputAttachment(generation, output, configurationLabel())
          )
        }
      )

  /** Free play keeps nothing by default
    * (`specs/22-free-play-and-scratch-generations.md`): this is the way out —
    * into the gallery alone, or into a project, where the recipe becomes one of
    * its versions like any submission's would.
    */
  private def keepElement: Node =
    if (!scratch) emptyNode
    else if (!generation.scratch)
      p(
        cls := "text-secondary is-size-7 mt-2",
        "✓ Saved — it is in the gallery now."
      )
    else
      div(
        cls := "field has-addons mt-2 mb-0",
        div(
          cls := "control",
          span(cls := "button is-small is-static", "Save into")
        ),
        div(
          cls := "control",
          div(
            cls := "select is-small",
            select(
              onChange.mapToValue --> keepProjectVar,
              option(value := "", "the gallery only"),
              children <-- projectService.projects.map(
                _.map(project =>
                  option(
                    value := project.id,
                    selected <-- keepProjectVar.signal.map(_ == project.id),
                    project.label
                  )
                )
              )
            )
          )
        ),
        div(
          cls := "control",
          button(
            cls := "button is-small is-primary",
            "💾 Save",
            onClick --> (_ =>
              service.push(
                GenerationService.Command.Keep(
                  generation.id,
                  Some(keepProjectVar.now()).filter(_.nonEmpty)
                )
              )
            )
          )
        )
      )

  lazy val element: HtmlElement =
    generation.status match {
      case GenerationStatus.Queued | GenerationStatus.Generating =>
        div(
          cls := "notification is-info is-light py-3 px-4",
          p(
            cls := "has-text-weight-bold mb-1",
            generation.status match {
              case GenerationStatus.Queued if generation.queuePosition > 0 =>
                s"Queued — position ${generation.queuePosition}"
              case GenerationStatus.Queued => "Queued"
              case _                       => "Generating…"
            }
          ),
          p(
            cls := "is-size-7",
            child.text <-- tick.map(_ =>
              s"${elapsedSeconds(generation.submittedAt, None)}s elapsed" +
                seedText
            )
          ),
          // The batch around the image in flight. The session's, so only once
          // this generation is the one it is running.
          if (generation.status == GenerationStatus.Generating)
            BatchProgressView(sessionSignal.map(_.flatMap(_.batch))).element
          else emptyNode,
          // The step count comes from the log, which spec 08 anticipated and
          // spec 13 delivered; without a match the bar stays indeterminate.
          LogProgressView(
            sessionSignal.map(session =>
              (session.flatMap(_.progress), session.flatMap(_.activity))
            )
          ).element,
          GenerationCancel.stopButton(
            service,
            capabilities,
            generation,
            label = "⏹ Cancel",
            classes = "button is-warning is-small mt-2",
            hint = "stop this job"
          )
        )
      case GenerationStatus.Completed =>
        div(
          p(
            cls := "text-secondary is-size-7 mb-2",
            s"Completed in ${elapsedSeconds(
                generation.startedAt.getOrElse(generation.submittedAt),
                generation.completedAt
              )}s" + seedText
          ),
          // A batch lays its images side by side (`specs/14`), so the
          // variations are compared rather than scrolled through.
          div(
            cls := (if (generation.outputs.size > 1) "batch-outputs" else ""),
            generation.outputs.zipWithIndex.map((output, index) =>
              div(
                outputElement(output, index),
                askAssistantButton(output)
              )
            )
          ),
          keepElement
        )
      case GenerationStatus.Failed =>
        div(
          cls := "notification is-danger is-light py-3 px-4",
          p(cls := "has-text-weight-bold mb-1", "Generation failed"),
          pre(
            cls := "is-size-7",
            styleAttr := "white-space: pre-wrap; word-break: break-all; background: transparent; padding: 0;",
            generation.error.getOrElse("no error reported")
          )
        )
      case GenerationStatus.Cancelled =>
        div(
          cls := "notification py-3 px-4",
          p("Generation cancelled")
        )
    }
}
