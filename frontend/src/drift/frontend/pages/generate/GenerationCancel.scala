package drift.frontend.pages.generate

import drift.frontend.services.GenerationService
import drift.shared.*

import com.raquo.laminar.api.L.*
import org.scalajs.dom.window

/** Stopping a job, wherever it is offered — the result under the form and the
  * queue beside it (`specs/08-inference-ui.md`).
  *
  * The rule is the server's own: `features_by_mode` says whether it can drop a
  * queued job and whether it can interrupt one under way. Where it can, the
  * cancel is a request like any other. Where it cannot, stopping still has to
  * be possible — the only way is to take sd-cpp down and bring it back — so the
  * user is told what it costs and decides.
  */
object GenerationCancel {

  /** Whether the server can stop this job where it stands. An absent feature is
    * read as "yes": a build that does not report the flag is the older one that
    * simply cancels.
    */
  def allowed(
      capabilities: Option[SessionCapabilities],
      generation: Generation
  ): Boolean = {
    val features = capabilities
      .map(_.featuresByMode.getOrElse(generation.kind, Map.empty))
      .getOrElse(Map.empty)
    generation.status match {
      case GenerationStatus.Queued => features.getOrElse("cancel_queued", true)
      case GenerationStatus.Generating =>
        features.getOrElse("cancel_generating", true)
      case _ => false
    }
  }

  /** What a restart costs, spelled out: the model reloads, and the server's
    * queue dies with it — everything waiting behind this job goes too, which is
    * the part a user would not otherwise expect.
    */
  val RestartWarning: String =
    "sd-cpp reports it cannot cancel this job.\n\n" +
      "drift can stop it anyway by killing sd-cpp and launching the same " +
      "configuration again — the model reloads, which takes as long as " +
      "starting it did, and every other job still waiting on this session is " +
      "dropped with it.\n\nStop it that way?"

  /** The button that stops a job, wherever it is shown — the result's says "⏹
    * Cancel", a queue row's is just "⏹", and both spin and stop taking clicks
    * while the stop is on its way. Nothing about a cancel is instant: a running
    * job ends at sd-cpp's next step boundary, a forced one once the server has
    * been killed and is loading again.
    */
  def stopButton(
      service: GenerationService,
      capabilities: Option[SessionCapabilities],
      generation: Generation,
      label: String,
      classes: String,
      hint: String
  ): HtmlElement = {
    val pending = service.cancelling.map(_.contains(generation.id))
    button(
      cls := classes,
      cls("is-loading") <-- pending,
      disabled <-- pending,
      title := hint,
      label,
      onClick --> (_ => stop(service, capabilities, generation))
    )
  }

  /** Stops `generation`: outright where the server allows it, else after the
    * user accepts the restart.
    */
  def stop(
      service: GenerationService,
      capabilities: Option[SessionCapabilities],
      generation: Generation
  ): Unit =
    if (allowed(capabilities, generation))
      service.push(GenerationService.Command.Cancel(generation.id))
    // The restart is offered for the job under way and for nothing else:
    // killing the server to remove something that has not started would take
    // the running generation and the rest of the queue with it, which is never
    // the trade a user is asking for. Upstream drops queued jobs on request
    // anyway (`cancel_queued` is true there), so this is the theoretical case.
    else if (generation.status == GenerationStatus.Generating) {
      // Declining the warning is an answer: nothing is sent, and the button
      // does not spin for a stop that was called off.
      if (window.confirm(RestartWarning))
        service.push(
          GenerationService.Command.Cancel(generation.id, force = true)
        )
    } else service.push(GenerationService.Command.Cancel(generation.id))
}
