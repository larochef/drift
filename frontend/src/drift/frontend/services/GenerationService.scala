package drift.frontend.services

import drift.shared.*

import scala.scalajs.js.timers
import scala.util.*

import com.raquo.laminar.api.L.*

object GenerationService {

  /** How long a stop may be pending before its button is offered again — long
    * enough for a queued job to leave and for a restart to take the server
    * down, short enough not to strand the button.
    */
  val CancelPendingMillis: Int = 30000

  enum Command {
    case LoadCapabilities(sessionId: String)
    case LoadGenerations(sessionId: String)
    case SubmitImage(
        sessionId: String,
        parameters: ImageGenerationParameters,
        context: SubmitContext = SubmitContext(),
        /** Free play (`specs/22-…`): the result is not kept unless asked for.
          */
        scratch: Boolean = false
    )
    case SubmitVideo(
        sessionId: String,
        parameters: VideoGenerationParameters,
        context: SubmitContext = SubmitContext(),
        scratch: Boolean = false
    )

    /** Stops a generation. `force` kills and relaunches the session when the
      * model cannot be interrupted any other way.
      */
    case Cancel(generationId: String, force: Boolean = false)

    /** Promotes a free-play result: into the gallery, and into a project when
      * one is named.
      */
    case Keep(generationId: String, projectId: Option[String])

    /** Throws away every free-play result of this session. */
    case ClearScratch
  }

  /** What the gallery hands the next generation panel to seed itself with
    * (`specs/12-gallery.md`).
    */
  enum Reuse {

    /** The recorded request field for field — honoured only by a panel of the
      * very configuration it was recorded with; elsewhere it becomes `Task`.
      */
    case Full(generation: Generation)

    /** The recorded *task* on whatever configuration is live, for comparing
      * models on the same job: prompt, negative prompt, input images, seed,
      * size, frames, and the sampling settings the form shows. Whatever the
      * form does not show (SLG, flow shift, …) is model-specific and stays the
      * live configuration's own default.
      */
    case Task(generation: Generation)
  }
}

/** Generation against a live session (`specs/08-inference-ui.md`).
  *
  * The backend owns the native job behind each generation and polls the session
  * port itself; the status socket pushes that mirror whenever it changes, so
  * queue position, progress and completion arrive without the browser ever
  * touching the session port.
  */
class GenerationService(statusSocket: StatusSocketService)
    extends ServiceErrors {
  import GenerationService.Command

  private val capabilitiesFn = ApiClient.stream(getSessionCapabilities)
  private val listFn = ApiClient.stream(drift.shared.listGenerations)
  private val submitImageFn = ApiClient.stream(submitImageGeneration)
  private val submitVideoFn = ApiClient.stream(submitVideoGeneration)
  private val cancelFn = ApiClient.stream(cancelGeneration)
  private val keepFn = ApiClient.stream(keepScratchGeneration)
  private val clearScratchFn = ApiClient.stream(drift.shared.clearScratch)

  /** Keyed by session id. A session that is not ready yet simply has no entry;
    * the panel keeps asking until one appears.
    */
  private val _capabilities = Var(Map.empty[String, SessionCapabilities])
  val capabilities: Signal[Map[String, SessionCapabilities]] =
    _capabilities.signal

  /** The current value, for event handlers — `Signal.now()` is not public. */
  def capabilitiesOf(sessionId: String): Option[SessionCapabilities] =
    _capabilities.now().get(sessionId)

  /** The loaded session's generations, oldest first. */
  private val _generations = Var(List.empty[Generation])
  val generations: Signal[List[Generation]] = _generations.signal

  /** Which session `_generations` belongs to — socket pushes for any other
    * session are not this panel's to display.
    */
  private val watchedSessionId = Var(Option.empty[String])

  /** The reuse the gallery requested, held until a panel seeds its form — the
    * panel is rebuilt per navigation, so the request outlives it here.
    */
  private val pendingReuse = Var(Option.empty[GenerationService.Reuse])

  def requestReuse(reuse: GenerationService.Reuse): Unit =
    pendingReuse.set(Some(reuse))

  /** Empties the Sandbox's results (`specs/47-sandbox.md`) on a subscription of
    * its own: the Sandbox calls it as it unmounts, when the `effects` that
    * serve `Command.ClearScratch` may already be torn down with it.
    */
  def clearScratch(): Unit =
    clearScratchFn(()).recoverToTry.foreach {
      case Success(cleared) =>
        _generations.update(_.filterNot(g => cleared.contains(g.id)))
      case Failure(err) => reportFailure("Clearing the Sandbox's results", err)
    }(using unsafeWindowOwner)

  /** Hands the pending reuse over exactly once. */
  def takePendingReuse(): Option[GenerationService.Reuse] = {
    val reuse = pendingReuse.now()
    pendingReuse.set(None)
    reuse
  }

  /** A prompt the assistant proposed (`specs/21-assistant-page.md`), held until
    * the next panel seeds its form — the same hand-off as reuse.
    */
  private val pendingPromptProposal =
    Var(Option.empty[PromptProposal])

  def requestPromptProposal(
      proposal: PromptProposal
  ): Unit = pendingPromptProposal.set(Some(proposal))

  def takePendingPromptProposal(): Option[PromptProposal] = {
    val proposal = pendingPromptProposal.now()
    pendingPromptProposal.set(None)
    proposal
  }

  private val cmdBus = new EventBus[Command]

  /** The generations whose stop has been asked for and has not landed yet. A
    * cancel is never instant — a queued job goes at once, a running one when
    * sd-cpp reaches a step boundary, a forced one when the server has been
    * killed and is coming back — so the button that asked for it says so and
    * stops taking clicks. Held here rather than in the button: the panel
    * rebuilds its rows on every status tick, which would clear anything the
    * element owned.
    */
  private val _cancelling = Var(Set.empty[String])
  val cancelling: Signal[Set[String]] = _cancelling.signal

  def push(command: Command): Unit = {
    command match {
      case Command.Cancel(generationId, _) =>
        _cancelling.update(_ + generationId)
        // A stop the server quietly refuses would otherwise spin until the job
        // ended of its own accord: after this, the button is offered again.
        timers.setTimeout(GenerationService.CancelPendingMillis)(
          _cancelling.update(_ - generationId)
        )
        ()
      case _ => ()
    }
    cmdBus.writer.onNext(command)
  }

  private def upsert(generation: Generation): Unit = {
    settle(List(generation))
    _generations.update(all =>
      (all.filterNot(_.id == generation.id) :+ generation)
        .sortBy(_.submittedAt)
    )
  }

  /** Forgets the pending cancels this listing answers: a generation it carries
    * as no longer active has stopped. Only what the listing names is judged —
    * `upsert` passes a single generation, and the others are still pending.
    */
  private def settle(listing: List[Generation]): Unit =
    _cancelling.update(pending =>
      if (pending.isEmpty) pending
      else
        pending.filterNot(id =>
          listing.exists(generation =>
            generation.id == id && !generation.status.isActive
          )
        )
    )

  // Declared before `effects`, which captures it — a val defined later would
  // still be null when the binder Seq is built.
  private val submissionObserver = Observer[Try[Generation]] {
    case Success(generation) =>
      clearError()
      upsert(generation)
    case Failure(err) => reportFailure("Submitting the generation", err)
  }

  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.LoadCapabilities(sessionId) => sessionId }
      .flatMapSwitch(sessionId =>
        capabilitiesFn(sessionId).map((sessionId, _)).recoverToTry
      ) --> Observer[Try[(String, Option[SessionCapabilities])]] {
      case Success((sessionId, Some(caps))) =>
        clearError()
        _capabilities.update(_ + (sessionId -> caps))
      // None means "not answering yet" — the panel retries, so not an error.
      case Success((_, None)) => ()
      case Failure(err) => reportFailure("Fetching model capabilities", err)
    },
    cmdBus.events
      .collect { case Command.LoadGenerations(sessionId) => sessionId }
      .flatMapSwitch { sessionId =>
        watchedSessionId.set(Some(sessionId))
        listFn(sessionId).recoverToTry
      } --> Observer[Try[List[Generation]]] {
      case Success(listing) =>
        clearError()
        settle(listing)
        _generations.set(listing.sortBy(_.submittedAt))
      case Failure(err) => reportFailure("Listing generations", err)
    },
    cmdBus.events
      .collect {
        case Command.SubmitImage(sessionId, parameters, context, scratch) =>
          (sessionId, context, Some(scratch), parameters)
      }
      .flatMapMerge(submitImageFn(_).recoverToTry) --> submissionObserver,
    cmdBus.events
      .collect {
        case Command.SubmitVideo(sessionId, parameters, context, scratch) =>
          (sessionId, context, Some(scratch), parameters)
      }
      .flatMapMerge(submitVideoFn(_).recoverToTry) --> submissionObserver,
    cmdBus.events
      .collect { case Command.Keep(generationId, projectId) =>
        (generationId, KeepRequest(projectId))
      }
      .flatMapMerge(keepFn(_).recoverToTry) --> Observer[
      Try[Option[Generation]]
    ] {
      case Success(Some(generation)) =>
        clearError()
        upsert(generation)
      case Success(None) =>
        reportFailure(
          "Keeping the result",
          "it is no longer a free-play result — already kept, still running, " +
            "or its session is gone."
        )
      case Failure(err) => reportFailure("Keeping the result", err)
    },
    cmdBus.events
      .collect { case Command.ClearScratch => () }
      .flatMapMerge(_ => clearScratchFn(()).recoverToTry) --> Observer[
      Try[List[String]]
    ] {
      case Success(cleared) =>
        clearError()
        _generations.update(_.filterNot(g => cleared.contains(g.id)))
      case Failure(err) => reportFailure("Clearing the free-play results", err)
    },
    cmdBus.events
      .collect { case Command.Cancel(generationId, force) =>
        (generationId, force)
      }
      .flatMapMerge(input =>
        cancelFn(input).map(result => (input._1, result)).recoverToTry
      ) --> Observer[Try[(String, Option[Generation])]] {
      case Success((_, Some(generation))) =>
        clearError()
        upsert(generation)
      case Success((id, None)) =>
        _cancelling.update(_ - id)
        reportFailure("Cancelling the generation", s"'$id' is unknown.")
      case Failure(err) =>
        // The stop never reached the backend: the button comes back rather
        // than spinning for a request that is not happening.
        _cancelling.set(Set.empty)
        reportFailure("Cancelling the generation", err)
    },
    // The socket pushes a session's list whenever any of its generations
    // change, so queue position and the finished result arrive on their own.
    statusSocket.generations
      .withCurrentValueOf(watchedSessionId.signal)
      .collect {
        case (bySession, Some(sessionId)) if bySession.contains(sessionId) =>
          bySession(sessionId)
      } --> Observer[List[Generation]] { listing =>
      settle(listing)
      _generations.set(listing.sortBy(_.submittedAt))
    }
  )
}
