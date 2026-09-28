package drift.frontend.services

import drift.shared.*

import scala.util.*

import com.raquo.laminar.api.L.*

object RuntimeService {

  enum Command {

    /** Runtimes and the default selection together — every view of one wants
      * the other.
      */
    case Load

    /** Both tools' release listings — the runtime list shows "update available"
      * per runtime, whichever tool it is.
      */
    case LoadReleases
    case LoadTargets
    case LoadInstalls
    case Install(request: InstallRuntimeRequest)

    /** Ask which TheRock build would pair with a ROCm version for a gfx target,
      * so the install form can show the choice (and offer to override it).
      */
    case ResolveTheRock(gfxTarget: String, rocmVersion: String)

    /** Install the newest release of a tool for a backend under
      * `latest-<backend>` (`llama-latest-<backend>`).
      */
    case InstallLatest(request: InstallLatestRequest)

    /** Whether this drift carries a runner to install (`specs/43`). */
    case LoadRunnerOffer

    /** The builds to offer per tool where a launch finds none (`specs/46`). */
    case LoadInstallOptions

    /** Install drift's own runner: both its runtimes, on one TheRock build. */
    case InstallRunner(theRockVersion: Option[String])

    /** Install a build picked where configurations could not launch; the
      * backend moves them to its engine once it validates (`specs/46`).
      */
    case InstallFor(request: InstallForConfigurationsRequest)

    /** Re-resolve the newest release for a latest runtime and install it if
      * newer. The TheRock pick matters only when that release declares a ROCm
      * version other than the paired one.
      */
    case Upgrade(runtimeId: String, theRockVersion: Option[String])

    /** Re-pair a ROCm runtime with another TheRock build, release kept. */
    case ChangeTheRock(runtimeId: String, theRockVersion: String)
    case CancelInstall(runtimeId: String)

    /** The adopt path: create the entity as given, then validate it. */
    case Register(runtime: Runtime)
    case Validate(runtimeId: String)
    case Delete(runtimeId: String)

    /** The default for one tool; the other tool's is untouched. */
    case SetDefault(tool: RuntimeTool, runtimeId: Option[String])
  }
  enum Event {

    /** An install left the active states — the runtime list changed (a failed
      * validation still registers the runtime, marked invalid).
      */
    case InstallSettled(runtimeId: String)

    /** An `InstallFor` was answered: configurations may have moved already (the
      * engine was installed), or the install is now under way.
      */
    case InstallForAnswered(runtimeId: String)
  }

  /** How long an install may go without a word from the socket before the list
    * is asked for over REST instead.
    */
  val InstallPollMillis: Int = 5000

  /** The key a [[TheRockResolution]] is stored under. */
  def theRockKey(gfxTarget: String, rocmVersion: String): String =
    s"$gfxTarget|$rocmVersion"
}

class RuntimeService(statusSocket: StatusSocketService) extends ServiceErrors {
  import RuntimeService.{Command, Event}

  private val listRuntimesFn = ApiClient.stream(drift.shared.listRuntimes)
  private val createRuntimeFn = ApiClient.stream(drift.shared.createRuntime)
  private val deleteRuntimeFn = ApiClient.stream(drift.shared.deleteRuntime)
  private val validateRuntimeFn =
    ApiClient.stream(drift.shared.validateRuntime)
  private val getSelectionFn =
    ApiClient.stream(drift.shared.getRuntimeSelection)
  private val setSelectionFn =
    ApiClient.stream(drift.shared.setRuntimeSelection)
  private val listReleasesFn =
    ApiClient.stream(drift.shared.listRuntimeReleases)
  private val listTargetsFn = ApiClient.stream(drift.shared.listTheRockTargets)
  private val resolveTheRockFn = ApiClient.stream(drift.shared.resolveTheRock)
  private val installFn = ApiClient.stream(drift.shared.installRuntime)
  private val installLatestFn =
    ApiClient.stream(drift.shared.installLatestRuntime)
  private val upgradeFn = ApiClient.stream(drift.shared.upgradeRuntime)
  private val runnerOfferFn = ApiClient.stream(drift.shared.runnerOffer)
  private val installOptionsFn =
    ApiClient.stream(drift.shared.runtimeInstallOptions)
  private val installRunnerFn = ApiClient.stream(drift.shared.installRunner)
  private val installForFn =
    ApiClient.stream(drift.shared.installForConfigurations)

  private val _requested = Var(Set.empty[String])

  /** Runtime ids whose `InstallFor` is sent but not answered: an install is
    * starting before any job says so, and the button shows it at once.
    */
  val requested: Signal[Set[String]] = _requested.signal
  private val changeTheRockFn =
    ApiClient.stream(drift.shared.changeRuntimeTheRock)
  private val listRulesFn = ApiClient.stream(drift.shared.listRuntimeRules)
  private val listInstallsFn =
    ApiClient.stream(drift.shared.listRuntimeInstalls)
  private val cancelInstallFn =
    ApiClient.stream(drift.shared.cancelRuntimeInstall)

  private val _runtimes = Var(List.empty[Runtime])
  private val _runtimesLoaded = Var(false)

  /** Whether the list has been read once: an empty list before that says
    * nothing about what is installed.
    */
  val runtimesLoaded: Signal[Boolean] = _runtimesLoaded.signal
  private val _selection = Var(RuntimeSelection())
  private val _releases = Var(Map.empty[RuntimeTool, List[RuntimeRelease]])
  private val _targets = Var(List.empty[String])
  private val _installs = Var(Map.empty[String, RuntimeInstallJob])
  private val _theRockResolutions = Var(Map.empty[String, TheRockResolution])
  private val _rules = Var(List.empty[RuntimeRule])
  private val _runnerOffer = Var(Option.empty[RunnerOffer])

  /** Whether drift's own runner can be installed; none until asked. */
  val runnerOffer: Signal[Option[RunnerOffer]] = _runnerOffer.signal

  private val _installOptions =
    Var(Map.empty[RuntimeTool, List[RuntimeInstallOption]])

  /** The builds to choose from where a launch finds no runtime of its tool, the
    * recommended one first.
    */
  val installOptions: Signal[Map[RuntimeTool, List[RuntimeInstallOption]]] =
    _installOptions.signal

  val runtimes: Signal[List[Runtime]] = _runtimes.signal
  val selection: Signal[RuntimeSelection] = _selection.signal

  /** What each build wants and refuses (`specs/16-parameter-resolution.md`) —
    * read-only reference data, loaded with the runtimes so the command-line
    * preview resolves exactly what the launcher will.
    */
  val rules: Signal[List[RuntimeRule]] = _rules.signal

  /** The sd-cpp default — what a generation launch uses when nothing pins a
    * runtime.
    */
  val defaultRuntimeId: Signal[Option[String]] =
    _selection.signal.map(_.defaultRuntimeId)

  /** Release listings per tool, newest first; a tool absent from the map has
    * not loaded yet (or failed to).
    */
  val releases: Signal[Map[RuntimeTool, List[RuntimeRelease]]] =
    _releases.signal
  val targets: Signal[List[String]] = _targets.signal
  val installs: Signal[Map[String, RuntimeInstallJob]] = _installs.signal

  /** TheRock pairings, keyed by [[RuntimeService.theRockKey]] (`gfx|rocm`), so
    * the specific-release form and the install-latest shortcut can each look up
    * their own without clobbering each other.
    */
  val theRockResolutions: Signal[Map[String, TheRockResolution]] =
    _theRockResolutions.signal

  private val cmdBus = new EventBus[Command]
  private val evtBus = new EventBus[Event]

  val events: EventStream[Event] = evtBus.events

  def push(command: Command): Unit = cmdBus.writer.onNext(command)

  /** When the socket last said anything about installs — what the fallback poll
    * below reads to tell a quiet socket from a working one.
    */
  private var lastInstallPush: Long = 0L

  /** Replaces the install map with the polled list, emitting `InstallSettled`
    * for every job that left the active states since the previous picture.
    */
  private def applyInstalls(listing: List[RuntimeInstallJob]): Unit = {
    val previous = _installs.now()
    val next = listing.map(job => job.runtimeId -> job).toMap
    _installs.set(next)
    next.values.foreach { job =>
      val wasActive = previous.get(job.runtimeId).forall(_.state.isActive)
      val settled = job.state == RuntimeInstallState.Completed ||
        job.state == RuntimeInstallState.Failed
      if (settled && wasActive)
        evtBus.writer.onNext(Event.InstallSettled(job.runtimeId))
    }
  }

  private def applyJob(job: RuntimeInstallJob): Unit = {
    _installs.update(_ + (job.runtimeId -> job))
    // A terminal answer ("already latest", a refusal) settles nothing the
    // socket would push, so nudge a reload for the runtime it names.
    if (!job.state.isActive) push(Command.Load)
  }

  // See ModelService.effects for why writes merge instead of switching.
  val effects: Modifier[HtmlElement] = Seq(
    cmdBus.events
      .collect { case Command.Load => () }
      .flatMapSwitch(_ =>
        listRuntimesFn(())
          .combineWith(getSelectionFn(()), listRulesFn(()))
          .recoverToTry
      )
      --> Observer[Try[(List[Runtime], RuntimeSelection, List[RuntimeRule])]] {
        case Success((runtimes, selection, rules)) =>
          clearError()
          _runtimes.set(runtimes.sortBy(_.label))
          _runtimesLoaded.set(true)
          _selection.set(selection)
          _rules.set(rules)
        case Failure(err) => reportFailure("Loading runtimes", err)
      },
    cmdBus.events
      .collect { case Command.LoadReleases => () }
      .flatMapSwitch(_ =>
        EventStream.merge(
          RuntimeTool.values.toList
            .map(tool => listReleasesFn(tool).map(tool -> _).recoverToTry)*
        )
      )
      --> Observer[Try[(RuntimeTool, List[RuntimeRelease])]] {
        case Success((tool, releases)) =>
          clearError()
          _releases.update(_ + (tool -> releases))
        case Failure(err) => reportFailure("Listing releases", err)
      },
    cmdBus.events
      .collect { case Command.LoadTargets => () }
      .flatMapSwitch(_ => listTargetsFn(()).recoverToTry)
      --> Observer[Try[List[String]]] {
        case Success(targets) =>
          clearError()
          _targets.set(targets)
        case Failure(err) => reportFailure("Listing TheRock targets", err)
      },
    cmdBus.events
      .collect { case Command.LoadInstalls => () }
      .flatMapSwitch(_ => listInstallsFn(()).recoverToTry)
      --> Observer[Try[List[RuntimeInstallJob]]] {
        case Success(listing) =>
          clearError()
          applyInstalls(listing)
        case Failure(err) => reportFailure("Listing runtime installs", err)
      },
    cmdBus.events
      .collect { case Command.Install(request) => request }
      .flatMapMerge(request => installFn(request).recoverToTry)
      --> Observer[Try[RuntimeInstallJob]] {
        case Success(job) =>
          clearError()
          applyJob(job)
        case Failure(err) => reportFailure("Starting the install", err)
      },
    cmdBus.events
      .collect { case Command.ResolveTheRock(gfx, rocm) => (gfx, rocm) }
      .flatMapMerge((gfx, rocm) => resolveTheRockFn((gfx, rocm)).recoverToTry)
      --> Observer[Try[TheRockResolution]] {
        case Success(resolution) =>
          clearError()
          _theRockResolutions.update(
            _ + (RuntimeService.theRockKey(
              resolution.gfxTarget,
              resolution.neededVersion
            ) -> resolution)
          )
        case Failure(err) => reportFailure("Resolving the ROCm build", err)
      },
    cmdBus.events
      .collect { case Command.LoadInstallOptions => () }
      .flatMapSwitch(_ => installOptionsFn(()).recoverToTry)
      --> Observer[Try[List[RuntimeInstallOption]]] {
        case Success(options) =>
          clearError()
          _installOptions.set(
            options
              .groupBy(_.tool)
              .view
              .mapValues(_.sortBy(!_.recommended))
              .toMap
          )
        case Failure(err) =>
          reportFailure("Reading the runtimes to install", err)
      },
    cmdBus.events
      .collect { case Command.LoadRunnerOffer => () }
      .flatMapMerge(_ => runnerOfferFn(()).recoverToTry)
      --> Observer[Try[RunnerOffer]] {
        case Success(offer) =>
          clearError()
          _runnerOffer.set(Some(offer))
        case Failure(err) => reportFailure("Asking about the runner", err)
      },
    cmdBus.events
      .collect { case Command.InstallRunner(theRock) => theRock }
      .flatMapMerge(theRock =>
        installRunnerFn(InstallRunnerRequest(theRock)).recoverToTry
      )
      --> Observer[Try[List[RuntimeInstallJob]]] {
        case Success(jobs) =>
          clearError()
          jobs.foreach(applyJob)
        case Failure(err) => reportFailure("Installing the runner", err)
      },
    cmdBus.events
      .collect { case Command.InstallFor(request) => request }
      .flatMapMerge { request =>
        val id = request.option.runtimeId
        _requested.update(_ + id)
        installForFn(request).recoverToTry.map(result => (id, result))
      } --> Observer[(String, Try[List[RuntimeInstallJob]])] { (id, result) =>
      _requested.update(_ - id)
      result match {
        case Success(jobs) =>
          clearError()
          jobs.foreach(applyJob)
          push(Command.Load)
          evtBus.writer.onNext(Event.InstallForAnswered(id))
        case Failure(err) => reportFailure("Installing the runtime", err)
      }
    },
    cmdBus.events
      .collect { case Command.InstallLatest(request) => request }
      .flatMapMerge(request => installLatestFn(request).recoverToTry)
      --> Observer[Try[RuntimeInstallJob]] {
        case Success(job) =>
          clearError()
          applyJob(job)
        case Failure(err) => reportFailure("Installing the latest runtime", err)
      },
    cmdBus.events
      .collect { case Command.Upgrade(id, theRock) => (id, theRock) }
      .flatMapMerge((id, theRock) =>
        upgradeFn((id, UpgradeRuntimeRequest(theRock))).recoverToTry
      )
      --> Observer[Try[RuntimeInstallJob]] {
        case Success(job) =>
          clearError()
          applyJob(job)
        case Failure(err) => reportFailure("Upgrading the runtime", err)
      },
    cmdBus.events
      .collect { case Command.ChangeTheRock(id, version) => (id, version) }
      .flatMapMerge((id, version) =>
        changeTheRockFn((id, ChangeTheRockRequest(version))).recoverToTry
      )
      --> Observer[Try[RuntimeInstallJob]] {
        case Success(job) =>
          clearError()
          applyJob(job)
        case Failure(err) => reportFailure("Changing the ROCm build", err)
      },
    cmdBus.events
      .collect { case Command.CancelInstall(id) => id }
      .flatMapMerge(id => cancelInstallFn(id).recoverToTry)
      --> Observer[Try[RuntimeInstallJob]] {
        case Success(job) =>
          clearError()
          _installs.update(_ + (job.runtimeId -> job))
        case Failure(err) => reportFailure("Cancelling the install", err)
      },
    cmdBus.events
      .collect { case Command.Register(runtime) => runtime }
      .flatMapMerge(runtime => createRuntimeFn(runtime).recoverToTry)
      --> Observer[Try[Runtime]] {
        case Success(runtime) =>
          clearError()
          _runtimes.update(_.filterNot(_.id == runtime.id) :+ runtime)
          push(Command.Validate(runtime.id))
        case Failure(err) => reportFailure("Registering the runtime", err)
      },
    cmdBus.events
      .collect { case Command.Validate(id) => id }
      .flatMapMerge(id => validateRuntimeFn(id).map((id, _)).recoverToTry)
      --> Observer[Try[(String, Option[Runtime])]] {
        case Success((_, Some(_))) =>
          clearError()
          // Validation may also have made this the first default; reload the
          // whole picture rather than patching it together.
          push(Command.Load)
        case Success((id, None)) =>
          reportFailure("Validating the runtime", s"'$id' no longer exists.")
        case Failure(err) => reportFailure("Validating the runtime", err)
      },
    cmdBus.events
      .collect { case Command.Delete(id) => id }
      .flatMapMerge(id =>
        deleteRuntimeFn(id).map(deleted => (id, deleted)).recoverToTry
      )
      --> Observer[Try[(String, Boolean)]] {
        case Success((id, true)) =>
          clearError()
          _runtimes.update(_.filterNot(_.id == id))
          _selection.update(current =>
            RuntimeTool.values.foldLeft(current)((sel, tool) =>
              if (sel.defaultFor(tool).contains(id)) sel.withDefault(tool, None)
              else sel
            )
          )
        case Success((id, false)) =>
          reportFailure(
            "Deleting the runtime",
            s"the server refused to delete '$id'."
          )
        case Failure(err) => reportFailure("Deleting the runtime", err)
      },
    cmdBus.events
      .collect { case Command.SetDefault(tool, id) => (tool, id) }
      .flatMapMerge((tool, id) =>
        setSelectionFn(_selection.now().withDefault(tool, id)).recoverToTry
      )
      --> Observer[Try[RuntimeSelection]] {
        case Success(selection) =>
          clearError()
          _selection.set(selection)
        case Failure(err) => reportFailure("Setting the default runtime", err)
      },
    // The socket pushes the install list whenever it changes, so progress
    // moves without the user touching anything — through `applyInstalls`, so
    // `InstallSettled` still fires.
    statusSocket.runtimeInstalls --> Observer[List[RuntimeInstallJob]] {
      listing =>
        lastInstallPush = System.currentTimeMillis()
        applyInstalls(listing)
    },
    // The fallback: while an install is running and the socket has said
    // nothing about installs for as long as this poll's interval, the list is
    // asked for over REST. An install moves only by push, so a socket that is
    // not connected — a dev server not forwarding the upgrade, a connection
    // dropped while the tab was away — froze the row on the state its POST
    // answered with and left the runtime list unaware it had finished, until
    // the page was reloaded (François, 2026-09-20). A pushing socket keeps
    // `lastInstallPush` fresh, so nothing is polled in the normal case.
    EventStream
      .periodic(RuntimeService.InstallPollMillis)
      .withCurrentValueOf(installs)
      .filter((_, jobs) =>
        jobs.values.exists(_.state.isActive) &&
          System.currentTimeMillis() - lastInstallPush >=
          RuntimeService.InstallPollMillis
      )
      .map(_ => Command.LoadInstalls) --> Observer[Command](push),
    // A settled install changed the runtime list (even a failed validation
    // registers the runtime, marked invalid).
    events --> Observer { case Event.InstallSettled(_) =>
      push(Command.Load)
    }
  )
}
