package drift.frontend.services

import drift.shared.*

import com.raquo.laminar.api.L.*

object LaunchPrerequisites {

  /** How a configuration's missing weights stand: `idle` are the models no
    * download has started for, `underway` counts the ones queued or
    * downloading.
    */
  case class PendingWeights(idle: List[String], underway: Int)

  /** What a tool's runtimes are for, as the notices say it — the use, not the
    * technology (François, 2026-09-28).
    */
  def useOf(architecture: Architecture): String =
    if (architecture.tool == RuntimeTool.LlamaCpp) "text"
    else if (
      architecture.tags.contains(ArchitectureTags.Video) &&
      !architecture.tags.contains(ArchitectureTags.Image)
    ) "video"
    else "image"

  /** "a video runtime", "an image runtime". */
  def aRuntime(use: String): String =
    (if ("aeiou".contains(use.head)) "an " else "a ") + s"$use runtime"

  /** The order builds are offered in, the first preselected: drift's runner
    * where it can run the model (François, 2026-09-28), then the recommended
    * upstream build, then the rest.
    */
  def preference(option: RuntimeInstallOption): (Boolean, Boolean) =
    (option.engine != RuntimeEngine.DriftRunner, !option.recommended)

  /** Configurations of one tool whose engine has no valid runtime. The builds
    * to choose from — those of their architectures' engines, recommended first
    * — are empty until read; one on another engine switches every configuration
    * here whose architecture runs on it.
    */
  case class RuntimeNeed(
      tool: RuntimeTool,
      /** What its models make, as the notice names the runtime: "image",
        * "video", "text" — "image or video" for a mix.
        */
      use: String,
      /** Each with the engines its architecture runs on. */
      configurations: List[(RunConfiguration, List[RuntimeEngine])],
      options: List[RuntimeInstallOption],
      /** Engines of the tool with a valid runtime already: choosing one of
        * those only switches the configurations.
        */
      installed: Set[RuntimeEngine],
      installing: Boolean,
      /** Why the last install of one of these builds failed, while none runs.
        */
      failure: Option[String]
  ) {

    /** Whether choosing `option` moves any configuration to another engine. */
    def switches(option: RuntimeInstallOption): Boolean =
      configurations.exists((configuration, runs) =>
        configuration.runner != option.engine && runs.contains(option.engine)
      )
  }
  object RuntimeNeed {

    /** One need for all of a picker's configurations: every build any of them
      * can take, recommended first.
      */
    def merge(needs: List[RuntimeNeed]): Option[RuntimeNeed] =
      needs
        .reduceOption((a, b) =>
          a.copy(
            configurations = a.configurations ++ b.configurations,
            options = (a.options ++ b.options).distinctBy(_.runtimeId),
            use =
              if (a.use.split(" or ").contains(b.use)) a.use
              else s"${a.use} or ${b.use}",
            installing = a.installing || b.installing,
            failure = a.failure.orElse(b.failure)
          )
        )
        .map(need => need.copy(options = need.options.sortBy(preference)))
  }

  /** What a configuration still lacks before it can launch; at least one of the
    * two is set.
    */
  case class Missing(
      runtime: Option[RuntimeNeed],
      weights: Option[PendingWeights]
  )

  /** What `missing` comes to, in the few words a picker entry holds: "install
    * an image runtime, download the weights".
    */
  def describe(missing: Missing): String =
    List(
      missing.runtime.map(need =>
        (if (need.installing) "installing " else "install ") +
          aRuntime(need.use)
      ),
      missing.weights.map(pending =>
        if (pending.idle.isEmpty) "downloading the weights"
        else "download the weights"
      )
    ).flatten.mkString(", ")

  /** Where each of `modelIds` stands against the download jobs; a completed job
    * counts as neither, as the cache is only re-read after it.
    */
  def pending(
      modelIds: List[String],
      jobs: Map[String, DownloadJob]
  ): PendingWeights = {
    val states = modelIds.map(id => id -> jobs.get(id).map(_.state))
    PendingWeights(
      states.collect {
        case (id, None) => id
        case (id, Some(state))
            if !state.isActive && state != DownloadState.Completed =>
          id
      },
      states.count(_._2.exists(_.isActive))
    )
  }
}

/** What run configurations lack before they can launch — a runtime of their
  * tool, downloaded weights — and the actions that supply it
  * (`specs/46-starter-configurations.md`): every place that starts a model
  * offers them in place of the launch.
  *
  * Its `effects` mount the cache, download and runtime services, so a page that
  * already mounts those reads the signals without mounting this.
  */
class LaunchPrerequisites(
    runConfigurationService: RunConfigurationService,
    cacheService: CacheService,
    downloadService: DownloadService,
    runtimeService: RuntimeService
) {
  import LaunchPrerequisites.*

  /** Configuration id → the models whose weights it lacks. Empty until the
    * cache has been read: every model has a status row, so none means "not
    * known yet" rather than "nothing on disk".
    */
  private val lacking: Signal[Map[String, List[String]]] =
    runConfigurationService.runConfigurations
      .combineWith(
        runConfigurationService.architectures,
        runConfigurationService.allModels,
        cacheService.statuses
      )
      .map { (configurations, architectures, models, cache) =>
        if (cache.isEmpty) Map.empty
        else
          configurations.flatMap { configuration =>
            val modelIds = CommandLine
              .blockers(configuration, architectures, models, cache)
              .collect { case LaunchBlocker.WeightsNotCached(_, modelId, _) =>
                modelId
              }
              .distinct
            Option.when(modelIds.nonEmpty)(configuration.id -> modelIds)
          }.toMap
      }
      .distinct

  private val weights: Signal[Map[String, PendingWeights]] =
    lacking
      .combineWith(downloadService.jobs)
      .map((lacking, jobs) => lacking.view.mapValues(pending(_, jobs)).toMap)

  /** Configuration id → the runtime it lacks: none valid of its tool and
    * engine. Only the upstream engines are offered; drift's own runner is
    * installed from Settings.
    */
  private val runtimes: Signal[Map[String, RuntimeNeed]] =
    runConfigurationService.runConfigurations
      .combineWith(
        runConfigurationService.architectures,
        runtimeService.runtimes,
        runtimeService.runtimesLoaded,
        runtimeService.installOptions,
        runtimeService.installs,
        runtimeService.requested
      )
      .map {
        (
            configurations,
            architectures,
            installed,
            loaded,
            offers,
            jobs,
            requested
        ) =>
          if (!loaded) Map.empty
          else
            configurations.flatMap { configuration =>
              architectures
                .find(_.id == configuration.architectureId)
                .flatMap { architecture =>
                  val tool = architecture.tool
                  val valid = installed
                    .filter(r => r.valid && r.tool == tool)
                    .map(_.engine)
                    .toSet
                  Option.when(!valid.contains(configuration.runner)) {
                    val options = offers
                      .getOrElse(tool, Nil)
                      .filter(o => architecture.runners.contains(o.engine))
                      .sortBy(preference)
                    val settled = options.flatMap(o => jobs.get(o.runtimeId))
                    val installing = options.exists(o =>
                      requested.contains(o.runtimeId) ||
                        jobs.get(o.runtimeId).exists(_.state.isActive)
                    )
                    configuration.id -> RuntimeNeed(
                      tool,
                      useOf(architecture),
                      List(configuration -> architecture.runners),
                      options,
                      valid,
                      installing,
                      Option
                        .when(!installing)(
                          settled
                            .filter(_.state == RuntimeInstallState.Failed)
                            .sortBy(-_.completedAt.getOrElse(0L))
                            .headOption
                            .map(job => job.error.getOrElse("it failed"))
                        )
                        .flatten
                    )
                  }
                }
            }.toMap
      }

  /** Configuration id → what it lacks, for the configurations lacking anything.
    * Changes only when a download or an install starts or ends, never with its
    * progress, so a select built from it stays open.
    */
  val byConfiguration: Signal[Map[String, Missing]] =
    runtimes
      .combineWith(weights)
      .map((runtimes, weights) =>
        (runtimes.keySet ++ weights.keySet)
          .map(id => id -> Missing(runtimes.get(id), weights.get(id)))
          .toMap
      )
      .distinct

  def of(configurationId: Signal[String]): Signal[Option[Missing]] =
    byConfiguration.combineWith(configurationId).map(_.get(_)).distinct

  def download(modelIds: List[String]): Unit =
    modelIds.foreach(id =>
      downloadService.push(DownloadService.Command.Start(id))
    )

  /** Installs the build the user chose — unless its engine already has a valid
    * runtime — and has the backend switch each configuration whose architecture
    * runs on that engine to it, once it validates (`specs/43`: a configuration
    * names its engine). The first valid runtime of a tool becomes its default.
    */
  def choose(need: RuntimeNeed, option: RuntimeInstallOption): Unit =
    runtimeService.push(
      RuntimeService.Command.InstallFor(
        InstallForConfigurationsRequest(
          option,
          need.configurations.collect {
            case (configuration, runs) if runs.contains(option.engine) =>
              configuration.id
          }
        )
      )
    )

  /** Configurations move on the backend once an install they asked for
    * validates: re-read them when an install settles or is answered. Part of
    * `effects`; a page mounting the services itself mounts this alone.
    */
  val followInstalls: Modifier[HtmlElement] =
    runtimeService.events --> Observer[RuntimeService.Event] { _ =>
      runConfigurationService.push(RunConfigurationService.Command.Load)
    }

  val effects: Modifier[HtmlElement] = Seq(
    followInstalls,
    cacheService.effects,
    downloadService.effects,
    runtimeService.effects,
    downloadService.events --> Observer[DownloadService.Event] {
      case DownloadService.Event.Finished(_) =>
        cacheService.push(CacheService.Command.Load)
    },
    onMountCallback { _ =>
      runConfigurationService.modelService.push(ModelService.Command.Load)
      cacheService.push(CacheService.Command.Load)
      downloadService.push(DownloadService.Command.Load)
      runtimeService.push(RuntimeService.Command.Load)
      runtimeService.push(RuntimeService.Command.LoadInstalls)
      runtimeService.push(RuntimeService.Command.LoadInstallOptions)
      runtimeService.push(RuntimeService.Command.LoadRunnerOffer)
    }
  )
}
