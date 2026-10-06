package drift.shared

/** Why a run configuration cannot be launched yet. */
enum LaunchBlocker derives CanEqual {

  /** The architecture the configuration names no longer exists. */
  case UnknownArchitecture(architectureId: String)

  /** A required checkpoint slot has nothing assigned. */
  case UnassignedCheckpoint(checkpointName: String, familyId: String)

  /** A slot names a model id that is no longer registered. */
  case MissingModel(checkpointName: String, modelId: String)

  /** The model is registered but its weights are not on disk. Distinct from the
    * others because it is the one the user can fix by downloading rather than
    * by editing the configuration.
    */
  case WeightsNotCached(
      checkpointName: String,
      modelId: String,
      bytes: Option[Long]
  )

  /** A `Local` model whose path does not exist. Nothing to download. */
  case WeightsUnreadable(checkpointName: String, path: String)

  def message: String = this match {
    case UnknownArchitecture(id) =>
      s"architecture '$id' no longer exists"
    case UnassignedCheckpoint(name, familyId) =>
      s"'$name' has no model assigned (needs one from family '$familyId')"
    case MissingModel(name, modelId) =>
      s"'$name' is assigned model '$modelId', which is no longer registered"
    case WeightsNotCached(name, modelId, bytes) =>
      val size =
        bytes.map(b => s" (${LaunchBlocker.humanBytes(b)})").getOrElse("")
      s"'$name' needs model '$modelId' downloaded$size"
    case WeightsUnreadable(name, path) =>
      s"'$name' points at '$path', which does not exist"
  }

  /** True when downloading would clear this blocker. */
  def isDownloadable: Boolean = this match {
    case WeightsNotCached(_, _, _) => true
    case _                         => false
  }
}

object LaunchBlocker {
  def humanBytes(bytes: Long): String =
    if (bytes < 1024L) s"$bytes B"
    else if (bytes < 1024L * 1024) f"${bytes / 1024.0}%.1f KB"
    else if (bytes < 1024L * 1024 * 1024) f"${bytes / (1024.0 * 1024)}%.1f MB"
    else f"${bytes / (1024.0 * 1024 * 1024)}%.1f GB"
}

/** Assembling the sd-cpp command line from a run configuration.
  *
  * This lives in `shared` on purpose: the frontend renders it as a preview and
  * the backend will spawn it (see `specs/07-launch-and-supervision.md`). Two
  * implementations would be two chances to disagree about what actually runs.
  */
object CommandLine {

  /** The architecture's slots that exist on the configuration's runner. */
  private def slots(
      configuration: RunConfiguration,
      architecture: Architecture
  ): List[CheckpointRef] =
    architecture.checkpoints.filter(_.appliesTo(configuration.runner))

  /** Everything standing between a configuration and a launch, in checkpoint
    * order. Empty means ready.
    *
    * `cache` carries whether each model's weights are on disk, keyed by model
    * id. This module cannot look at a filesystem — it is cross-compiled to JS —
    * so the caller supplies it: the backend from `ModelCache`, the frontend
    * from `/api/cache/status`. An empty map means "do not check", which is what
    * the command-line preview wants.
    */
  def blockers(
      configuration: RunConfiguration,
      architectures: List[Architecture],
      models: List[Model],
      cache: Map[String, ModelCacheStatus] = Map.empty
  ): List[LaunchBlocker] =
    architectures.find(_.id == configuration.architectureId) match {
      case None =>
        List(LaunchBlocker.UnknownArchitecture(configuration.architectureId))
      case Some(architecture) =>
        slots(configuration, architecture).flatMap { checkpoint =>
          configuration.assignments.get(checkpoint.name) match {
            case None if checkpoint.required =>
              Some(
                LaunchBlocker
                  .UnassignedCheckpoint(checkpoint.name, checkpoint.familyId)
              )
            case None          => None
            case Some(modelId) =>
              models.find(_.id == modelId) match {
                case None =>
                  Some(LaunchBlocker.MissingModel(checkpoint.name, modelId))
                case Some(model) =>
                  cache.get(modelId).map(_.state) match {
                    case Some(CacheState.Missing) =>
                      Some(
                        LaunchBlocker.WeightsNotCached(
                          checkpoint.name,
                          modelId,
                          cache.get(modelId).flatMap(_.bytes)
                        )
                      )
                    case Some(CacheState.Broken) =>
                      Some(
                        LaunchBlocker.WeightsUnreadable(
                          checkpoint.name,
                          model.source match {
                            case Local(p) => p
                            case other    => other.toString
                          }
                        )
                      )
                    case _ => None
                  }
              }
          }
        }
    }

  /** The parameters actually passed, and why
    * (`specs/16-parameter-resolution.md`). Four layers, each overriding the one
    * before it:
    *
    *   1. the architecture's defaults — true for every model of this shape;
    *   2. the assigned models' own parameters, in checkpoint order — what THIS
    *      checkpoint wants (Krea 2 Raw at 52 steps, Turbo at 4);
    *   3. the runtime's defaults — what THIS build wants;
    *   4. the run configuration's overrides — the user's explicit choice, which
    *      beats every suggestion below it.
    *
    * A layer does not only override values: a model and a run configuration can
    * each *remove* a flag (`removedParameters`), which is how a checkpoint that
    * cannot take one of its architecture's defaults says so. A removal
    * overrides like a value does, so a layer above can set the flag again.
    *
    * Then the runtime's vetoes are applied on top of all four, because a flag
    * the build cannot take is not a preference. Every veto, every removal of a
    * flag something below had set, and every flag the runtime added by itself
    * becomes a note: nothing disappears from a command line without the UI
    * being able to say who dropped it and why.
    *
    * `rule` is the entry matching the runtime that will actually be launched,
    * `None` when no runtime is known yet — the resolution then has three layers
    * and no notes.
    */
  def resolveParameters(
      configuration: RunConfiguration,
      architecture: Architecture,
      models: List[Model],
      rule: Option[RuntimeRule] = None
  ): ParameterResolution = {
    val (sources, notes) =
      resolveSources(configuration, architecture, models, rule)
    ParameterResolution(
      arguments = sources.map(parameter => parameter.flag -> parameter.value),
      notes = notes
    )
  }

  /** What the layers *below* a configuration's own overrides come to: the same
    * fold, with `overriddenParameters` and `removedParameters` emptied.
    *
    * This is what a parameter form has to show — the flags already set, the
    * value each would pass, and the layer that asked for it — and it goes
    * through `resolveSources` so that the form and the launch cannot disagree.
    * `rule` is usually `None` here: a form does not know which runtime will be
    * picked, so the runtime layer is left out rather than guessed.
    */
  def inheritedParameters(
      configuration: RunConfiguration,
      architecture: Architecture,
      models: List[Model],
      rule: Option[RuntimeRule] = None
  ): List[ResolvedParameter] =
    resolveSources(
      configuration.copy(
        overriddenParameters = Map.empty,
        removedParameters = List.empty
      ),
      architecture,
      models,
      rule
    )._1

  /** The layers folded and the vetoes applied: every flag that survives, with
    * the layer that asked for it, and every note about the ones that did not.
    * [[resolveParameters]] is this without the layers.
    */
  private def resolveSources(
      configuration: RunConfiguration,
      architecture: Architecture,
      models: List[Model],
      rule: Option[RuntimeRule]
  ): (List[ResolvedParameter], List[ResolutionNote]) = {
    val byId = models.map(m => m.id -> m).toMap

    // A layer either gives a flag a value or takes it off the command line
    // altogether (`removedParameters`); `None` is the second. An empty string
    // cannot mean it — that is the convention for a valueless flag, which is
    // passed.
    def set(pairs: Iterable[(String, String)]): List[(String, Option[String])] =
      pairs.toList.map((flag, value) => flag -> Some(value))

    // Every assigned model contributes, in checkpoint order, rather than
    // special-casing "the diffusion slot": in practice only diffusion models
    // carry parameters, and a uniform rule needs no exceptions. A model's own
    // removals come after its values, so one that both sets and removes a flag
    // removes it.
    val modelEntries = slots(configuration, architecture).flatMap(checkpoint =>
      configuration.assignments
        .get(checkpoint.name)
        .flatMap(byId.get)
        .toList
        .flatMap(model =>
          set(model.parameters) ++ model.removedParameters.map(_ -> None)
        )
    )

    val layers: List[(ParameterLayer, List[(String, Option[String])])] = List(
      ParameterLayer.Architecture -> set(architecture.defaultParameters),
      ParameterLayer.Model -> modelEntries,
      ParameterLayer.Runtime -> set(rule.map(_.defaults).getOrElse(Map.empty)),
      ParameterLayer.Configuration ->
        (set(configuration.overriddenParameters) ++
          configuration.removedParameters.map(_ -> None))
    )

    // Folded in layer order rather than merged: a removal has to be noted
    // against the layer it removes from, and what a flag was worth before it
    // is exactly what the accumulator holds when the removal is applied. A
    // removal overrides like any other entry, so a layer above can set the
    // flag again.
    val (merged, removals) = layers
      .flatMap((source, entries) =>
        entries.map((flag, value) => (flag, value, source))
      )
      .foldLeft(
        (
          Map.empty[String, (Option[String], ParameterLayer)],
          List.empty[ResolutionNote]
        )
      ) { case ((values, notes), (flag, value, source)) =>
        val removedFrom =
          if (value.isDefined) None
          else values.get(flag).filter(_._1.isDefined).map(_._2)
        (
          values.updated(flag, (value, source)),
          notes ++ removedFrom.map(from =>
            ResolutionNote.Removed(flag, from, source): ResolutionNote
          )
        )
      }

    val vetoes =
      rule.map(_.unsupported).getOrElse(List.empty).map(u => u.flag -> u).toMap

    val kept = merged.collect {
      case (flag, (Some(value), source)) if !vetoes.contains(flag) =>
        flag -> (value, source)
    }

    // A flag nobody asks for any more is not "dropped by the runtime": the
    // removal above already explains its absence.
    val dropped = merged.toList
      .flatMap((flag, entry) =>
        vetoes
          .get(flag)
          .filter(_ => entry._1.isDefined)
          .map(veto =>
            ResolutionNote.Dropped(flag, entry._2, veto.reason): ResolutionNote
          )
      )
      .sortBy(_.message)

    val added = rule
      .map(_.defaults)
      .getOrElse(Map.empty)
      .toList
      .filter((flag, _) =>
        kept.get(flag).exists(_._2 == ParameterLayer.Runtime)
      )
      .map((flag, value) =>
        ResolutionNote.AddedByRuntime(flag, value): ResolutionNote
      )
      .sortBy(_.message)

    (
      kept.toList
        .map((flag, entry) => ResolvedParameter(flag, entry._1, entry._2))
        .sortBy(_.flag),
      dropped ++ removals.sortBy(_.message) ++ added
    )
  }

  /** The parameters actually passed, without the explanation — for callers that
    * only need the argv.
    */
  def effectiveParameters(
      configuration: RunConfiguration,
      architecture: Architecture,
      models: List[Model] = List.empty,
      rule: Option[RuntimeRule] = None
  ): List[(String, String)] =
    resolveParameters(configuration, architecture, models, rule).arguments

  /** The full argv, or the reasons it cannot be built.
    *
    * `resolvePath` turns an assigned model into the path sd-cpp should load.
    * Until the cache exists it can render the source for display; afterwards it
    * returns the real local file.
    */
  def argv(
      configuration: RunConfiguration,
      architectures: List[Architecture],
      models: List[Model],
      resolvePath: Model => String,
      port: Option[Int] = None,
      /** The LoRA root (`specs/09-lora-management.md`); its architecture folder
        * is emitted as `--lora-model-dir`. The backend passes the absolute
        * path; the preview passes the `~`-form since it cannot know the home
        * directory.
        */
      loraModelDirectory: Option[String] = None,
      /** The flat upscaler store (`specs/10-generation-time-upscaling.md`),
        * emitted as `--hires-upscalers-dir` — same convention as the LoRA root.
        */
      hiresUpscalersDirectory: Option[String] = None,
      /** The rule for the runtime this will launch on
        * (`specs/16-parameter-resolution.md`), when one is known.
        */
      rule: Option[RuntimeRule] = None,
      /** For llama-server, which scans no folder: the architecture's installed
        * GGUF LoRA files, each emitted as `--lora` and loaded unapplied, so
        * every assistant request applies its own
        * (`specs/35-assistant-loras.md`). Absolute from the backend, `~`-forms
        * in the preview, like the LoRA root.
        */
      loraAdapters: List[String] = Nil
  ): Either[List[LaunchBlocker], List[String]] =
    resolve(
      configuration,
      architectures,
      models,
      resolvePath,
      port,
      loraModelDirectory,
      hiresUpscalersDirectory,
      rule,
      loraAdapters
    ).map(_._1)

  /** The argv *and* the notes explaining it — what the preview and the launcher
    * both want. [[argv]] is this without the second half.
    */
  def resolve(
      configuration: RunConfiguration,
      architectures: List[Architecture],
      models: List[Model],
      resolvePath: Model => String,
      port: Option[Int] = None,
      loraModelDirectory: Option[String] = None,
      hiresUpscalersDirectory: Option[String] = None,
      rule: Option[RuntimeRule] = None,
      loraAdapters: List[String] = Nil
  ): Either[List[LaunchBlocker], (List[String], List[ResolutionNote])] = {
    val problems = blockers(configuration, architectures, models)
    if (problems.nonEmpty) Left(problems)
    else {
      val architecture =
        architectures.find(_.id == configuration.architectureId).get
      val byId = models.map(m => m.id -> m).toMap

      val checkpointArguments = slots(configuration, architecture).flatMap {
        checkpoint =>
          configuration.assignments
            .get(checkpoint.name)
            .flatMap(byId.get)
            .toList
            .flatMap(model => List(checkpoint.flag, resolvePath(model)))
      }

      // An empty value means a valueless flag; that is the convention in
      // `architectures.json`, e.g. `"--diffusion-fa": ""`.
      val resolution =
        resolveParameters(configuration, architecture, models, rule)
      val parameterArguments = resolution.arguments.flatMap {
        case (flag, "")    => List(flag)
        case (flag, value) => List(flag, value)
      }

      // The model directories are sd-server's scans; llama-server has no
      // equivalent and would refuse the flags.
      val sdCpp = architecture.tool == RuntimeTool.SdCpp
      // Only the architecture's own LoRAs are scanned, so a model cannot be
      // handed another architecture's (François, 2026-09-12).
      val loraArguments =
        if (sdCpp)
          loraModelDirectory.toList
            .flatMap(root =>
              List("--lora-model-dir", s"$root/${architecture.id}")
            )
        else if (loraAdapters.isEmpty) Nil
        else
          loraAdapters.flatMap(path => List("--lora", path)) :+
            "--lora-init-without-apply"

      val upscalerArguments = hiresUpscalersDirectory.toList
        .filter(_ => sdCpp)
        .flatMap(directory => List("--hires-upscalers-dir", directory))

      val listenArguments = port.toList.flatMap(p =>
        architecture.tool match {
          case RuntimeTool.SdCpp =>
            List("--listen-ip", "127.0.0.1", "--listen-port", p.toString)
          case RuntimeTool.LlamaCpp =>
            List("--host", "127.0.0.1", "--port", p.toString)
        }
      )

      Right(
        (
          checkpointArguments ++ parameterArguments ++ loraArguments ++
            upscalerArguments ++ listenArguments,
          resolution.notes
        )
      )
    }
  }

  /** The argv as a single shell-ish line, for display and copy-paste. */
  def render(arguments: List[String], executable: String): String =
    (executable :: arguments)
      .map(a => if (a.exists(_.isWhitespace)) "\"" + a + "\"" else a)
      .mkString(" ")
}
