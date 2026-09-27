package drift.backend.session

import drift.backend.cache.{CacheEntry, ModelCache}
import drift.backend.runtime.LaunchRuntime
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.Path

/** The argv a run configuration launches with
  * (`specs/16-parameter-resolution.md`): its assigned weights resolved to
  * cached paths, the rule of the runtime it runs on applied, or the blockers
  * that stop it.
  */
final private[session] class LaunchArguments(
    storage: StorageService,
    modelCache: ModelCache,
    /** The LoRA root; a server is given its architecture's folder in it as
      * `--lora-model-dir` (`specs/09-lora-management.md`).
      */
    lorasRoot: Path,
    /** The flat upscaler store passed as `--hires-upscalers-dir`
      * (`specs/10-generation-time-upscaling.md`).
      */
    upscaleRoot: Path
) {

  /** The configuration, its architecture and the argv of a one-shot `sd-cli`
    * run of it on `launchRuntime` — resolved as a launch would, blockers
    * refused by name — for the PiD upscale's tiles (`specs/26-tiled-pid.md`)
    * and the configuration a tiled job checks.
    */
  def resolve(
      runConfigurationId: String,
      launchRuntime: LaunchRuntime
  ): Either[String, (RunConfiguration, Architecture, List[String])] =
    storage.get[RunConfiguration](
      "run-configurations",
      runConfigurationId
    ) match {
      case None =>
        Left(s"run configuration '$runConfigurationId' does not exist")
      case Some(configuration) =>
        val architectures = storage.list[Architecture]("architectures")
        argumentsFor(
          configuration,
          architectures,
          storage.list[Model]("models"),
          launchRuntime,
          None
        ).left
          .map(problems =>
            "cannot run: " + problems.map(_.message).mkString("; ")
          )
          .map((arguments, _) =>
            (
              configuration,
              architectures.find(_.id == configuration.architectureId).get,
              arguments
            )
          )
    }

  /** Whether each assigned model's weights are on disk, in the shape
    * `CommandLine.blockers` wants — and, for the ones present, the local path
    * the argv needs.
    */
  def cacheStatuses(
      configuration: RunConfiguration,
      models: List[Model]
  ): Map[String, ModelCacheStatus] =
    configuration.assignments.values.toList.distinct.flatMap { modelId =>
      models.find(_.id == modelId).map { model =>
        modelId -> (modelCache.resolve(model.source) match {
          case CacheEntry.Present(path, bytes) =>
            ModelCacheStatus(
              modelId,
              CacheState.Cached,
              Some(path.toString),
              Some(bytes)
            )
          case CacheEntry.Absent =>
            ModelCacheStatus(modelId, CacheState.Missing)
          case CacheEntry.BrokenLocal(path) =>
            ModelCacheStatus(modelId, CacheState.Broken, Some(path.toString))
        })
      }
    }.toMap

  /** The argv `configuration` launches with on `launchRuntime` — cached weight
    * paths, the rule of the runtime it actually runs on — or the blockers. With
    * a `port` it is a server's, listening there with the LoRA and upscaler
    * roots it scans; a session and a job server launch the same way. Without, a
    * one-shot `sd-cli` run's, which scans neither.
    */
  def argumentsFor(
      configuration: RunConfiguration,
      architectures: List[Architecture],
      models: List[Model],
      launchRuntime: LaunchRuntime,
      port: Option[Int]
  ): Either[List[LaunchBlocker], (List[String], List[ResolutionNote])] = {
    val statuses = cacheStatuses(configuration, models)
    val problems =
      CommandLine.blockers(configuration, architectures, models, statuses)
    if (problems.nonEmpty) Left(problems)
    else
      CommandLine.resolve(
        configuration,
        architectures,
        models,
        model =>
          statuses
            .get(model.id)
            .flatMap(_.path)
            // Unreachable: the blockers above proved every assigned model
            // cached, and refusing loudly beats a broken argv.
            .getOrElse(
              throw IllegalStateException(
                s"model '${model.id}' has no cached path"
              )
            ),
        port,
        loraModelDirectory = port.map(_ => lorasRoot.toString),
        hiresUpscalersDirectory = port.map(_ => upscaleRoot.toString),
        rule =
          RuntimeRule.forRuntime(launchRuntime.runtime, storage.runtimeRules),
        // Only files on disk: llama-server refuses to start on a missing one.
        loraAdapters = port.toList.flatMap(_ =>
          LoraAdapters
            .relativePaths(
              configuration.architectureId,
              storage.list[Lora]("loras")
            )
            .map(lorasRoot.resolve)
            .filter(java.nio.file.Files.isRegularFile(_))
            .map(_.toString)
        )
      )
  }
}
