package drift.backend.runtime

import drift.backend.storage.StorageService
import drift.shared.*

import com.typesafe.scalalogging.Logger

/** An install picked where configurations could not launch
  * (`specs/46-starter-configurations.md`), carried through to a launchable
  * state: the build installs — unless its engine already has a valid runtime —
  * and the configurations move to its engine only once it has validated. A
  * failed or cancelled install leaves them on the engine they had, so no
  * configuration ever names an engine with nothing behind it. Held here rather
  * than in the page, so leaving the page does not lose the switch.
  */
final private[runtime] class RuntimeInstallResolution(
    storage: StorageService,
    installs: RuntimeInstalls
) {
  private val logger = Logger[RuntimeInstallResolution]

  def installFor(
      request: InstallForConfigurationsRequest
  ): List[RuntimeInstallJob] = {
    val option = request.option
    val alreadyValid = storage
      .list[Runtime]("runtimes")
      .exists(r =>
        r.valid && r.tool == option.tool && r.engine == option.engine
      )
    if (alreadyValid) {
      switch(option, request.configurationIds)
      Nil
    } else {
      val jobs = option.request.fold(installs.installRunner(None))(latest =>
        List(
          installs.installLatest(
            latest.tool,
            latest.backend,
            latest.gfxTarget,
            latest.theRockVersion
          )
        )
      )
      val switchNow = () => switch(option, request.configurationIds)
      // Registered before the job can finish; when it has already, its outcome
      // is on disk to read.
      if (!installs.whenValid(option.runtimeId)(switchNow))
        if (storage.get[Runtime]("runtimes", option.runtimeId).exists(_.valid))
          switchNow()
      jobs
    }
  }

  /** Moves each configuration its architecture lets run on the engine. */
  private def switch(
      option: RuntimeInstallOption,
      configurationIds: List[String]
  ): Unit =
    configurationIds
      .flatMap(storage.get[RunConfiguration]("run-configurations", _))
      .filter(_.runner != option.engine)
      .filter(configuration =>
        storage
          .get[Architecture]("architectures", configuration.architectureId)
          .exists(_.runners.contains(option.engine))
      )
      .foreach { configuration =>
        storage.save(
          "run-configurations",
          configuration.id,
          configuration.copy(runner = option.engine)
        )
        logger.info(
          s"'${configuration.id}' now runs on ${option.engine.displayName}"
        )
      }
}
