package drift.backend.runtime

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.Paths

/** The default runtime per tool (`settings/runtime-selection.json`), and the
  * runtime a launch runs: the pin, else the engine's (`specs/43`).
  */
final private[runtime] class RuntimeSelections(storage: StorageService) {

  def current: RuntimeSelection =
    storage
      .get[RuntimeSelection]("settings", "runtime-selection")
      .getOrElse(RuntimeSelection())

  def save(next: RuntimeSelection): RuntimeSelection =
    storage.save("settings", "runtime-selection", next)

  /** The runtime a launch uses — the pin, else `engine`'s: the tool's default
    * when it is that engine's, else the engine's newest valid build — refusing
    * with a named reason if none resolves to something that runs, or if it is a
    * build of the other tool.
    */
  def resolveForLaunch(
      tool: RuntimeTool,
      engine: RuntimeEngine,
      pinnedId: Option[String]
  ): Either[String, LaunchRuntime] =
    pinnedId.orElse(engineRuntime(tool, engine)) match {
      case None =>
        Left(
          s"no ${engine.displayName} runtime for ${tool.displayName} is installed"
        )
      case Some(id) =>
        storage.get[Runtime]("runtimes", id) match {
          case None => Left(s"runtime '$id' is not registered")
          case Some(runtime) if runtime.tool != tool =>
            Left(
              s"runtime '${runtime.label}' is a ${runtime.tool.displayName} build, not ${tool.displayName}"
            )
          case Some(runtime) if pinnedId.isEmpty && runtime.engine != engine =>
            Left(s"runtime '${runtime.label}' is not ${engine.displayName}")
          case Some(runtime) if !runtime.valid =>
            Left(
              s"runtime '${runtime.label}' failed validation: ${runtime.validationError
                  .getOrElse("never validated")}"
            )
          case Some(runtime) =>
            RuntimeManager
              .executableIn(Paths.get(runtime.installedAt), runtime.tool)
              .toRight(
                s"runtime '${runtime.label}': ${runtime.tool.executableName} not found under ${runtime.installedAt}"
              )
              .map(executable =>
                LaunchRuntime(
                  runtime,
                  executable,
                  Map(
                    "LD_LIBRARY_PATH" -> RuntimeManager
                      .libraryPath(runtime, executable)
                  )
                )
              )
        }
    }

  /** The tool's default when it is `engine`'s, else the engine's newest valid
    * build of the tool.
    */
  private def engineRuntime(
      tool: RuntimeTool,
      engine: RuntimeEngine
  ): Option[String] = {
    val runtimes = storage
      .list[Runtime]("runtimes")
      .filter(r => r.tool == tool && r.engine == engine)
    current
      .defaultFor(tool)
      .filter(id => runtimes.exists(_.id == id))
      .orElse(
        runtimes.filter(_.valid).sortBy(-_.createdAt).headOption.map(_.id)
      )
  }
}
