package drift.backend.runtime

import drift.backend.process.ChildProcesses
import drift.backend.storage.{Collections, StorageService}
import drift.shared.*

import java.nio.file.Paths
import scala.concurrent.duration.DurationInt
import scala.util.control.NonFatal

/** Validates a runtime by running its executable (`sd-server --help`,
  * `llama-server --version`) with the runtime's own `LD_LIBRARY_PATH`, and
  * saves the outcome. The only honest check: the binary on the target machine
  * fails immediately without the environment (`error while loading shared
  * libraries: libstable-diffusion.so`), so executing is the proof that binary,
  * libraries and environment agree. A failing runtime is kept and marked
  * invalid with the error, not silently unusable at launch.
  */
final private[runtime] class RuntimeValidation(
    storage: StorageService,
    selections: RuntimeSelections
) {

  def validateAndSave(runtime: Runtime): Runtime = {
    val validated = validate(runtime)
    storage.save(Collections.runtimes, validated.id, validated)
    // The first runtime of a tool that proves itself becomes that tool's
    // default; switching afterwards is explicit.
    if (validated.valid)
      selections.update(current =>
        if (current.defaultFor(validated.tool).isEmpty)
          current.withDefault(validated.tool, Some(validated.id))
        else current
      )
    validated
  }

  def revalidate(id: String): Option[Runtime] =
    storage.get(Collections.runtimes, id).map(validateAndSave)

  /** Tries each validation flag of the tool in turn (`--version` is the
    * informative one for llama.cpp, `--help` the fallback that every build
    * answers); the first that exits 0 wins.
    */
  private def validate(runtime: Runtime): Runtime =
    runtime.engine match {
      // drift's runner: `--version` proves this GPU runs its kernels, and
      // `--model-kinds` says what it runs (`specs/43`)
      case RuntimeEngine.DriftRunner =>
        val validated = validateWith(runtime, "--version")
        if (!validated.valid) validated.copy(modelKinds = None)
        else
          run(validated, "--model-kinds") match {
            case Right(kinds) =>
              validated.copy(modelKinds =
                Some(kinds.linesIterator.map(_.trim).filter(_.nonEmpty).toList)
              )
            case Left(error) =>
              validated.copy(
                valid = false,
                validationError = Some(s"--model-kinds: $error"),
                modelKinds = None
              )
          }
      case _ =>
        val flags = RuntimeManager.validationFlags(runtime.tool)
        val attempts =
          flags.iterator.map(flag => (flag, validateWith(runtime, flag)))
        attempts
          .find(_._2.valid)
          .map(_._2)
          .getOrElse(validateWith(runtime, flags.head))
    }

  private def validateWith(runtime: Runtime, flag: String): Runtime =
    run(runtime, flag) match {
      case Right(text) =>
        runtime.copy(
          valid = true,
          validationError = None,
          reportedVersion =
            Some(RuntimeManager.reportedVersion(runtime.tool, text))
        )
      case Left(error) =>
        runtime.copy(
          valid = false,
          validationError = Some(error),
          reportedVersion = None
        )
    }

  /** Runs the runtime's executable with `flag` under its own environment: its
    * output when it exits 0, else why not.
    */
  private def run(runtime: Runtime, flag: String): Either[String, String] = {
    val executableName = runtime.tool.executableName
    RuntimeManager.executableIn(
      Paths.get(runtime.installedAt),
      runtime.tool
    ) match {
      case None =>
        Left(s"$executableName not found under ${runtime.installedAt}")
      case Some(executable) =>
        try
          ChildProcesses
            .answer(
              List(executable.toString, flag),
              within = 30.seconds,
              // --help is bigger than nothing
              most = 64 * 1024,
              environment = Map(
                "LD_LIBRARY_PATH" ->
                  RuntimeManager.libraryPath(runtime, executable)
              )
            )
            .left
            .map(_ => s"$executableName $flag did not finish within 30s")
            .flatMap { answer =>
              if (answer.exit == 0) Right(answer.text)
              else
                Left(
                  s"exit code ${answer.exit}: ${answer.text.trim.take(300)}"
                )
            }
        catch {
          case NonFatal(err) =>
            Left(Option(err.getMessage).getOrElse(err.toString))
        }
    }
  }
}
