package drift.backend.runtime

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.charset.StandardCharsets
import java.nio.file.Paths
import java.util.concurrent.TimeUnit
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
    storage.save("runtimes", validated.id, validated)
    // The first runtime of a tool that proves itself becomes that tool's
    // default; switching afterwards is explicit.
    val current = selections.current
    if (validated.valid && current.defaultFor(validated.tool).isEmpty)
      selections.save(current.withDefault(validated.tool, Some(validated.id)))
    validated
  }

  def revalidate(id: String): Option[Runtime] =
    storage.get[Runtime]("runtimes", id).map(validateAndSave)

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
        try {
          val builder = ProcessBuilder(executable.toString, flag)
            .redirectErrorStream(true)
          builder
            .environment()
            .put(
              "LD_LIBRARY_PATH",
              RuntimeManager.libraryPath(runtime, executable)
            )
          val process = builder.start()
          // Read on a separate thread: --help is bigger than nothing, and a
          // full pipe would deadlock a read-after-wait.
          val output = StringBuilder()
          val reader = Thread(new Runnable {
            def run(): Unit =
              try {
                val text = String(
                  process.getInputStream.readNBytes(64 * 1024),
                  StandardCharsets.UTF_8
                )
                output.synchronized(output.append(text))
              } catch { case NonFatal(_) => () }
          })
          reader.start()
          val finished = process.waitFor(30, TimeUnit.SECONDS)
          if (!finished) {
            process.destroyForcibly()
            Left(s"$executableName $flag did not finish within 30s")
          } else {
            reader.join(5000)
            val text = output.synchronized(output.toString)
            if (process.exitValue() == 0) Right(text)
            else
              Left(s"exit code ${process.exitValue()}: ${text.trim.take(300)}")
          }
        } catch {
          case NonFatal(err) =>
            Left(Option(err.getMessage).getOrElse(err.toString))
        }
    }
  }
}
