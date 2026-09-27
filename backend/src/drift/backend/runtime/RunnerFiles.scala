package drift.backend.runtime

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.util.Arrays
import java.util.jar.JarInputStream
import scala.util.Using

import com.typesafe.scalalogging.Logger

/** drift's own runner as drift ships it (`specs/43`): the jar among its
  * resources (`runner/drift-runner.jar`, kernels included) and the ROCm version
  * its kernels were compiled with (`runner/rocm-version`). An install writes it
  * into a runtime's folder with a launcher running drift's own Java; at startup
  * the installed runner runtimes are brought to the jar this drift carries.
  */
final private[runtime] class RunnerFiles(
    storage: StorageService,
    validation: RuntimeValidation
) {
  private val logger = Logger(getClass)

  private def resource(path: String): Option[Array[Byte]] =
    Option(getClass.getClassLoader.getResourceAsStream(path))
      .map(stream => Using.resource(stream)(_.readAllBytes()))

  private lazy val jar: Option[Array[Byte]] = resource(RunnerFiles.Jar)

  /** The ROCm version the kernels were compiled with. */
  lazy val rocmVersion: Option[String] =
    resource(RunnerFiles.RocmVersion)
      .map(String(_, StandardCharsets.UTF_8).trim)
      .filter(_.nonEmpty)

  /** Whether this drift can install the runner, and why not. */
  def offer: RunnerOffer = {
    val reason =
      if (jar.isEmpty) Some("this build of drift carries no runner")
      else if (!jar.exists(RunnerFiles.hasKernels))
        Some(
          "the runner was built without its kernels (no TheRock at build time)"
        )
      else if (java.lang.Runtime.version().feature() < 25)
        Some(
          s"the runner needs Java 25; drift runs on ${java.lang.Runtime.version()}"
        )
      else if (rocmVersion.isEmpty)
        Some("the runner does not say which ROCm its kernels were built with")
      else None
    RunnerOffer(reason.isEmpty, rocmVersion, reason)
  }

  /** Writes the runner for `tool` into `folder`, on the TheRock tree `rock`;
    * answers whether a file changed.
    */
  def write(folder: Path, tool: RuntimeTool, rock: Path): Boolean = {
    val bytes = jar.getOrElse(
      throw new IllegalStateException(offer.reason.getOrElse("no runner"))
    )
    Files.createDirectories(folder)
    val java = Paths.get(System.getProperty("java.home"), "bin", "java")
    val mainClass = RunnerFiles.mainClass(tool)
    val launcher =
      s"""#!/usr/bin/env bash
         |# drift's own runner, started as drift starts ${tool.executableName} (specs/43).
         |# Written by drift from the runner it ships.
         |here="$$(cd "$$(dirname "$$0")" && pwd)"
         |export DRIFT_ROCM_ROOT="$${DRIFT_ROCM_ROOT:-$rock}"
         |exec "$java" --enable-native-access=ALL-UNNAMED -cp "$$here/drift-runner.jar" $mainClass "$$@"
         |""".stripMargin.getBytes(StandardCharsets.UTF_8)
    val wroteJar = writeIfDifferent(folder.resolve("drift-runner.jar"), bytes)
    val launcherPath = folder.resolve(tool.executableName)
    val wroteLauncher = writeIfDifferent(launcherPath, launcher)
    launcherPath.toFile.setExecutable(true)
    wroteJar || wroteLauncher
  }

  /** Brings each installed runner runtime to the runner this drift carries,
    * validating again the ones that changed. Never installs one: a deleted
    * runner stays deleted.
    */
  def refresh(): Unit =
    if (offer.available)
      storage
        .list[drift.shared.Runtime]("runtimes")
        .filter(_.engine == RuntimeEngine.DriftRunner)
        .foreach { runtime =>
          runtime.theRockPath.map(Paths.get(_)) match {
            case Some(rock) if Files.isDirectory(rock) =>
              if (write(Paths.get(runtime.installedAt), runtime.tool, rock)) {
                val validated = validation.validateAndSave(runtime)
                if (validated.valid)
                  logger.info(s"updated the runner '${runtime.id}'")
                else
                  logger.warn(
                    s"the updated runner '${runtime.id}' does not run: ${validated.validationError.getOrElse("")}"
                  )
              }
            case _ =>
              logger.warn(
                s"the runner '${runtime.id}' has lost its TheRock tree; change its ROCm build"
              )
          }
        }

  /** Writes `bytes` unless the file holds exactly them; answers whether it
    * wrote. A new file moved in place: a runner still running keeps the old.
    */
  private def writeIfDifferent(path: Path, bytes: Array[Byte]): Boolean = {
    val same = Files
      .isRegularFile(path) && Arrays.equals(Files.readAllBytes(path), bytes)
    if (!same) {
      val fresh = path.resolveSibling(s"${path.getFileName}.new")
      Files.write(fresh, bytes)
      Files.move(fresh, path, StandardCopyOption.REPLACE_EXISTING)
    }
    !same
  }
}

private[runtime] object RunnerFiles {

  val Jar = "runner/drift-runner.jar"
  val RocmVersion = "runner/rocm-version"

  /** The runner's kernels are built for Strix Halo only. */
  val Gfx = "gfx1151"

  /** Each tool's runner runtime: its id, label and main class. */
  def id(tool: RuntimeTool): String = tool match {
    case RuntimeTool.LlamaCpp => "drift-runner"
    case RuntimeTool.SdCpp    => "drift-runner-images"
  }

  def label(tool: RuntimeTool): String = tool match {
    case RuntimeTool.LlamaCpp => "drift runner (gfx1151)"
    case RuntimeTool.SdCpp    => "drift runner, images (gfx1151)"
  }

  def mainClass(tool: RuntimeTool): String = tool match {
    case RuntimeTool.LlamaCpp => "drift.runner.server.Main"
    case RuntimeTool.SdCpp    => "drift.runner.server.ImageMain"
  }

  /** Whether the jar carries its compiled kernels. */
  def hasKernels(jar: Array[Byte]): Boolean =
    Using.resource(JarInputStream(java.io.ByteArrayInputStream(jar))) { in =>
      Iterator
        .continually(in.getNextJarEntry)
        .takeWhile(_ != null)
        .exists(_.getName.startsWith("kernels/"))
    }
}
