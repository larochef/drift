package drift.backend.runtime

import drift.backend.download.Downloader
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.*
import java.util.Comparator
import scala.jdk.CollectionConverters.*

/** Everything a launch needs from a runtime
  * (`specs/07-launch-and-supervision.md` consumes this): the binary, and the
  * environment that lets it find its libraries. A run configuration's argv is
  * useless without them.
  */
case class LaunchRuntime(
    runtime: Runtime,
    executable: Path,
    environment: Map[String, String]
)

/** Owns drift's runtimes — sd-cpp (`specs/06-sdcpp-runtime.md`) and llama.cpp
  * (`specs/17-assistant-runtime.md`), the same code with the tool as a
  * parameter: fetches a chosen release together with a matching TheRock ROCm
  * build, keeps several side by side, validates each by actually running it,
  * and resolves which one a launch uses.
  *
  * Layout, all re-downloadable and therefore under `~/.cache`:
  * {{{
  * <runtimesRoot>/sd-cpp/<tag>-<backend>/...      unpacked sd-cpp release
  * <runtimesRoot>/llama-cpp/<tag>-<backend>/...   unpacked llama.cpp release
  * <runtimesRoot>/therock/<gfx>-<version>/...     unpacked TheRock tarball
  * <runtimesRoot>/downloads/...                   archives in flight
  * }}}
  * TheRock builds are shared between releases — of either tool — that want the
  * same ROCm, so they are keyed separately rather than nested. The `Runtime`
  * entities live in `~/.config/drift` with the rest.
  *
  * The work lives beside this file (`specs/29-split-oversized-files.md`):
  * installing (`RuntimeInstalls`, fetching through `RuntimeArchives`), pairing
  * a ROCm build (`TheRockPairing`), validating (`RuntimeValidation`), the
  * default per tool and what a launch runs (`RuntimeSelections`), and removing
  * what nothing uses any more (`RuntimeCleanup`).
  */
final class RuntimeManager(
    storage: StorageService,
    downloader: Downloader,
    catalog: RuntimeCatalog,
    val runtimesRoot: Path
) {
  private val selections = RuntimeSelections(storage)
  private val validation = RuntimeValidation(storage, selections)
  private val pairing = TheRockPairing(catalog, runtimesRoot)
  private val cleanup = RuntimeCleanup(storage, runtimesRoot, selections)
  private val runnerFiles = RunnerFiles(storage, validation)
  private val installs = RuntimeInstalls(
    storage,
    catalog,
    runtimesRoot,
    pairing,
    validation,
    cleanup,
    RuntimeArchives(downloader, runtimesRoot),
    runnerFiles
  )

  private val resolution = RuntimeInstallResolution(storage, installs)

  /** Installs a build picked where configurations could not launch, and moves
    * them to its engine once it validates (`specs/46`).
    */
  def installFor(
      request: InstallForConfigurationsRequest
  ): List[RuntimeInstallJob] =
    resolution.installFor(request)

  /** The installed drift runner brought to the one this build ships
    * (`specs/43`); run at startup.
    */
  def refreshRunner(): Unit = runnerFiles.refresh()

  def runnerOffer: RunnerOffer = runnerFiles.offer

  def installRunner(request: InstallRunnerRequest): List[RuntimeInstallJob] =
    installs.installRunner(request.theRockVersion)

  def listInstalls: List[RuntimeInstallJob] = installs.list

  def cancelInstall(runtimeId: String): RuntimeInstallJob =
    installs.cancel(runtimeId)

  def install(request: InstallRuntimeRequest): RuntimeInstallJob =
    installs.install(request)

  def installLatest(
      tool: RuntimeTool,
      backend: RuntimeBackend,
      gfxTarget: Option[String],
      theRockVersion: Option[String] = None
  ): RuntimeInstallJob =
    installs.installLatest(tool, backend, gfxTarget, theRockVersion)

  def upgrade(
      id: String,
      theRockVersion: Option[String] = None
  ): RuntimeInstallJob = installs.upgrade(id, theRockVersion)

  /** Re-pairs a runtime with another TheRock build; the drift runner's two
    * runtimes move together.
    */
  def changeTheRock(id: String, theRockVersion: String): RuntimeInstallJob = {
    val job = installs.changeTheRock(id, theRockVersion)
    runnerSibling(id).foreach(installs.changeTheRock(_, theRockVersion))
    job
  }

  /** The drift runner's other runtime, when `id` is one of its two. */
  private def runnerSibling(id: String): Option[String] =
    storage
      .get[Runtime]("runtimes", id)
      .filter(_.engine == RuntimeEngine.DriftRunner)
      .flatMap(runtime =>
        RuntimeTool.values
          .find(_ != runtime.tool)
          .map(RunnerFiles.id)
          .filter(storage.get[Runtime]("runtimes", _).isDefined)
      )

  def resolveTheRock(gfx: String, rocmVersion: String): TheRockResolution =
    pairing.resolve(gfx, rocmVersion)

  def validateAndSave(runtime: Runtime): Runtime =
    validation.validateAndSave(runtime)

  def revalidate(id: String): Option[Runtime] = validation.revalidate(id)

  def selection: RuntimeSelection = selections.current

  def saveSelection(next: RuntimeSelection): RuntimeSelection =
    selections.save(next)

  def resolveForLaunch(
      tool: RuntimeTool,
      engine: RuntimeEngine,
      pinnedId: Option[String]
  ): Either[String, LaunchRuntime] =
    selections.resolveForLaunch(tool, engine, pinnedId)

  /** The builds to offer per tool where a launch finds none
    * (`specs/46-starter-configurations.md`): ROCm only where the ROCm driver
    * exposes an AMD GPU, paired for its gfx target; Vulkan and CPU always;
    * drift's runner where this drift carries it and the GPU is the one its
    * kernels are built for. Recommended: ROCm for sd-cpp when there is one,
    * Vulkan otherwise.
    */
  def installOptions: List[RuntimeInstallOption] = {
    val gfx = GpuDetection.amdGfxTarget()
    val runner = runnerFiles.offer.available && gfx.contains(RunnerFiles.Gfx)
    RuntimeTool.values.toList.flatMap { tool =>
      val preferred =
        if (tool == RuntimeTool.SdCpp && gfx.isDefined) RuntimeBackend.Rocm
        else RuntimeBackend.Vulkan
      val backends = gfx.fold(List.empty[RuntimeBackend])(_ =>
        List(RuntimeBackend.Rocm)
      ) ++ List(RuntimeBackend.Vulkan, RuntimeBackend.Cpu)
      backends.map { backend =>
        val target = Option.when(backend == RuntimeBackend.Rocm)(gfx).flatten
        RuntimeInstallOption(
          tool,
          RuntimeEngine.upstream(tool),
          Some(InstallLatestRequest(tool, backend, target)),
          RuntimeManager.latestId(tool, backend),
          s"${tool.displayName} on ${backend match {
              case RuntimeBackend.Rocm   => "ROCm"
              case RuntimeBackend.Vulkan => "Vulkan"
              case RuntimeBackend.Cpu    => "CPU"
            }}" + target.fold("")(t => s" ($t)"),
          recommended = backend == preferred
        )
      } ++ Option.when(runner)(
        RuntimeInstallOption(
          tool,
          RuntimeEngine.DriftRunner,
          None,
          RunnerFiles.id(tool),
          RunnerFiles.label(tool),
          recommended = false
        )
      )
    }
  }

  /** After a delete: the files nothing uses any more; deleting one of the drift
    * runner's runtimes deletes the other too.
    */
  def cleanupDeleted(runtime: Runtime): Unit = {
    if (runtime.engine == RuntimeEngine.DriftRunner)
      RuntimeTool.values
        .filter(_ != runtime.tool)
        .map(RunnerFiles.id)
        .flatMap(storage.get[Runtime]("runtimes", _))
        .foreach { sibling =>
          storage.delete("runtimes", sibling.id)
          cleanup.cleanupDeleted(sibling)
        }
    cleanup.cleanupDeleted(runtime)
  }
}

object RuntimeManager {

  /** The `sd-cli` shipped beside a runtime's sd-server — what one-shot work
    * (post-processing, conversion) spawns instead of a session.
    */
  def sdCliOf(launch: LaunchRuntime): Either[String, Path] = {
    val cli = launch.executable.getParent.resolve("sd-cli")
    Either.cond(
      Files.isRegularFile(cli),
      cli,
      s"runtime '${launch.runtime.label}' ships no sd-cli beside sd-server"
    )
  }

  /** Installs allowed to run together. Matched to
    * `Downloader.MaxTransfersPerHost` on purpose: the two limits govern the
    * same resource from different ends, and a runtime download had no reason to
    * be the only kind that waits its turn.
    */
  val MaxConcurrentInstalls: Int = Downloader.MaxTransfersPerHost

  /** Strix Halo, the target machine — detection is an open question in the
    * spec, so a predictable default fills in.
    */
  val DefaultGfxTarget: String = "gfx1151"

  def runtimeId(request: InstallRuntimeRequest): String =
    s"${request.tool.idPrefix}${request.releaseTag}-${request.asset.backend.toString.toLowerCase}"

  /** The stable id of the latest-tracking runtime for a tool and backend: it
    * outlives the tag it currently points at, so the default selection and any
    * launch survive an upgrade.
    */
  def latestId(tool: RuntimeTool, backend: RuntimeBackend): String =
    s"${tool.idPrefix}latest-${backend.toString.toLowerCase}"

  /** Where a tool's releases unpack under the runtimes root. */
  def toolDirectory(tool: RuntimeTool): String = tool match {
    case RuntimeTool.SdCpp    => "sd-cpp"
    case RuntimeTool.LlamaCpp => "llama-cpp"
  }

  private val SdCppBanner =
    """stable-diffusion\.cpp version (\S+), commit (\w+).*""".r
  private val LlamaCppBanner =
    """version: (\S+) \(build (\d+), commit (\w+)\).*""".r

  /** The prefix llama.cpp's logger puts in front of a line — `0.00.000.283 I
    * srv ` — which older builds did not print at all. Stripped before a line is
    * matched, so a banner logged rather than printed still reads.
    */
  private val LlamaCppLogPrefix =
    """^[\d.]+ [A-Z] \S+\s+(.*)$""".r

  /** What the runtime row shows for a validated build: the banner's useful
    * part. sd-cpp's release builds print `stable-diffusion.cpp version unknown,
    * commit 6b3edaa` — the version is always "unknown", so the commit is the
    * identity; llama.cpp prints `version: 0.4.1-dev (build 11060, commit
    * 426090367)`. Anything unrecognised is kept as printed.
    *
    * The banner is looked for in the *whole* output rather than its first line:
    * llama.cpp b11060 logs `llama_server: initializing ...` before it, and
    * reading only the first line put that log line in the version chip
    * (François, 2026-09-20). A build whose banner is still the first thing it
    * prints reads exactly as it did.
    */
  def reportedVersion(tool: RuntimeTool, output: String): String = {
    val lines = output.linesIterator
      .map(line =>
        line.trim match {
          case LlamaCppLogPrefix(logged) => logged.trim
          case plain                     => plain
        }
      )
      .filter(_.nonEmpty)
      .toList
    lines.iterator
      .map(line => (tool, line))
      .collectFirst {
        case (RuntimeTool.SdCpp, SdCppBanner("unknown", commit)) =>
          s"commit $commit"
        case (RuntimeTool.SdCpp, SdCppBanner(version, commit)) =>
          s"$version ($commit)"
        case (RuntimeTool.LlamaCpp, LlamaCppBanner(version, build, commit)) =>
          s"$version (b$build, $commit)"
      }
      // Nothing recognisable: the first line as printed, which is what an
      // older build — or a build whose banner has changed again — shows.
      .getOrElse(lines.headOption.getOrElse("").take(200))
  }

  /** The flags that prove a build runs, in order of preference. */
  def validationFlags(tool: RuntimeTool): List[String] = tool match {
    case RuntimeTool.SdCpp    => List("--help")
    case RuntimeTool.LlamaCpp => List("--version", "--help")
  }

  /** The (major, minor, patch) of a version, ignoring any build stamp. */
  private[runtime] def triple(version: String): (Int, Int, Int) = {
    val key = RuntimeCatalog.versionSortKey(version)
    (key._1, key._2, key._3)
  }

  /** Whether a TheRock build satisfies a declared ROCm version — see
    * [[RocmVersions.satisfies]], shared with the UI's pairing display.
    */
  private[runtime] def sameRocm(declared: String, candidate: String): Boolean =
    RocmVersions.satisfies(declared, candidate)

  /** The build stamp after the triple ("…a20260612" -> 20260612), 0 for a
    * tagged release. Orders builds within one triple, newest first.
    */
  private[runtime] def buildStamp(version: String): Long =
    RuntimeCatalog.versionSortKey(version)._4

  /** The channel a bare version string implies, for a build known only from an
    * on-disk directory: a plain triple is a tagged stable release, an `rc`
    * marker a release candidate, anything else a nightly.
    */
  private[runtime] def inferChannel(version: String): TheRockChannel =
    if (version.matches("""\d+\.\d+\.\d+""")) TheRockChannel.Stable
    else if (version.contains("rc")) TheRockChannel.ReleaseCandidate
    else TheRockChannel.Nightly

  /** Where the executable is: the sd-cpp ROCm zip unpacks to `build/bin/`, the
    * Vulkan and CPU zips unpack flat (verified against the live archives);
    * llama.cpp tarballs are probed the same way.
    */
  def executableIn(installDir: Path, tool: RuntimeTool): Option[Path] = {
    val name = tool.executableName
    List(
      installDir.resolve("build").resolve("bin").resolve(name),
      installDir.resolve("bin").resolve(name),
      installDir.resolve(name)
    ).find(Files.isRegularFile(_))
  }

  /** For Rocm, TheRock's `lib` first; then the binary's own directory (which
    * holds `libstable-diffusion.so` and the `libggml-*` family) and, when an
    * archive keeps its libraries apart, a `lib` beside `bin`.
    */
  def libraryPath(runtime: Runtime, executable: Path): String = {
    val own = executable.getParent
    val siblingLib = own.resolveSibling("lib")
    val ownEntries =
      own.toString :: Option
        .when(Files.isDirectory(siblingLib))(siblingLib.toString)
        .toList
    val rock = runtime.backend match {
      case RuntimeBackend.Rocm =>
        runtime.theRockPath
          .map(r => Paths.get(r).resolve("lib").toString)
          .toList
      case _ => Nil
    }
    (rock ++ ownEntries).mkString(":")
  }

  def deleteTree(path: Path): Unit =
    if (Files.exists(path))
      Files
        .walk(path)
        .sorted(Comparator.reverseOrder())
        .iterator()
        .asScala
        .foreach(Files.deleteIfExists(_))
}
