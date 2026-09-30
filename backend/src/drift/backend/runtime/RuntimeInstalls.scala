package drift.backend.runtime

import drift.backend.{Background, WorkQueue}
import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.*
import java.util.concurrent.*
import java.util.concurrent.atomic.AtomicBoolean
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.typesafe.scalalogging.Logger

/** One install's job, and the flag its cancel raises. */
final private[runtime] class RuntimeInstallEntry(
    @volatile var job: RuntimeInstallJob,
    val cancelled: AtomicBoolean = AtomicBoolean(false)
)

/** Installing runtimes: the jobs, the queue they wait in, and each install's
  * run — the TheRock pairing resolved first, the release and the build fetched
  * unless already unpacked, then the runtime validated and saved. Also the
  * requests built on an install: the newest release of a tool, an upgrade of a
  * latest runtime, and a re-pairing with another TheRock build.
  */
final private[runtime] class RuntimeInstalls(
    storage: StorageService,
    catalog: RuntimeCatalog,
    runtimesRoot: Path,
    pairing: TheRockPairing,
    validation: RuntimeValidation,
    cleanup: RuntimeCleanup,
    archives: RuntimeArchives,
    runnerFiles: RunnerFiles,
    background: Background
) {
  private val logger = Logger[RuntimeInstalls]

  private val entries = ConcurrentHashMap[String, RuntimeInstallEntry]()

  /** What runs once an install validates, before its job reads `Completed`
    * (`RuntimeInstallResolution`): whoever sees the job complete sees its
    * effects too. Dropped when the install fails.
    */
  private val onValid = ConcurrentHashMap[String, () => Unit]()

  /** Runs `action` when the running install of `runtimeId` validates; answers
    * false when none is running, leaving the caller to act on the outcome it
    * can read.
    */
  def whenValid(runtimeId: String)(action: () => Unit): Boolean = {
    onValid.put(runtimeId, action)
    val running =
      Option(entries.get(runtimeId)).exists(_.job.state.isActive)
    if (!running) onValid.remove(runtimeId, action)
    running
  }

  // Two at a time, the same figure the model downloads use
  // (`Downloader.MaxTransfersPerHost`) - a runtime install is a download like
  // any other and had no business being the one thing that queued
  // (François, 2026-09-10). Bandwidth needs no separate rule here: installs
  // fetch through that same `Downloader`, so they share its per-host permits
  // with everything else drift is pulling.
  private val installs = WorkQueue(
    background,
    "drift-runtime-install",
    RuntimeManager.MaxConcurrentInstalls
  )

  /** Serialises anything that writes the same directory. Two installs may now
    * run together, and two of them wanting the same files is the ordinary case
    * rather than a corner: a pinned runtime and a `latest` one can resolve to
    * one release directory, and two ROCm runtimes - sd-cpp and llama.cpp -
    * routinely want the same TheRock build. The lock is held across the
    * "already unpacked?" check as well as the fetch, so the second install
    * still sees the first one's work and skips it instead of racing it.
    */
  private val destinationLocks = ConcurrentHashMap[String, Object]()

  private def atDestination[A](target: Path)(body: => A): A =
    destinationLocks
      .computeIfAbsent(target.toAbsolutePath.toString, _ => Object())
      .synchronized(body)

  def list: List[RuntimeInstallJob] =
    entries.values.asScala.map(_.job).toList.sortBy(_.runtimeId)

  def cancel(runtimeId: String): RuntimeInstallJob =
    Option(entries.get(runtimeId)) match {
      case None =>
        RuntimeInstallJob(
          runtimeId,
          RuntimeInstallState.Failed,
          error = Some("no install for this runtime")
        )
      case Some(entry) =>
        entry.cancelled.set(true)
        if (entry.job.state == RuntimeInstallState.Queued)
          entry.job = entry.job.copy(
            state = RuntimeInstallState.Cancelled,
            completedAt = Some(System.currentTimeMillis())
          )
        entry.job
    }

  def install(request: InstallRuntimeRequest): RuntimeInstallJob =
    install(request, RuntimeManager.runtimeId(request), tracksLatest = false)

  private def install(
      request: InstallRuntimeRequest,
      id: String,
      tracksLatest: Boolean
  ): RuntimeInstallJob =
    synchronized {
      Option(entries.get(id)).map(_.job).filter(_.state.isActive) match {
        case Some(active) => active
        case None         =>
          val entry = RuntimeInstallEntry(
            RuntimeInstallJob(id, RuntimeInstallState.Queued)
          )
          entries.put(id, entry)
          installs.submit(install(id, request, entry, tracksLatest))
          entry.job
      }
    }

  /** Installs drift's own runner (`specs/43`): both its runtimes, each an
    * install whose files come from the jar drift ships instead of a download,
    * paired with TheRock as a ROCm release is, against the ROCm the kernels
    * were built with. The two share the TheRock build.
    */
  def installRunner(theRockVersion: Option[String]): List[RuntimeInstallJob] = {
    val offer = runnerFiles.offer
    RuntimeTool.values.toList.map { tool =>
      val id = RunnerFiles.id(tool)
      if (!offer.available)
        RuntimeInstallJob(
          id,
          RuntimeInstallState.Failed,
          error = offer.reason,
          completedAt = Some(System.currentTimeMillis())
        )
      else
        install(runnerRequest(tool, theRockVersion), id, tracksLatest = false)
    }
  }

  /** An install of the runner for `tool`: no release to fetch, its ROCm the
    * kernels'.
    */
  private def runnerRequest(
      tool: RuntimeTool,
      theRockVersion: Option[String]
  ): InstallRuntimeRequest =
    InstallRuntimeRequest(
      tool = tool,
      releaseTag = Runtime.DriftRunnerTag,
      asset = RuntimeReleaseAsset(
        name = "",
        sizeBytes = 0L,
        downloadUrl = "",
        sha256 = None,
        backend = RuntimeBackend.Rocm,
        rocmVersion = runnerFiles.rocmVersion
      ),
      gfxTarget = Some(RunnerFiles.Gfx),
      theRockVersion = theRockVersion,
      label = Some(RunnerFiles.label(tool))
    )

  /** Installs — or, on an existing latest runtime, upgrades — the newest
    * release of a tool for a backend, under the stable id `latest-<backend>`
    * (`llama-latest-<backend>`). Already on the newest tag: a terminal
    * `Completed` job saying so, not a reinstall.
    */
  def installLatest(
      tool: RuntimeTool,
      backend: RuntimeBackend,
      gfxTarget: Option[String],
      theRockVersion: Option[String]
  ): RuntimeInstallJob =
    synchronized {
      val id = RuntimeManager.latestId(tool, backend)
      resolveNewestRequest(tool, backend, gfxTarget, theRockVersion) match {
        case Left(reason) =>
          val job = RuntimeInstallJob(
            id,
            RuntimeInstallState.Failed,
            error = Some(reason),
            completedAt = Some(System.currentTimeMillis())
          )
          // Record it so the UI shows the refusal like any settled job.
          entries.put(id, RuntimeInstallEntry(job))
          job
        case Right(request) =>
          storage.get[Runtime]("runtimes", id) match {
            case Some(existing)
                if existing.valid &&
                  existing.releaseTag == request.releaseTag &&
                  // An explicit different TheRock pick means the user wants to
                  // re-pair, so it is not a no-op even on the same release.
                  theRockVersion.forall(existing.theRockVersion.contains) &&
                  RuntimeManager
                    .executableIn(Paths.get(existing.installedAt), tool)
                    .isDefined =>
              RuntimeInstallJob(
                id,
                RuntimeInstallState.Completed,
                step = s"already on the latest release (${request.releaseTag})",
                completedAt = Some(System.currentTimeMillis())
              )
            case _ => install(request, id, tracksLatest = true)
          }
      }
    }

  /** Upgrades a latest-tracking runtime by re-resolving the newest release for
    * its tool and backend. Refuses a runtime that is pinned or unknown.
    *
    * The ROCm pairing follows the spec's rule: when the newest release declares
    * the same ROCm version the runtime already has, the current TheRock build
    * is kept — a deliberate pick included; when it declares another, the
    * request's pick applies, or the declared version resolves as on install.
    */
  def upgrade(id: String, theRockVersion: Option[String]): RuntimeInstallJob =
    storage.get[Runtime]("runtimes", id) match {
      case None =>
        RuntimeInstallJob(
          id,
          RuntimeInstallState.Failed,
          error = Some(s"runtime '$id' is not registered"),
          completedAt = Some(System.currentTimeMillis())
        )
      case Some(runtime) if !runtime.tracksLatest =>
        RuntimeInstallJob(
          id,
          RuntimeInstallState.Failed,
          error = Some(
            s"runtime '${runtime.label}' is pinned to ${runtime.releaseTag}; only a latest runtime upgrades"
          ),
          completedAt = Some(System.currentTimeMillis())
        )
      case Some(runtime) =>
        val declaredNow = newestAsset(runtime.tool, runtime.backend)
          .flatMap(_._2.rocmVersion)
        val declaredUnchanged = declaredNow.exists(now =>
          runtime.rocmVersion.exists(RuntimeManager.sameRocm(_, now))
        )
        val pairing =
          if (declaredUnchanged) runtime.theRockVersion else theRockVersion
        installLatest(runtime.tool, runtime.backend, runtime.gfxTarget, pairing)
    }

  /** Re-pairs an installed ROCm runtime with another TheRock build, keeping the
    * release on disk: the install job finds the executable already there and
    * only fetches the TheRock tarball if needed, then re-validates and drops
    * the previous TheRock directory when nothing else references it. Refuses
    * non-ROCm runtimes and adopted ones (their ROCm directory is not drift's to
    * swap).
    */
  def changeTheRock(id: String, theRockVersion: String): RuntimeInstallJob = {
    def refuse(reason: String) =
      RuntimeInstallJob(
        id,
        RuntimeInstallState.Failed,
        error = Some(reason),
        completedAt = Some(System.currentTimeMillis())
      )
    storage.get[Runtime]("runtimes", id) match {
      case None => refuse(s"runtime '$id' is not registered")
      case Some(runtime) if runtime.backend != RuntimeBackend.Rocm =>
        refuse(s"runtime '${runtime.label}' is not a ROCm build")
      case Some(runtime) if runtime.adopted =>
        refuse(
          s"runtime '${runtime.label}' is adopted; its ROCm directory is not drift's to change"
        )
      case Some(runtime) if runtime.engine == RuntimeEngine.DriftRunner =>
        install(
          runnerRequest(runtime.tool, Some(theRockVersion)),
          id,
          tracksLatest = false
        )
      case Some(runtime) =>
        // The catalog's asset for this tag, so a missing directory can still be
        // fetched; else a placeholder that carries only what the pairing needs.
        val asset = catalog
          .releases(runtime.tool)
          .find(_.tag == runtime.releaseTag)
          .flatMap(_.assets.find(_.backend == RuntimeBackend.Rocm))
          .getOrElse(
            RuntimeReleaseAsset(
              name = "",
              sizeBytes = 0L,
              downloadUrl = "",
              sha256 = None,
              backend = RuntimeBackend.Rocm,
              rocmVersion = runtime.rocmVersion
            )
          )
        val installDir = Paths.get(runtime.installedAt)
        if (
          asset.downloadUrl.isEmpty &&
          RuntimeManager.executableIn(installDir, runtime.tool).isEmpty
        )
          refuse(
            s"runtime '${runtime.label}': ${runtime.tool.executableName} is missing under ${runtime.installedAt} and release ${runtime.releaseTag} is no longer listed; reinstall it"
          )
        else
          install(
            InstallRuntimeRequest(
              tool = runtime.tool,
              releaseTag = runtime.releaseTag,
              asset = asset,
              gfxTarget = runtime.gfxTarget,
              theRockVersion = Some(theRockVersion),
              label = Some(runtime.label)
            ),
            id,
            tracksLatest = runtime.tracksLatest
          )
    }
  }

  /** The newest release carrying an asset for this tool and backend. Releases
    * come back newest first, so the first match is the newest.
    */
  private def newestAsset(
      tool: RuntimeTool,
      backend: RuntimeBackend
  ): Option[(String, RuntimeReleaseAsset)] =
    catalog
      .releases(tool)
      .iterator
      .flatMap(release =>
        release.assets
          .find(_.backend == backend)
          .map(asset => (release.tag, asset))
      )
      .nextOption()

  private def resolveNewestRequest(
      tool: RuntimeTool,
      backend: RuntimeBackend,
      gfxTarget: Option[String],
      theRockVersion: Option[String]
  ): Either[String, InstallRuntimeRequest] = {
    val backendName = backend.toString.toLowerCase
    newestAsset(tool, backend)
      .toRight(
        s"no $backendName build of ${tool.displayName} is published in the latest releases (or GitHub could not be listed)"
      )
      .map((tag, asset) =>
        InstallRuntimeRequest(
          tool = tool,
          releaseTag = tag,
          asset = asset,
          gfxTarget = Option.when(backend == RuntimeBackend.Rocm)(
            gfxTarget.getOrElse(RuntimeManager.DefaultGfxTarget)
          ),
          theRockVersion = Option
            .when(backend == RuntimeBackend.Rocm)(theRockVersion)
            .flatten,
          label = Some(s"latest ${tool.displayName} $backendName")
        )
      )
  }

  private def install(
      id: String,
      request: InstallRuntimeRequest,
      entry: RuntimeInstallEntry,
      tracksLatest: Boolean
  ): Unit =
    try {
      if (entry.cancelled.get()) return
      // Captured before the entity is overwritten: an upgrade moves the id to
      // a new tag's directory, and the old one is cleaned up afterwards.
      val previous = storage.get[Runtime]("runtimes", id)
      entry.job = entry.job.copy(
        state = RuntimeInstallState.Downloading,
        startedAt = Some(System.currentTimeMillis())
      )

      val tool = request.tool
      val backendName = request.asset.backend.toString.toLowerCase
      val installDir =
        runtimesRoot
          .resolve(RuntimeManager.toolDirectory(tool))
          .resolve(s"${request.releaseTag}-$backendName")
      // drift's own runner: its files are written once TheRock is there
      val runner = request.releaseTag == Runtime.DriftRunnerTag

      // Resolve the TheRock pairing before any bytes move: if the bucket
      // has no matching build there is no point downloading the release.
      val theRockBuild: Option[TheRockBuild] =
        if (request.asset.backend == RuntimeBackend.Rocm) {
          val gfx =
            request.gfxTarget.getOrElse(RuntimeManager.DefaultGfxTarget)
          entry.job = entry.job.copy(
            step =
              s"matching a TheRock build for ROCm ${request.asset.rocmVersion
                  .getOrElse("?")} on $gfx"
          )
          pairing.buildFor(
            request.asset,
            gfx,
            request.theRockVersion
          ) match {
            case Right(build) => Some(build)
            case Left(reason) =>
              entry.job = entry.job.copy(
                state = RuntimeInstallState.Failed,
                error = Some(reason),
                completedAt = Some(System.currentTimeMillis())
              )
              return
          }
        } else None

      // Fetch and unpack the release, unless a previous install already left
      // a working directory — the entity may have been deleted while the
      // files were kept, or this is a re-pairing of the same release.
      val releaseReady = atDestination(installDir) {
        if (runner || RuntimeManager.executableIn(installDir, tool).nonEmpty)
          true
        else
          archives.fetchAndUnpack(
            entry,
            step = s"${tool.displayName} ${request.releaseTag} ($backendName)",
            url = request.asset.downloadUrl,
            archiveName = request.asset.name,
            sha256 = request.asset.sha256,
            target = installDir
          )
      }
      if (!releaseReady) return

      // The ROCm runtime, shared between releases wanting the same build,
      // so an already-unpacked one is simply reused.
      val theRockDir = theRockBuild.map { build =>
        runtimesRoot
          .resolve("therock")
          .resolve(s"${build.gfxTarget}-${build.version}")
      }
      val theRockReady =
        theRockBuild.zip(theRockDir).forall { (build, target) =>
          atDestination(target) {
            if (Files.isDirectory(target)) true
            else
              archives.fetchAndUnpack(
                entry,
                step = s"ROCm ${build.version} (${build.gfxTarget})",
                url = build.downloadUrl,
                archiveName =
                  s"therock-dist-linux-${build.gfxTarget}-${build.version}.tar.gz",
                // The bucket publishes no content hash (its ETags are
                // multipart), so this transfer goes unverified.
                sha256 = None,
                target = target
              )
          }
        }
      if (!theRockReady) return
      if (runner)
        theRockDir.foreach(rock =>
          atDestination(installDir)(runnerFiles.write(installDir, tool, rock))
        )

      if (entry.cancelled.get()) {
        entry.job = entry.job.copy(
          state = RuntimeInstallState.Cancelled,
          completedAt = Some(System.currentTimeMillis())
        )
        return
      }

      entry.job = entry.job.copy(
        state = RuntimeInstallState.Validating,
        step =
          s"${tool.executableName} ${RuntimeManager.validationFlags(tool).head}"
      )
      val runtime = Runtime(
        id = id,
        label = request.label.getOrElse(
          s"${tool.displayName} ${request.releaseTag} $backendName"
        ),
        tool = tool,
        backend = request.asset.backend,
        releaseTag = request.releaseTag,
        rocmVersion = request.asset.rocmVersion,
        gfxTarget = theRockBuild.map(_.gfxTarget),
        theRockVersion = theRockBuild.map(_.version),
        installedAt = installDir.toString,
        theRockPath = theRockDir.map(_.toString),
        // the runner is drift's own, whatever registered it before
        adopted = !runner && previous.exists(_.adopted),
        tracksLatest = tracksLatest,
        createdAt =
          previous.map(_.createdAt).getOrElse(System.currentTimeMillis()),
        modelKinds = None
      )
      val validated = validation.validateAndSave(runtime)
      // Only once the replacement proves itself: an upgrade that fails
      // validation must not delete the working files it was replacing.
      if (validated.valid)
        previous.foreach(cleanup.cleanupSuperseded(_, validated))
      Option(onValid.remove(id)).filter(_ => validated.valid).foreach { act =>
        try act()
        catch {
          case NonFatal(err) =>
            logger.warn(s"After installing $id: ${err.getMessage}")
        }
      }
      entry.job = entry.job.copy(
        state =
          if (validated.valid) RuntimeInstallState.Completed
          else RuntimeInstallState.Failed,
        error = validated.validationError,
        completedAt = Some(System.currentTimeMillis())
      )
    } catch {
      case NonFatal(err) =>
        logger.warn(s"Install of $id blew up", err)
        entry.job = entry.job.copy(
          state = RuntimeInstallState.Failed,
          error = Some(Option(err.getMessage).getOrElse(err.toString)),
          completedAt = Some(System.currentTimeMillis())
        )
    }
}
