package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}
import sttp.tapir.*
import sttp.tapir.generic.auto.*
import sttp.tapir.json.jsoniter.*

/** Which program a runtime is a build of (`specs/17-assistant-runtime.md`):
  * sd-cpp's `sd-server` for generation, llama.cpp's `llama-server` for the
  * prompt assistant. Same entity, same install and pairing code; only the
  * release source, the executable name and the validation command differ.
  */
enum RuntimeTool derives CanEqual {
  case SdCpp, LlamaCpp

  /** What the tool is called in labels and messages. */
  def displayName: String = this match {
    case SdCpp    => "sd-cpp"
    case LlamaCpp => "llama.cpp"
  }

  def executableName: String = this match {
    case SdCpp    => "sd-server"
    case LlamaCpp => "llama-server"
  }

  /** Prefix keeping the two tools' runtime ids apart (`<tag>-<backend>` for
    * sd-cpp, `llama-<tag>-<backend>` for llama.cpp).
    */
  def idPrefix: String = this match {
    case SdCpp    => ""
    case LlamaCpp => "llama-"
  }
}
object RuntimeTool {
  given Schema[RuntimeTool] =
    Schema.derivedEnumeration[RuntimeTool].defaultStringBased
  // As a query parameter (`?tool=LlamaCpp`), the same plain name.
  given Codec[String, RuntimeTool, CodecFormat.TextPlain] =
    Codec.derivedEnumeration[String, RuntimeTool].defaultStringBased
}

/** What does a runtime's work (`specs/43-runners-per-architecture.md`): an
  * upstream program, or drift's own runner, which speaks both tools' APIs. A
  * runtime's engine is derived (`Runtime.engine`); an architecture lists the
  * engines that run it, and a run configuration names the one it runs on.
  */
enum RuntimeEngine derives CanEqual {
  case SdCpp, LlamaCpp, DriftRunner

  def displayName: String = this match {
    case SdCpp       => "sd-cpp"
    case LlamaCpp    => "llama.cpp"
    case DriftRunner => "drift runner"
  }
}
object RuntimeEngine {

  /** The upstream engine of a tool. */
  def upstream(tool: RuntimeTool): RuntimeEngine = tool match {
    case RuntimeTool.SdCpp    => SdCpp
    case RuntimeTool.LlamaCpp => LlamaCpp
  }

  given Schema[RuntimeEngine] =
    Schema.derivedEnumeration[RuntimeEngine].defaultStringBased
  given JsonValueCodec[List[RuntimeEngine]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** ROCm version arithmetic shared by the pairing (backend) and its display
  * (frontend): whether a TheRock build satisfies what a release declares.
  */
object RocmVersions {
  private val Pattern = """(\d+)\.(\d+)(?:\.(\d+))?.*""".r

  /** (major, minor, patch) of "7.14.0", "7.14.0a20260612" or "10.0" (patch 0
    * when absent); (0, 0, 0) for anything else.
    */
  def components(version: String): (Int, Int, Int) = version.trim match {
    case Pattern(major, minor, patch) =>
      (major.toInt, minor.toInt, Option(patch).map(_.toInt).getOrElse(0))
    case _ => (0, 0, 0)
  }

  /** By triple when the declaration has three components ("7.14.0" — build
    * stamps ignored), by major.minor when it has two ("10.0", llama.cpp's
    * naming, meaning any 10.0.x).
    */
  def satisfies(declared: String, candidate: String): Boolean = {
    val wanted = components(declared)
    val found = components(candidate)
    if (declared.trim.count(_ == '.') == 1)
      wanted._1 == found._1 && wanted._2 == found._2
    else wanted == found
  }
}

/** Which build flavour a runtime is (`specs/06-sdcpp-runtime.md`). */
enum RuntimeBackend derives CanEqual {
  case Rocm, Vulkan, Cpu
}
object RuntimeBackend {
  // jsoniter encodes singleton-case enums as plain strings (see the
  // discriminator notes below); the tapir schema must agree — and auto
  // derivation cannot reach an enum nested two products deep anyway.
  given Schema[RuntimeBackend] =
    Schema.derivedEnumeration[RuntimeBackend].defaultStringBased
}

/** One installed runtime — of either tool: a binary, the libraries it links
  * against, and the environment that lets it find them. The executable and
  * `LD_LIBRARY_PATH` are derived from `installedAt` / `theRockPath` by the
  * backend rather than stored — the archive layout differs per backend (the
  * ROCm zip unpacks to `build/bin/`, the Vulkan and CPU zips unpack flat).
  *
  * Whether the ROCm pairing is the one the release declared or a deliberate
  * pick is not stored: it is [[theRockVersion]] matching [[rocmVersion]] or
  * not, which the UI derives.
  */
case class Runtime(
    id: String,
    label: String,
    tool: RuntimeTool,
    backend: RuntimeBackend,
    /** The release tag, e.g. "master-841-6b3edaa" (sd-cpp) or "b10844"
      * (llama.cpp).
      */
    releaseTag: String,
    /** The ROCm version the release was built against (from the asset name,
      * e.g. "7.14.0"). Rocm only.
      */
    rocmVersion: Option[String] = None,
    /** e.g. "gfx1151" (Strix Halo). Rocm only. */
    gfxTarget: Option[String] = None,
    /** The TheRock build actually installed, e.g. "7.14.0a20260612" — the
      * newest bucket build matching [[rocmVersion]], resolved by the install.
      */
    theRockVersion: Option[String] = None,
    /** Directory holding the unpacked release. */
    installedAt: String,
    /** Directory holding the unpacked TheRock distribution; its `lib` goes on
      * `LD_LIBRARY_PATH`. Rocm only.
      */
    theRockPath: Option[String] = None,
    /** Registered from an existing directory rather than downloaded — drift
      * does not own the files and never deletes them.
      */
    adopted: Boolean = false,
    /** This runtime follows the newest release of its tool for its backend
      * rather than a pinned tag: its id is stable (`latest-<backend>`,
      * `llama-latest-<backend>`) while [[releaseTag]] moves as it is upgraded.
      * "Update available" is decided by comparing [[releaseTag]] against the
      * newest release the catalog offers for [[tool]] and [[backend]].
      */
    tracksLatest: Boolean = false,
    /** The executable ran successfully under the composed environment. */
    valid: Boolean = false,
    validationError: Option[String] = None,
    /** The banner the executable printed, e.g. "stable-diffusion.cpp version
      * unknown, commit 6b3edaa" or "version: 10844 (…)".
      */
    reportedVersion: Option[String] = None,
    createdAt: Long = 0L,
    /** The model kinds it runs, as its validation reported them (`specs/43`):
      * the drift runner's; none for an upstream build, which runs whatever its
      * architectures say.
      */
    modelKinds: Option[List[String]]
) {

  /** What does the work: the drift runner by its release tag, else the tool's
    * upstream program.
    */
  def engine: RuntimeEngine =
    if (releaseTag == Runtime.DriftRunnerTag) RuntimeEngine.DriftRunner
    else RuntimeEngine.upstream(tool)
}
object Runtime {

  /** The release tag of drift's own runner (installed from the runtimes page,
    * `specs/43`): its image runtime speaks sd-server's API but is no sd-cpp
    * build, so what it can do is known by this tag.
    */
  val DriftRunnerTag: String = "drift-runner"
  // No discriminator: `RuntimeBackend` has only singleton cases, so it encodes
  // as a plain string ("Rocm"). The config must be inline -- the macro needs a
  // constant expression.
  given JsonValueCodec[Runtime] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[Runtime]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[Option[Runtime]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** Which runtime a launch uses when nothing pins one — one default per tool, so
  * switching the assistant's never touches sd-cpp's.
  */
case class RuntimeSelection(
    defaultRuntimeId: Option[String] = None,
    defaultAssistantRuntimeId: Option[String] = None
) {
  def defaultFor(tool: RuntimeTool): Option[String] = tool match {
    case RuntimeTool.SdCpp    => defaultRuntimeId
    case RuntimeTool.LlamaCpp => defaultAssistantRuntimeId
  }
  def withDefault(tool: RuntimeTool, id: Option[String]): RuntimeSelection =
    tool match {
      case RuntimeTool.SdCpp    => copy(defaultRuntimeId = id)
      case RuntimeTool.LlamaCpp => copy(defaultAssistantRuntimeId = id)
    }
}
object RuntimeSelection {
  given JsonValueCodec[RuntimeSelection] = JsonCodecMaker.make
}

/** Where a TheRock ROCm build comes from, in order of how much drift trusts it
  * (`specs/06-sdcpp-runtime.md`). A `Stable` build is a tagged AMD release; the
  * others are dated builds off the respective branch.
  */
enum TheRockChannel derives CanEqual {
  case Stable, ReleaseCandidate, Nightly

  /** Preference order for automatic resolution: a tagged stable release beats a
    * release candidate beats a nightly.
    */
  def rank: Int = this match {
    case Stable           => 0
    case ReleaseCandidate => 1
    case Nightly          => 2
  }
}
object TheRockChannel {
  given Schema[TheRockChannel] =
    Schema.derivedEnumeration[TheRockChannel].defaultStringBased
}

/** One TheRock build that could satisfy a runtime's ROCm requirement: which
  * channel it is from, its full version (e.g. "10.0.0" tagged, or
  * "7.14.0a20260612" dated), whether it is a tagged release, and whether drift
  * already has it unpacked on disk.
  */
case class TheRockChoice(
    channel: TheRockChannel,
    version: String,
    tagged: Boolean,
    onDisk: Boolean
)

/** What drift would pair with a ROCm sd-cpp build for a gfx target: the version
  * the build asks for, the choice drift prefers (already-on-disk first, then
  * the most stable channel with a match), and every matching build across
  * channels so the UI can show the options and let the user override.
  */
case class TheRockResolution(
    gfxTarget: String,
    neededVersion: String,
    chosen: Option[TheRockChoice],
    alternatives: List[TheRockChoice]
)
object TheRockResolution {
  given JsonValueCodec[TheRockResolution] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** The job states of an install, mirroring [[DownloadState]] but honest about
  * the extra phases: a runtime is fetched, unpacked *and* proven to run before
  * it is done.
  */
enum RuntimeInstallState derives CanEqual {
  case Queued, Downloading, Unpacking, Validating, Completed, Failed, Cancelled

  def isActive: Boolean = this match {
    case Queued | Downloading | Unpacking | Validating => true
    case _                                             => false
  }
}
object RuntimeInstallState {
  given Schema[RuntimeInstallState] =
    Schema.derivedEnumeration[RuntimeInstallState].defaultStringBased
}

/** One runtime install, keyed by the runtime it produces. `step` names what is
  * currently being fetched or unpacked — the sd-cpp archive and the TheRock
  * tarball are two separate transfers under one job.
  */
case class RuntimeInstallJob(
    runtimeId: String,
    state: RuntimeInstallState,
    step: String = "",
    downloadedBytes: Long = 0L,
    totalBytes: Option[Long] = None,
    error: Option[String] = None,
    startedAt: Option[Long] = None,
    completedAt: Option[Long] = None
)
object RuntimeInstallJob {
  given JsonValueCodec[RuntimeInstallJob] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
  given JsonValueCodec[List[RuntimeInstallJob]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** One Linux asset of a release, already classified by backend. `sha256` comes
  * from the GitHub API's asset digest and lets the download be verified like
  * any weight.
  */
case class RuntimeReleaseAsset(
    name: String,
    sizeBytes: Long,
    downloadUrl: String,
    sha256: Option[String],
    backend: RuntimeBackend,
    /** From the asset name, e.g. "7.14.0" (sd-cpp) or "10.0" (llama.cpp names
      * only major.minor). Rocm only.
      */
    rocmVersion: Option[String] = None
)

case class RuntimeRelease(
    tool: RuntimeTool,
    tag: String,
    publishedAt: String,
    assets: List[RuntimeReleaseAsset]
)
object RuntimeRelease {
  given JsonValueCodec[List[RuntimeRelease]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** What to install: a chosen release asset, nothing more. For Rocm the matching
  * TheRock build is resolved server-side from the ROCm version the asset name
  * carries — the whole point of the pairing being drift's job.
  */
case class InstallRuntimeRequest(
    tool: RuntimeTool,
    releaseTag: String,
    asset: RuntimeReleaseAsset,
    /** Rocm only; defaults to the target machine's gfx1151. */
    gfxTarget: Option[String] = None,
    /** Rocm only. An explicit TheRock version to pair, overriding automatic
      * resolution — the "move to a less-stable build on purpose" escape hatch.
      * Empty means drift resolves it: already-on-disk first, then the most
      * stable channel that matches the asset's ROCm version.
      */
    theRockVersion: Option[String] = None,
    label: Option[String] = None
)
object InstallRuntimeRequest {
  given JsonValueCodec[InstallRuntimeRequest] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** Install (or, on an existing latest runtime, upgrade) the newest release of a
  * tool for a backend. drift resolves which release that is — the whole point
  * of "latest" is not naming a tag — and for Rocm pairs the matching TheRock
  * build as always.
  */
case class InstallLatestRequest(
    tool: RuntimeTool,
    backend: RuntimeBackend,
    /** Rocm only; defaults to the target machine's gfx1151. */
    gfxTarget: Option[String] = None,
    /** Rocm only. An explicit TheRock version to pair; empty resolves it the
      * same way a pinned install does (a tagged stable release matching the
      * newest sd-cpp release's ROCm version).
      */
    theRockVersion: Option[String] = None
)
object InstallLatestRequest {
  given JsonValueCodec[InstallLatestRequest] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** A build drift offers to install where a launch finds no runtime of its
  * engine (`specs/46-starter-configurations.md`): each upstream backend the
  * machine can run, installed as any latest-tracking runtime is, and drift's
  * own runner where this drift carries it for the machine's GPU. The user
  * picks; `recommended` marks the preselected one — ROCm for sd-cpp on an AMD
  * GPU the ROCm driver exposes, Vulkan otherwise.
  */
case class RuntimeInstallOption(
    tool: RuntimeTool,
    engine: RuntimeEngine,
    /** The upstream install; none for the drift runner, which installs both its
      * runtimes from the jar drift ships (`specs/43`).
      */
    request: Option[InstallLatestRequest],
    runtimeId: String,
    /** What the choice names: "sd-cpp on ROCm (gfx1151)". */
    label: String,
    recommended: Boolean
)
object RuntimeInstallOption {
  given JsonValueCodec[List[RuntimeInstallOption]] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** Install `option` for configurations that could not launch, and move those of
  * them whose architecture runs on its engine to it once it validates
  * (`specs/46-starter-configurations.md`).
  */
case class InstallForConfigurationsRequest(
    option: RuntimeInstallOption,
    configurationIds: List[String]
)
object InstallForConfigurationsRequest {
  given JsonValueCodec[InstallForConfigurationsRequest] =
    JsonCodecMaker.make(CodecMakerConfig.withDiscriminatorFieldName(None))
}

/** Install drift's own runner (`specs/43`): both its runtimes, chat and images,
  * on one TheRock build; empty pairs the one matching the ROCm its kernels were
  * built with.
  */
case class InstallRunnerRequest(theRockVersion: Option[String])
object InstallRunnerRequest {
  given JsonValueCodec[InstallRunnerRequest] = JsonCodecMaker.make
}

/** Whether this drift carries a runner to install, and the ROCm version its
  * kernels were compiled with (what the TheRock build is paired against);
  * `reason` says why not.
  */
case class RunnerOffer(
    available: Boolean,
    rocmVersion: Option[String],
    reason: Option[String]
)
object RunnerOffer {
  given JsonValueCodec[RunnerOffer] = JsonCodecMaker.make
}

/** Upgrade a latest runtime. `theRockVersion` matters only when the newest
  * release declares a ROCm version different from the one the runtime is paired
  * with: then it is the user's pick from the same select an install shows, or
  * empty for the declared version. When the declared version is unchanged the
  * current pairing is kept regardless.
  */
case class UpgradeRuntimeRequest(theRockVersion: Option[String] = None)
object UpgradeRuntimeRequest {
  given JsonValueCodec[UpgradeRuntimeRequest] = JsonCodecMaker.make
}

/** Re-pair an installed ROCm runtime with another TheRock build, keeping the
  * release already on disk. The version comes from the same select an install
  * shows (`specs/17-assistant-runtime.md`).
  */
case class ChangeTheRockRequest(theRockVersion: String)
object ChangeTheRockRequest {
  given JsonValueCodec[ChangeTheRockRequest] = JsonCodecMaker.make
}

// `JsonValueCodec[List[String]]` is declared once for the package, in
// Api.scala; a second anonymous given of the same shape clashes by name and
// makes every `jsonBody[Boolean]` in the package ambiguous.

// `base` in Api.scala is private to that file (tracked as bug 16); the same
// prefix is rebuilt here so these endpoints live under /api like the rest.
private val runtimesBase = endpoint.in("api")

val listRuntimes: PublicEndpoint[Unit, Unit, List[Runtime], Any] =
  runtimesBase.get.in("runtimes").out(jsonBody[List[Runtime]])

val getRuntime: PublicEndpoint[String, Unit, Option[Runtime], Any] =
  runtimesBase.get.in("runtimes" / path[String]).out(jsonBody[Option[Runtime]])

/** Registers a runtime entity as given — the "adopt an existing directory"
  * path. Installed runtimes are created by their install job instead.
  */
val createRuntime: PublicEndpoint[Runtime, Unit, Runtime, Any] =
  runtimesBase.post.in("runtimes").in(jsonBody[Runtime]).out(jsonBody[Runtime])

val updateRuntime
    : PublicEndpoint[(String, Runtime), Unit, Option[Runtime], Any] =
  runtimesBase.put
    .in("runtimes" / path[String])
    .in(jsonBody[Runtime])
    .out(jsonBody[Option[Runtime]])

/** Deletes the registration, and — for a runtime drift installed itself — the
  * files under drift's runtimes directory. Adopted directories are never
  * touched.
  */
val deleteRuntime: PublicEndpoint[String, Unit, Boolean, Any] =
  runtimesBase.delete.in("runtimes" / path[String]).out(jsonBody[Boolean])

/** Re-runs the executable under the runtime's composed environment and saves
  * the outcome. The only honest check: it proves the binary, the libraries and
  * the environment agree.
  */
val validateRuntime: PublicEndpoint[String, Unit, Option[Runtime], Any] =
  runtimesBase.post
    .in("runtimes" / path[String] / "validate")
    .out(jsonBody[Option[Runtime]])

val getRuntimeSelection: PublicEndpoint[Unit, Unit, RuntimeSelection, Any] =
  runtimesBase.get.in("runtime-selection").out(jsonBody[RuntimeSelection])

val setRuntimeSelection
    : PublicEndpoint[RuntimeSelection, Unit, RuntimeSelection, Any] =
  runtimesBase.put
    .in("runtime-selection")
    .in(jsonBody[RuntimeSelection])
    .out(jsonBody[RuntimeSelection])

/** Available releases of a tool from GitHub, newest first, so the UI can offer
  * them rather than making the user paste a tag.
  */
val listRuntimeReleases
    : PublicEndpoint[RuntimeTool, Unit, List[RuntimeRelease], Any] =
  runtimesBase.get
    .in("runtime-releases")
    .in(query[RuntimeTool]("tool"))
    .out(jsonBody[List[RuntimeRelease]])

/** The gfx targets TheRock publishes builds for, e.g. "gfx1151". */
val listTheRockTargets: PublicEndpoint[Unit, Unit, List[String], Any] =
  runtimesBase.get.in("therock" / "targets").out(jsonBody[List[String]])

/** For a gfx target and the ROCm version a sd-cpp build needs, what TheRock
  * build drift would pair — preferring an already-downloaded match, then the
  * most stable channel — and every matching build across channels, so the UI
  * can show the choice and let the user pick a different one.
  */
val resolveTheRock
    : PublicEndpoint[(String, String), Unit, TheRockResolution, Any] =
  runtimesBase.get
    .in("therock" / "resolve")
    .in(query[String]("gfx"))
    .in(query[String]("rocm"))
    .out(jsonBody[TheRockResolution])

val installRuntime
    : PublicEndpoint[InstallRuntimeRequest, Unit, RuntimeInstallJob, Any] =
  runtimesBase.post
    .in("runtime-installs")
    .in(jsonBody[InstallRuntimeRequest])
    .out(jsonBody[RuntimeInstallJob])

/** The builds offered per tool (`RuntimeInstallOption`). */
val runtimeInstallOptions
    : PublicEndpoint[Unit, Unit, List[RuntimeInstallOption], Any] =
  runtimesBase.get
    .in("runtime-installs" / "options")
    .out(jsonBody[List[RuntimeInstallOption]])

val listRuntimeInstalls
    : PublicEndpoint[Unit, Unit, List[RuntimeInstallJob], Any] =
  runtimesBase.get.in("runtime-installs").out(jsonBody[List[RuntimeInstallJob]])

val cancelRuntimeInstall: PublicEndpoint[String, Unit, RuntimeInstallJob, Any] =
  runtimesBase.post
    .in("runtime-installs" / path[String]("runtimeId") / "cancel")
    .out(jsonBody[RuntimeInstallJob])

/** Installs the newest release for a backend under the stable id
  * `latest-<backend>`, or attaches to the install already in flight for it.
  * When that runtime is already on the newest release, answers with a terminal
  * `Completed` job saying so rather than reinstalling — the same "no-op returns
  * a terminal job" shape [[startDownload]] uses.
  */
val installLatestRuntime
    : PublicEndpoint[InstallLatestRequest, Unit, RuntimeInstallJob, Any] =
  runtimesBase.post
    .in("runtime-installs" / "latest")
    .in(jsonBody[InstallLatestRequest])
    .out(jsonBody[RuntimeInstallJob])

/** Whether drift's own runner can be installed (`specs/43`). */
val runnerOffer: PublicEndpoint[Unit, Unit, RunnerOffer, Any] =
  runtimesBase.get.in("runtime-installs" / "runner").out(jsonBody[RunnerOffer])

/** Installs drift's own runner: one job per runtime, chat and images. */
val installForConfigurations: PublicEndpoint[
  InstallForConfigurationsRequest,
  Unit,
  List[RuntimeInstallJob],
  Any
] =
  runtimesBase.post
    .in("runtime-installs" / "for-configurations")
    .in(jsonBody[InstallForConfigurationsRequest])
    .out(jsonBody[List[RuntimeInstallJob]])

val installRunner
    : PublicEndpoint[InstallRunnerRequest, Unit, List[RuntimeInstallJob], Any] =
  runtimesBase.post
    .in("runtime-installs" / "runner")
    .in(jsonBody[InstallRunnerRequest])
    .out(jsonBody[List[RuntimeInstallJob]])

/** Re-resolves the newest release for a latest-tracking runtime and installs it
  * if a newer tag exists, keeping the same id and default status. Refuses a
  * runtime that does not track latest.
  */
val upgradeRuntime: PublicEndpoint[
  (String, UpgradeRuntimeRequest),
  Unit,
  RuntimeInstallJob,
  Any
] =
  runtimesBase.post
    .in("runtimes" / path[String] / "upgrade")
    .in(jsonBody[UpgradeRuntimeRequest])
    .out(jsonBody[RuntimeInstallJob])

/** Re-pairs a ROCm runtime with another TheRock build without re-downloading
  * the release: only the TheRock tarball moves if it is not on disk, then the
  * runtime is re-validated. Refuses a non-ROCm or adopted runtime.
  */
val changeRuntimeTheRock: PublicEndpoint[
  (String, ChangeTheRockRequest),
  Unit,
  RuntimeInstallJob,
  Any
] =
  runtimesBase.post
    .in("runtimes" / path[String] / "therock")
    .in(jsonBody[ChangeTheRockRequest])
    .out(jsonBody[RuntimeInstallJob])
