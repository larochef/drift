package drift.backend

import java.nio.file.{Path, Paths}

/** Where drift keeps its files. Every root is resolved once at startup, in this
  * order: drift's own variable, the well-known variable of the convention the
  * root belongs to (XDG base directories, the HuggingFace hub), then the
  * conventional place under the home directory. So a systemd user unit, a
  * container run as the user, or a test on scratch copies all relocate drift
  * with the variables they already set — no path is hardcoded anywhere else in
  * the backend (`docs/files-and-folders.md`).
  *
  * `DRIFT_APP_NAME` (default `drift`) is the folder name under each convention
  * root, so one variable gives a second instance (`drift-test`) its own
  * configuration, cache and outputs. Blank values count as unset, as the XDG
  * specification asks.
  */
final case class Locations(
    /** The folder name under each convention root. */
    appName: String,
    /** Entities, one JSON file each (`StorageService`). */
    configDir: Path,
    /** Model weights, LoRAs, upscalers, runtimes, logs, assistant uploads. */
    cacheRoot: Path,
    /** Installed runtimes and their TheRock builds. */
    runtimesRoot: Path,
    /** Generated images and videos with their sidecars. */
    outputsRoot: Path,
    /** The HuggingFace hub cache drift shares with every other HF tool. */
    huggingFaceRoot: Path
) {
  def logsRoot: Path = cacheRoot.resolve("logs")
  def lorasRoot: Path = cacheRoot.resolve("loras")
  def upscaleRoot: Path = cacheRoot.resolve("upscale")
  def uploadsRoot: Path = cacheRoot.resolve("assistant-uploads")
}

object Locations {

  def fromEnvironment(
      env: Map[String, String] = sys.env,
      home: Path = Paths.get(sys.props("user.home"))
  ): Locations = {
    def variable(name: String): Option[Path] =
      env.get(name).map(_.trim).filter(_.nonEmpty).map(Paths.get(_))

    val xdgConfig =
      variable("XDG_CONFIG_HOME").getOrElse(home.resolve(".config"))
    val xdgCache = variable("XDG_CACHE_HOME").getOrElse(home.resolve(".cache"))
    val xdgData =
      variable("XDG_DATA_HOME").getOrElse(
        home.resolve(".local").resolve("share")
      )

    val appName =
      env
        .get("DRIFT_APP_NAME")
        .map(_.trim)
        .filter(_.nonEmpty)
        .getOrElse("drift")
    val cacheRoot =
      variable("DRIFT_CACHE_ROOT").getOrElse(xdgCache.resolve(appName))
    Locations(
      appName = appName,
      configDir =
        variable("DRIFT_CONFIG_DIR").getOrElse(xdgConfig.resolve(appName)),
      cacheRoot = cacheRoot,
      runtimesRoot = variable("DRIFT_RUNTIMES_ROOT").getOrElse(
        cacheRoot.resolve("runtimes")
      ),
      outputsRoot = variable("DRIFT_OUTPUTS_ROOT")
        .getOrElse(xdgData.resolve(appName).resolve("outputs")),
      huggingFaceRoot = variable("HF_HUB_CACHE")
        .orElse(variable("HF_HOME").map(_.resolve("hub")))
        .getOrElse(xdgCache.resolve("huggingface").resolve("hub"))
    )
  }
}
