package drift.backend.runtime

import drift.shared.*

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/** Which TheRock ROCm build a ROCm runtime pairs with
  * (`specs/06-sdcpp-runtime.md`): the stable releases for the gfx target and
  * whatever is already unpacked, and the preferred choice among them.
  */
final private[runtime] class TheRockPairing(
    catalog: RuntimeCatalog,
    runtimesRoot: Path
) {

  /** What TheRock build drift would pair with a ROCm requirement for a gfx
    * target: everything matching the declared version across channels and
    * already on disk, with the preferred choice — reuse what is downloaded,
    * else the most stable channel. The version carries a build stamp after the
    * triple ("7.14.0a20260612"), so matching is by triple, not by exact string
    * — and a declared "10.0" (llama.cpp names only major.minor) matches the
    * newest 10.0.x.
    */
  def resolve(gfx: String, rocmVersion: String): TheRockResolution = {
    def wanted(candidate: String): Boolean =
      RuntimeManager.sameRocm(rocmVersion, candidate)
    val installed = installedVersions(gfx)

    // Stable (tagged) releases only — the user's rule is "only stable, not
    // nightly". Every version is offered, not just the one the asset asks for,
    // so a mismatched-but-worth-testing build (e.g. 7.13.0 against a 7.14.0
    // binary) can be chosen deliberately.
    val stableBuilds = catalog.theRockBuilds(gfx, TheRockChannel.Stable)
    // Anything already unpacked but no longer in the stable listing (a leftover
    // nightly, say) is still offered so a broken-looking build can be retested.
    val onDiskExtra = installed
      .filterNot(version => stableBuilds.exists(_.version == version))
      .toList
      .map(version =>
        TheRockChoice(
          RuntimeManager.inferChannel(version),
          version,
          tagged = version.matches("""\d+\.\d+\.\d+"""),
          onDisk = true
        )
      )

    val choices = (stableBuilds.map(build =>
      TheRockChoice(
        build.channel,
        build.version,
        build.tagged,
        onDisk = installed.contains(build.version)
      )
    ) ++ onDiskExtra)
      .sortBy { choice =>
        val key = RuntimeCatalog.versionSortKey(choice.version)
        // Tagged releases first, then newest version.
        (if (choice.tagged) 0 else 1, -key._1, -key._2, -key._3, -key._4)
      }

    // Prefer a tagged stable release for the exact version the asset needs —
    // that is both stable and correct, and moves off a broken on-disk build
    // rather than reusing it. Fall back to an on-disk match; otherwise leave it
    // unset so the user picks rather than drift guessing a different version.
    val chosen = choices
      .find(choice => choice.tagged && wanted(choice.version))
      .orElse(choices.find(choice => wanted(choice.version)))

    TheRockResolution(gfx, rocmVersion, chosen, choices)
  }

  /** The TheRock build to install for a Rocm asset: an explicit version if the
    * request pinned one, else the resolver's preferred choice. Refuses, naming
    * the ROCm version, when nothing matches in any channel.
    */
  def buildFor(
      asset: RuntimeReleaseAsset,
      gfx: String,
      explicitVersion: Option[String]
  ): Either[String, TheRockBuild] =
    asset.rocmVersion match {
      case None =>
        Left(
          s"asset '${asset.name}' names no ROCm version to match a TheRock build against"
        )
      case Some(rocm) =>
        explicitVersion match {
          case Some(version) =>
            buildOfVersion(gfx, version).toRight(
              s"TheRock build '$version' is not available for $gfx in any channel"
            )
          case None =>
            val resolution = resolve(gfx, rocm)
            resolution.chosen match {
              case None =>
                Left(
                  s"no TheRock build matches ROCm $rocm for $gfx in any channel " +
                    "(stable, release-candidate or nightly)"
                )
              case Some(choice) =>
                Right(
                  buildOfVersion(gfx, choice.version).getOrElse(
                    // On disk but no longer listed by any channel: reuse it,
                    // the install skips the download when the directory exists.
                    TheRockBuild(gfx, choice.version, 0L, "", choice.channel)
                  )
                )
            }
        }
    }

  /** A concrete build for an exact version, preferring the most stable channel
    * that carries it (so its download URL is used if a fetch is needed).
    */
  private def buildOfVersion(
      gfx: String,
      version: String
  ): Option[TheRockBuild] =
    catalog
      .theRockBuilds(gfx, TheRockChannel.Stable)
      .find(_.version == version)
      .orElse(
        Option.when(installedVersions(gfx).contains(version))(
          TheRockBuild(
            gfx,
            version,
            0L,
            "",
            RuntimeManager.inferChannel(version)
          )
        )
      )

  /** The TheRock versions already unpacked for a gfx target, from the directory
    * names `therock/<gfx>-<version>`.
    */
  private def installedVersions(gfx: String): Set[String] = {
    val dir = runtimesRoot.resolve("therock")
    if (!Files.isDirectory(dir)) Set.empty
    else {
      val stream = Files.list(dir)
      try
        stream
          .iterator()
          .asScala
          .filter(Files.isDirectory(_))
          .map(_.getFileName.toString)
          .filter(_.startsWith(s"$gfx-"))
          .map(_.stripPrefix(s"$gfx-"))
          .toSet
      finally stream.close()
    }
  }
}
