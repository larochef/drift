package drift.backend.runtime

import drift.shared.*

import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.github.plokhotnyuk.jsoniter_scala.macros.JsonCodecMaker
import com.typesafe.scalalogging.Logger
import sttp.client3.{basicRequest, HttpURLConnectionBackend}
import sttp.model.Uri.*

/** The GitHub release listing, reduced to what asset classification needs. */
private case class GitHubAsset(
    name: String,
    size: Long = 0L,
    browser_download_url: String = "",
    /** "sha256:<hex>" — present on current releases, and what lets a 252 MB
      * archive be verified like any weight.
      */
    digest: Option[String] = None
)

private case class GitHubRelease(
    tag_name: String,
    published_at: String = "",
    assets: List[GitHubAsset] = Nil
)

private given com.github.plokhotnyuk.jsoniter_scala.core.JsonValueCodec[
  List[GitHubRelease]
] = JsonCodecMaker.make

/** One TheRock portable ROCm build from the public bucket, e.g. version
  * "7.9.0rc20251006" for target "gfx1151". Backend-internal: the UI never picks
  * one — [[RuntimeManager]] resolves the match itself.
  */
case class TheRockBuild(
    gfxTarget: String,
    version: String,
    sizeBytes: Long,
    downloadUrl: String,
    channel: TheRockChannel
) {

  /** A tagged AMD release (e.g. "10.0.0"), as opposed to a dated build off a
    * branch ("7.14.0a20260612", "6.4.0rc20250514"): the triple carries no
    * trailing letter marker.
    */
  def tagged: Boolean =
    version.matches("""\d+\.\d+\.\d+""")
}

/** Where runtimes come from (`specs/06-sdcpp-runtime.md`,
  * `specs/17-assistant-runtime.md`):
  *
  *   - the tools themselves: GitHub releases, published per commit — which is
  *     exactly the cadence this feature exists for. sd-cpp on
  *     `leejet/stable-diffusion.cpp`, llama.cpp on `ggml-org/llama.cpp`.
  *   - the ROCm runtime: AMD's TheRock publishes portable per-gfx-target
  *     tarballs, tagged releases on `repo.amd.com` indexes and dated builds in
  *     a public S3 bucket (its GitHub releases carry no assets).
  *
  * All listings are read live; nothing here downloads.
  */
class RuntimeCatalog(
    githubToken: Option[String] = None,
    theRockBucket: String = RuntimeCatalog.DefaultTheRockBucket
) {
  private val logger = Logger[RuntimeCatalog]
  private val backend = HttpURLConnectionBackend()

  /** Releases of a tool newest first, each with only its Linux x64 assets,
    * classified. A release whose assets are all unrecognized is dropped.
    */
  def releases(tool: RuntimeTool): List[RuntimeRelease] =
    try {
      // Two segments, not one: the interpolator would encode the slash of
      // "owner/name" and GitHub would answer 404.
      val (owner, name) = RuntimeCatalog.repository(tool)
      var request = basicRequest
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "drift/0.1.0")
        .get(
          uri"https://api.github.com/repos/$owner/$name/releases?per_page=30"
        )
      githubToken.foreach(t =>
        request = request.header("Authorization", s"Bearer $t")
      )
      request.send(backend).body match {
        case Right(json) =>
          readFromString[List[GitHubRelease]](json).flatMap { release =>
            val assets = release.assets.flatMap(classify(tool, _))
            if (assets.isEmpty) None
            else
              Some(
                RuntimeRelease(
                  tool,
                  release.tag_name,
                  release.published_at,
                  assets
                )
              )
          }
        case Left(err) =>
          logger.warn(s"GitHub release listing failed: ${err.take(200)}")
          Nil
      }
    } catch {
      case NonFatal(err) =>
        logger.warn(s"GitHub release listing failed", err)
        Nil
    }

  /** The Linux x64 assets drift manages, per tool (verified against the live
    * listings, 2026-09-02 and 2026-09-07):
    *
    *   - sd-cpp:
    *     `sd-master-<sha>-bin-Linux-Ubuntu-24.04-x86_64[-rocm-<ver>|-vulkan].zip`
    *   - llama.cpp: `llama-<tag>-bin-ubuntu[-rocm-<ver>|-vulkan]-x64.tar.gz`
    *
    * Anything else (Windows, macOS, CUDA, arm64, SYCL, OpenVINO) is not a
    * runtime drift manages.
    */
  private def classify(
      tool: RuntimeTool,
      asset: GitHubAsset
  ): Option[RuntimeReleaseAsset] = {
    val sha256 =
      asset.digest.map(_.stripPrefix("sha256:").toLowerCase).filter(_.nonEmpty)
    def make(backend: RuntimeBackend, rocmVersion: Option[String]) =
      RuntimeReleaseAsset(
        name = asset.name,
        sizeBytes = asset.size,
        downloadUrl = asset.browser_download_url,
        sha256 = sha256,
        backend = backend,
        rocmVersion = rocmVersion
      )
    tool match {
      case RuntimeTool.SdCpp =>
        if (!asset.name.contains("-bin-Linux") || !asset.name.endsWith(".zip"))
          return None
        asset.name.stripSuffix(".zip") match {
          case s if s.contains("-rocm-") =>
            Some(make(RuntimeBackend.Rocm, Some(s.split("-rocm-").last)))
          case s if s.endsWith("-vulkan") =>
            Some(make(RuntimeBackend.Vulkan, None))
          case s if s.endsWith("x86_64") => Some(make(RuntimeBackend.Cpu, None))
          case _                         => None
        }
      case RuntimeTool.LlamaCpp =>
        asset.name match {
          case RuntimeCatalog.LlamaCppAssetPattern(flavour) =>
            Option(flavour) match {
              case None            => Some(make(RuntimeBackend.Cpu, None))
              case Some("-vulkan") => Some(make(RuntimeBackend.Vulkan, None))
              case Some(rocm) if rocm.startsWith("-rocm-") =>
                Some(
                  make(RuntimeBackend.Rocm, Some(rocm.stripPrefix("-rocm-")))
                )
              case Some(_) => None
            }
          case _ => None
        }
    }
  }

  // ------------------------------------------------------------------ TheRock

  private val keyPrefix = "therock-dist-linux-"

  /** The gfx targets the bucket holds builds for, via an S3 delimiter listing —
    * one common prefix per target.
    */
  def theRockTargets: List[String] =
    fetchXml(
      s"$theRockBucket/?list-type=2&prefix=$keyPrefix&delimiter=-&max-keys=100"
    ) match {
      case None      => Nil
      case Some(xml) =>
        RuntimeCatalog.CommonPrefixPattern
          .findAllMatchIn(xml)
          .map(_.group(1))
          .toList
          .sorted
    }

  /** Builds for one gfx target from one channel, newest first. Nightly is the
    * S3 bucket; the stable and release-candidate channels are AMD's official
    * `repo.amd.com` tarball indexes, which serve tagged releases.
    */
  def theRockBuilds(
      gfxTarget: String,
      channel: TheRockChannel
  ): List[TheRockBuild] = channel match {
    case TheRockChannel.Nightly => nightlyBuilds(gfxTarget)
    case TheRockChannel.Stable  =>
      // Two AMD tarball indexes carry tagged per-gfx releases between them
      // (7.13.0 in both, 7.14.x only in multi-arch); merge and de-duplicate by
      // version, newest first.
      RuntimeCatalog.StableBases
        .flatMap(base => repoAmdBuilds(base, gfxTarget, channel))
        .groupBy(_.version)
        .values
        .map(_.head)
        .toList
        .sortBy(build => RuntimeCatalog.versionSortKey(build.version))
        .reverse
    case TheRockChannel.ReleaseCandidate =>
      repoAmdBuilds(RuntimeCatalog.ReleaseCandidateBase, gfxTarget, channel)
  }

  /** Every matching build across all channels for a gfx target. */
  def theRockBuilds(gfxTarget: String): List[TheRockBuild] =
    TheRockChannel.values.toList.flatMap(theRockBuilds(gfxTarget, _))

  /** The `repo.amd.com` tarball index is a flat HTML directory listing; the
    * filenames match the bucket's, minus the `-tests-` variants (which the
    * digit guard drops — their "version" starts with `tests`).
    */
  private def repoAmdBuilds(
      base: String,
      gfxTarget: String,
      channel: TheRockChannel
  ): List[TheRockBuild] =
    fetchXml(base) match {
      case None       => Nil
      case Some(html) =>
        val pattern =
          s"""therock-dist-linux-$gfxTarget-([^"<>\\s]+)\\.tar\\.gz""".r
        pattern
          .findAllMatchIn(html)
          .map(_.group(1))
          .toList
          .distinct
          .filter(_.headOption.exists(_.isDigit))
          .map(version =>
            TheRockBuild(
              gfxTarget = gfxTarget,
              version = version,
              sizeBytes = 0L,
              downloadUrl =
                s"$base" + s"therock-dist-linux-$gfxTarget-$version.tar.gz",
              channel = channel
            )
          )
          .sortBy(build => RuntimeCatalog.versionSortKey(build.version))
          .reverse
          .take(50)
    }

  /** Builds for one gfx target from the nightly S3 bucket, newest first. The
    * bucket lists ascending by key, and "7.14" sorts before "7.9"
    * lexicographically, so the whole listing is paged in and sorted by parsed
    * version.
    */
  private def nightlyBuilds(gfxTarget: String): List[TheRockBuild] = {
    val prefix = s"$keyPrefix$gfxTarget-"
    var keys = List.empty[(String, Long)]
    var startAfter = Option.empty[String]
    var pages = 0
    var truncated = true
    while (truncated && pages < 10) {
      pages += 1
      val after = startAfter.map(k => s"&start-after=$k").getOrElse("")
      fetchXml(
        s"$theRockBucket/?list-type=2&prefix=$prefix&max-keys=1000$after"
      ) match {
        case None      => truncated = false
        case Some(xml) =>
          val page = RuntimeCatalog.KeyPattern
            .findAllMatchIn(xml)
            .map(m => (m.group(1), m.group(2).toLong))
            .toList
          keys = keys ++ page
          truncated =
            xml.contains("<IsTruncated>true</IsTruncated>") && page.nonEmpty
          startAfter = page.lastOption.map(_._1)
      }
    }
    keys
      .flatMap { case (key, size) =>
        val version = key.stripPrefix(prefix).stripSuffix(".tar.gz")
        // Skips ADHOCBUILD and anything else that is not a plain version.
        Option.when(version.headOption.exists(_.isDigit))(
          TheRockBuild(
            gfxTarget = gfxTarget,
            version = version,
            sizeBytes = size,
            downloadUrl = s"$theRockBucket/$key",
            channel = TheRockChannel.Nightly
          )
        )
      }
      .sortBy(build => RuntimeCatalog.versionSortKey(build.version))
      .reverse
      .take(50)
  }

  private def fetchXml(url: String): Option[String] =
    try
      basicRequest
        .header("User-Agent", "drift/0.1.0")
        .get(uri"$url")
        .send(backend)
        .body match {
        case Right(xml) => Some(xml)
        case Left(err)  =>
          logger.warn(s"TheRock bucket listing failed: ${err.take(200)}")
          None
      }
    catch {
      case NonFatal(err) =>
        logger.warn(s"TheRock bucket listing failed", err)
        None
    }
}

object RuntimeCatalog {

  /** GitHub (owner, repository) of a tool's releases. */
  def repository(tool: RuntimeTool): (String, String) = tool match {
    case RuntimeTool.SdCpp    => ("leejet", "stable-diffusion.cpp")
    case RuntimeTool.LlamaCpp => ("ggml-org", "llama.cpp")
  }

  /** `llama-b10844-bin-ubuntu-x64.tar.gz`, `…-bin-ubuntu-vulkan-x64.tar.gz`,
    * `…-bin-ubuntu-rocm-10.0-x64.tar.gz`; the group is the flavour between
    * `ubuntu` and `x64`, absent for the CPU build.
    */
  private val LlamaCppAssetPattern =
    """llama-b\d+-bin-ubuntu(-vulkan|-rocm-[\d.]+)?-x64\.tar\.gz""".r

  val DefaultTheRockBucket: String =
    sys.env.getOrElse(
      "DRIFT_THEROCK_BUCKET",
      "https://therock-nightly-tarball.s3.amazonaws.com"
    )

  /** AMD's official tarball indexes carrying tagged per-gfx releases. Three of
    * them, because they split the versions between them: `tarball/` holds
    * 7.9–7.13, `tarball-multi-arch/` holds 7.13–7.14.1, and
    * `stable.repo.amd.com/rocm/core/tarball/` holds 10.0.0 — which matches no
    * sd-cpp build but is what the llama.cpp ROCm build declares (verified live,
    * 2026-09-07). Trailing slash on purpose: build URLs are this base
    * concatenated with the filename.
    */
  val StableBases: List[String] =
    sys.env
      .get("DRIFT_THEROCK_STABLE_BASES")
      .map(_.split(",").toList.map(_.trim).filter(_.nonEmpty))
      .getOrElse(
        List(
          "https://repo.amd.com/rocm/tarball/",
          "https://repo.amd.com/rocm/tarball-multi-arch/",
          "https://stable.repo.amd.com/rocm/core/tarball/"
        )
      )
  val ReleaseCandidateBase: String =
    sys.env.getOrElse(
      "DRIFT_THEROCK_RC_BASE",
      "https://rc.repo.amd.com/rocm/core/tarball/"
    )

  private val CommonPrefixPattern =
    """<CommonPrefixes><Prefix>therock-dist-linux-([^<]+)-</Prefix></CommonPrefixes>""".r

  /** `<Key>` and `<Size>` inside one `<Contents>` block; the fields between
    * them vary, hence the lazy gap.
    */
  private val KeyPattern =
    """<Contents><Key>([^<]+)</Key>.*?<Size>(\d+)</Size>""".r

  /** "7.9.0rc20251006" or "7.14.0a20260612" -> sortable (7, 9, 0, 20251006):
    * the bucket stamps a date after a letter marker ("rc", "a").
    */
  private[runtime] def versionSortKey(
      version: String
  ): (Int, Int, Int, Long) = {
    val pattern = """(\d+)\.(\d+)\.(\d+)(?:[a-z]+(\d+))?.*""".r
    version match {
      case pattern(major, minor, patch, rc) =>
        (
          major.toInt,
          minor.toInt,
          patch.toInt,
          Option(rc).map(_.toLong).getOrElse(0L)
        )
      case _ => (0, 0, 0, 0L)
    }
  }
}
