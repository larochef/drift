package drift.backend.runtime

import drift.backend.download.{DownloadOutcome, Downloader}
import drift.shared.*

import java.nio.charset.StandardCharsets
import java.nio.file.*
import java.util.concurrent.TimeUnit
import scala.jdk.CollectionConverters.*

/** A runtime's archives — a release, a TheRock build — downloaded through the
  * shared `Downloader` into `<runtimesRoot>/downloads` and unpacked into place,
  * the install's job following each step.
  */
final private[runtime] class RuntimeArchives(
    downloader: Downloader,
    runtimesRoot: Path
) {

  /** One archive: download (resumable, verified when a hash exists), then
    * unpack into place atomically. False means the job is already in a terminal
    * state and the caller must stop.
    */
  def fetchAndUnpack(
      entry: RuntimeInstallEntry,
      step: String,
      url: String,
      archiveName: String,
      sha256: Option[String],
      target: Path
  ): Boolean = {
    val archive = runtimesRoot.resolve("downloads").resolve(archiveName)
    entry.job = entry.job.copy(
      state = RuntimeInstallState.Downloading,
      step = step,
      downloadedBytes = 0L,
      totalBytes = None
    )
    val outcome = downloader.fetch(
      url = url,
      partFile = archive.resolveSibling(archiveName + ".part"),
      finalFile = archive,
      expectedSha256 = sha256,
      isCancelled = () => entry.cancelled.get(),
      onProgress = (done, total) =>
        entry.job = entry.job.copy(
          downloadedBytes = done,
          totalBytes = total.orElse(entry.job.totalBytes)
        )
    )
    outcome match {
      case DownloadOutcome.Completed(_, _) =>
        entry.job = entry.job.copy(state = RuntimeInstallState.Unpacking)
        unpack(archive, target)
        Files.deleteIfExists(archive)
        true
      case DownloadOutcome.Cancelled =>
        entry.job = entry.job.copy(
          state = RuntimeInstallState.Cancelled,
          completedAt = Some(System.currentTimeMillis())
        )
        false
      case DownloadOutcome.Failed(reason) =>
        entry.job = entry.job.copy(
          state = RuntimeInstallState.Failed,
          error = Some(s"$step: $reason"),
          completedAt = Some(System.currentTimeMillis())
        )
        false
    }
  }

  /** Unpacks into a temp sibling and moves into place, so `target` either does
    * not exist or is complete. Shells out to `unzip`/`tar` because
    * `java.util.zip` drops the executable bit, which is the one thing a runtime
    * archive cannot lose.
    */
  private def unpack(archive: Path, target: Path): Unit = {
    Files.createDirectories(target.getParent)
    val tmp = Files.createTempDirectory(target.getParent, ".unpack-")
    try {
      val command =
        if (archive.getFileName.toString.endsWith(".zip"))
          List("unzip", "-q", "-o", archive.toString, "-d", tmp.toString)
        else List("tar", "-xzf", archive.toString, "-C", tmp.toString)
      val process = ProcessBuilder(command*).redirectErrorStream(true).start()
      val output = String(
        process.getInputStream.readNBytes(16 * 1024),
        StandardCharsets.UTF_8
      )
      if (!process.waitFor(15, TimeUnit.MINUTES)) {
        process.destroyForcibly()
        throw RuntimeException(s"${command.head} timed out on $archive")
      }
      if (process.exitValue() != 0)
        throw RuntimeException(
          s"${command.head} failed on $archive: ${output.take(300)}"
        )
      // A tarball may wrap everything in a single top-level directory; the
      // runtime directory should be the content either way.
      val children = Files.list(tmp).iterator().asScala.toList
      val root = children match {
        case only :: Nil if Files.isDirectory(only) => only
        case _                                      => tmp
      }
      RuntimeManager.deleteTree(target)
      moveIntoPlace(root, target)
    } finally RuntimeManager.deleteTree(tmp)
  }

  private def moveIntoPlace(from: Path, to: Path): Unit =
    try
      Files.move(
        from,
        to,
        StandardCopyOption.REPLACE_EXISTING,
        StandardCopyOption.ATOMIC_MOVE
      )
    catch {
      case _: AtomicMoveNotSupportedException =>
        Files.move(from, to, StandardCopyOption.REPLACE_EXISTING)
    }
}
