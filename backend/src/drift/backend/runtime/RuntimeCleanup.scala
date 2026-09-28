package drift.backend.runtime

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.{Path, Paths}

import com.typesafe.scalalogging.Logger

/** Removes the files of runtimes drift installed itself once nothing references
  * them — never an adopted directory, never anything outside the runtimes root.
  */
final private[runtime] class RuntimeCleanup(
    storage: StorageService,
    runtimesRoot: Path,
    selections: RuntimeSelections
) {
  private val logger = Logger[RuntimeCleanup]

  /** After the registration is deleted: for a runtime drift installed itself,
    * remove the unpacked files too — but only what nothing else references. A
    * `latest-<backend>` runtime and a pinned one can share a release directory
    * (both keyed by the same tag), and TheRock builds are shared by design, so
    * neither is deleted while another runtime still points at it. Adopted
    * directories are never touched. A default pointing at the deleted runtime
    * is cleared rather than silently reassigned.
    */
  def cleanupDeleted(runtime: Runtime): Unit = {
    if (!runtime.adopted) {
      // onDeleted fires after the entity is removed, so `list` already excludes
      // it; any hit is a genuine other owner.
      val remaining = storage.list[Runtime]("runtimes")
      if (!remaining.exists(_.installedAt == runtime.installedAt))
        deleteOwnedTree(Paths.get(runtime.installedAt))
      runtime.theRockPath.foreach { rock =>
        if (!remaining.exists(_.theRockPath.contains(rock)))
          deleteOwnedTree(Paths.get(rock))
      }
    }
    selections.update(current =>
      if (current.defaultFor(runtime.tool).contains(runtime.id))
        current.withDefault(runtime.tool, None)
      else current
    )
  }

  /** After a latest runtime upgrades to a new tag's directory: remove the tag
    * it left behind, and its TheRock build, unless another runtime still points
    * at either. Adopted directories and paths the replacement itself uses are
    * never touched.
    */
  def cleanupSuperseded(previous: Runtime, replacement: Runtime): Unit =
    if (!previous.adopted) {
      def referencedElsewhere(
          path: String,
          pick: Runtime => Option[String]
      ): Boolean =
        storage.list[Runtime]("runtimes").exists(r => pick(r).contains(path))

      if (
        previous.installedAt != replacement.installedAt &&
        !referencedElsewhere(previous.installedAt, r => Some(r.installedAt))
      ) deleteOwnedTree(Paths.get(previous.installedAt))

      previous.theRockPath.foreach { rock =>
        if (
          !replacement.theRockPath.contains(rock) &&
          !referencedElsewhere(rock, _.theRockPath)
        ) deleteOwnedTree(Paths.get(rock))
      }
    }

  /** Refuses anything outside the runtimes root, whatever the entity says. */
  private def deleteOwnedTree(path: Path): Unit = {
    val normalized = path.toAbsolutePath.normalize
    if (
      normalized.startsWith(runtimesRoot.toAbsolutePath.normalize) &&
      normalized != runtimesRoot.toAbsolutePath.normalize
    ) RuntimeManager.deleteTree(normalized)
    else logger.info(s"Leaving $normalized alone: not under $runtimesRoot")
  }
}
