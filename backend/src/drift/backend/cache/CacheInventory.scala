package drift.backend.cache

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.readFromString
import com.typesafe.scalalogging.Logger

/** Walks the cache roots and reports what is actually on disk — including files
  * no registered model references, which is the whole point of the on-disk
  * view: orphans are what you look for when reclaiming space, and they are
  * invisible from everywhere else in the app.
  */
final class CacheInventory(cache: ModelCache, storage: StorageService) {
  private val logger = Logger[CacheInventory]

  private val auxiliaryNames = Set("drift-metadata.json", "drift-lora.json")

  private def lorasRoot: Path = cache.driftRoot.resolve("loras")

  def list: List[CachedFileEntry] = {
    val referencedBy = referenceIndex()
    listDrift(referencedBy) ++ listLoras(referencedBy) ++
      listHuggingFace(referencedBy) ++ listModelScope(referencedBy)
  }

  /** Deletes one listed file. Refuses anything outside the two roots.
    *
    * A drift entry takes its whole version directory with it — weights, sidecar
    * and previews are one unit — pruning emptied parents. A HuggingFace blob
    * also removes the snapshot symlinks pointing at it, so nothing dangles.
    */
  def delete(rawPath: String): Boolean =
    try {
      val path = Path.of(rawPath).toAbsolutePath.normalize
      val underDrift = path.startsWith(cache.driftRoot.toAbsolutePath.normalize)
      val underHuggingFace =
        path.startsWith(cache.huggingFaceRoot.toAbsolutePath.normalize)
      if (!underDrift && !underHuggingFace) {
        logger.warn(s"Refusing to delete outside the cache roots: $path")
        false
      } else if (!Files.exists(path)) false
      else if (underDrift) deleteDriftEntry(path)
      else deleteHuggingFaceEntry(path)
    } catch {
      case NonFatal(err) =>
        logger.warn(s"Cache delete failed for $rawPath", err)
        false
    }

  // ---------------------------------------------------------------- listing

  private def listDrift(
      referencedBy: Map[Path, List[String]]
  ): List[CachedFileEntry] = {
    val modelsRoot = cache.driftRoot.resolve("models")
    if (!Files.isDirectory(modelsRoot)) return Nil
    walkFiles(modelsRoot)
      .filter { file =>
        val name = file.getFileName.toString
        !auxiliaryNames.contains(name) && !name.startsWith("preview-")
      }
      .map { file =>
        val name = file.getFileName.toString
        val metadata = sidecarMetadata(file.getParent)
        // A Civitai download is stored as `<fileId>-<filename>` beside the
        // sidecar naming its model and version (`ModelCache.civitaiTarget`).
        val fileId = name.takeWhile(_.isDigit)
        val civitai = metadata
          .filter(_.source == "civitai")
          .filter(_ => fileId.nonEmpty && name.startsWith(s"$fileId-"))
          .map(m =>
            Civitai(
              modelId = m.modelId,
              versionId = m.versionId,
              fileId = fileId,
              filename = name.drop(fileId.length + 1)
            )
          )
        CachedFileEntry(
          path = file.toString,
          bytes = size(file),
          root = "drift",
          group = modelsRoot.relativize(file).getName(0).toString,
          label = metadata
            .map(_.name)
            .filter(_.nonEmpty)
            .map(modelName => s"$modelName — $name")
            .getOrElse(name),
          // The sidecar is what remembers where a download came from; a file
          // without one is simply local content in the drift directory.
          kind =
            if (metadata.exists(_.source == "civitai")) CachedFileKind.Civitai
            else CachedFileKind.Local,
          referencedBy = referencedBy.getOrElse(realPath(file), Nil),
          partial = isPartial(name),
          source = civitai.filter(resolvesTo(file))
        )
      }
  }

  /** The LoRA store (`specs/09-lora-management.md`):
    * `loras/<architectureId>/<sfw|nsfw>/<folder>/<fileId>-<name>`. Grouped by
    * architecture; the sfw/nsfw placement rides in the label.
    */
  private def listLoras(
      referencedBy: Map[Path, List[String]]
  ): List[CachedFileEntry] = {
    if (!Files.isDirectory(lorasRoot)) return Nil
    walkFiles(lorasRoot)
      .filter { file =>
        val name = file.getFileName.toString
        !auxiliaryNames.contains(name) && !name.startsWith("preview-")
      }
      .map { file =>
        val name = file.getFileName.toString
        val relative = lorasRoot.relativize(file)
        val architectureId = relative.getName(0).toString
        val nsfw = relative.getNameCount > 1 &&
          relative.getName(1).toString == "nsfw"
        val metadata = sidecarMetadata(file.getParent, "drift-lora.json")
        val display = metadata
          .map(_.name)
          .filter(_.nonEmpty)
          .map(loraName => s"$loraName — $name")
          .getOrElse(name)
        CachedFileEntry(
          path = file.toString,
          bytes = size(file),
          root = "drift",
          group = architectureId,
          label = if (nsfw) s"$display (nsfw)" else display,
          kind = CachedFileKind.Lora,
          referencedBy = referencedBy.getOrElse(realPath(file), Nil),
          partial = isPartial(name)
        )
      }
  }

  /** ModelScope downloads (`specs/37-modelscope.md`):
    * `modelscope/<owner>/<name>/<path>`, grouped by repository. Deleting one
    * removes that file alone — the drift entry rule for anything outside
    * `models/` and `loras/`.
    */
  private def listModelScope(
      referencedBy: Map[Path, List[String]]
  ): List[CachedFileEntry] =
    walkFiles(cache.modelScopeRoot)
      .filter(file => cache.modelScopeRoot.relativize(file).getNameCount > 2)
      .map { file =>
        val relative = cache.modelScopeRoot.relativize(file)
        val repo = s"${relative.getName(0)}/${relative.getName(1)}"
        val path = relative.subpath(2, relative.getNameCount).toString
        CachedFileEntry(
          path = file.toString,
          bytes = size(file),
          root = "drift",
          group = repo,
          label = path,
          kind = CachedFileKind.ModelScope,
          referencedBy = referencedBy.getOrElse(realPath(file), Nil),
          partial = isPartial(file.getFileName.toString),
          source = Some(ModelScope(repo, path.stripSuffix(".part")))
            .filter(resolvesTo(file))
        )
      }

  private def listHuggingFace(
      referencedBy: Map[Path, List[String]]
  ): List[CachedFileEntry] = {
    val root = cache.huggingFaceRoot
    if (!Files.isDirectory(root)) return Nil
    childDirectories(root)
      .filter(_.getFileName.toString.startsWith("models--"))
      .flatMap { repoDirectory =>
        val repo = repoDirectory.getFileName.toString
          .stripPrefix("models--")
          .replace("--", "/")
        // Readable names come from the snapshot symlinks pointing at each blob.
        val snapshotsDirectory = repoDirectory.resolve("snapshots")
        val names: Map[Path, List[String]] = walkFiles(
          snapshotsDirectory,
          followLinks = false
        )
          .filter(Files.isSymbolicLink)
          .groupMap(link => realPath(link)) { link =>
            // snapshots/<revision>/sub/dirs/file -> sub/dirs/file
            val relative = snapshotsDirectory.relativize(link)
            if (relative.getNameCount > 1)
              relative.subpath(1, relative.getNameCount).toString
            else relative.toString
          }
          .view
          .mapValues(_.distinct)
          .toMap

        val blobsDirectory = repoDirectory.resolve("blobs")
        val blobEntries =
          if (!Files.isDirectory(blobsDirectory)) Nil
          else
            childFiles(blobsDirectory).map { blob =>
              val real = realPath(blob)
              val blobName = blob.getFileName.toString
              CachedFileEntry(
                path = blob.toString,
                bytes = size(blob),
                root = "huggingface",
                group = repo,
                label = names
                  .get(real)
                  .map(_.mkString(", "))
                  .getOrElse(blobName.take(12) + "…"),
                kind = CachedFileKind.HuggingFace,
                referencedBy = referencedBy.getOrElse(real, Nil),
                partial = isPartial(blobName),
                // Any snapshot name of the blob addresses it; a blob no
                // snapshot links to has no name to reference it by.
                source = names
                  .get(real)
                  .flatMap(_.headOption)
                  .map(HuggingFace(repo, _))
                  .filter(resolvesTo(blob))
              )
            }

        // Some tools write real files straight into snapshots/ with no blob
        // and no symlink -- lemonade's downloads have this shape. They are
        // content too, and skipping them hid ~90 GB from the view whose whole
        // job is answering "what is using my disk?".
        val flatSnapshotEntries = walkFiles(
          snapshotsDirectory,
          followLinks = false
        )
          .filter(file =>
            Files.isRegularFile(
              file,
              java.nio.file.LinkOption.NOFOLLOW_LINKS
            ) && !Files.isSymbolicLink(file)
          )
          .map { file =>
            val relative = snapshotsDirectory.relativize(file)
            val label =
              if (relative.getNameCount > 1)
                relative.subpath(1, relative.getNameCount).toString
              else relative.toString
            CachedFileEntry(
              path = file.toString,
              bytes = size(file),
              root = "huggingface",
              group = repo,
              label = label,
              kind = CachedFileKind.HuggingFace,
              referencedBy = referencedBy.getOrElse(realPath(file), Nil),
              partial = isPartial(file.getFileName.toString),
              source = Some(HuggingFace(repo, label)).filter(resolvesTo(file))
            )
          }

        blobEntries ++ flatSnapshotEntries
      }
  }

  /** Real path of every registered model's resolved file -> the model ids. Real
    * paths, because a HuggingFace model resolves to a snapshot symlink while
    * the listing reports the blob it points at — same file, and it must count
    * as referenced.
    */
  private def referenceIndex(): Map[Path, List[String]] = {
    // A split model's shards are its content as much as its index is.
    val fromModels = storage
      .list[Model]("models")
      .flatMap { model =>
        cache.resolve(model.source) match {
          case CacheEntry.Present(path, _) =>
            (path :: SafetensorsIndex.completeShards(path).getOrElse(Nil))
              .map(file => realPath(file) -> model.id)
          case _ => Nil
        }
      }
    // LoRA entities claim their files the same way, so an installed LoRA's
    // weights read "used by <id>" instead of orphan.
    val fromLoras = storage
      .list[Lora]("loras")
      .flatMap(lora =>
        lora.files.map(file =>
          realPath(lorasRoot.resolve(lora.storagePathOf(file))) -> lora.id
        )
      )
    (fromModels ++ fromLoras).groupMap(_._1)(_._2)
  }

  // --------------------------------------------------------------- deleting

  private def deleteDriftEntry(path: Path): Boolean = {
    val modelsRoot = cache.driftRoot.resolve("models").toAbsolutePath.normalize
    val normalizedLorasRoot = lorasRoot.toAbsolutePath.normalize
    val parentDirectory = path.getParent
    if (parentDirectory != null && parentDirectory.startsWith(modelsRoot)) {
      // The version directory is the deletion unit; stop at the family level.
      deleteRecursively(parentDirectory)
      pruneEmptyUpTo(parentDirectory.getParent, modelsRoot)
      true
    } else if (
      parentDirectory != null &&
      parentDirectory.startsWith(normalizedLorasRoot) &&
      parentDirectory != normalizedLorasRoot
    ) {
      // The LoRA folder is the unit: metadata, previews and a wan pair's two
      // files travel together.
      deleteRecursively(parentDirectory)
      pruneEmptyUpTo(parentDirectory.getParent, normalizedLorasRoot)
      true
    } else Files.deleteIfExists(path)
  }

  private def deleteHuggingFaceEntry(path: Path): Boolean =
    if (path.getParent.getFileName.toString == "blobs") {
      // A blob: also remove the snapshot symlinks pointing at it.
      val repoDirectory = path.getParent.getParent
      val real = realPath(path)
      walkFiles(repoDirectory.resolve("snapshots"), followLinks = false)
        .filter(Files.isSymbolicLink)
        .filter(link => realPath(link) == real)
        .foreach(Files.deleteIfExists(_))
      Files.deleteIfExists(path)
    } else
      // A real file under snapshots/ (the flat layout): nothing links to it.
      Files.deleteIfExists(path)

  // ---------------------------------------------------------------- helpers

  /** Whether a source resolves to this very file — the condition for an adopted
    * model to keep the file's origin rather than its path.
    */
  private def resolvesTo(file: Path)(source: ModelSource): Boolean =
    cache.resolve(source) match {
      case CacheEntry.Present(path, _) => realPath(path) == realPath(file)
      case _                           => false
    }

  private def isPartial(name: String): Boolean =
    name.endsWith(".part") || name.endsWith(".incomplete")

  private def sidecarMetadata(
      directory: Path,
      fileName: String = "drift-metadata.json"
  ): Option[ModelMetadata] =
    try {
      val sidecar = directory.resolve(fileName)
      if (!Files.isRegularFile(sidecar)) None
      else Some(readFromString[ModelMetadata](Files.readString(sidecar)))
    } catch { case NonFatal(_) => None }

  private def walkFiles(
      directory: Path,
      followLinks: Boolean = true
  ): List[Path] =
    if (!Files.isDirectory(directory)) Nil
    else {
      val stream = Files.walk(directory)
      try
        stream
          .iterator()
          .asScala
          .filter(p =>
            if (followLinks) Files.isRegularFile(p)
            else
              Files.isRegularFile(
                p,
                java.nio.file.LinkOption.NOFOLLOW_LINKS
              ) || Files.isSymbolicLink(p)
          )
          .toList
      finally stream.close()
    }

  private def childDirectories(directory: Path): List[Path] =
    if (!Files.isDirectory(directory)) Nil
    else {
      val stream = Files.list(directory)
      try stream.iterator().asScala.filter(Files.isDirectory(_)).toList
      finally stream.close()
    }

  private def childFiles(directory: Path): List[Path] = {
    val stream = Files.list(directory)
    try stream.iterator().asScala.filter(Files.isRegularFile(_)).toList
    finally stream.close()
  }

  private def realPath(path: Path): Path =
    try path.toRealPath()
    catch { case NonFatal(_) => path.toAbsolutePath.normalize }

  private def size(path: Path): Long =
    try Files.size(path)
    catch { case NonFatal(_) => 0L }

  private def deleteRecursively(directory: Path): Unit = {
    val stream = Files.walk(directory)
    try
      stream
        .iterator()
        .asScala
        .toList
        .reverse
        .foreach(Files.deleteIfExists(_))
    finally stream.close()
  }

  private def pruneEmptyUpTo(start: Path, stopExclusive: Path): Unit = {
    var current = start
    while (
      current != null && current.startsWith(stopExclusive) &&
      current != stopExclusive && isEmptyDirectory(current)
    ) {
      Files.deleteIfExists(current)
      current = current.getParent
    }
  }

  private def isEmptyDirectory(path: Path): Boolean =
    Files.isDirectory(path) && {
      val stream = Files.list(path)
      try !stream.iterator().hasNext
      finally stream.close()
    }
}
