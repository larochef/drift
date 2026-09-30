package drift.backend.cache

import drift.shared.*

import java.nio.file.*
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** Where a model's weights are, or why they are not here yet. */
enum CacheEntry derives CanEqual {

  /** The file is on disk and readable. */
  case Present(path: Path, bytes: Long)

  /** Nothing on disk for this source. */
  case Absent

  /** A `Local` source pointing at a path that does not exist. Distinguished
    * from `Absent` because there is nothing to download — the user gave a bad
    * path.
    */
  case BrokenLocal(path: Path)
}

/** Resolves a [[ModelSource]] to a file on disk, and never downloads anything.
  *
  * drift indexes rather than owns: weights are measured in hundreds of GB, so
  * the cache is an ordered list of roots searched before anything is fetched.
  * See `specs/05-model-cache-and-downloads.md`.
  */
class ModelCache(
    /** Where drift's own downloads live. */
    val driftRoot: Path,
    /** The shared HuggingFace cache, read and written in HuggingFace's own
      * layout.
      */
    val huggingFaceRoot: Path,
    /** Extra read-only roots holding files the user already has. */
    val extraRoots: List[Path] = Nil
) {

  def resolve(source: ModelSource): CacheEntry = source match {
    case Local(path) =>
      val p = Paths.get(path)
      if (isComplete(p)) present(ModelCache.preferSnapshotPath(p))
      else CacheEntry.BrokenLocal(p)

    case hf: HuggingFace => firstPresent(huggingFacePaths(hf))
    case ms: ModelScope  => firstPresent(List(modelScopePath(ms)))
    case cv: Civitai     => firstPresent(civitaiPaths(cv))
  }

  /** ModelScope files, in drift's own tree (`specs/37-modelscope.md`). */
  val modelScopeRoot: Path = driftRoot.resolve("modelscope")

  /** `<modelScopeRoot>/<owner>/<name>/<path in the repository>`. */
  def modelScopePath(source: ModelScope): Path =
    modelScopeRoot.resolve(source.repo).resolve(source.filename)

  /** Where a Civitai download should be written.
    *
    * `<driftRoot>/models/<familyId>/<slug>-<modelId>/<slug>-<versionId>/<fileId>-<filename>`.
    * Grouped by family, not architecture: a family is what makes models
    * interchangeable in a slot, and the same VAE is deliberately shared between
    * architectures — `wan-2.1-vae` serves both `wan-2.2-14B` and `krea2`, and
    * must be stored once.
    *
    * The file id prefixes the name because a version can publish the *same
    * filename* in several precisions (int8, bf16…, seen live); on the bare
    * filename they would overwrite each other and, worse, resolve as each
    * other.
    */
  def civitaiTarget(
      source: Civitai,
      familyId: String,
      modelName: String,
      versionName: String
  ): Path =
    driftRoot
      .resolve("models")
      .resolve(familyId)
      .resolve(s"${ModelCache.slug(modelName)}-${source.modelId}")
      .resolve(s"${ModelCache.slug(versionName)}-${source.versionId}")
      .resolve(s"${source.fileId}-${source.filename}")

  /** Candidate locations for a Civitai file.
    *
    * The slug segments are decoration — the ids carry identity — so lookup
    * globs past them. That is what lets an upstream rename leave a
    * stale-looking directory rather than an unfindable file. Extra roots are
    * foreign stores with their own naming, so only the bare filename is tried
    * there.
    */
  private def civitaiPaths(source: Civitai): List[Path] = {
    val modelsRoot = driftRoot.resolve("models")
    val matches =
      for {
        familyDir <- childDirectories(modelsRoot)
        modelDir <- childDirectories(familyDir)
        if modelDir.getFileName.toString.endsWith(s"-${source.modelId}")
        versionDir <- childDirectories(modelDir)
        if source.versionId.isEmpty ||
          versionDir.getFileName.toString.endsWith(s"-${source.versionId}")
      } yield versionDir.resolve(s"${source.fileId}-${source.filename}")
    matches ++ extraRoots.map(_.resolve(source.filename))
  }

  /** The Civitai file ids already on disk for one model, whichever family and
    * version directories hold them — read straight off the `<fileId>-<name>`
    * scheme, so the browser can mark what is downloaded without a sidecar.
    */
  def civitaiCachedFileIds(modelId: String): List[String] =
    (for {
      familyDir <- childDirectories(driftRoot.resolve("models"))
      modelDir <- childDirectories(familyDir)
      if modelDir.getFileName.toString.endsWith(s"-$modelId")
      versionDir <- childDirectories(modelDir)
      file <- childFiles(versionDir)
      name = file.getFileName.toString
      if !name.endsWith(".part") && !name.endsWith(".incomplete")
      fileId = name.takeWhile(_.isDigit)
      if fileId.nonEmpty && name.startsWith(s"$fileId-")
    } yield fileId).distinct

  /** Candidate locations for a HuggingFace file.
    *
    * The shared cache stores content under `blobs/<sha256>` with
    * `snapshots/<commit>/<path>` symlinking to it. Any snapshot will do: the
    * blob is the same file whichever revision references it. A file the shared
    * cache could not take (no blob name or no commit) is in drift's own tree,
    * `<driftRoot>/models/huggingface/<repo dir>/<path>`.
    */
  private def huggingFacePaths(source: HuggingFace): List[Path] = {
    val repoDirectoryName = ModelCache.repoDirectoryName(source.repo)
    val repoDir = huggingFaceRoot.resolve(repoDirectoryName)
    val snapshots = childDirectories(repoDir.resolve("snapshots"))
    val fromSnapshots = snapshots.map(_.resolve(source.filename))
    val fromDriftTree = driftRoot
      .resolve("models")
      .resolve("huggingface")
      .resolve(repoDirectoryName)
      .resolve(source.filename)
    val fromExtras = extraRoots.map(_.resolve(source.filename))
    fromSnapshots ++ (fromDriftTree :: fromExtras)
  }

  private def firstPresent(candidates: List[Path]): CacheEntry =
    candidates.find(isComplete).map(present).getOrElse(CacheEntry.Absent)

  /** A readable file and, for a split model's index, every shard beside it
    * (`specs/34-sharded-safetensors.md`): an index whose shards are still
    * downloading is not a model yet.
    */
  private def isComplete(p: Path): Boolean =
    isReadableFile(p) &&
      (!Option(p.getFileName).exists(name =>
        ShardedSafetensors.isIndex(name.toString)
      ) || SafetensorsIndex.completeShards(p).isDefined)

  /** A split model weighs its shards. */
  private def present(p: Path): CacheEntry =
    CacheEntry.Present(
      p,
      SafetensorsIndex
        .completeShards(p)
        .getOrElse(List(p))
        .map(file =>
          try Files.size(file)
          catch { case NonFatal(_) => 0L }
        )
        .sum
    )

  /** Follows symlinks on purpose: the HuggingFace cache stores every snapshot
    * entry as a link into `blobs/`.
    */
  private def isReadableFile(p: Path): Boolean =
    try Files.isRegularFile(p) && Files.isReadable(p)
    catch { case NonFatal(_) => false }

  private def childDirectories(dir: Path): List[Path] =
    if (!Files.isDirectory(dir)) Nil
    else
      try {
        val stream = Files.list(dir)
        try stream.iterator().asScala.filter(Files.isDirectory(_)).toList
        finally stream.close()
      } catch { case NonFatal(_) => Nil }

  private def childFiles(dir: Path): List[Path] =
    if (!Files.isDirectory(dir)) Nil
    else
      try {
        val stream = Files.list(dir)
        try stream.iterator().asScala.filter(Files.isRegularFile(_)).toList
        finally stream.close()
      } catch { case NonFatal(_) => Nil }
}

object ModelCache {

  /** HuggingFace names a repo directory `models--{owner}--{name}`. */
  def repoDirectoryName(repo: String): String =
    "models--" + repo.replace("/", "--")

  /** A `Local` path aimed straight at a HuggingFace `blobs/<sha256>` file is
    * rewritten to the snapshot symlink referencing it. The symlink carries the
    * file's real name, and sd-cpp sniffs the weight format from the extension
    * (`.safetensors` vs `.gguf`) — a bare content hash has none. Same bytes
    * either way; anything that is not a blob path passes through untouched.
    */
  def preferSnapshotPath(p: Path): Path = {
    val isBlob =
      Option(p.getParent).exists(_.getFileName.toString == "blobs")
    val snapshots =
      Option(p.getParent)
        .flatMap(b => Option(b.getParent))
        .map(_.resolve("snapshots"))
    snapshots.filter(s => isBlob && Files.isDirectory(s)) match {
      case None               => p
      case Some(snapshotsDir) =>
        try {
          val real = p.toRealPath()
          val stream = Files.walk(snapshotsDir)
          try
            stream
              .iterator()
              .asScala
              .find(candidate =>
                Files.isSymbolicLink(candidate) && {
                  try candidate.toRealPath() == real
                  catch { case NonFatal(_) => false }
                }
              )
              .getOrElse(p)
          finally stream.close()
        } catch { case NonFatal(_) => p }
    }
  }

  /** A readable path segment: lowercase, non-alphanumerics collapsed to `-`.
    *
    * Falls back to the empty string when nothing survives — Civitai names like
    * `Premium 臻享版` slug to nothing — and callers append an id, so the segment
    * is still unique.
    */
  def slug(raw: String, maxLength: Int = 48): String = {
    val cleaned = raw.trim.toLowerCase
      .map(c => if (c.isLetterOrDigit && c < 128) c else '-')
      .mkString
    val collapsed = cleaned.split("-+").filter(_.nonEmpty).mkString("-")
    collapsed.take(maxLength).stripSuffix("-")
  }
}
