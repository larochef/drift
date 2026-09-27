package drift.backend.modelscope

import drift.backend.cache.SafetensorsIndex
import drift.backend.download.{DownloadOutcome, Downloader}
import drift.shared.*

import java.nio.file.Path

/** Fetches ModelScope files into drift's own tree (`specs/37-modelscope.md`):
  * `<root>/<owner>/<name>/<path>`, verified against the sha256 ModelScope lists
  * for every file. A split safetensors model, named by its index, brings its
  * shards beside it (`specs/34`). Transfers go through the shared
  * [[Downloader]] (resume, hash, per-host limit); what to fetch is asked of
  * ModelScope's own client.
  */
final class ModelScopeDownloads(
    api: ModelScopeApi,
    downloader: Downloader,
    root: Path
) {

  /** Where a file lands; `None` for a path that would leave the tree. */
  def targetOf(source: ModelScope): Option[Path] = {
    val base = root.toAbsolutePath.normalize
    val target =
      base
        .resolve(source.repo)
        .resolve(source.filename)
        .toAbsolutePath
        .normalize
    Option.when(target.startsWith(base) && target != base)(target)
  }

  /** The file as ModelScope lists it: size and sha256. */
  def lookup(source: ModelScope): Either[String, ModelScopeFileInfo] =
    api
      .files(source.repo)
      .flatMap(
        _.find(_.path == source.filename)
          .toRight(
            s"'${source.filename}' is not in ${source.repo} on ModelScope"
          )
      )

  /** One file to `target` through a `.part` file, verified when a hash is given
    * — what a LoRA install uses for its own folder.
    */
  def downloadTo(
      source: ModelScope,
      target: Path,
      expectedSha256: Option[String],
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    api.downloadRequest(source.repo, source.filename) match {
      case Left(reason)          => DownloadOutcome.Failed(reason)
      case Right((url, headers)) =>
        downloader.fetch(
          url = url,
          headers = headers,
          partFile =
            target.resolveSibling(target.getFileName.toString + ".part"),
          finalFile = target,
          expectedSha256 = expectedSha256,
          isCancelled = isCancelled,
          onProgress = onProgress
        )
    }

  /** A model's file into the tree, or, for a safetensors index, the index and
    * every shard it lists, one after the other.
    */
  def download(
      source: ModelScope,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    (api.files(source.repo), targetOf(source)) match {
      case (Left(reason), _) => DownloadOutcome.Failed(reason)
      case (_, None)         =>
        DownloadOutcome.Failed(s"'${source.filename}' is not a file path")
      case (Right(files), Some(target)) =>
        files.find(_.path == source.filename) match {
          case None =>
            DownloadOutcome.Failed(
              s"'${source.filename}' is not in ${source.repo} on ModelScope"
            )
          case Some(file) if !ShardedSafetensors.isIndex(source.filename) =>
            downloadTo(source, target, file.sha256, isCancelled, onProgress)
          case Some(index) =>
            downloadTo(
              source,
              target,
              index.sha256,
              isCancelled,
              (_, _) => ()
            ) match {
              case DownloadOutcome.Completed(_, _) =>
                shards(source, target, files, isCancelled, onProgress)
              case other => other
            }
        }
    }

  private def shards(
      index: ModelScope,
      indexPath: Path,
      files: List[ModelScopeFileInfo],
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome = {
    val folder = index.filename.lastIndexOf('/') match {
      case -1    => ""
      case slash => index.filename.take(slash + 1)
    }
    SafetensorsIndex.shardNames(indexPath) match {
      case Left(reason) =>
        DownloadOutcome.Failed(
          s"'${index.filename}' cannot be read as a safetensors index: $reason"
        )
      case Right(names) =>
        val listed =
          names.map(name => files.find(_.path == folder + name).toRight(name))
        listed.collectFirst { case Left(missing) => missing } match {
          case Some(missing) =>
            DownloadOutcome.Failed(
              s"'$missing', listed by '${index.filename}', is not in ${index.repo} on ModelScope"
            )
          case None =>
            val shardFiles = listed.collect { case Right(file) => file }
            val total = shardFiles.flatMap(_.size).sum
            onProgress(0L, Some(total))
            fetchShards(
              index.repo,
              shardFiles,
              0L,
              total,
              indexPath,
              isCancelled,
              onProgress
            )
        }
    }
  }

  private def fetchShards(
      repo: String,
      remaining: List[ModelScopeFileInfo],
      before: Long,
      total: Long,
      indexPath: Path,
      isCancelled: () => Boolean,
      onProgress: (Long, Option[Long]) => Unit
  ): DownloadOutcome =
    remaining match {
      case Nil                => DownloadOutcome.Completed(indexPath, before)
      case _ if isCancelled() => DownloadOutcome.Cancelled
      case file :: rest       =>
        val shard = ModelScope(repo, file.path)
        targetOf(shard) match {
          case None =>
            DownloadOutcome.Failed(s"'${file.path}' is not a file path")
          case Some(target) =>
            downloadTo(
              shard,
              target,
              file.sha256,
              isCancelled,
              (bytes, _) => onProgress(before + bytes, Some(total))
            ) match {
              case DownloadOutcome.Completed(_, bytes) =>
                fetchShards(
                  repo,
                  rest,
                  before + bytes,
                  total,
                  indexPath,
                  isCancelled,
                  onProgress
                )
              case other => other
            }
        }
    }
}
