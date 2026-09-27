package drift.backend.cache

import drift.shared.ShardedSafetensors

import java.nio.file.*
import scala.util.control.NonFatal

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.{
  CodecMakerConfig,
  JsonCodecMaker
}

/** Reads a split safetensors model's index (`specs/34-sharded-safetensors.md`):
  * `weight_map` names, for every tensor, the shard holding it, relative to the
  * index's own folder — where sd-cpp looks for them.
  */
object SafetensorsIndex {
  private case class IndexFile(weight_map: Map[String, String] = Map.empty)
  // jsoniter caps a decoded map at 1024 entries by default, against hostile
  // input; an index maps every tensor, and SenseNova U1.5's alone has 1116.
  private given JsonValueCodec[IndexFile] =
    JsonCodecMaker.make(CodecMakerConfig.withMapMaxInsertNumber(1048576))

  /** An index is a few hundred kilobytes; anything much larger named like one
    * is not worth reading into memory.
    */
  private val MaxIndexBytes = 16L * 1024 * 1024

  /** The shard file names an index lists, each once, or why there are none. */
  def shardNames(index: Path): Either[String, List[String]] =
    try
      if (Files.size(index) > MaxIndexBytes)
        Left(s"${Files.size(index)} bytes is too large for an index")
      else {
        val names = readFromArray[IndexFile](
          Files.readAllBytes(index)
        ).weight_map.values.toList.distinct.sorted
        if (names.isEmpty) Left("its weight_map lists no shards")
        else Right(names)
      }
    catch {
      case NonFatal(err) =>
        Left(Option(err.getMessage).getOrElse(err.toString))
    }

  /** The shards beside an index when every one of them is readable; None for a
    * path that is not an index, or an index with a shard missing.
    */
  def completeShards(index: Path): Option[List[Path]] =
    Option(index.getFileName)
      .map(_.toString)
      .filter(ShardedSafetensors.isIndex)
      .flatMap(_ => shardNames(index).toOption)
      .map(_.map(index.resolveSibling))
      .filter(
        _.forall(shard =>
          try Files.isRegularFile(shard) && Files.isReadable(shard)
          catch { case NonFatal(_) => false }
        )
      )
}
