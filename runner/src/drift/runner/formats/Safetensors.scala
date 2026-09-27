package drift.runner.formats

import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** One safetensors file: eight bytes of little-endian header length, the JSON
  * header, then the data section every `data_offsets` is relative to.
  */
final class SafetensorsFile(
    val source: String,
    val metadata: Map[String, String],
    val tensors: Map[String, StoredTensor],
    val unsupported: Map[String, String]
) extends TensorSource {

  /** The scale tensor paired with a weight (ComfyUI int8 and Ideogram fp8:
    * `<module>.weight_scale`; ComfyUI scaled fp8: `<module>.scale_weight`;
    * upstream's script: `<name>_scale`), the same rule as the backend's
    * dequantizer (`specs/25`).
    */
  def scaleOf(weightName: String): Option[StoredTensor] = {
    val candidates =
      if (weightName.endsWith(".weight")) {
        val module = weightName.stripSuffix(".weight")
        List(s"$module.weight_scale", s"$module.scale_weight")
      } else if (weightName.endsWith("weight")) List(s"${weightName}_scale")
      else Nil
    candidates.collectFirst(Function.unlift(tensors.get))
  }
}

object Safetensors {

  private val MaxHeaderBytes = 256L * 1024 * 1024

  /** Reads the header of `file`, the whole file's bytes. */
  def read(file: MemorySegment, source: String): SafetensorsFile = {
    def fail(message: String) = throw new FormatException(s"$source: $message")
    if (file.byteSize() < 8) fail("too short for a safetensors header")
    val headerBytes = LittleEndian.int64(file, 0)
    if (headerBytes < 2 || headerBytes > MaxHeaderBytes)
      fail(s"header length $headerBytes is not plausible")
    if (8 + headerBytes > file.byteSize())
      fail(s"header of $headerBytes bytes runs past the end of the file")
    val dataStart = 8 + headerBytes
    val json = ujson.read(
      new String(
        file
          .asSlice(8, headerBytes)
          .toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),
        StandardCharsets.UTF_8
      )
    )
    val entries = json.objOpt.getOrElse(fail("header is not a JSON object"))

    val metadata = entries
      .get("__metadata__")
      .flatMap(_.objOpt)
      .fold(Map.empty[String, String])(_.view.mapValues(_.str).toMap)

    val described = entries.view
      .filterKeys(_ != "__metadata__")
      .map { (name, entry) =>
        val dtypeName = entry("dtype").str
        val shape = Shape(entry("shape").arr.map(_.num.toLong).toVector)
        val offsets = entry("data_offsets").arr.map(_.num.toLong)
        if (offsets.size != 2 || offsets(0) > offsets(1))
          fail(s"$name has data offsets $offsets")
        if (dataStart + offsets(1) > file.byteSize())
          fail(s"$name runs past the end of the file")
        name -> DType.fromSafetensorsName(dtypeName).map { dtype =>
          val length = offsets(1) - offsets(0)
          if (length != dtype.byteSize(shape.elementCount))
            fail(s"$name is $dtypeName $shape but holds $length bytes")
          StoredTensor(name, dtype, shape, dataStart + offsets(0), file)
        }
      }
      .toMap

    new SafetensorsFile(
      source,
      metadata,
      described.collect { case (name, Right(tensor)) => name -> tensor },
      described.collect { case (name, Left(reason)) => name -> reason }
    )
  }
}

/** A model in one safetensors file, or sharded over several by
  * `model.safetensors.index.json` (`specs/34`). Every shard stays mapped until
  * the model closes.
  */
final class SafetensorsModel private (
    val files: Seq[SafetensorsFile],
    mappings: Seq[MappedFile]
) extends TensorSource
    with AutoCloseable {

  val tensors: Map[String, StoredTensor] = files.flatMap(_.tensors).toMap
  val unsupported: Map[String, String] = files.flatMap(_.unsupported).toMap

  def close(): Unit = mappings.foreach(_.close())
}

object SafetensorsModel {

  /** Opens `path`: a `.safetensors` file, or a shard index whose shards sit
    * beside it.
    */
  def open(path: Path): SafetensorsModel =
    if (path.getFileName.toString.endsWith(".json")) {
      val index = ujson.read(Files.readString(path))
      val shards =
        index("weight_map").obj.values.map(_.str).toSeq.distinct.sorted
      val mappings = shards.map { shard =>
        val shardPath = path.resolveSibling(shard)
        if (!Files.isRegularFile(shardPath))
          throw new FormatException(
            s"$path names shard $shard, which is missing"
          )
        new MappedFile(shardPath)
      }
      val model = new SafetensorsModel(
        mappings.map(m => Safetensors.read(m.segment, m.path.toString)),
        mappings
      )
      val missing =
        index("weight_map").obj.keys.filterNot(model.tensors.contains).toSeq
      if (missing.nonEmpty) {
        model.close()
        throw new FormatException(
          s"$path lists tensors no shard holds: ${missing.sorted.take(5).mkString(", ")}"
        )
      }
      model
    } else {
      val mapped = new MappedFile(path)
      new SafetensorsModel(
        Seq(Safetensors.read(mapped.segment, path.toString)),
        Seq(mapped)
      )
    }
}
