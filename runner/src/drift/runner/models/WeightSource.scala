package drift.runner.models

import drift.runner.formats.*
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, MappedFile, Shape, Tensor}

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.nio.file.{Files, Path}
import scala.collection.mutable

/** A model's weights where the backend reads them in place: one GGUF, one
  * safetensors file, or a shard index. Tensors are views over the mapped files,
  * never copies; closing unmaps them. `copied` weights are the exception, and
  * so are fp8 E4M3 safetensors weights, dequantized to BF16 (times their scale
  * tensor, ComfyUI's scaled fp8) the first time they are asked for. Every
  * tensor goes by the name the loaders read (`WeightNames`), whatever prefix or
  * naming the file stores it under.
  */
final class WeightSource private (
    stored: Map[String, StoredTensor],
    /** A stored tensor as the backend reads it. */
    load: StoredTensor => Tensor,
    closing: () => Unit,
    /** The GGUF, when the weights are one. */
    val gguf: Option[GgufFile],
    /** `config.json` beside safetensors weights, when there is one. */
    val config: Option[ModelConfig]
) extends AutoCloseable {

  def has(name: String): Boolean = stored.contains(name)

  def apply(name: String): Tensor =
    stored.get(name) match {
      case Some(tensor) => load(tensor)
      case None         =>
        throw new NoSuchElementException(s"the weights have no tensor $name")
    }

  def names: Iterable[String] = stored.keys

  /** A tensor's shape, nothing loaded. */
  def shape(name: String): Shape =
    stored
      .getOrElse(
        name,
        throw new NoSuchElementException(s"the weights have no tensor $name")
      )
      .shape

  /** A tensor's bytes as the file holds them, on the host. */
  def bytes(name: String): MemorySegment =
    stored
      .getOrElse(
        name,
        throw new NoSuchElementException(s"the weights have no tensor $name")
      )
      .bytes

  def close(): Unit = closing()
}

object WeightSource {

  def open(ops: Ops, path: Path): WeightSource = {
    val name = path.getFileName.toString
    if (name.endsWith(".gguf")) {
      val mapped = ops.mapFile(path)
      val file = Gguf.read(mapped.segment, path.toString)
      new WeightSource(
        renamed(file.tensors),
        _.tensor(mapped.storage),
        () => mapped.close(),
        Some(file),
        None
      )
    } else {
      val shards =
        if (name.endsWith(".json"))
          ujson
            .read(Files.readString(path))("weight_map")
            .obj
            .values
            .map(_.str)
            .toSeq
            .distinct
            .sorted
            .map(path.resolveSibling)
        else Seq(path)
      val mappings = shards.map(ops.mapFile)
      val stored = renamed(
        mappings
          .zip(shards)
          .flatMap { (mapped, shard) =>
            val file = Safetensors.read(mapped.segment, shard.toString)
            file.tensors.view.mapValues { tensor =>
              val scale = Option
                .when(tensor.dtype == DType.F8E4M3)(file.scaleOf(tensor.name))
                .flatten
                .fold(1f)(_.decode().head)
              (tensor, mapped, scale)
            }
          }
          .toMap
      )
      val config = Option(path.resolveSibling("config.json"))
        .filter(Files.isRegularFile(_))
        .map(ModelConfig.read)
      // by the stored name, which a tensor keeps
      val placed =
        stored.values.map((tensor, mapped, scale) => tensor.name -> (mapped, scale)).toMap
      val dequantized = mutable.Map.empty[String, Tensor]
      new WeightSource(
        stored.view.mapValues(_._1).toMap,
        tensor => {
          val (mapped, scale) = placed(tensor.name)
          val inPlace = tensor.tensor(mapped.storage)
          if (tensor.dtype != DType.F8E4M3) inPlace
          else
            dequantized.getOrElseUpdate(
              tensor.name,
              toBf16(ops, inPlace, scale)
            )
        },
        () => {
          dequantized.values.foreach(ops.release)
          mappings.foreach(_.close())
        },
        None,
        config
      )
    }
  }

  /** A GGUF mapped on the host only, each tensor copied to the backend the
    * first time it is asked for: a draft model's MTP layer, a few hundred MB,
    * then comes out of a whole model's file without registering all of it.
    */
  def copied(ops: Ops, path: Path): WeightSource = {
    val mapped = new MappedFile(path)
    try {
      val file = Gguf.read(mapped.segment, path.toString)
      val copies = mutable.Map.empty[String, Tensor]
      new WeightSource(
        renamed(file.tensors),
        tensor =>
          copies.getOrElseUpdate(
            tensor.name,
            ops.fromBytes(
              tensor.dtype,
              tensor.shape,
              tensor.bytes.toArray(JAVA_BYTE)
            )
          ),
        () => {
          copies.values.foreach(ops.release)
          mapped.close()
        },
        Some(file),
        None
      )
    } catch {
      case error: Throwable =>
        mapped.close()
        throw error
    }
  }

  /** An fp8 E4M3 weight times `scale`, in BF16: through F32 a few million
    * values at a time.
    */
  private def toBf16(ops: Ops, fp8: Tensor, scale: Float): Tensor = {
    val columns = fp8.shape.dimensions.last
    val rows = fp8.shape.elementCount / columns
    val chunk = math.min(rows, math.max(1L, (1L << 24) / columns))
    val out = ops.allocate(DType.BF16, fp8.shape)
    val floats = ops.allocate(DType.F32, Shape.of(chunk, columns))
    try {
      var row = 0L
      while (row < rows) {
        val count = math.min(chunk, rows - row)
        val part = floats.rows(0, count)
        ops.convert(fp8.view(rows, columns).rows(row, count), part)
        if (scale != 1f) ops.scale(part, scale, part)
        ops.convert(part, out.view(rows, columns).rows(row, count))
        row += count
      }
    } finally ops.release(floats)
    out
  }

  /** `tensors` by the names the loaders read. */
  private def renamed[A](tensors: Map[String, A]): Map[String, A] =
    WeightNames
      .canonical(tensors.keys)
      .map((name, stored) => name -> tensors(stored))
}
