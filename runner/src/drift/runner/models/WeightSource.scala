package drift.runner.models

import drift.runner.formats.*
import drift.runner.ops.{ConvRotLevels, ConvRotParts, Ops}
import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.nio.file.{Files, Path}
import scala.collection.mutable

/** A model's weights where the backend reads them in place: one GGUF, one
  * safetensors file, or a shard index. Tensors are views over the mapped files,
  * never copies; closing unmaps them. `copied` weights are the exception, and
  * so are fp8 E4M3 safetensors weights, dequantized to BF16 (times their scale
  * tensor, ComfyUI's scaled fp8) the first time they are asked for, and
  * ComfyUI's int8 and 4-bit linears (`ComfyQuant`), decoded to BF16 likewise or
  * kept undecoded (`ComfyQuantStorage`). Every tensor goes by the name the
  * loaders read (`WeightNames`), whatever prefix or naming the file stores it
  * under.
  */
final class WeightSource private (
    stored: Map[String, StoredTensor],
    /** A stored tensor as the backend reads it. */
    load: StoredTensor => Tensor,
    /** A stored linear's weight as `Ops.linear` reads it. */
    loadLinear: StoredTensor => Tensor,
    /** Rows of a stored matrix, decoded on the host. */
    rowsOf: (StoredTensor, Long, Int) => Array[Float],
    /** A stored tensor's shape as the backend reads it. */
    shapeOf: StoredTensor => Shape,
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

  private def named(name: String): StoredTensor =
    stored.getOrElse(
      name,
      throw new NoSuchElementException(s"the weights have no tensor $name")
    )

  /** A linear layer's weight `[N, K]`, for `Ops.linear` and `Ops.linears`
    * alone: left where the file holds it when the backend decodes it there
    * (ComfyUI's rotated linears, `ComfyQuantStorage.InPlace`), else as `apply`
    * gives it. Nothing else reads such a weight: an embedding table, or a
    * weight transformed at load, is asked with `apply`.
    */
  def linear(name: String): Tensor = loadLinear(named(name))

  /** Rows of a matrix as floats on the host, whatever stores them: for weights
    * transformed at load.
    */
  def hostRows(name: String, first: Long, count: Int): Array[Float] =
    rowsOf(named(name), first, count)

  /** The type the file stores a tensor as; `apply` may give another (fp8 and
    * ComfyUI's quantized weights come decoded).
    */
  def storedAs(name: String): DType = named(name).dtype

  def names: Iterable[String] = stored.keys

  /** A tensor's shape, nothing loaded. */
  def shape(name: String): Shape =
    shapeOf(
      stored.getOrElse(
        name,
        throw new NoSuchElementException(s"the weights have no tensor $name")
      )
    )

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

/** How ComfyUI's rotated linears are held (`bugs/53`):
  * `-Ddrift.comfyQuant=inplace`, `gpu` or `cpu`.
  */
enum ComfyQuantStorage {

  /** Where the file holds them, decoded by each product: no memory but the
    * file's.
    */
  case InPlace

  /** BF16, decoded by the GPU at load. */
  case DecodedOnGpu

  /** BF16, decoded on the CPU at load: the reference. */
  case Decoded
}

object ComfyQuantStorage {
  def fromProperty: ComfyQuantStorage =
    sys.props.get("drift.comfyQuant") match {
      case None | Some("inplace") => InPlace
      case Some("gpu")            => DecodedOnGpu
      case Some("cpu")            => Decoded
      case Some(other)            =>
        throw new IllegalArgumentException(
          s"drift.comfyQuant is inplace, gpu or cpu, not $other"
        )
    }
}

object WeightSource {

  def open(
      ops: Ops,
      path: Path,
      comfyQuant: ComfyQuantStorage = ComfyQuantStorage.fromProperty
  ): WeightSource = {
    val name = path.getFileName.toString
    if (name.endsWith(".gguf")) {
      val mapped = ops.mapFile(path)
      val file = Gguf.read(mapped.segment, path.toString)
      new WeightSource(
        renamed(file.tensors),
        _.tensor(mapped.storage),
        _.tensor(mapped.storage),
        plainRows(_, _, _),
        _.shape,
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
      val files = mappings
        .zip(shards)
        .map((mapped, shard) =>
          mapped -> Safetensors.read(mapped.segment, shard.toString)
        )
      // by the stored name, which a tensor keeps
      val quantized = files.flatMap((_, file) => ComfyQuant.weights(file)).toMap
      val stored = renamed(
        files.flatMap { (mapped, file) =>
          file.tensors.view.mapValues { tensor =>
            val scale = Option
              .when(tensor.dtype == DType.F8E4M3)(file.scaleOf(tensor.name))
              .flatten
              .fold(1f)(_.decode().head)
            (tensor, mapped, scale)
          }
        }.toMap
      )
      val config = Option(path.resolveSibling("config.json"))
        .filter(Files.isRegularFile(_))
        .map(ModelConfig.read)
      // by the stored name, which a tensor keeps
      val placed =
        stored.values
          .map((tensor, mapped, scale) => tensor.name -> (mapped, scale))
          .toMap
      val dequantized = mutable.Map.empty[String, Tensor]
      // the rotated weights left in the file, and the scales made for them
      val rotated = mutable.Map.empty[String, (Tensor, Seq[Tensor])]
      val load = (tensor: StoredTensor) => {
        val (mapped, scale) = placed(tensor.name)
        quantized.get(tensor.name) match {
          case Some(weight) =>
            dequantized.getOrElseUpdate(
              tensor.name,
              weight.stored
                .filter(_ => comfyQuant != ComfyQuantStorage.Decoded)
                .fold(toBf16(ops, weight)) { stored =>
                  val (codes, parts) =
                    rotatedInPlace(ops, stored, mapped.storage)
                  try {
                    val out = ops.allocate(DType.BF16, weight.shape)
                    ops.convert(codes, out)
                    out
                  } finally {
                    ops.forgetRotated(codes)
                    parts.foreach(ops.release)
                  }
                }
            )
          case None =>
            val inPlace = tensor.tensor(mapped.storage)
            if (tensor.dtype != DType.F8E4M3) inPlace
            else
              dequantized.getOrElseUpdate(
                tensor.name,
                toBf16(ops, inPlace, scale)
              )
        }
      }
      new WeightSource(
        stored.view.mapValues(_._1).toMap,
        load,
        tensor =>
          quantized
            .get(tensor.name)
            .flatMap(_.stored)
            .filter(_ => comfyQuant == ComfyQuantStorage.InPlace)
            .fold(load(tensor))(stored =>
              rotated
                .getOrElseUpdate(
                  tensor.name,
                  rotatedInPlace(ops, stored, placed(tensor.name)._1.storage)
                )
                ._1
            ),
        (tensor, first, count) =>
          quantized.get(tensor.name) match {
            case Some(weight) => weight.rows(first, count)
            case None         =>
              val values = plainRows(tensor, first, count)
              val scale = placed(tensor.name)._2
              if (scale == 1f) values else values.map(_ * scale)
          },
        tensor => quantized.get(tensor.name).fold(tensor.shape)(_.shape),
        () => {
          rotated.values.foreach { (codes, parts) =>
            ops.forgetRotated(codes)
            parts.foreach(ops.release)
          }
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
      val copy = (tensor: StoredTensor) =>
        copies.getOrElseUpdate(
          tensor.name,
          ops.fromBytes(
            tensor.dtype,
            tensor.shape,
            tensor.bytes.toArray(JAVA_BYTE)
          )
        )
      new WeightSource(
        renamed(file.tensors),
        copy,
        copy,
        plainRows(_, _, _),
        _.shape,
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

  /** Rows `first until first + count` of a matrix stored in a plain type,
    * decoded on the host.
    */
  private def plainRows(
      tensor: StoredTensor,
      first: Long,
      count: Int
  ): Array[Float] = {
    val columns = tensor.shape.elementCount / tensor.shape.dimensions.head
    tensor.dtype.decode(
      tensor.bytes.asSlice(
        tensor.dtype.byteSize(first * columns),
        tensor.dtype.byteSize(count * columns)
      ),
      (count * columns).toInt
    )
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

  /** A ComfyUI quantized linear's weight as it was before quantization, in
    * BF16: decoded on the CPU a few million values at a time.
    */
  private def toBf16(ops: Ops, weight: ComfyQuant.Weight): Tensor = {
    val rows = weight.shape.dimensions(0)
    val columns = weight.shape.dimensions(1)
    val chunk = math.min(rows, math.max(1L, (1L << 24) / columns)).toInt
    val out = ops.allocate(DType.BF16, weight.shape)
    var row = 0L
    while (row < rows) {
      val count = math.min(chunk.toLong, rows - row).toInt
      val part =
        ops.fromFloats(Shape.of(count, columns), weight.rows(row, count))
      try ops.convert(part, out.rows(row, count))
      finally ops.release(part)
      row += count
    }
    out
  }

  /** A ComfyUI rotated linear left where the file holds it (`Ops.rotated`), and
    * the tensors made for its scales, the caller's to release.
    */
  private def rotatedInPlace(
      ops: Ops,
      stored: ComfyQuant.Stored,
      storage: Storage
  ): (Tensor, Seq[Tensor]) = {
    val scaleValues = stored.scales()
    val scales = ops.fromFloats(Shape.of(scaleValues.length), scaleValues)
    val levels = stored.levels.map(levels =>
      ConvRotLevels(
        ops.fromFloats(Shape.of(16), levels.codebook()),
        levels.relative.tensor(storage),
        levels.groupSize
      )
    )
    (
      ops.rotated(stored.codes.tensor(storage), ConvRotParts(scales, levels)),
      scales +: levels.map(_.codebook).toSeq
    )
  }

  /** `tensors` by the names the loaders read. */
  private def renamed[A](tensors: Map[String, A]): Map[String, A] =
    WeightNames
      .canonical(tensors.keys)
      .map((name, stored) => name -> tensors(stored))
}
