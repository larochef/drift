package drift.runner.formats

import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.nio.charset.StandardCharsets
import java.nio.file.Path

/** One GGUF metadata value. Integers of every width are `Integer`; a `u64`
  * above `Long.MaxValue` keeps its value.
  */
enum GgufValue {
  case Integer(value: BigInt)
  case Real(value: Double)
  case Bool(value: Boolean)
  case Text(value: String)
  case Array(values: Vector[GgufValue])
}

/** A GGUF file (versions 2 and 3, little-endian): metadata, then tensor infos,
  * then the data section, each tensor at a multiple of `general.alignment`.
  *
  * GGUF lists dimensions innermost first (`ne[0]` is the row length); a `Shape`
  * is outermost first, so the order is reversed here, once.
  */
final class GgufFile(
    val source: String,
    val version: Int,
    val metadata: Map[String, GgufValue],
    val alignment: Long,
    val tensors: Map[String, StoredTensor],
    val unsupported: Map[String, String]
) extends TensorSource {

  private def value(key: String): GgufValue =
    metadata.getOrElse(key, throw new FormatException(s"$source has no $key"))

  def long(key: String): Long = value(key) match {
    case GgufValue.Integer(number) if number.isValidLong => number.toLong
    case other => throw new FormatException(s"$source: $key is $other")
  }

  def double(key: String): Double = value(key) match {
    case GgufValue.Real(number)    => number
    case GgufValue.Integer(number) => number.toDouble
    case other => throw new FormatException(s"$source: $key is $other")
  }

  def string(key: String): String = value(key) match {
    case GgufValue.Text(text) => text
    case other => throw new FormatException(s"$source: $key is $other")
  }

  def boolean(key: String): Boolean = value(key) match {
    case GgufValue.Bool(flag) => flag
    case other => throw new FormatException(s"$source: $key is $other")
  }

  def strings(key: String): Seq[String] = value(key) match {
    case GgufValue.Array(values) =>
      values.map {
        case GgufValue.Text(text) => text
        case other => throw new FormatException(s"$source: $key holds $other")
      }
    case other => throw new FormatException(s"$source: $key is $other")
  }

  def longs(key: String): Seq[Long] = value(key) match {
    case GgufValue.Array(values) =>
      values.map {
        case GgufValue.Integer(number) if number.isValidLong => number.toLong
        case other => throw new FormatException(s"$source: $key holds $other")
      }
    case other => throw new FormatException(s"$source: $key is $other")
  }

  /** `general.architecture`: the key prefix of the model's hyperparameters. */
  def architecture: String = string("general.architecture")
}

object Gguf {

  private val Magic = "GGUF".getBytes(StandardCharsets.US_ASCII)
  private val DefaultAlignment = 32L

  /** Opens and maps a GGUF file; the mapping lives as long as the result. */
  def open(path: Path): (GgufFile, MappedFile) = {
    val mapped = new MappedFile(path)
    try (read(mapped.segment, path.toString), mapped)
    catch {
      case error: Throwable =>
        mapped.close()
        throw error
    }
  }

  /** Reads `file`, the whole file's bytes. */
  def read(file: MemorySegment, source: String): GgufFile =
    new Reader(file, source).read()

  final private class Reader(file: MemorySegment, source: String) {
    private var position = 0L

    private def fail(message: String): Nothing =
      throw new FormatException(s"$source: $message")

    private def need(bytes: Long): Unit =
      if (bytes < 0 || position + bytes > file.byteSize())
        fail(s"truncated at byte $position (needs $bytes more)")

    private def u8(): Int = {
      need(1); val v = LittleEndian.uint8(file, position); position += 1; v
    }
    private def i8(): Int = {
      need(1); val v = LittleEndian.int8(file, position); position += 1; v
    }
    private def u16(): Int = {
      need(2); val v = LittleEndian.uint16(file, position); position += 2; v
    }
    private def i16(): Int = {
      need(2); val v = LittleEndian.int16(file, position); position += 2; v
    }
    private def u32(): Long = {
      need(4); val v = LittleEndian.uint32(file, position); position += 4; v
    }
    private def i32(): Int = {
      need(4); val v = LittleEndian.int32(file, position); position += 4; v
    }
    private def i64(): Long = {
      need(8); val v = LittleEndian.int64(file, position); position += 8; v
    }
    private def f32(): Float = {
      need(4); val v = LittleEndian.float32(file, position); position += 4; v
    }
    private def f64(): Double = {
      need(8); val v = LittleEndian.float64(file, position); position += 8; v
    }

    private def u64(): BigInt = {
      val v = i64()
      if (v >= 0) BigInt(v) else BigInt(v) + (BigInt(1) << 64)
    }

    /** A count or a length: a `u64` that must fit what follows it. */
    private def count(what: String): Long = {
      val v = i64()
      if (v < 0 || v > file.byteSize()) fail(s"$what of $v is not plausible")
      v
    }

    private def text(): String = {
      val length = count("string length")
      need(length)
      val bytes = file.asSlice(position, length).toArray(JAVA_BYTE)
      position += length
      new String(bytes, StandardCharsets.UTF_8)
    }

    private def value(valueType: Long): GgufValue = valueType match {
      case 0 => GgufValue.Integer(u8())
      case 1 => GgufValue.Integer(i8())
      case 2 => GgufValue.Integer(u16())
      case 3 => GgufValue.Integer(i16())
      case 4 => GgufValue.Integer(u32())
      case 5 => GgufValue.Integer(i32())
      case 6 => GgufValue.Real(f32())
      case 7 => GgufValue.Bool(u8() != 0)
      case 8 => GgufValue.Text(text())
      case 9 =>
        val elementType = u32()
        val size = count("array length")
        GgufValue.Array(Vector.fill(size.toInt)(value(elementType)))
      case 10    => GgufValue.Integer(u64())
      case 11    => GgufValue.Integer(i64())
      case 12    => GgufValue.Real(f64())
      case other => fail(s"unknown metadata value type $other")
    }

    final private case class Info(
        name: String,
        dimensions: Vector[Long],
        typeId: Int,
        offset: Long
    )

    def read(): GgufFile = {
      need(8)
      if (!file.asSlice(0, 4).toArray(JAVA_BYTE).sameElements(Magic))
        fail("not a GGUF file")
      position = 4
      val version = u32()
      if (version != 2 && version != 3)
        fail(
          if (version > 0xffff) "big-endian GGUF is not supported"
          else s"GGUF version $version is not supported"
        )
      val tensorCount = count("tensor count")
      val metadataCount = count("metadata count")
      val metadata = (0L until metadataCount).map { _ =>
        val key = text()
        key -> value(u32())
      }.toMap

      val alignment = metadata.get("general.alignment") match {
        case Some(GgufValue.Integer(number))
            if number > 0 && number.bitCount == 1 =>
          number.toLong
        case None  => DefaultAlignment
        case other => fail(s"general.alignment is $other")
      }

      val infos = (0L until tensorCount).map { _ =>
        val name = text()
        val rank = u32()
        if (rank > 8) fail(s"$name has $rank dimensions")
        val dimensions = Vector.fill(rank.toInt)(count(s"$name dimension"))
        Info(name, dimensions, u32().toInt, i64())
      }

      val dataStart = (position + alignment - 1) / alignment * alignment
      val dataBytes = file.byteSize() - dataStart
      if (dataBytes < 0) fail("truncated before the data section")

      val described = infos.map { info =>
        val shape = Shape(info.dimensions.reverse)
        info.name -> DType.fromGgmlId(info.typeId).map { dtype =>
          if (info.dimensions.headOption.exists(_ % dtype.blockElements != 0))
            fail(
              s"${info.name}: rows of ${info.dimensions.head} are not whole $dtype blocks"
            )
          if (info.offset % alignment != 0)
            fail(s"${info.name} is not aligned to $alignment")
          val tensor =
            StoredTensor(info.name, dtype, shape, dataStart + info.offset, file)
          if (info.offset + tensor.byteSize > dataBytes)
            fail(s"${info.name} runs past the end of the file")
          tensor
        }
      }

      checkSpans(infos, described.toMap, dataBytes, alignment)

      val tensors = described.collect { case (name, Right(tensor)) =>
        name -> tensor
      }.toMap
      tensors.values.foreach { tensor =>
        tensor.dtype
          .rejects(tensor.bytes)
          .foreach(reason => fail(s"${tensor.name}: $reason"))
      }

      new GgufFile(
        source,
        version.toInt,
        metadata,
        alignment,
        tensors,
        described.collect { case (name, Left(reason)) => name -> reason }.toMap
      )
    }

    /** Each tensor's type fixes its byte size; the file fixes the room between
      * its offset and the next tensor's. They must agree up to the alignment
      * padding: a type id read as the wrong type (a fork's id taken by an
      * official type, say) nearly always breaks that.
      */
    private def checkSpans(
        infos: Seq[Info],
        described: Map[String, Either[String, StoredTensor]],
        dataBytes: Long,
        alignment: Long
    ): Unit = {
      val byOffset = infos.sortBy(_.offset)
      byOffset.zipWithIndex.foreach { (info, index) =>
        described(info.name).foreach { tensor =>
          val end = byOffset.lift(index + 1).fold(dataBytes)(_.offset)
          val room = end - info.offset
          if (room < tensor.byteSize || room - tensor.byteSize >= alignment)
            fail(
              s"${info.name} is ${tensor.dtype} ${tensor.shape}, ${tensor.byteSize} bytes, but the file gives it $room"
            )
        }
      }
    }
  }
}
