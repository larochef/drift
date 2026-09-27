package drift.backend.cache

import drift.shared.ComfyQuantInfo

import java.io.EOFException
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Path, StandardOpenOption}

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.{named, JsonCodecMaker}

/** One tensor of a safetensors header: its dtype as the format spells it
  * (`BF16`, `F8_E4M3`, `I8`, …), its shape, and where its bytes lie in the data
  * section.
  */
case class SafetensorsEntry(
    dtype: String,
    shape: List[Long],
    @named("data_offsets") dataOffsets: List[Long]
) {
  def elements: Long = shape.product
  def byteLength: Long = dataOffsets(1) - dataOffsets(0)
}
object SafetensorsEntry {
  given JsonValueCodec[SafetensorsEntry] = JsonCodecMaker.make
}

/** The header of a safetensors file, and how its bytes are reached: eight bytes
  * of little-endian length, the JSON, then the data section. Shared by the
  * inspector, which only reads it, and the dequantizer, which rewrites the
  * file.
  */
final class SafetensorsHeader(
    val path: Path,
    /** Where the data section starts: 8 + the header's byte length. */
    val dataStart: Long,
    /** Every tensor, `__metadata__` left out. */
    val entries: Map[String, SafetensorsEntry]
) {

  /** Entries in file order — the order a streaming rewrite reads them in. */
  def inFileOrder: List[(String, SafetensorsEntry)] =
    entries.toList.sortBy(_._2.dataOffsets.head)

  /** The bytes of one tensor, read through an open channel on the file. The
    * whole tensor lands in memory, so it must fit an array.
    */
  def bytesOf(channel: FileChannel, entry: SafetensorsEntry): Array[Byte] = {
    val length = entry.byteLength
    require(
      length >= 0 && length <= Int.MaxValue - 8,
      s"tensor of $length bytes does not fit in memory"
    )
    val buffer = ByteBuffer.allocate(length.toInt)
    var position = dataStart + entry.dataOffsets.head
    while (buffer.hasRemaining) {
      val read = channel.read(buffer, position)
      if (read < 0) throw EOFException()
      position += read
    }
    buffer.array()
  }

  def withChannel[A](f: FileChannel => A): A = {
    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try f(channel)
    finally channel.close()
  }

  /** ComfyUI's quantization configs, keyed by module name: every
    * `<module>.comfy_quant` tensor holds a small JSON document.
    */
  def comfyQuantConfigs: Map[String, ComfyQuantInfo] = withChannel { channel =>
    entries.collect {
      case (name, entry) if name.endsWith(SafetensorsHeader.ComfyQuantSuffix) =>
        val config = readFromArray[SafetensorsHeader.ComfyQuantJson](
          bytesOf(channel, entry)
        )
        name.stripSuffix(SafetensorsHeader.ComfyQuantSuffix) ->
          ComfyQuantInfo(config.format, config.convrot, config.groupSize)
    }
  }

  /** The scale tensor paired with a weight, if the file has one: ComfyUI's int8
    * format and Ideogram's fp8 use `<module>.weight_scale`, ComfyUI's scaled
    * fp8 uses `<module>.scale_weight`.
    */
  def scaleOf(weightName: String): Option[String] =
    SafetensorsHeader.scaleNamesOf(weightName).find(entries.contains)
}

object SafetensorsHeader {
  val ComfyQuantSuffix = ".comfy_quant"

  /** A `scaled_fp8` file's marker tensor (ComfyUI). */
  val ScaledFp8Marker = "scaled_fp8"

  val Fp8Types: Set[String] = Set("F8_E4M3", "F8_E4M3FN", "F8_E5M2")

  /** The JSON a `<module>.comfy_quant` tensor carries. */
  case class ComfyQuantJson(
      format: String = "",
      convrot: Boolean = false,
      @named("convrot_groupsize") groupSize: Int = 0
  )
  given JsonValueCodec[ComfyQuantJson] = JsonCodecMaker.make

  private val MaxHeaderBytes = 256L * 1024 * 1024

  /** Scale names a weight may be paired with, most common first. Upstream's
    * script also accepts `<name>_scale` for a name ending in `weight` without
    * the dot.
    */
  def scaleNamesOf(weightName: String): List[String] =
    if (weightName.endsWith(".weight")) {
      val module = weightName.stripSuffix(".weight")
      List(s"$module.weight_scale", s"$module.scale_weight")
    } else if (weightName.endsWith("weight")) List(s"${weightName}_scale")
    else Nil

  /** The header is a map of tensor name to entry, plus a `__metadata__` map of
    * strings that no entry codec would accept — hence a hand-written reader
    * that skips it.
    */
  private val headerCodec: JsonValueCodec[Map[String, SafetensorsEntry]] =
    new JsonValueCodec[Map[String, SafetensorsEntry]] {
      private val entryCodec = summon[JsonValueCodec[SafetensorsEntry]]
      override def decodeValue(
          in: JsonReader,
          default: Map[String, SafetensorsEntry]
      ): Map[String, SafetensorsEntry] =
        if (in.isNextToken('{')) {
          if (in.isNextToken('}')) Map.empty
          else {
            in.rollbackToken()
            val entries = Map.newBuilder[String, SafetensorsEntry]
            while ({
              val key = in.readKeyAsString()
              if (key == "__metadata__") in.skip()
              else entries += key -> entryCodec.decodeValue(in, null)
              in.isNextToken(',')
            }) ()
            if (in.isCurrentToken('}')) entries.result()
            else in.objectEndOrCommaError()
          }
        } else in.readNullOrTokenError(default, '{')
      override def encodeValue(
          value: Map[String, SafetensorsEntry],
          out: JsonWriter
      ): Unit = out.writeNull()
      override def nullValue: Map[String, SafetensorsEntry] = Map.empty
    }

  /** Reads a file's header. Fails with the reason when the file is not a
    * safetensors file or its header is unreadable.
    */
  def read(path: Path): Either[String, SafetensorsHeader] = {
    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try {
      val lengthBytes = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
      if (channel.read(lengthBytes, 0) < 8)
        return Left(
          s"'${path.getFileName}' is too short for a safetensors file"
        )
      val headerLength = lengthBytes.flip().getLong
      if (headerLength <= 0 || headerLength > MaxHeaderBytes)
        return Left(s"'${path.getFileName}' has no safetensors header")
      val header = ByteBuffer.allocate(headerLength.toInt)
      var position = 8L
      while (header.hasRemaining) {
        val read = channel.read(header, position)
        if (read < 0)
          return Left(s"'${path.getFileName}' ends inside its header")
        position += read
      }
      val entries = readFromArray(header.array())(using headerCodec)
      Right(SafetensorsHeader(path, 8 + headerLength, entries))
    } catch {
      case err: JsonReaderException =>
        Left(s"'${path.getFileName}' has a malformed header: ${err.getMessage}")
    } finally channel.close()
  }

  /** The bytes of a header for `entries`, ready to write: the JSON with a
    * `__metadata__` block first, padded with spaces to a multiple of eight
    * bytes as the format requires, preceded by its length.
    */
  def encode(
      entries: List[(String, SafetensorsEntry)],
      metadata: Map[String, String]
  ): Array[Byte] = {
    val body = writeToString(entries.toMap)(using
      JsonCodecMaker.make[Map[String, SafetensorsEntry]]
    )
    val meta = writeToString(metadata)(using
      JsonCodecMaker.make[Map[String, String]]
    )
    val json =
      if (entries.isEmpty) s"""{"__metadata__":$meta}"""
      else s"""{"__metadata__":$meta,${body.drop(1)}"""
    val bytes = json.getBytes(StandardCharsets.UTF_8)
    val padding = (8 - bytes.length % 8) % 8
    val padded = bytes ++ Array.fill(padding)(' '.toByte)
    val out = ByteBuffer
      .allocate(8 + padded.length)
      .order(ByteOrder.LITTLE_ENDIAN)
      .putLong(padded.length.toLong)
      .put(padded)
    out.array()
  }
}
