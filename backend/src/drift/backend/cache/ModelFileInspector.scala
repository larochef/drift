package drift.backend.cache

import drift.shared.*

import java.io.*
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.util.control.NonFatal

/** What a weight file holds, read from its header alone
  * (`specs/25-model-conversion.md`): the dtype mix, the parameter count, and
  * the quantization markers — enough to say whether sd-cpp can convert the file
  * as it is, whether drift must dequantize it first, and how big the result
  * will be, without reading the weights.
  */
object ModelFileInspector {

  /** Storage size per element of a safetensors dtype, named as sd-cpp names the
    * type.
    */
  private val safetensorsTypes: Map[String, (String, Int)] = Map(
    "F32" -> ("f32", 4),
    "F16" -> ("f16", 2),
    "BF16" -> ("bf16", 2),
    "F64" -> ("f64", 8),
    "F8_E4M3" -> ("f8_e4m3", 1),
    "F8_E4M3FN" -> ("f8_e4m3", 1),
    "F8_E5M2" -> ("f8_e5m2", 1),
    "I8" -> ("i8", 1),
    "U8" -> ("u8", 1),
    "I16" -> ("i16", 2),
    "I32" -> ("i32", 4),
    "I64" -> ("i64", 8),
    "BOOL" -> ("bool", 1)
  )

  /** ggml's type ids: name, block size, bytes per block. */
  private val ggmlTypes: Map[Int, (String, Int, Int)] = Map(
    0 -> ("f32", 1, 4),
    1 -> ("f16", 1, 2),
    2 -> ("q4_0", 32, 18),
    3 -> ("q4_1", 32, 20),
    6 -> ("q5_0", 32, 22),
    7 -> ("q5_1", 32, 24),
    8 -> ("q8_0", 32, 34),
    9 -> ("q8_1", 32, 36),
    10 -> ("q2_K", 256, 84),
    11 -> ("q3_K", 256, 110),
    12 -> ("q4_K", 256, 144),
    13 -> ("q5_K", 256, 176),
    14 -> ("q6_K", 256, 210),
    15 -> ("q8_K", 256, 292),
    16 -> ("iq2_xxs", 256, 66),
    17 -> ("iq2_xs", 256, 74),
    18 -> ("iq3_xxs", 256, 98),
    19 -> ("iq1_s", 256, 50),
    20 -> ("iq4_nl", 32, 18),
    21 -> ("iq3_s", 256, 110),
    22 -> ("iq2_s", 256, 82),
    23 -> ("iq4_xs", 256, 136),
    24 -> ("i8", 1, 1),
    25 -> ("i16", 1, 2),
    26 -> ("i32", 1, 4),
    27 -> ("i64", 1, 8),
    28 -> ("f64", 1, 8),
    29 -> ("iq1_m", 256, 56),
    30 -> ("bf16", 1, 2),
    34 -> ("tq1_0", 256, 54),
    35 -> ("tq2_0", 256, 66)
  )

  private val MaxStringBytes = 256L * 1024 * 1024

  def inspect(path: Path): Either[String, ModelFileInfo] =
    try {
      if (!Files.isRegularFile(path)) Left(s"'$path' is not a file")
      else {
        val name = path.getFileName.toString.toLowerCase
        if (name.endsWith(".safetensors")) inspectSafetensors(path)
        else if (name.endsWith(".gguf")) inspectGguf(path)
        else Left(s"'${path.getFileName}' is neither safetensors nor GGUF")
      }
    } catch {
      case _: EOFException =>
        Left(s"'${path.getFileName}' ends before its header does")
      case NonFatal(err) =>
        Left(s"'${path.getFileName}' could not be read: ${err.getMessage}")
    }

  // ----------------------------------------------------------- safetensors

  private def inspectSafetensors(path: Path): Either[String, ModelFileInfo] =
    SafetensorsHeader.read(path).map { header =>
      val tensors = header.entries.filterNot((name, _) =>
        name.endsWith(SafetensorsHeader.ComfyQuantSuffix)
      )
      val bytesByType = tensors.values
        .groupMapReduce(entry =>
          safetensorsTypes.get(entry.dtype).map(_._1).getOrElse(entry.dtype)
        )(_.byteLength)(_ + _)
      // The first marker says what every quantized tensor of the file is;
      // ComfyUI writes one per module, all alike.
      val comfyQuant = header.comfyQuantConfigs.values.headOption
      val scaledFp8 =
        header.entries.contains(SafetensorsHeader.ScaledFp8Marker) ||
          tensors.exists((name, entry) =>
            SafetensorsHeader.Fp8Types(entry.dtype) &&
              header.scaleOf(name).isDefined
          )
      ModelFileInfo(
        format = "safetensors",
        tensorCount = tensors.size,
        parameterCount = tensors.values.map(_.elements).sum,
        bytesByType = bytesByType,
        comfyQuant = comfyQuant,
        scaledFp8 = scaledFp8
      )
    }

  // ------------------------------------------------------------------ gguf

  private def inspectGguf(path: Path): Either[String, ModelFileInfo] = {
    val stream = BufferedInputStream(Files.newInputStream(path))
    try {
      val input = LittleEndianInput(stream)
      val magic = input.bytes(4)
      if (!(magic sameElements "GGUF".getBytes(StandardCharsets.US_ASCII)))
        return Left(s"'${path.getFileName}' is not a GGUF file")
      val version = input.u32
      if (version < 2 || version > 3)
        return Left(s"'${path.getFileName}' is GGUF version $version")
      val tensorCount = input.u64
      val keyValueCount = input.u64
      var read = 0L
      while (read < keyValueCount) {
        input.string
        input.skipValue(input.u32.toInt)
        read += 1
      }
      var bytesByType = Map.empty[String, Long]
      var parameterCount = 0L
      var index = 0L
      while (index < tensorCount) {
        input.string
        val dimensions = input.u32.toInt
        val elements = (0 until dimensions).map(_ => input.u64).product
        val typeId = input.u32.toInt
        input.u64
        val (typeName, blockSize, blockBytes) =
          ggmlTypes.getOrElse(typeId, (s"type$typeId", 1, 0))
        parameterCount += elements
        bytesByType = bytesByType.updatedWith(typeName)(previous =>
          Some(previous.getOrElse(0L) + elements / blockSize * blockBytes)
        )
        index += 1
      }
      Right(
        ModelFileInfo(
          format = "gguf",
          tensorCount = tensorCount.toInt,
          parameterCount = parameterCount,
          bytesByType = bytesByType
        )
      )
    } finally stream.close()
  }

  /** GGUF is little-endian throughout. */
  final private class LittleEndianInput(stream: InputStream) {
    private val scratch = Array.ofDim[Byte](8)

    private def fill(count: Int): ByteBuffer = {
      var read = 0
      while (read < count) {
        val got = stream.read(scratch, read, count - read)
        if (got < 0) throw EOFException()
        read += got
      }
      ByteBuffer.wrap(scratch, 0, count).order(ByteOrder.LITTLE_ENDIAN)
    }

    def u8: Int = fill(1).get & 0xff
    def u16: Int = fill(2).getShort & 0xffff
    def u32: Long = fill(4).getInt & 0xffffffffL
    def u64: Long = fill(8).getLong

    def bytes(count: Int): Array[Byte] = {
      val result = Array.ofDim[Byte](count)
      var read = 0
      while (read < count) {
        val got = stream.read(result, read, count - read)
        if (got < 0) throw EOFException()
        read += got
      }
      result
    }

    def string: String = {
      val length = u64
      if (length < 0 || length > MaxStringBytes)
        throw IllegalStateException(s"string of $length bytes")
      String(bytes(length.toInt), StandardCharsets.UTF_8)
    }

    /** Skips one metadata value of GGUF type `valueType`. */
    def skipValue(valueType: Int): Unit = valueType match {
      case 0 | 1 | 7    => u8
      case 2 | 3        => u16
      case 4 | 5 | 6    => u32
      case 10 | 11 | 12 => u64
      case 8            => string
      case 9            =>
        val elementType = u32.toInt
        val length = u64
        var index = 0L
        while (index < length) {
          skipValue(elementType)
          index += 1
        }
      case other =>
        throw IllegalStateException(s"unknown GGUF value type $other")
    }
  }
}
