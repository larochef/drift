package drift.backend.conversion

import drift.backend.cache.{SafetensorsEntry, SafetensorsHeader}

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.*
import scala.util.boundary
import scala.util.boundary.break
import scala.util.control.NonFatal

import ox.flow.Flow

/** The pre-step of `specs/25-model-conversion.md` (step 2): a safetensors file
  * whose weights sd-cpp's converter would not requantize — ComfyUI
  * `int8_tensorwise` (with or without `convrot`), or fp8 with scale tensors —
  * rewritten as a plain float safetensors the converter can quantize.
  *
  * int8: `w_rot = int8 × weight_scale` (one scale per output row, or one for
  * the tensor), and for `convrot` the rotation undone: the file holds `W·H` for
  * a "regular Hadamard" `H` applied in groups along the input dimension, and
  * the runtime rotates activations the same way. `H` is symmetric and
  * orthonormal, so `W = W_rot·H` — the same routine again, copied from
  * `ggml_regular_hadamard_group_f32` in leejet/ggml (`ggml-cpu.c`, the
  * submodule sd-cpp pins): scale by 1/√group, then radix-4 butterflies with the
  * 4×4 block `[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`.
  *
  * fp8: `w = fp8 × scale`, e4m3 (no infinities, NaN at all-ones) or e5m2.
  *
  * Dequantized tensors come out as F16 — a 10-bit mantissa where bf16 has 7, on
  * values a quantized source cannot push past 65504 — or as F32 when the scales
  * say a value could overflow F16. Every other tensor is copied byte for byte.
  * Scale and marker tensors are dropped.
  */
object Dequantizer {

  /** How one tensor of the source becomes one of the output. */
  private[conversion] enum Action {
    case Copy

    /** int8 × per-row scale, the rotation undone in groups when `group` > 0.
      */
    case Int8(scale: String, group: Int)
    case Fp8(scale: String, e5m2: Boolean)
  }

  private[conversion] case class Planned(
      name: String,
      source: SafetensorsEntry,
      action: Action,
      outputType: String,
      outputBytes: Long
  )

  /** What the pre-step will do to a file, decided from its header and its scale
    * tensors alone.
    */
  case class Plan(
      header: SafetensorsHeader,
      tensors: List[Planned],
      /** Scale and marker tensors the output leaves out. */
      dropped: List[String]
  ) {
    def dequantized: Int = tensors.count(_.action != Action.Copy)
    def outputBytes: Long = tensors.map(_.outputBytes).sum
  }

  /** The reason `run` gives up when asked to stop. */
  val Cancelled = "cancelled"

  private val Int8Format = "int8_tensorwise"
  private val Float16Max = 65504.0
  private val E4m3Max = 448.0
  private val E5m2Max = 57344.0

  def plan(source: Path): Either[String, Plan] =
    SafetensorsHeader.read(source).flatMap { header =>
      try header.withChannel(channel => planWith(header, channel))
      catch {
        case NonFatal(err) =>
          Left(
            s"'${source.getFileName}' could not be planned: ${err.getMessage}"
          )
      }
    }

  private def planWith(
      header: SafetensorsHeader,
      channel: FileChannel
  ): Either[String, Plan] = boundary[Either[String, Plan]] {
    val configs = header.comfyQuantConfigs
    // Some ComfyUI writers leave the `format` key out (a Flux.2 Klein 9B
    // file: `{"convrot": true, "convrot_groupsize": 256, "per_row": true}`);
    // the I8 dtype and the scale tensor are what the plan checks anyway.
    configs
      .find((_, config) =>
        config.format.nonEmpty && config.format != Int8Format
      )
      .foreach { (module, config) =>
        break(
          Left(
            s"'$module' uses ComfyUI format '${config.format}', which drift cannot dequantize."
          )
        )
      }
    // The tensors that change: int8 weights with a ComfyUI config, fp8
    // weights with a scale. A refusal on any of them refuses the file.
    val handled = header.entries.flatMap { (name, entry) =>
      val module = name.stripSuffix(".weight")
      configs.get(module) match {
        case Some(config) if name.endsWith(".weight") =>
          Some(
            name -> planInt8(header, channel, name, entry, config)
              .fold(reason => break(Left(reason)), identity)
          )
        case _
            if SafetensorsHeader.Fp8Types(entry.dtype) &&
              header.scaleOf(name).isDefined =>
          Some(
            name -> planFp8(header, channel, name, entry)
              .fold(reason => break(Left(reason)), identity)
          )
        case _ => None
      }
    }
    val scales = handled.values.collect {
      case Planned(_, _, Action.Int8(scale, _), _, _) => scale
      case Planned(_, _, Action.Fp8(scale, _), _, _)  => scale
    }.toSet
    val inputScales =
      handled.keys.map(_.stripSuffix(".weight") + ".scale_input").toSet
    def dropped(name: String): Boolean =
      name == SafetensorsHeader.ScaledFp8Marker ||
        name.endsWith(SafetensorsHeader.ComfyQuantSuffix) ||
        scales(name) || inputScales(name)
    val ordered = header.inFileOrder
    val tensors = ordered.collect {
      case (name, _) if handled.contains(name) => handled(name)
      case (name, entry) if !dropped(name)     =>
        Planned(name, entry, Action.Copy, entry.dtype, entry.byteLength)
    }
    Right(Plan(header, tensors, ordered.map(_._1).filter(dropped)))
  }

  private def planInt8(
      header: SafetensorsHeader,
      channel: FileChannel,
      name: String,
      entry: SafetensorsEntry,
      config: drift.shared.ComfyQuantInfo
  ): Either[String, Planned] =
    for {
      _ <- Either.cond(
        entry.dtype == "I8",
        (),
        s"'$name' is ComfyUI int8 but stored as ${entry.dtype}."
      )
      _ <- Either.cond(
        entry.shape.size == 2,
        (),
        s"'$name' has ${entry.shape.size} dimensions; only 2-D int8 weights are handled."
      )
      scaleName <- header.scaleOf(name).toRight(s"'$name' has no weight_scale.")
      scale <- scaleValues(header, channel, scaleName, entry.shape.head)
      // sd-cpp's loader takes any power of four that divides the input
      // width (real files use 1024); the butterflies below work for any.
      group <-
        if (!config.convrot) Right(0)
        else
          Either.cond(
            isPowerOfFour(config.groupSize) &&
              entry.shape(1) % config.groupSize == 0,
            config.groupSize,
            s"'$name' has a convrot group of ${config.groupSize} on ${entry.shape(1)} columns, which sd-cpp's loader would refuse too."
          )
    } yield {
      val bound = 127.0 * maxAbs(scale) *
        (if (group > 0) math.sqrt(group.toDouble) else 1.0)
      planned(name, entry, Action.Int8(scaleName, group), bound)
    }

  private def planFp8(
      header: SafetensorsHeader,
      channel: FileChannel,
      name: String,
      entry: SafetensorsEntry
  ): Either[String, Planned] = {
    val e5m2 = entry.dtype == "F8_E5M2"
    for {
      scaleName <- header.scaleOf(name).toRight(s"'$name' has no scale.")
      scale <- scaleValues(
        header,
        channel,
        scaleName,
        entry.shape.headOption.getOrElse(1L)
      )
    } yield {
      val bound = (if (e5m2) E5m2Max else E4m3Max) * maxAbs(scale)
      planned(name, entry, Action.Fp8(scaleName, e5m2), bound)
    }
  }

  private def planned(
      name: String,
      entry: SafetensorsEntry,
      action: Action,
      bound: Double
  ): Planned = {
    val half = bound < Float16Max
    Planned(
      name,
      entry,
      action,
      if (half) "F16" else "F32",
      entry.elements * (if (half) 2 else 4)
    )
  }

  /** A weight's scales as floats: one per output row, or a single one. */
  private def scaleValues(
      header: SafetensorsHeader,
      channel: FileChannel,
      scaleName: String,
      rows: Long
  ): Either[String, Array[Float]] = {
    val entry = header.entries(scaleName)
    val values = floatsOf(header.bytesOf(channel, entry), entry.dtype)
    Either.cond(
      values.length == 1 || values.length == rows,
      values,
      s"'$scaleName' has ${values.length} values for $rows rows."
    )
  }

  private def floatsOf(bytes: Array[Byte], dtype: String): Array[Float] = {
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    dtype match {
      case "F32" => Array.fill(bytes.length / 4)(buffer.getFloat)
      case "F16" =>
        Array.fill(bytes.length / 2)(
          java.lang.Float.float16ToFloat(buffer.getShort)
        )
      case "BF16" =>
        Array.fill(bytes.length / 2)(
          java.lang.Float.intBitsToFloat((buffer.getShort & 0xffff) << 16)
        )
      case "F64" => Array.fill(bytes.length / 8)(buffer.getDouble.toFloat)
      case other =>
        throw IllegalArgumentException(s"a scale stored as $other")
    }
  }

  private def maxAbs(values: Array[Float]): Double =
    values.foldLeft(0.0)((max, v) => math.max(max, math.abs(v.toDouble)))

  private def isPowerOfFour(n: Int): Boolean = {
    var remainder = n
    while (remainder > 1 && remainder % 4 == 0) remainder /= 4
    n >= 1 && remainder == 1
  }

  // ------------------------------------------------------------------- run

  /** Writes the dequantized file to `output`, tensor by tensor in file order,
    * reporting `(done, total)` after each. `keepGoing` is consulted before
    * every tensor; once it says no, the run ends with [[Cancelled]] and the
    * caller removes the partial output.
    */
  def run(
      source: Path,
      output: Path,
      keepGoing: () => Boolean,
      onProgress: (Int, Int) => Unit
  ): Either[String, Plan] =
    plan(source).flatMap { plan =>
      try {
        val (entries, _) = plan.tensors.foldLeft(
          (List.empty[(String, SafetensorsEntry)], 0L)
        ) { case ((done, offset), tensor) =>
          val entry = SafetensorsEntry(
            tensor.outputType,
            tensor.source.shape,
            List(offset, offset + tensor.outputBytes)
          )
          ((tensor.name -> entry) :: done, offset + tensor.outputBytes)
        }
        val headerBytes = SafetensorsHeader.encode(
          entries.reverse,
          Map(
            "format" -> "pt",
            "drift" -> s"dequantized from ${source.getFileName}"
          )
        )
        val in = FileChannel.open(source, StandardOpenOption.READ)
        try {
          val out = FileChannel.open(
            output,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING,
            StandardOpenOption.WRITE
          )
          try {
            writeFully(out, ByteBuffer.wrap(headerBytes))
            val total = plan.tensors.size
            val remaining = plan.tensors.iterator.zipWithIndex
            var cancelled = false
            while (!cancelled && remaining.hasNext) {
              val (tensor, index) = remaining.next()
              if (!keepGoing()) cancelled = true
              else {
                writeTensor(plan, in, out, tensor)
                onProgress(index + 1, total)
              }
            }
            if (cancelled) Left(Cancelled) else Right(plan)
          } finally out.close()
        } finally in.close()
      } catch {
        case NonFatal(err) =>
          Left(
            s"dequantizing '${source.getFileName}' failed: ${err.getMessage}"
          )
      }
    }

  /** One tensor of the plan, read at its source offset and written at the
    * output's current position: copied, or dequantized row by row.
    */
  private def writeTensor(
      plan: Plan,
      in: FileChannel,
      out: FileChannel,
      tensor: Planned
  ): Unit = {
    val position = plan.header.dataStart + tensor.source.dataOffsets.head
    tensor.action match {
      case Action.Copy =>
        transfer(in, position, tensor.source.byteLength, out)
      case Action.Int8(scaleName, group) =>
        val scale = scaleValues(
          plan.header,
          in,
          scaleName,
          tensor.source.shape.head
        ).fold(reason => throw IllegalStateException(reason), identity)
        val block = readFully(in, position, tensor.source.byteLength)
        writeRows(
          out,
          rows = tensor.source.shape.head.toInt,
          columns = tensor.source.shape(1).toInt,
          half = tensor.outputType == "F16"
        ) { (values, row, columns) =>
          val rowScale = if (scale.length == 1) scale(0) else scale(row)
          val base = row * columns
          var c = 0
          while (c < columns) {
            values(c) = block(base + c).toFloat * rowScale
            c += 1
          }
          if (group > 0) {
            var g = 0
            while (g < columns) {
              regularHadamard(values, g, group)
              g += group
            }
          }
        }
      case Action.Fp8(scaleName, e5m2) =>
        val rows = tensor.source.shape.headOption.getOrElse(1L)
        val scale = scaleValues(plan.header, in, scaleName, rows)
          .fold(reason => throw IllegalStateException(reason), identity)
        val block = readFully(in, position, tensor.source.byteLength)
        val table = if (e5m2) e5m2Table else e4m3Table
        writeRows(
          out,
          rows = rows.toInt,
          columns = (tensor.source.elements / math.max(rows, 1L)).toInt,
          half = tensor.outputType == "F16"
        ) { (values, row, columns) =>
          val rowScale = if (scale.length == 1) scale(0) else scale(row)
          val base = row * columns
          var c = 0
          while (c < columns) {
            values(c) = table(block(base + c) & 0xff) * rowScale
            c += 1
          }
        }
    }
  }

  /** Runs `fill` on every row (values of `columns` floats) in parallel chunks
    * and writes the rows in order as F16 or F32.
    */
  private def writeRows(
      out: FileChannel,
      rows: Int,
      columns: Int,
      half: Boolean
  )(
      fill: (Array[Float], Int, Int) => Unit
  ): Unit = {
    val rowsPerChunk = math.max(1, (4 << 20) / math.max(columns, 1))
    val starts = (0 until rows by rowsPerChunk).toList
    val width = if (half) 2 else 4
    Flow
      .fromIterable(starts)
      .mapPar(Runtime.getRuntime.availableProcessors()) { start =>
        val end = math.min(start + rowsPerChunk, rows)
        val values = new Array[Float](columns)
        val buffer = ByteBuffer
          .allocate((end - start) * columns * width)
          .order(ByteOrder.LITTLE_ENDIAN)
        var row = start
        while (row < end) {
          fill(values, row, columns)
          var c = 0
          if (half)
            while (c < columns) {
              buffer.putShort(java.lang.Float.floatToFloat16(values(c)))
              c += 1
            }
          else
            while (c < columns) {
              buffer.putFloat(values(c))
              c += 1
            }
          row += 1
        }
        buffer.flip()
        buffer
      }
      .runForeach(buffer => writeFully(out, buffer))
  }

  /** In-place regular Hadamard on `values(offset until offset + size)`, as the
    * runtime applies it to activations — and therefore what undoes the weight's
    * rotation.
    */
  private[conversion] def regularHadamard(
      values: Array[Float],
      offset: Int,
      size: Int
  ): Unit = {
    val scale = (1.0 / math.sqrt(size.toDouble)).toFloat
    var i = 0
    while (i < size) {
      values(offset + i) *= scale
      i += 1
    }
    var stride = 1
    while (stride < size) {
      var base = 0
      while (base < size) {
        var j = 0
        while (j < stride) {
          val i0 = offset + base + j
          val i1 = i0 + stride
          val i2 = i1 + stride
          val i3 = i2 + stride
          val a = values(i0)
          val b = values(i1)
          val c = values(i2)
          val d = values(i3)
          values(i0) = a + b + c - d
          values(i1) = a + b - c + d
          values(i2) = a - b + c + d
          values(i3) = -a + b + c + d
          j += 1
        }
        base += 4 * stride
      }
      stride *= 4
    }
  }

  /** fp8 e4m3 (the `fn` variant torch uses: no infinities, NaN at all ones)
    * decoded for every byte value.
    */
  private val e4m3Table: Array[Float] = Array.tabulate(256) { bits =>
    val sign = if ((bits & 0x80) != 0) -1.0f else 1.0f
    val exponent = (bits >> 3) & 0xf
    val mantissa = bits & 0x7
    if (exponent == 0) sign * (mantissa / 8.0f) * math.pow(2, -6).toFloat
    else if (exponent == 15 && mantissa == 7) Float.NaN
    else sign * (1 + mantissa / 8.0f) * math.pow(2, exponent - 7).toFloat
  }

  /** fp8 e5m2: IEEE-like, with infinities and NaNs at exponent 31. */
  private val e5m2Table: Array[Float] = Array.tabulate(256) { bits =>
    val sign = if ((bits & 0x80) != 0) -1.0f else 1.0f
    val exponent = (bits >> 2) & 0x1f
    val mantissa = bits & 0x3
    if (exponent == 0) sign * (mantissa / 4.0f) * math.pow(2, -14).toFloat
    else if (exponent == 31)
      if (mantissa == 0) sign * Float.PositiveInfinity else Float.NaN
    else sign * (1 + mantissa / 4.0f) * math.pow(2, exponent - 15).toFloat
  }

  private def readFully(
      channel: FileChannel,
      position: Long,
      length: Long
  ): Array[Byte] = {
    require(
      length <= Int.MaxValue - 8,
      s"a tensor of $length bytes does not fit in memory"
    )
    val buffer = ByteBuffer.allocate(length.toInt)
    var at = position
    while (buffer.hasRemaining) {
      val read = channel.read(buffer, at)
      if (read < 0) throw java.io.EOFException()
      at += read
    }
    buffer.array()
  }

  private def writeFully(channel: FileChannel, buffer: ByteBuffer): Unit =
    while (buffer.hasRemaining) channel.write(buffer)

  private def transfer(
      in: FileChannel,
      position: Long,
      length: Long,
      out: FileChannel
  ): Unit = {
    var at = position
    var left = length
    while (left > 0) {
      val moved = in.transferTo(at, left, out)
      if (moved <= 0) throw java.io.EOFException()
      at += moved
      left -= moved
    }
  }
}
