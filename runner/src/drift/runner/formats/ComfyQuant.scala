package drift.runner.formats

import drift.runner.tensor.*

import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.nio.charset.StandardCharsets
import java.util.stream.IntStream

/** ComfyUI's quantized linears whose stored weight is not the weight: a
  * `<module>.comfy_quant` marker (JSON in a U8 tensor) names the format, and
  * two of them are decoded here, on the CPU, back to the float weight the model
  * was quantized from (`specs/42`, LTX 2.5's mixed checkpoints):
  *
  *   - `int8_tensorwise`: `weight` I8 `[out, in]` times `weight_scale` (one a
  *     row, or one for all);
  *   - `asym_w4a8_int8`: `weight` holds two 4-bit codes a byte (`[out, in /
  *     2]`, the even column in the low nibble); a code is a level of
  *     `weight_codebook` (16), times `weight_s_rel` (fp8, one a `group_size`
  *     columns), rounded to an int8, times `weight_s_channel` (one a row).
  *
  * Both are stored in ConvRot's rotated basis (`convrot`, always for the 4-bit
  * one): every `convrot_groupsize` columns were multiplied by a normalized
  * regular Hadamard matrix, symmetric and its own inverse, so the same product
  * undoes it. ComfyUI's kernels rotate and quantize the activations instead;
  * the runner computes in floats, on the weights as they were.
  */
object ComfyQuant {

  /** A weight to decode: `shape` is the decoded one, `rows(first, count)` its
    * values for those rows.
    */
  final class Weight(
      val shape: Shape,
      /** The parts a backend decodes the weight from where the file holds it,
        * when it is rotated.
        */
      val stored: Option[Stored],
      decode: (Long, Int) => Array[Float]
  ) {
    def rows(first: Long, count: Int): Array[Float] = decode(first, count)
  }

  /** A rotated weight where the file holds it, for `Ops.rotated`: the codes,
    * the scales (one a row, or one for all), and a 4-bit weight's levels.
    */
  final class Stored(
      val codes: StoredTensor,
      val scales: () => Array[Float],
      val levels: Option[Levels]
  )

  /** What makes an int8 of a 4-bit code: the codebook and the relative scales,
    * one a `groupSize` columns.
    */
  final class Levels(
      val codebook: () => Array[Float],
      val relative: StoredTensor,
      val groupSize: Int
  )

  private val RotationGroup = ConvRot.Group

  /** The weights of `file` to decode, by their stored names. */
  def weights(file: TensorSource): Map[String, Weight] =
    file.tensors.keys
      .filter(_.endsWith(".comfy_quant"))
      .flatMap { markerName =>
        val module = markerName.stripSuffix(".comfy_quant")
        val marker = ujson
          .read(
            new String(
              file(markerName).bytes.toArray(JAVA_BYTE),
              StandardCharsets.UTF_8
            )
          )
          .obj
        val format = marker.get("format").flatMap(_.strOpt)
        val decoded = format.collect {
          case "int8_tensorwise" =>
            int8(
              file,
              module,
              Option.when(marker.get("convrot").exists(_.bool))(
                rotationGroup(module, marker)
              )
            )
          case "asym_w4a8_int8" =>
            fourBit(
              file,
              module,
              marker.get("group_size").map(_.num.toInt).getOrElse(16),
              Option.when(marker.get("convrot").forall(_.bool))(
                rotationGroup(module, marker)
              )
            )
        }
        decoded.map(s"$module.weight" -> _)
      }
      .toMap

  private def fail(message: String) = throw new FormatException(message)

  private def rotationGroup(
      module: String,
      marker: collection.Map[String, ujson.Value]
  ): Int = {
    val group =
      marker.get("convrot_groupsize").map(_.num.toInt).getOrElse(RotationGroup)
    if (group != RotationGroup)
      fail(
        s"$module: a ConvRot group of $group columns, where the runner only undoes $RotationGroup"
      )
    group
  }

  /** The stored weight of `module`, I8 and two-dimensional, as (tensor, rows,
    * stored columns).
    */
  private def stored(
      file: TensorSource,
      module: String
  ): (StoredTensor, Long, Int) = {
    val weight = file(s"$module.weight")
    if (weight.dtype != DType.I8 || weight.shape.dimensions.size != 2)
      fail(
        s"$module.weight is ${weight.dtype.name} ${weight.shape}, not the I8 matrix its comfy_quant marker announces"
      )
    (weight, weight.shape.dimensions(0), weight.shape.dimensions(1).toInt)
  }

  private def checkRotation(
      module: String,
      columns: Int,
      rotation: Option[Int]
  ): Unit =
    rotation.foreach(group =>
      if (columns % group != 0)
        fail(
          s"$module: $columns columns are no multiple of its ConvRot group, $group"
        )
    )

  private def eachRow(count: Int)(body: Int => Unit): Unit =
    IntStream.range(0, count).parallel().forEach(row => body(row))

  private def int8(
      file: TensorSource,
      module: String,
      rotation: Option[Int]
  ): Weight = {
    val (weight, rows, columns) = stored(file, module)
    checkRotation(module, columns, rotation)
    lazy val scales = {
      val values = file(s"$module.weight_scale").decode()
      if (values.length != 1 && values.length != rows)
        fail(
          s"$module.weight_scale holds ${values.length} scales for $rows rows"
        )
      values
    }
    new Weight(
      weight.shape,
      rotation.map(_ => new Stored(weight, () => scales, None)),
      (first, count) => {
        val bytes = weight.bytes
          .asSlice(first * columns, count.toLong * columns)
          .toArray(JAVA_BYTE)
        val out = new Array[Float](count * columns)
        eachRow(count) { row =>
          val scale =
            if (scales.length == 1) scales(0) else scales((first + row).toInt)
          val base = row * columns
          var column = 0
          while (column < columns) {
            out(base + column) = bytes(base + column) * scale
            column += 1
          }
          rotation.foreach(unrotate(out, base, columns, _))
        }
        out
      }
    )
  }

  private def fourBit(
      file: TensorSource,
      module: String,
      groupSize: Int,
      rotation: Option[Int]
  ): Weight = {
    val (weight, rows, packed) = stored(file, module)
    val columns = packed * 2
    checkRotation(module, columns, rotation)
    if (groupSize <= 0 || columns % groupSize != 0)
      fail(s"$module: $columns columns in groups of $groupSize")
    val groups = columns / groupSize
    lazy val codebook = {
      val levels = file(s"$module.weight_codebook").decode()
      if (levels.length != 16)
        fail(s"$module.weight_codebook holds ${levels.length} levels, not 16")
      levels
    }
    lazy val channel = {
      val values = file(s"$module.weight_s_channel").decode()
      if (values.length != rows)
        fail(
          s"$module.weight_s_channel holds ${values.length} scales for $rows rows"
        )
      values
    }
    val relative = file(s"$module.weight_s_rel")
    if (relative.shape != Shape.of(rows, groups.toLong))
      fail(
        s"$module.weight_s_rel is ${relative.shape}, not $rows rows of $groups groups"
      )
    new Weight(
      Shape.of(rows, columns.toLong),
      Option.when(
        rotation.nonEmpty && relative.dtype == DType.F8E4M3 &&
          groupSize % 16 == 0
      )(
        new Stored(
          weight,
          () => channel,
          Some(new Levels(() => codebook, relative, groupSize))
        )
      ),
      (first, count) => {
        val bytes = weight.bytes
          .asSlice(first * packed, count.toLong * packed)
          .toArray(JAVA_BYTE)
        val scales = relative.dtype.decode(
          relative.bytes.asSlice(
            relative.dtype.byteSize(first * groups),
            relative.dtype.byteSize(count.toLong * groups)
          ),
          count * groups
        )
        val out = new Array[Float](count * columns)
        eachRow(count) { row =>
          val rowScale = channel((first + row).toInt)
          val base = row * columns
          var column = 0
          while (column < columns) {
            val byte = bytes(row * packed + column / 2)
            val code = if (column % 2 == 0) byte & 0xf else (byte >> 4) & 0xf
            // the int8 the 4-bit kernel multiplies by, as its reference rounds it
            val level = math
              .rint(codebook(code) * scales(row * groups + column / groupSize))
              .toFloat
            out(base + column) =
              math.max(-127f, math.min(127f, level)) * rowScale
            column += 1
          }
          rotation.foreach(unrotate(out, base, columns, _))
        }
        out
      }
    )
  }

  private[runner] def unrotate(
      values: Array[Float],
      from: Int,
      length: Int,
      group: Int
  ): Unit = ConvRot.rotate(values, from, length, group)
}
