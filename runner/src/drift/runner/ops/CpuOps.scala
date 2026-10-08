package drift.runner.ops

import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.lang.foreign.{Arena, MemorySegment}
import java.lang.foreign.ValueLayout.*

/** The reference backend: plain loops, accumulating in `Double`. Slow on
  * purpose; it is what every kernel is checked against, so it stays simple
  * enough to be read as the maths.
  */
final class CpuOps extends Ops {

  private val arena = Arena.ofShared()

  def name: String = "cpu"

  def allocate(dtype: DType, shape: Shape): Tensor = {
    val bytes = dtype.byteSize(shape.elementCount)
    Tensor(
      dtype,
      shape,
      Storage.Host(arena.allocate(math.max(bytes, 1), 64)),
      0
    )
  }

  def release(tensor: Tensor): Unit = () // the arena frees everything at close

  def mapFile(path: java.nio.file.Path): MappedWeights = {
    val mapped = new drift.runner.tensor.MappedFile(path)
    new MappedWeights(
      mapped.segment,
      Storage.Host(mapped.segment),
      () => mapped.close()
    )
  }

  def embedding(table: Tensor, ids: Tensor, out: Tensor): Unit = {
    Ops.checkEmbedding(table, ids, out)
    val (rows, target, columns) =
      (segment(table), segment(out), table.shape.last.toInt)
    val rowBytes = table.dtype.byteSize(columns.toLong)
    val index = segment(ids)
    (0L until ids.shape.elementCount).foreach { t =>
      val row = index.getAtIndex(JAVA_INT, t).toLong
      require(
        row >= 0 && row < table.shape.dimensions.head,
        s"token id $row outside the table"
      )
      val values =
        table.dtype.decode(rows.asSlice(row * rowBytes, rowBytes), columns)
      MemorySegment.copy(
        values,
        0,
        target,
        JAVA_FLOAT,
        t * columns * 4,
        columns
      )
    }
  }

  def fromFloats(shape: Shape, values: Array[Float]): Tensor = {
    require(
      values.length == shape.elementCount,
      s"${values.length} values for $shape"
    )
    val tensor = allocate(DType.F32, shape)
    MemorySegment.copy(values, 0, segment(tensor), JAVA_FLOAT, 0, values.length)
    tensor
  }

  def fromInts(shape: Shape, values: Array[Int]): Tensor = {
    require(
      values.length == shape.elementCount,
      s"${values.length} values for $shape"
    )
    val tensor = allocate(DType.I32, shape)
    MemorySegment.copy(values, 0, segment(tensor), JAVA_INT, 0, values.length)
    tensor
  }

  def writeInts(tensor: Tensor, values: Array[Int]): Unit = {
    require(
      tensor.dtype == DType.I32 && values.length <= tensor.shape.elementCount,
      s"${values.length} ints into $tensor"
    )
    MemorySegment.copy(values, 0, segment(tensor), JAVA_INT, 0, values.length)
  }

  def fromBytes(dtype: DType, shape: Shape, bytes: Array[Byte]): Tensor = {
    Ops.checkBytes(dtype, shape, bytes)
    val tensor = allocate(dtype, shape)
    MemorySegment.copy(
      MemorySegment.ofArray(bytes),
      0,
      segment(tensor),
      0,
      bytes.length
    )
    tensor
  }

  def toFloats(tensor: Tensor): Array[Float] = {
    Ops.requireF32("toFloats", tensor)
    segment(tensor).toArray(JAVA_FLOAT)
  }

  def toInts(tensor: Tensor): Array[Int] = {
    require(tensor.dtype == DType.I32, s"toInts of ${tensor.dtype}")
    segment(tensor).toArray(JAVA_INT)
  }

  def argmax(values: Tensor): Int = {
    Ops.requireF32("argmax", values)
    val read = reader(values)
    var best = 0L
    var i = 1L
    while (i < values.shape.elementCount) {
      if (read(i) > read(best)) best = i
      i += 1
    }
    best.toInt
  }

  def candidates(values: Tensor, count: Int): Seq[Candidates] = {
    Ops.requireCandidates(values, count)
    Candidates.of(toFloats(values), values.shape.last.toInt, count)
  }

  /** `out(i) = f(i)` over every element. */
  private def fill(out: Tensor)(f: Long => Double): Unit = {
    val target = segment(out)
    var index = 0L
    while (index < out.shape.elementCount) {
      target.setAtIndex(JAVA_FLOAT, index, f(index).toFloat)
      index += 1
    }
  }

  private def reader(tensor: Tensor): Long => Double = {
    val source = segment(tensor)
    index => source.getAtIndex(JAVA_FLOAT, index).toDouble
  }

  def add(a: Tensor, b: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("add", a, b, out)
    val (left, right) = (reader(a), reader(b))
    fill(out)(i => left(i) + right(i))
  }

  def mul(a: Tensor, b: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("mul", a, b, out)
    val (left, right) = (reader(a), reader(b))
    fill(out)(i => left(i) * right(i))
  }

  def scale(x: Tensor, factor: Float, out: Tensor): Unit = {
    Ops.checkElementwise("scale", x, out)
    val input = reader(x)
    fill(out)(i => input(i) * factor)
  }

  def addRow(x: Tensor, row: Tensor, out: Tensor): Unit = {
    Ops.checkAddRow(x, row, out)
    val (input, vector, columns) = (reader(x), reader(row), x.shape.last)
    fill(out)(i => input(i) + vector(i % columns))
  }

  private def activate(kind: Activation, v: Double): Double = kind match {
    case Activation.Silu     => v / (1 + math.exp(-v))
    case Activation.GeluTanh =>
      0.5 * v * (1 + math.tanh(
        math.sqrt(2 / math.Pi) * (v + 0.044715 * v * v * v)
      ))
    case Activation.GeluErf => 0.5 * v * (1 + CpuOps.erf(v / math.sqrt(2)))
    case Activation.Sigmoid => 1 / (1 + math.exp(-v))
  }

  def scaledActivation(
      kind: Activation,
      inputScale: Float,
      x: Tensor,
      out: Tensor
  ): Unit = {
    Ops.checkElementwise("activation", x, out)
    val input = reader(x)
    fill(out)(i => activate(kind, inputScale * input(i)))
  }

  def gated(kind: Activation, gate: Tensor, up: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("gated", gate, up, out)
    val (g, u) = (reader(gate), reader(up))
    fill(out)(i => activate(kind, g(i)) * u(i))
  }

  /** Rows `first until first + count` of a rotated linear (`rotated`), decoded
    * as `ComfyQuant` decodes them: a code times its scales, each group of 256
    * rotated back.
    */
  private def rotatedRows(
      weight: Tensor,
      first: Long,
      count: Int
  ): Array[Float] = {
    val parts = partsOf(weight)
    val k = weight.shape.last.toInt
    val source = segment(weight)
    val scales = reader(parts.scales)
    val single = parts.scales.shape.elementCount == 1
    val out = new Array[Float](count * k)
    val level: (Long, Int) => Float = parts.levels match {
      case None =>
        (row, column) => source.get(JAVA_BYTE, row * k + column).toFloat
      case Some(l) =>
        val codebook = reader(l.codebook)
        val groups = k / l.groupSize
        val relative = DType.F8E4M3.decode(
          segment(l.relative).asSlice(first * groups, count.toLong * groups),
          count * groups
        )
        (row, column) => {
          val byte = source.get(JAVA_BYTE, row * (k / 2) + column / 2)
          val code = if (column % 2 == 0) byte & 0xf else (byte >> 4) & 0xf
          val scale =
            relative(((row - first) * groups + column / l.groupSize).toInt)
          math
            .max(
              -127.0,
              math.min(127.0, math.rint(codebook(code).toFloat * scale))
            )
            .toFloat
        }
    }
    var row = 0
    while (row < count) {
      val scale = scales(if (single) 0 else first + row).toFloat
      var column = 0
      while (column < k) {
        out(row * k + column) = level(first + row, column) * scale
        column += 1
      }
      row += 1
    }
    ConvRot.rotate(out, 0, out.length, ConvRot.Group)
    out
  }

  def convert(x: Tensor, out: Tensor): Unit = {
    Ops.checkConvert(x, out)
    val (source, target) = (segment(x), segment(out))
    val count = x.shape.elementCount
    if (out.dtype == DType.F32) {
      val values = x.dtype.decode(source, count.toInt)
      MemorySegment.copy(values, 0, target, JAVA_FLOAT, 0, values.length)
    } else if (x.dtype != DType.F32) {
      val values = rotatedRows(x, 0, x.shape.dimensions.head.toInt)
      var index = 0
      while (index < count) {
        target.setAtIndex(JAVA_SHORT, index, CpuOps.toBfloat16(values(index)))
        index += 1
      }
    } else {
      var index = 0L
      while (index < count) {
        val value = source.getAtIndex(JAVA_FLOAT, index)
        val bits =
          if (out.dtype == DType.F16) java.lang.Float.floatToFloat16(value)
          else CpuOps.toBfloat16(value)
        target.setAtIndex(JAVA_SHORT, index, bits)
        index += 1
      }
    }
  }

  /** Runs `body(start, columns)` for each row. */
  private def rows(x: Tensor, count: Long)(body: (Long, Long) => Unit): Unit = {
    val columns = x.shape.last
    var row = 0L
    while (row < count) {
      body(row * columns, columns)
      row += 1
    }
  }

  def rmsNorm(
      x: Tensor,
      weight: Tensor,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit = {
    val count = Ops.checkRowWise("rmsNorm", x, out, weight)
    val (input, scale, target) = (reader(x), reader(weight), segment(out))
    rows(x, count) { (start, columns) =>
      val squares =
        (0L until columns).map(c => input(start + c) * input(start + c)).sum
      val factor = 1 / math.sqrt(squares / columns + epsilon)
      (0L until columns).foreach { c =>
        val value = input(start + c) * factor * (scale(c) + weightOffset)
        target.setAtIndex(JAVA_FLOAT, start + c, value.toFloat)
      }
    }
  }

  def groupRmsNorm(
      x: Tensor,
      weight: Tensor,
      groups: Int,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit = {
    val count = Ops.checkRowWise("groupRmsNorm", x, out, weight)
    require(x.shape.last % groups == 0, s"${x.shape} in $groups groups")
    val (input, scale, target) = (reader(x), reader(weight), segment(out))
    val width = x.shape.last / groups
    rows(x, count) { (start, columns) =>
      (0 until groups).foreach { g =>
        val first = start + g * width
        val squares =
          (0L until width).map(c => input(first + c) * input(first + c)).sum
        val factor = 1 / math.sqrt(squares / width + epsilon)
        (0L until width).foreach { c =>
          val value =
            input(first + c) * factor * (scale(g * width + c) + weightOffset)
          target.setAtIndex(JAVA_FLOAT, first + c, value.toFloat)
        }
      }
    }
  }

  def modulate(x: Tensor, scale: Tensor, shift: Tensor, out: Tensor): Unit = {
    Ops.checkRowWise("modulate", x, out, scale, shift)
    val (input, scales, shifts) = (reader(x), reader(scale), reader(shift))
    val cols = x.shape.last
    fill(out)(i => input(i) * (1 + scales(i % cols)) + shifts(i % cols))
  }

  def gatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit = {
    Ops.checkRowWise("gatedAdd", x, y, gate)
    val (before, added, gates) = (reader(x), reader(y), reader(gate))
    val cols = x.shape.last
    val values =
      Array.tabulate(x.shape.elementCount.toInt)(i =>
        before(i) + gates(i % cols) * added(i)
      )
    fill(x)(i => values(i.toInt))
  }

  def shortAttention(
      q: Tensor,
      k: Tensor,
      v: Tensor,
      scale: Float,
      out: Tensor,
      bias: Option[Tensor]
  ): Unit = {
    val (s, b, heads, kvHeads, d) = Ops.checkShortAttention(q, k, v, out)
    bias.foreach(t => Ops.checkAttentionBias(t, heads, s))
    val biases = bias.map(reader)
    val (queries, keys, values, target) =
      (reader(q), reader(k), reader(v), segment(out))
    for {
      position <- 0 until s
      sequence <- 0 until b
      h <- 0 until heads
    } {
      val row = position.toLong * b + sequence
      val kv = h / (heads / kvHeads)
      def keyAt(other: Int) = (other.toLong * b + sequence) * kvHeads + kv
      val scores = (0 until s).map { other =>
        (0 until d)
          .map(i =>
            queries((row * heads + h) * d + i) * keys(keyAt(other) * d + i)
          )
          .sum * scale + biases.fold(0.0)(table =>
          table((h.toLong * s + position) * s + other)
        )
      }
      val largest = scores.max
      val weights = scores.map(score => math.exp(score - largest))
      (0 until d).foreach { i =>
        val value = (0 until s)
          .map(other => weights(other) * values(keyAt(other) * d + i))
          .sum / weights.sum
        target.setAtIndex(JAVA_FLOAT, (row * heads + h) * d + i, value.toFloat)
      }
    }
  }

  def conv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor,
      stride: Int,
      replicate: Boolean,
      reflect: Boolean
  ): Unit = {
    val (height, width, in, outChannels) =
      Ops.checkConv3x3(x, weight, bias, out, stride)
    def reflected(i: Int, size: Int) =
      if (i < 0) -i else if (i >= size) 2 * size - 2 - i else i
    val pad = if (stride == 1) 1 else 0
    val outWidth = width / stride
    val input = reader(x)
    val kernel =
      weight.dtype.decode(segment(weight), weight.shape.elementCount.toInt)
    val biases = reader(bias)
    val target = segment(out)
    for {
      y <- 0 until height / stride
      xx <- 0 until outWidth
      o <- 0 until outChannels
    } {
      var sum = biases(o)
      for {
        c <- 0 until in
        ky <- 0 until 3
        kx <- 0 until 3
      } {
        val (sy, sx) =
          if (replicate)
            (
              math.min(math.max(y * stride + ky - pad, 0), height - 1),
              math.min(math.max(xx * stride + kx - pad, 0), width - 1)
            )
          else if (reflect)
            (
              reflected(y * stride + ky - pad, height),
              reflected(xx * stride + kx - pad, width)
            )
          else (y * stride + ky - pad, xx * stride + kx - pad)
        if (sy >= 0 && sy < height && sx >= 0 && sx < width)
          sum += kernel((o * in + c) * 9 + ky * 3 + kx) * input(
            (sy.toLong * width + sx) * in + c
          )
      }
      target.setAtIndex(
        JAVA_FLOAT,
        (y.toLong * outWidth + xx) * outChannels + o,
        sum.toFloat
      )
    }
  }

  def groupNorm(
      x: Tensor,
      groups: Int,
      weight: Tensor,
      bias: Tensor,
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val (rows, channels) = Ops.checkGroupNorm(x, groups, weight, bias, out)
    val (input, scale, shift) = (reader(x), reader(weight), reader(bias))
    val width = channels / groups
    val statistics = (0 until groups).map { g =>
      def values =
        (0L until rows).iterator.flatMap(r =>
          (0 until width).iterator.map(c => input(r * channels + g * width + c))
        )
      val count = rows * width
      val mean = values.sum / count
      val variance = values.map(v => (v - mean) * (v - mean)).sum / count
      (mean, 1 / math.sqrt(variance + epsilon))
    }
    fill(out) { index =>
      val c = (index % channels).toInt
      val (mean, factor) = statistics(c / width)
      (input(index) - mean) * factor * scale(c) + shift(c)
    }
  }

  def ropeTable(
      x: Tensor,
      cosines: Tensor,
      sines: Tensor,
      out: Tensor,
      halves: Boolean
  ): Unit = {
    val (_, heads, d, pairs, perHead) =
      Ops.checkRopeTable(x, cosines, sines, out)
    val (input, cosAt, sinAt) = (reader(x), reader(cosines), reader(sines))
    fill(out) { index =>
      val dim = (index % d).toInt
      // the table's row: the token's, or the token's head's
      val token =
        if (perHead) index / d else index / (heads.toLong * d)
      val (cos, sin) = (cosAt, sinAt)
      if (halves) {
        if (dim >= 2 * pairs) input(index)
        else {
          val m = dim % pairs
          val (c, s) = (cos(token * pairs + m), sin(token * pairs + m))
          val (a, b) = (input(index - dim + m), input(index - dim + m + pairs))
          if (dim < pairs) a * c - b * s else b * c + a * s
        }
      } else {
        val m = dim / 2
        if (m >= pairs) input(index)
        else {
          val (c, s) = (cos(token * pairs + m), sin(token * pairs + m))
          val (a, b) = (input(index - dim % 2), input(index - dim % 2 + 1))
          if (dim % 2 == 0) a * c - b * s else a * s + b * c
        }
      }
    }
  }

  /** The image index of patch-ordered row value `index`. */
  private def patchPixel(
      index: Long,
      width: Int,
      channels: Int,
      patch: Int
  ): Long = {
    val c = index % channels
    val row = index / channels
    val area = patch * patch
    val (l, p) = (row / area, row % area)
    val gridWidth = width / patch
    val (y, x) =
      (l / gridWidth * patch + p / patch, l % gridWidth * patch + p % patch)
    (y * width + x) * channels + c
  }

  def pixelsToPatches(image: Tensor, patch: Int, out: Tensor): Unit = {
    val (_, width, channels) = Ops.checkPatchPixels(image, out, patch)
    val input = reader(image)
    fill(out)(index => input(patchPixel(index, width, channels, patch)))
  }

  def patchesToPixels(rows: Tensor, patch: Int, out: Tensor): Unit = {
    val (_, width, channels) = Ops.checkPatchPixels(out, rows, patch)
    val input = reader(rows)
    val target = segment(out)
    (0L until rows.shape.elementCount).foreach { index =>
      target.setAtIndex(
        JAVA_FLOAT,
        patchPixel(index, width, channels, patch),
        input(index).toFloat
      )
    }
  }

  def modulateChunks(
      x: Tensor,
      table: Tensor,
      shift: Int,
      scale: Int,
      out: Tensor
  ): Unit = {
    val (_, channels, chunks) =
      Ops.checkChunks("modulateChunks", x, table, out, shift, scale)
    val (input, values) = (reader(x), reader(table))
    fill(out) { index =>
      val (r, c) = (index / channels, index % channels)
      val row = r * chunks * channels
      input(index) * (1 + values(row + scale * channels + c)) +
        values(row + shift * channels + c)
    }
  }

  def gatedAddChunk(x: Tensor, y: Tensor, table: Tensor, gate: Int): Unit = {
    val (_, channels, chunks) =
      Ops.checkChunks("gatedAddChunk", x, table, y, gate)
    val (input, other, values) = (reader(x), reader(y), reader(table))
    fill(x) { index =>
      val (r, c) = (index / channels, index % channels)
      input(index) + values(
        r * chunks * channels + gate * channels + c
      ) * other(index)
    }
  }

  def rowGatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit = {
    val (_, cols) = Ops.checkRowGatedAdd(x, y, gate)
    val (input, other, gates) = (reader(x), reader(y), reader(gate))
    fill(x)(index => input(index) + gates(index / cols) * other(index))
  }

  def maxAbs(x: Tensor): Float = {
    Ops.requireF32("maxAbs", x)
    val input = reader(x)
    (0L until x.shape.elementCount)
      .map(i => math.abs(input(i)))
      .filterNot(_.isNaN)
      .maxOption
      .getOrElse(0.0)
      .toFloat
  }

  def packPatches(image: Tensor, patch: Int, out: Tensor): Unit = {
    val (_, gridWidth, channels) = Ops.checkPackPatches(image, patch, out)
    val input = reader(image)
    val features = channels.toLong * patch * patch
    fill(out) { index =>
      val (token, feature) = (index / features, index % features)
      val c = feature / (patch * patch)
      val (py, px) = (feature % (patch * patch) / patch, feature % patch)
      val (y, xx) =
        (token / gridWidth * patch + py, token % gridWidth * patch + px)
      input((y * gridWidth * patch + xx) * channels + c)
    }
  }

  def concatColumns(parts: Seq[Tensor], out: Tensor): Unit = {
    val (_, columns) = Ops.checkConcatColumns(parts, out)
    val starts = columns.scanLeft(0)(_ + _)
    val inputs = parts.map(reader)
    val width = columns.sum
    fill(out) { index =>
      val (row, column) = (index / width, (index % width).toInt)
      val part = starts.lastIndexWhere(_ <= column)
      inputs(part)(row * columns(part) + column - starts(part))
    }
  }

  def conv1d(
      x: Tensor,
      weight: Tensor,
      bias: Option[Tensor],
      taps: Int,
      dilation: Int,
      stride: Int,
      padLeft: Int,
      padRight: Int,
      out: Tensor
  ): Unit = {
    val (length, in, _, outChannels) = Ops.checkConv1d(
      x,
      weight,
      bias,
      taps,
      dilation,
      stride,
      padLeft,
      padRight,
      out
    )
    val input = reader(x)
    val kernel =
      weight.dtype.decode(segment(weight), weight.shape.elementCount.toInt)
    val biases = bias.map(reader)
    fill(out) { index =>
      val (t, o) = (index / outChannels, (index % outChannels).toInt)
      var sum = biases.fold(0.0)(_(o))
      for {
        c <- 0 until in
        k <- 0 until taps
      } {
        val s = t * stride + k * dilation - padLeft
        if (s >= 0 && s < length)
          sum += kernel((o * in + c) * taps + k) * input(s * in + c)
      }
      sum
    }
  }

  def overlapAdd(
      columns: Tensor,
      taps: Int,
      stride: Int,
      pad: Int,
      bias: Tensor,
      out: Tensor
  ): Unit = {
    val (length, _, outChannels) =
      Ops.checkOverlapAdd(columns, taps, stride, pad, bias, out)
    val (input, biases) = (reader(columns), reader(bias))
    fill(out) { index =>
      val (p, o) = (index / outChannels, (index % outChannels).toInt)
      var sum = biases(o)
      (0 until taps).foreach { k =>
        val shifted = p + pad - k
        if (shifted >= 0 && shifted % stride == 0 && shifted / stride < length)
          sum += input((shifted / stride) * outChannels * taps + o * taps + k)
      }
      sum
    }
  }

  def snake(x: Tensor, alpha: Tensor, out: Tensor): Unit = {
    val (length, channels) = Ops.checkSnake(x, alpha, out)
    val (input, alphas, target) = (reader(x), reader(alpha), segment(out))
    (0L until length.toLong * channels).foreach { i =>
      val a = alphas(i % channels)
      val value = input(i)
      val s = math.sin(a * value)
      target.setAtIndex(JAVA_FLOAT, i, (value + s * s / (a + 1e-9)).toFloat)
    }
  }

  def antiAliasedSnake(
      x: Tensor,
      frequency: Tensor,
      inverseMagnitude: Tensor,
      upFilter: Tensor,
      downFilter: Tensor,
      out: Tensor
  ): Unit = {
    val (length, channels) = Ops.checkAntiAliasedSnake(
      x,
      frequency,
      inverseMagnitude,
      upFilter,
      downFilter,
      out
    )
    val input = reader(x)
    val (frequencies, inverses) = (reader(frequency), reader(inverseMagnitude))
    val (up, down) = (reader(upFilter), reader(downFilter))
    val taps = Ops.SnakeTaps
    def clamp(i: Long, limit: Long) = math.min(math.max(i, 0L), limit - 1)
    // the upsampled, activated value n of channel c
    def activated(n: Long, c: Int): Double = {
      var u = 0.0
      (0 until taps).foreach { k =>
        val q = n + 15 - k
        if (q % 2 == 0)
          u += up(k) * input(clamp(q / 2 - 5, length) * channels + c)
      }
      u *= 2
      val sine = math.sin(frequencies(c) * u)
      u + inverses(c) * sine * sine
    }
    fill(out) { index =>
      val (t, c) = (index / channels, (index % channels).toInt)
      (0 until taps).foldLeft(0.0)((sum, j) =>
        sum + down(j) * activated(clamp(2 * t + j - 5, 2L * length), c)
      )
    }
  }

  def upsample2x(x: Tensor, out: Tensor): Unit = {
    val (_, width, channels) = Ops.checkUpsample2x(x, out)
    val input = reader(x)
    fill(out) { index =>
      val c = index % channels
      val p = index / channels
      val (y, xx) = (p / (2 * width), p % (2 * width))
      input(((y / 2) * width + xx / 2) * channels + c)
    }
  }

  def duplicateUp(x: Tensor, frames: Int, out: Tensor): Unit = {
    val (_, width, channels, outChannels) =
      Ops.checkDuplicateUp(x, frames, out)
    val repeats = outChannels * frames * 4 / channels
    val input = reader(x)
    fill(out) { index =>
      val o = index % outChannels
      val p = index / outChannels
      val (y, xx) = (p / (2 * width), p % (2 * width))
      val j = ((o * frames + frames - 1) * 2 + y % 2) * 2 + xx % 2
      input(((y / 2) * width + xx / 2) * channels + j / repeats)
    }
  }

  def averageDown(x: Tensor, frames: Int, out: Tensor): Unit = {
    val (_, width, channels, outChannels) =
      Ops.checkAverageDown(x, frames, out)
    val group = channels * frames * 4 / outChannels
    val input = reader(x)
    fill(out) { index =>
      val o = index % outChannels
      val p = index / outChannels
      val (y, xx) = (p / (width / 2), p % (width / 2))
      var sum = 0.0
      (0 until group).foreach { g =>
        val k = o * group + g
        if ((k / 4) % frames == frames - 1)
          sum += input(
            ((2 * y + k / 2 % 2) * width + 2 * xx + k % 2) * channels +
              k / (4 * frames)
          )
      }
      sum / group
    }
  }

  def transpose(x: Tensor, out: Tensor): Unit = {
    val Seq(rows, cols) = Ops.checkTranspose(x, out)
    if (x.dtype == DType.F32) {
      val input = reader(x)
      fill(out)(index => input((index % rows) * cols + index / rows))
    } else {
      val (from, to) = (segment(x), segment(out))
      (0L until rows.toLong * cols).foreach { index =>
        val source = (index % rows) * cols + index / rows
        to.setAtIndex(JAVA_SHORT, index, from.getAtIndex(JAVA_SHORT, source))
      }
    }
  }

  def scatterAddRows(
      rows: Tensor,
      ids: Tensor,
      weights: Tensor,
      out: Tensor
  ): Unit = {
    val (slots, columns) = Ops.checkScatterAddRows(rows, ids, weights, out)
    val (input, factors, index, target) =
      (reader(rows), reader(weights), segment(ids), segment(out))
    for {
      slot <- 0L until slots
      column <- 0 until columns
    } {
      val at = index.getAtIndex(JAVA_INT, slot).toLong * columns + column
      target.setAtIndex(
        JAVA_FLOAT,
        at,
        (target.getAtIndex(JAVA_FLOAT, at) +
          factors(slot) * input(slot * columns + column)).toFloat
      )
    }
  }

  def depthwiseConv3x3(
      x: Tensor,
      weight: Tensor,
      bias: Tensor,
      out: Tensor
  ): Unit = {
    val (height, width, channels) =
      Ops.checkDepthwiseConv3x3(x, weight, bias, out)
    val (input, taps, offset) = (reader(x), reader(weight), reader(bias))
    val result = new Array[Float](height * width * channels)
    result.indices.foreach { index =>
      val c = index % channels
      val (y, xx) = (index / channels / width, index / channels % width)
      var sum = offset(c)
      for {
        ky <- 0 until 3
        kx <- 0 until 3
        (sy, sx) = (y + ky - 1, xx + kx - 1)
        if sy >= 0 && sy < height && sx >= 0 && sx < width
      } sum += taps(c * 9L + ky * 3 + kx) *
        input((sy.toLong * width + sx) * channels + c)
      result(index) = sum.toFloat
    }
    // `out` may be `x`
    fill(out)(index => result(index.toInt))
  }

  def columnMean(x: Tensor, out: Tensor): Unit = {
    val (rows, columns) = Ops.checkColumnMean(x, out)
    val input = reader(x)
    fill(out)(c => (0L until rows).map(r => input(r * columns + c)).sum / rows)
  }

  def unpackPatches(
      packed: Tensor,
      gridHeight: Int,
      patch: Int,
      out: Tensor
  ): Unit = {
    val (gridWidth, channels) =
      Ops.checkUnpackPatches(packed, gridHeight, patch, out)
    val input = reader(packed)
    val width = gridWidth.toLong * patch
    fill(out) { index =>
      val c = index % channels
      val p = index / channels
      val (y, xx) = (p / width, p % width)
      val token = (y / patch) * gridWidth + xx / patch
      input(
        token * channels * patch * patch + c * patch * patch + (y % patch) * patch + xx % patch
      )
    }
  }

  def layerNorm(
      x: Tensor,
      weight: Option[Tensor],
      bias: Option[Tensor],
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val count = Ops.checkRowWise("layerNorm", x, out, (weight ++ bias).toSeq*)
    val (input, target) = (reader(x), segment(out))
    val (scale, shift) = (weight.map(reader), bias.map(reader))
    rows(x, count) { (start, columns) =>
      val mean = (0L until columns).map(c => input(start + c)).sum / columns
      val variance =
        (0L until columns)
          .map(c => math.pow(input(start + c) - mean, 2))
          .sum / columns
      val factor = 1 / math.sqrt(variance + epsilon)
      (0L until columns).foreach { c =>
        val normal = (input(start + c) - mean) * factor
        val value = normal * scale.fold(1.0)(_(c)) + shift.fold(0.0)(_(c))
        target.setAtIndex(JAVA_FLOAT, start + c, value.toFloat)
      }
    }
  }

  def softmax(x: Tensor, scale: Float, out: Tensor): Unit = {
    val count = Ops.checkRowWise("softmax", x, out)
    val (input, target) = (reader(x), segment(out))
    rows(x, count) { (start, columns) =>
      val scaled = (0L until columns).map(c => input(start + c) * scale)
      val largest = scaled.max
      val exponentials = scaled.map(v => math.exp(v - largest))
      val sum = exponentials.sum
      exponentials.zipWithIndex.foreach { (value, c) =>
        target.setAtIndex(JAVA_FLOAT, start + c, (value / sum).toFloat)
      }
    }
  }

  /** The angle is computed in float, as transformers does (`inv_freq` in
    * float32, then `position × inv_freq`), and turned in double.
    */
  def rope(x: Tensor, positions: Tensor, rope: Rope, out: Tensor): Unit = {
    val (tokens, heads, dimension) = Ops.checkRope(x, positions, rope, out)
    val (input, target, position) =
      (reader(x), segment(out), segment(positions))
    val rotary = rope.rotaryDimensions
    val half = rotary / 2
    def section(pair: Int): Int = rope.sections match {
      case RopeSections.Single                          => 0
      case RopeSections.Contiguous(temporal, height, _) =>
        if (pair < temporal) 0 else if (pair < temporal + height) 1 else 2
      case RopeSections.Axes(pairs) =>
        pairs.scanLeft(0)(_ + _).tail.indexWhere(pair < _)
      case RopeSections.Interleaved(_, height, width) =>
        if (pair % 3 == 1 && pair < 3 * height) 1
        else if (pair % 3 == 2 && pair < 3 * width) 2
        else 0
    }
    for {
      t <- 0 until tokens
      h <- 0 until heads
    } {
      val base = (t.toLong * heads + h) * dimension
      for (d <- 0 until dimension) {
        val value =
          if (d >= rotary) input(base + d)
          else {
            val (pair, first, partner) = rope.layout match {
              case RopeLayout.Neox =>
                (d % half, d < half, if (d < half) d + half else d - half)
              case RopeLayout.Interleaved =>
                (d / 2, d % 2 == 0, if (d % 2 == 0) d + 1 else d - 1)
            }
            val inverseFrequency = rope.sections match {
              case RopeSections.Axes(pairs) =>
                val axis = section(pair)
                val (start, count) = (pairs.take(axis).sum, pairs(axis))
                1f / math
                  .pow(rope.theta, (pair - start).toDouble / count)
                  .toFloat
              case _ =>
                1f / math.pow(rope.theta, 2.0 * pair / rotary).toFloat
            }
            val p =
              position.getAtIndex(JAVA_INT, section(pair).toLong * tokens + t)
            val angle = (p.toFloat * inverseFrequency).toDouble
            val (a, b) =
              if (first) (input(base + d), input(base + partner))
              else (input(base + partner), input(base + d))
            if (first) a * math.cos(angle) - b * math.sin(angle)
            else a * math.sin(angle) + b * math.cos(angle)
          }
        target.setAtIndex(JAVA_FLOAT, base + d, value.toFloat)
      }
    }
  }

  def linear(x: Tensor, weight: Tensor, out: Tensor): Unit = {
    val (m, n, k) = Ops.checkLinear(x, weight, out)
    val (input, weights, target) = (segment(x), segment(weight), segment(out))
    val rowBytes = weight.dtype.byteSize(k)
    var row = 0L
    while (row < n) {
      val decoded =
        if (ConvRot.stored(weight.dtype)) rotatedRows(weight, row, 1)
        else
          weight.dtype
            .decode(weights.asSlice(row * rowBytes, rowBytes), k.toInt)
      var sample = 0L
      while (sample < m) {
        var sum = 0.0
        var column = 0
        while (column < k) {
          sum += decoded(column).toDouble * input.getAtIndex(
            JAVA_FLOAT,
            sample * k + column
          )
          column += 1
        }
        target.setAtIndex(JAVA_FLOAT, sample * n + row, sum.toFloat)
        sample += 1
      }
      row += 1
    }
  }

  def zero(tensor: Tensor): Unit = segment(tensor).fill(0.toByte)

  def splitHalves(x: Tensor, first: Tensor, second: Tensor): Unit = {
    Ops.checkSplitHalves(x, first, second)
    val input = reader(x)
    val d = first.shape.last
    fill(first)(i => input(i / d * 2 * d + i % d))
    fill(second)(i => input(i / d * 2 * d + d + i % d))
  }

  def joinHalves(first: Tensor, second: Tensor, x: Tensor): Unit = {
    Ops.checkSplitHalves(x, first, second)
    val (left, right) = (reader(first), reader(second))
    val d = first.shape.last
    fill(x) { i =>
      val (row, column) = (i / (2 * d), i % (2 * d))
      if (column < d) left(row * d + column) else right(row * d + column - d)
    }
  }

  def copy(from: Tensor, to: Tensor): Unit = {
    Ops.checkCopy(from, to)
    MemorySegment.copy(segment(from), 0, segment(to), 0, from.byteSize)
  }

  def causalConv(
      x: Tensor,
      weight: Tensor,
      dilation: Int,
      state: Tensor,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    val (tokens, channels, taps) =
      Ops.checkCausalConv(x, weight, dilation, state, out)
    history.foreach(Ops.checkHistory(_, tokens, state))
    val span = (taps - 1) * dilation
    val (input, w, target, memory) =
      (reader(x), reader(weight), segment(out), segment(state))
    (0 until channels).foreach { c =>
      // the inputs this channel has seen, oldest first: the state, then x
      val seen = (0 until span).map(i =>
        memory.getAtIndex(JAVA_FLOAT, i.toLong * channels + c).toDouble
      ) ++
        (0 until tokens).map(t => input(t.toLong * channels + c))
      (0 until tokens).foreach { t =>
        // tap i reads (taps − 1 − i) × dilation tokens back
        val sum = (0 until taps)
          .map(i =>
            w(c.toLong * taps + i) * seen(t + span - (taps - 1 - i) * dilation)
          )
          .sum
        target.setAtIndex(
          JAVA_FLOAT,
          t.toLong * channels + c,
          (sum / (1 + math.exp(-sum))).toFloat
        )
      }
      (0 until span).foreach(i =>
        memory.setAtIndex(
          JAVA_FLOAT,
          i.toLong * channels + c,
          seen(tokens + i).toFloat
        )
      )
      history.map(segment).foreach { kept =>
        for {
          t <- 0 until tokens
          i <- 0 until span
        } kept.setAtIndex(
          JAVA_FLOAT,
          (t.toLong * span + i) * channels + c,
          seen(t + 1 + i).toFloat
        )
      }
    }
  }

  /** transformers' `torch_recurrent_gated_delta_rule`, in double. */
  def gatedDeltaRule(
      qkv: Tensor,
      a: Tensor,
      b: Tensor,
      decay: Tensor,
      dtBias: Tensor,
      state: Tensor,
      rule: DeltaRule,
      history: Option[Tensor],
      out: Tensor
  ): Unit = {
    val tokens = Ops.checkDeltaRule(qkv, a, b, decay, dtBias, state, rule, out)
    history.foreach(Ops.checkHistory(_, tokens, state))
    val (row, as, bs, decays, biases) =
      (reader(qkv), reader(a), reader(b), reader(decay), reader(dtBias))
    val (memory, target) = (segment(state), segment(out))
    val (d, keys, values, width) =
      (rule.dimension, rule.keyHeads, rule.valueHeads, rule.qkvWidth.toLong)
    (0 until values).foreach { h =>
      val kh = rule.keyHeadOf(h)
      val s = Array.tabulate(d, d)((i, j) =>
        memory.getAtIndex(JAVA_FLOAT, (h.toLong * d + i) * d + j).toDouble
      )
      (0 until tokens).foreach { t =>
        def vector(offset: Long) =
          Array.tabulate(d)(i => row(t * width + offset + i))
        def normalized(v: Array[Double]) = {
          val n = 1 / math.sqrt(v.map(x => x * x).sum + 1e-6); v.map(_ * n)
        }
        val q = normalized(vector(kh.toLong * d)).map(_ / math.sqrt(d))
        val k = normalized(vector(keys.toLong * d + kh.toLong * d))
        val v = vector(2L * keys * d + h.toLong * d)
        val x = as(t.toLong * values + h) + biases(h)
        val g =
          math.exp(decays(h) * (if (x > 20) x else math.log1p(math.exp(x))))
        val beta = 1 / (1 + math.exp(-bs(t.toLong * values + h)))
        for {
          i <- 0 until d
          j <- 0 until d
        } s(i)(j) *= g
        val delta = Array.tabulate(d)(j =>
          (v(j) - (0 until d).map(i => s(i)(j) * k(i)).sum) * beta
        )
        for {
          i <- 0 until d
          j <- 0 until d
        } s(i)(j) += k(i) * delta(j)
        (0 until d).foreach { j =>
          val o = (0 until d).map(i => s(i)(j) * q(i)).sum
          target.setAtIndex(
            JAVA_FLOAT,
            (t.toLong * values + h) * d + j,
            o.toFloat
          )
        }
        history.map(segment).foreach { kept =>
          for {
            i <- 0 until d
            j <- 0 until d
          } kept.setAtIndex(
            JAVA_FLOAT,
            ((t.toLong * values + h) * d + i) * d + j,
            s(i)(j).toFloat
          )
        }
      }
      for {
        i <- 0 until d
        j <- 0 until d
      } memory.setAtIndex(
        JAVA_FLOAT,
        (h.toLong * d + i) * d + j,
        s(i)(j).toFloat
      )
    }
  }

  def route(logits: Tensor, ids: Tensor, weights: Tensor): Unit = {
    val (tokens, experts, k) = Ops.checkRoute(logits, ids, weights)
    val (scores, chosen, weight) =
      (reader(logits), segment(ids), segment(weights))
    (0 until tokens).foreach { t =>
      val top = (0 until experts)
        .sortBy(e => (-scores(t.toLong * experts + e), e))
        .take(k)
      val largest = scores(t.toLong * experts + top.head)
      val exponentials =
        top.map(e => math.exp(scores(t.toLong * experts + e) - largest))
      top.zipWithIndex.foreach { (e, j) =>
        chosen.setAtIndex(JAVA_INT, t.toLong * k + j, e)
        weight.setAtIndex(
          JAVA_FLOAT,
          t.toLong * k + j,
          (exponentials(j) / exponentials.sum).toFloat
        )
      }
    }
  }

  def expertsLinear(
      x: Tensor,
      weight: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit = {
    val (slots, n, k, divisor) = Ops.checkExperts(x, weight, ids, out)
    val (input, weights, chosen, target) =
      (segment(x), segment(weight), segment(ids), segment(out))
    val rowBytes = weight.dtype.byteSize(k.toLong)
    (0 until slots).foreach { s =>
      val expert = chosen.getAtIndex(JAVA_INT, s.toLong)
      (0 until n).foreach { row =>
        val decoded = weight.dtype.decode(
          weights.asSlice((expert.toLong * n + row) * rowBytes, rowBytes),
          k
        )
        val sum = (0 until k)
          .map(c =>
            decoded(c).toDouble * input
              .getAtIndex(JAVA_FLOAT, (s / divisor).toLong * k + c)
          )
          .sum
        target.setAtIndex(JAVA_FLOAT, s.toLong * n + row, sum.toFloat)
      }
    }
  }

  def expertsGatedLinear(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit = {
    Ops.requireSameShape("expertsGatedLinear", gate, up)
    val (gated, raised) =
      (allocate(DType.F32, out.shape), allocate(DType.F32, out.shape))
    expertsLinear(x, gate, ids, gated)
    expertsLinear(x, up, ids, raised)
    this.gated(Activation.Silu, gated, raised, out)
  }

  def gatedRmsNorm(
      x: Tensor,
      weight: Tensor,
      gate: Tensor,
      gateActivation: Activation,
      epsilon: Float,
      out: Tensor
  ): Unit = {
    Ops.requireSameShape("gatedRmsNorm", x, gate)
    rmsNorm(x, weight, epsilon, 0f, out)
    gated(gateActivation, gate, out, out)
  }

  def streamsMix(
      gates: Tensor,
      normed: Tensor,
      streams: Int,
      out: Tensor
  ): Unit = {
    val (_, found, width) =
      Ops.checkStreams("streamsMix", Seq(gates, normed), out)
    require(found == streams, s"streamsMix: $found streams, expected $streams")
    val (g, n) = (reader(gates), reader(normed))
    fill(out) { index =>
      val (t, c) = (index / width, index % width)
      (0 until streams).map { j =>
        val at = (t * streams + j) * width + c
        n(at) / (1 + math.exp(-g(at)))
      }.sum / streams
    }
  }

  def streamsCombine(x: Tensor, y: Tensor, logits: Tensor): Unit = {
    val (_, streams, width) = Ops.checkStreams("streamsCombine", Seq(x), y)
    val (before, added, logit) = (reader(x), reader(y), reader(logits))
    val values = Array.tabulate(x.shape.elementCount.toInt) { index =>
      val tj = index / width
      val weight = 2 / (1 + math.exp(-logit(tj) / streams))
      before(index) + weight * added(tj / streams * width + index % width)
    }
    fill(x)(index => values(index.toInt))
  }

  def pleGate(key: Tensor, query: Tensor, value: Tensor, out: Tensor): Unit = {
    val (tokens, streams, width) =
      Ops.checkStreams("pleGate", Seq(key, query, out), value)
    val (k, q, v) = (reader(key), reader(query), reader(value))
    val gates = Array.tabulate(tokens * streams) { tj =>
      val s = (0L until width)
        .map(c => k(tj.toLong * width + c) * q(tj.toLong * width + c))
        .sum /
        math.sqrt(width)
      val root = math.sqrt(math.max(math.abs(s), 1e-6))
      1 / (1 + math.exp(-(if (s < 0) -root else root)))
    }
    fill(out) { index =>
      val tj = (index / width).toInt
      gates(tj) * v(tj / streams * width.toLong + index % width)
    }
  }

  def ngramRows(
      tokens: Tensor,
      state: Tensor,
      hash: NgramHash,
      history: Option[Tensor],
      rows: Tensor
  ): Unit = {
    val count = Ops.checkNgramRows(tokens, state, hash, rows)
    history.foreach(Ops.checkHistory(_, count, state))
    val (ids, memory, target) = (segment(tokens), segment(state), segment(rows))
    // the stored predecessors, oldest first, then the call's tokens
    val seen = (0 until hash.kept).map { i =>
      val stored = memory.getAtIndex(JAVA_FLOAT, i.toLong).toInt
      if (stored == 0) hash.eos else stored - 1
    } ++ (0 until count).map(t => ids.getAtIndex(JAVA_INT, t.toLong))
    (0 until count).foreach { t =>
      val at = t + hash.kept
      val context = (0 until hash.ngramSize)
        .scanLeft((seen(at), false)) { case ((_, cut), k) =>
          if (k == 0) (seen(at), false)
          else {
            val previous = seen(at - k)
            val nowCut = cut || previous == hash.eos
            (if (nowCut) hash.eos else previous, nowCut)
          }
        }
        .tail
        .map(_._1)
      (0 until hash.heads).foreach { h =>
        val n = 2 + h / hash.headsPerNgram
        val mixed = (0 until n)
          .map(j => context(j).toLong * hash.multipliers(j))
          .reduce(_ ^ _)
        val row = java.lang.Long.remainderUnsigned(mixed, hash.sizes(h)) +
          hash.offsets(h)
        target.setAtIndex(JAVA_INT, t.toLong * hash.heads + h, row.toInt)
      }
      history.map(segment).foreach { kept =>
        (0 until hash.kept).foreach(i =>
          kept.setAtIndex(
            JAVA_FLOAT,
            t.toLong * hash.kept + i,
            (seen(at - (hash.kept - 1 - i)) + 1).toFloat
          )
        )
      }
    }
    (0 until hash.kept).foreach(i =>
      memory.setAtIndex(
        JAVA_FLOAT,
        i.toLong,
        (seen(count + i) + 1).toFloat
      )
    )
  }

  def moeCombine(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Option[Tensor],
      out: Tensor
  ): Unit = {
    val Seq(tokens, hidden) = out.shape.dimensions
    val k = weights.shape.elementCount / tokens
    val (experts, weight) = (reader(expertOut), reader(weights))
    val gates = shared.map { expert =>
      val (input, router) = (reader(expert.input), reader(expert.router))
      Array.tabulate(tokens.toInt)(t =>
        1 / (1 + math.exp(
          -(0L until hidden).map(c => input(t * hidden + c) * router(c)).sum
        ))
      )
    }
    val sharedOutput = shared.map(expert => reader(expert.output))
    val before =
      residual.map(r => CpuOps.copy(reader(r), out.shape.elementCount))
    fill(out) { i =>
      val (t, c) = (i / hidden, i % hidden)
      val routed = (0L until k)
        .map(j => weight(t * k + j) * experts((t * k + j) * hidden + c))
        .sum
      before.fold(0.0)(_(i.toInt)) + routed + gates.fold(0.0)(g =>
        g(t.toInt) * sharedOutput.get(i)
      )
    }
  }

  def allocateCache(
      pages: Int,
      pageSize: Int,
      kvHeads: Int,
      headDimension: Int
  ): KvCache =
    KvCache(
      allocate(DType.F16, Shape.of(pages, pageSize, kvHeads, headDimension)),
      allocate(DType.F16, Shape.of(pages, kvHeads, headDimension, pageSize)),
      pageSize
    )

  /** Where position `position`'s key for `head`, dimension `d`, lies in the key
    * pool, and its value in the value pool, in elements.
    */
  private def cacheIndices(
      cache: KvCache,
      table: MemorySegment,
      position: Int,
      head: Int,
      d: Int
  ): (Long, Long) = {
    val page =
      table.getAtIndex(JAVA_INT, (position / cache.pageSize).toLong).toLong
    val offset = position % cache.pageSize
    val (heads, dimension, size) =
      (cache.kvHeads, cache.headDimension, cache.pageSize)
    (
      ((page * size + offset) * heads + head) * dimension + d,
      ((page * heads + head) * dimension + d) * size + offset
    )
  }

  def cacheWrite(
      k: Tensor,
      v: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      start: Int
  ): Unit = {
    val tokens = Ops.checkCacheWrite(k, v, cache, pageTable, start)
    val (keys, values, table) =
      (segment(cache.keys), segment(cache.values), segment(pageTable))
    val (key, value) = (reader(k), reader(v))
    for {
      t <- 0 until tokens
      h <- 0 until cache.kvHeads
      d <- 0 until cache.headDimension
    } {
      val source = (t.toLong * cache.kvHeads + h) * cache.headDimension + d
      val (keyAt, valueAt) = cacheIndices(cache, table, start + t, h, d)
      keys.setAtIndex(
        JAVA_SHORT,
        keyAt,
        java.lang.Float.floatToFloat16(key(source).toFloat)
      )
      values.setAtIndex(
        JAVA_SHORT,
        valueAt,
        java.lang.Float.floatToFloat16(value(source).toFloat)
      )
    }
  }

  /** Scores, softmax and weighted values in double, one query row at a time,
    * over the F16 keys and values as stored.
    */
  def attention(
      q: Tensor,
      cache: KvCache,
      pageTable: Tensor,
      queryStart: Int,
      keyCount: Int,
      attention: Attention,
      out: Tensor
  ): Unit = {
    val (tokens, qHeads) = Ops.checkAttention(
      q,
      cache,
      pageTable,
      queryStart,
      keyCount,
      attention,
      out
    )
    val (keys, values, table, target) = (
      segment(cache.keys),
      segment(cache.values),
      segment(pageTable),
      segment(out)
    )
    val query = reader(q)
    val sinks = attention.sinks.map(reader)
    val dimension = cache.headDimension
    val group = qHeads / cache.kvHeads
    def half(pool: MemorySegment, index: Long) =
      java.lang.Float
        .float16ToFloat(pool.getAtIndex(JAVA_SHORT, index))
        .toDouble
    for {
      t <- 0 until tokens
      head <- 0 until qHeads
    } {
      val kvHead = head / group
      val position = queryStart + t
      val queryAt = (t.toLong * qHeads + head) * dimension
      val visible = (0 until keyCount).filter { key =>
        (!attention.causal || key <= position) && attention.window.forall(w =>
          key > position - w
        )
      }
      val scores = visible.map { key =>
        val dot = (0 until dimension).map { d =>
          query(queryAt + d) * half(
            keys,
            cacheIndices(cache, table, key, kvHead, d)._1
          )
        }.sum
        val score = dot * attention.scale
        attention.softcap.fold(score)(cap => cap * math.tanh(score / cap))
      }
      val sink = sinks.map(_(head.toLong))
      val largest = (scores ++ sink).maxOption.getOrElse(0.0)
      val weights = scores.map(s => math.exp(s - largest))
      val total = weights.sum + sink.fold(0.0)(s => math.exp(s - largest))
      (0 until dimension).foreach { d =>
        val sum = visible.indices.map { i =>
          weights(i) * half(
            values,
            cacheIndices(cache, table, visible(i), kvHead, d)._2
          )
        }.sum
        target.setAtIndex(
          JAVA_FLOAT,
          queryAt + d,
          (if (total > 0) sum / total else 0.0).toFloat
        )
      }
    }
  }

  /** The host view of a tensor; a registered mapping is readable here too. */
  private def segment(tensor: Tensor): MemorySegment = {
    val whole = tensor.storage match {
      case Storage.Host(segment)       => segment
      case Storage.Registered(host, _) => host
      case Storage.Device(_, _)        =>
        throw new IllegalArgumentException(
          "the cpu backend cannot read device memory; download the tensor first"
        )
    }
    whole.asSlice(tensor.byteOffset, tensor.byteSize)
  }

  def close(): Unit = arena.close()
}

object CpuOps {

  /** A tensor's values, read before `out` (which may be the same) is written.
    */
  def copy(read: Long => Double, count: Long): Array[Double] =
    Array.tabulate(count.toInt)(i => read(i.toLong))

  /** The error function, to about 1e-15: its Taylor series near zero, the
    * continued fraction of erfc beyond.
    */
  def erf(x: Double): Double =
    if (x < 0) -erf(-x)
    else if (x < 2.5) {
      var term = x
      var sum = x
      var n = 0
      while (math.abs(term) > 1e-17 * math.abs(sum)) {
        n += 1
        term *= -x * x / n
        sum += term / (2 * n + 1)
      }
      2 / math.sqrt(math.Pi) * sum
    } else {
      // erfc(x) = exp(−x²)/√π · 1/(x + 1/2/(x + 1/(x + 3/2/(x + …))))
      var fraction = 0.0
      var n = 60
      while (n >= 1) {
        fraction = (n / 2.0) / (x + fraction)
        n -= 1
      }
      1 - math.exp(-x * x) / math.sqrt(math.Pi) / (x + fraction)
    }

  /** A float to bfloat16 bits, rounding to nearest even; NaN stays NaN. */
  def toBfloat16(value: Float): Short = {
    val bits = java.lang.Float.floatToRawIntBits(value)
    if (value.isNaN) ((bits >>> 16) | 0x40).toShort
    else ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16).toShort
  }
}
