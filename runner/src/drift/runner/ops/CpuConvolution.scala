package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.ValueLayout.*

/** The reference backend's convolutions and resampling.
  */
private[ops] trait CpuConvolution extends CpuBase {

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
}
