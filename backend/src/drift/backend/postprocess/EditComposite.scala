package drift.backend.postprocess

import java.awt.image.BufferedImage

/** What drift keeps of an edited tile (`specs/39-seamless-edit.md`). An edit
  * model re-renders the whole tile, the parts the instruction never mentioned
  * included — Flux.2 Klein gave untouched skin a greyer tone and a grain of its
  * own — so drift takes the source's pixels back wherever the edit did not
  * really change anything:
  *
  *   1. colour match — a per-channel gain and offset, fitted by least squares
  *      on the pixels whose blurred difference is in the lowest 60 %, maps the
  *      edit onto the source's colours;
  *   2. change mask — the largest per-channel difference of the two images' 6
  *      px Gaussian means, ramped from 0 at 6 % to 1 at 16 %, grown by 4 px and
  *      feathered by 4 px: grain is not a change, a new colour or shape is;
  *   3. composite — the source where the mask is 0, the matched edit where it
  *      is 1.
  *
  * The constants come from one measured subject; they stay constants until
  * another shows them wrong.
  */
private[postprocess] object EditComposite {

  /** The composite, the change mask as a grey image, and the mask's mean — the
    * share of the tile the edit changed.
    */
  case class Result(
      image: BufferedImage,
      mask: BufferedImage,
      changedShare: Double
  )

  private val MeanRadius = 6.0
  private val CalmQuantile = 0.6
  private val MaskFloor = 0.06f
  private val MaskCeiling = 0.16f
  private val MaskGrowth = 4
  private val MaskFeather = 4.0

  /** `edited` — what the model returned for `source` — kept where it changed
    * something and brought to the source's colours there; the source everywhere
    * else. Both images are the same size.
    */
  def apply(source: BufferedImage, edited: BufferedImage): Result = {
    val width = source.getWidth
    val height = source.getHeight
    val sourceChannels = channels(source)
    val editedChannels = channels(edited)
    val sourceMeans =
      sourceChannels.map(gaussian(_, width, height, MeanRadius))
    val editedMeans =
      editedChannels.map(gaussian(_, width, height, MeanRadius))

    // The pixels the edit barely moved are the ones that say how its colours
    // relate to the source's; what it did change would skew the fit.
    val firstDifference = largestDifference(editedMeans, sourceMeans)
    val calmLimit = quantile(firstDifference, CalmQuantile)
    val calm = firstDifference.map(_ < calmLimit)
    val fits = sourceChannels.indices.map(c =>
      fit(editedChannels(c), sourceChannels(c), calm)
    )
    def matchedOf(values: Array[Float], c: Int): Array[Float] = {
      val (gain, offset) = fits(c)
      values.map(value => clamp(gain * value + offset))
    }
    val matched =
      editedChannels.indices.map(c => matchedOf(editedChannels(c), c))
    // Gain and offset are linear, so the mean of the matched edit is the
    // matched mean of the edit, up to the clamp.
    val matchedMeans =
      editedMeans.indices.map(c => matchedOf(editedMeans(c), c))

    val ramp = largestDifference(matchedMeans, sourceMeans).map(difference =>
      clamp((difference - MaskFloor) / (MaskCeiling - MaskFloor))
    )
    val mask = gaussian(
      grown(ramp, width, height, MaskGrowth),
      width,
      height,
      MaskFeather
    )

    val composite = new Array[Int](width * height)
    val grey = new Array[Int](width * height)
    for (index <- composite.indices) {
      val weight = mask(index)
      def channel(c: Int): Int =
        math
          .round(
            (sourceChannels(c)(index) * (1 - weight) +
              matched(c)(index) * weight) * 255
          )
          .toInt
          .max(0)
          .min(255)
      composite(index) =
        0xff000000 | channel(0) << 16 | channel(1) << 8 | channel(2)
      val level = math.round(weight * 255).toInt.max(0).min(255)
      grey(index) = 0xff000000 | level << 16 | level << 8 | level
    }
    Result(
      image(composite, width, height),
      image(grey, width, height),
      mask.map(_.toDouble).sum / mask.length
    )
  }

  private def image(pixels: Array[Int], width: Int, height: Int) = {
    val result = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    result.setRGB(0, 0, width, height, pixels, 0, width)
    result
  }

  /** Red, green and blue, each from 0 to 1. */
  private def channels(picture: BufferedImage): IndexedSeq[Array[Float]] = {
    val pixels = picture.getRGB(
      0,
      0,
      picture.getWidth,
      picture.getHeight,
      null,
      0,
      picture.getWidth
    )
    IndexedSeq(16, 8, 0).map(shift =>
      pixels.map(pixel => ((pixel >> shift) & 0xff) / 255f)
    )
  }

  private def clamp(value: Float): Float = value.max(0f).min(1f)

  /** Per pixel, the largest of the three channels' absolute differences. */
  private def largestDifference(
      one: IndexedSeq[Array[Float]],
      other: IndexedSeq[Array[Float]]
  ): Array[Float] = {
    val largest = new Array[Float](one.head.length)
    for {
      index <- largest.indices
      c <- one.indices
    } largest(index) =
      largest(index) max math.abs(one(c)(index) - other(c)(index))
    largest
  }

  private def quantile(values: Array[Float], share: Double): Float = {
    val sorted = values.sorted
    sorted(((sorted.length - 1) * share).toInt)
  }

  /** The least-squares gain and offset taking `from` to `to` over the pixels
    * `use` marks; no change at all when those pixels do not vary.
    */
  private def fit(
      from: Array[Float],
      to: Array[Float],
      use: Array[Boolean]
  ): (Float, Float) = {
    var count = 0L
    var sumX, sumY, sumXX, sumXY = 0.0
    for (index <- from.indices if use(index)) {
      val x = from(index).toDouble
      val y = to(index).toDouble
      count += 1
      sumX += x
      sumY += y
      sumXX += x * x
      sumXY += x * y
    }
    if (count == 0) (1f, 0f)
    else {
      val meanX = sumX / count
      val meanY = sumY / count
      val variance = sumXX / count - meanX * meanX
      val gain =
        if (variance < 1e-6) 1.0
        else (sumXY / count - meanX * meanY) / variance
      (gain.toFloat, (meanY - gain * meanX).toFloat)
    }
  }

  /** A Gaussian blur of standard deviation `sigma`, as three box blurs of the
    * width that matches it, sampling clamped at the edges.
    */
  private def gaussian(
      values: Array[Float],
      width: Int,
      height: Int,
      sigma: Double
  ): Array[Float] = {
    val radius =
      math.round((math.sqrt(4 * sigma * sigma + 1) - 1) / 2).toInt.max(1)
    var blurred = values
    for (_ <- 0 until 3) {
      blurred = box(blurred, width, height, radius, horizontal = true)
      blurred = box(blurred, width, height, radius, horizontal = false)
    }
    blurred
  }

  /** One box-blur pass along one axis, as a running sum. */
  private def box(
      values: Array[Float],
      width: Int,
      height: Int,
      radius: Int,
      horizontal: Boolean
  ): Array[Float] = {
    val out = new Array[Float](values.length)
    val (lines, length) = if (horizontal) (height, width) else (width, height)
    def at(line: Int, position: Int): Int = {
      val clamped = position.max(0).min(length - 1)
      if (horizontal) line * width + clamped else clamped * width + line
    }
    val span = 2 * radius + 1
    for (line <- 0 until lines) {
      var sum = 0.0
      for (position <- -radius to radius) sum += values(at(line, position))
      for (position <- 0 until length) {
        out(at(line, position)) = (sum / span).toFloat
        sum += values(at(line, position + radius + 1)) -
          values(at(line, position - radius))
      }
    }
    out
  }

  /** Every pixel raised to the largest value within `radius` of it. */
  private def grown(
      values: Array[Float],
      width: Int,
      height: Int,
      radius: Int
  ): Array[Float] = {
    def pass(input: Array[Float], horizontal: Boolean): Array[Float] =
      Array.tabulate(input.length) { index =>
        val x = index % width
        val y = index / width
        var largest = 0f
        for (offset <- -radius to radius) {
          val sample =
            if (horizontal) y * width + (x + offset).max(0).min(width - 1)
            else (y + offset).max(0).min(height - 1) * width + x
          largest = largest max input(sample)
        }
        largest
      }
    pass(pass(values, horizontal = true), horizontal = false)
  }
}
