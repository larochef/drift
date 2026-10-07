package drift.backend.postprocess

import drift.shared.ImageRegion

import java.awt.image.BufferedImage
import scala.collection.mutable

/** What drift keeps of an edited picture (`specs/39-seamless-edit.md`). An edit
  * model re-renders everything it is given, the parts the instruction never
  * mentioned included — Flux.2 Klein hands the picture back a pixel or two to
  * the side, relit, with a grain of its own — so drift takes the source's
  * pixels back wherever the edit did not really change anything:
  *
  *   1. alignment — the edit is moved back by the whole-pixel shift that fits
  *      it best on the source;
  *   2. colour match — a per-channel gain and offset, fitted by least squares
  *      on the pixels whose blurred difference is in the lowest 60 %, then the
  *      slow drift that is left — read where nothing changed and carried across
  *      what did — is taken off: a relit wall is not an edit, and the edit
  *      meets its surroundings at their own colour;
  *   3. change mask — a pixel changed if its 6 px mean moved by 16 % over a
  *      patch, or by 5 % while joined to such a patch, where "moved" also
  *      counts a change of local contrast: cream wool over a beige shirt moves
  *      the colour little and the texture a lot. Holes up to 48 px are closed —
  *      an object is edited whole or not at all — and the mask is grown by 6 px
  *      and feathered by as much;
  *   4. composite — the source where the mask is 0, the matched edit where it
  *      is 1. With a selection (`keep`), only the changed parts that touch it
  *      are kept, each as far as it goes.
  *
  * Measured the night of 2026-10-05
  * (`~/dev/redraw-experiments/2026-10-05-edit`, `ec.composite2`): the mask
  * before — the colour mean alone, ramped from 6 % to 16 % — left a sweater
  * half shirt, and called a relit wall an edit.
  */
private[postprocess] object EditComposite {

  /** The composite, the change mask as a grey image, the mask's mean — the
    * share of the picture the edit changed — and the shift, in pixels, that put
    * the edit back on the source.
    */
  case class Result(
      image: BufferedImage,
      mask: BufferedImage,
      changedShare: Double,
      shift: (Int, Int)
  )

  private val MeanRadius = 6.0
  private val CalmQuantile = 0.6
  private val ChangeFloor = 0.05f
  private val ChangeSeed = 0.16f
  private val TextureWeight = 1.0f
  private val ChangeSmoothing = 3.0
  private val DriftRadius = 32.0
  private val SeedSide = 4
  private val JoinReach = 6
  private val CloseRadius = 48
  private val MaskGrowth = 6
  private val MaskFeather = 6.0
  private val LargestShift = 8

  /** `edited` — what the model returned for `source` — kept where it changed
    * something and brought to the source's colours there; the source everywhere
    * else. Both images are the same size.
    */
  def apply(
      source: BufferedImage,
      edited: BufferedImage,
      keep: Option[ImageRegion] = None
  ): Result = {
    val width = source.getWidth
    val height = source.getHeight
    val sourceChannels = channels(source)
    val returned = channels(edited)
    val shift = shiftOf(sourceChannels, returned, width, height)
    val editedChannels =
      returned.map(moved(_, width, height, shift._1, shift._2))
    val Matching(matched, sourceMeans, matchedMeans) =
      matching(sourceChannels, editedChannels, width, height, None, DriftRadius)
    val colour = largestDifference(matchedMeans, sourceMeans)
    val sourceContrast = contrast(grey(sourceChannels), width, height)
    val matchedContrast = contrast(grey(matched), width, height)
    val change = gaussian(
      Array.tabulate(colour.length)(index =>
        colour(index) max TextureWeight * math.abs(
          matchedContrast(index) - sourceContrast(index)
        )
      ),
      width,
      height,
      ChangeSmoothing
    )
    // A seed is a patch, not a line: an edge the model moved by a fraction of
    // a pixel is not an edit.
    val seeds = shrunk(
      change.map(value => if (value > ChangeSeed) 1f else 0f),
      width,
      height,
      SeedSide
    )
    val joined = reached(
      seeds.map(_ > 0.5f),
      change.map(_ > ChangeFloor),
      width,
      height
    )
    val whole = shrunk(
      grown(joined.map(if (_) 1f else 0f), width, height, CloseRadius),
      width,
      height,
      CloseRadius
    )
    // Of what changed, a selection keeps the parts that touch it, whole.
    val wanted = keep.fold(whole) { region =>
      val changed = whole.map(_ > 0.5f)
      val seeds = Array.tabulate(changed.length) { index =>
        val x = index % width
        val y = index / width
        changed(index) && x >= region.x && x < region.x + region.width &&
        y >= region.y && y < region.y + region.height
      }
      reached(seeds, changed, width, height).map(if (_) 1f else 0f)
    }
    val mask = gaussian(
      grown(wanted, width, height, MaskGrowth),
      width,
      height,
      MaskFeather
    ).map(clamp)
    composited(sourceChannels, matched, mask, width, height, shift)
  }

  /** How far a carried-up edit's difference must go, in the tight mask read at
    * the picture's own size, to count from nothing to fully: the ramp the
    * composite had before it was read at the model's size.
    */
  private val TightFloor = 0.06f
  private val TightCeiling = 0.16f
  private val TightGrowth = 4
  private val TightFeather = 4.0

  /** How wide of the object the model-size mask is, in its own pixels: its
    * growth and feather, and what closing its holes rounds off.
    */
  private val MaskMargin = 10

  /** What a carried-up edit keeps (`specs/39-seamless-edit.md`): `upscaled` is
    * the edited part brought back to the size of `source` by an upscaler, and
    * `changed` the mask read where the edit was made, enlarged `by` times to
    * that size. That mask says which object changed — whole, its holes closed —
    * but is some pixels wide of it, `by` times as many here: pasted through it,
    * a band of the wall as the model drew it comes along with the garment. The
    * difference read at this size hugs the edge and is full of holes. So the
    * mask kept is the enlarged one pulled in by its own margin, plus the
    * difference wherever the enlarged one allows it.
    */
  def carried(
      source: BufferedImage,
      upscaled: BufferedImage,
      changed: BufferedImage,
      by: Double
  ): Result = {
    val width = source.getWidth
    val height = source.getHeight
    val sourceChannels = channels(source)
    val inside =
      channels(changed).head.map(value => if (value > 0.5f) 1f else 0f)
    val allowed =
      grown(inside, width, height, math.round(JoinReach * by).toInt.max(1))
    val Matching(matched, sourceMeans, matchedMeans) = matching(
      sourceChannels,
      channels(upscaled),
      width,
      height,
      Some(allowed),
      DriftRadius * by
    )
    val tight = gaussian(
      grown(
        largestDifference(matchedMeans, sourceMeans).map(difference =>
          clamp((difference - TightFloor) / (TightCeiling - TightFloor))
        ),
        width,
        height,
        TightGrowth
      ),
      width,
      height,
      TightFeather
    )
    val margin = math.round(MaskMargin * by).toInt.max(1)
    val core = gaussian(
      shrunk(inside, width, height, margin),
      width,
      height,
      MaskFeather
    )
    val mask = Array.tabulate(tight.length)(index =>
      clamp(core(index) max tight(index) * allowed(index))
    )
    composited(sourceChannels, matched, mask, width, height, (0, 0))
  }

  /** The edit at the source's colours, and the 6 px means the masks are read
    * from.
    */
  private case class Matching(
      matched: IndexedSeq[Array[Float]],
      sourceMeans: IndexedSeq[Array[Float]],
      matchedMeans: IndexedSeq[Array[Float]]
  )

  /** The largest move a pixel may have made and still say how the edit's
    * colours relate to the source's, and the least share of the picture that
    * must be left to say it: on an edit that takes most of the picture the
    * calmest 60 % are the new garment itself, and a fit on them pulled cream
    * wool towards the grey shirt it replaced; a gain read off a sliver of
    * background is a tint over everything else.
    */
  private val CalmCap = 0.08f

  /** How much of its neighbourhood must be unchanged for the drift read there
    * to count in full inside a real change: with none, it counts for nothing.
    */
  private val DriftTrust = 0.25f
  private val CalmLeast = 0.2

  /** `changed` is what is known to have changed already, 1 where it has — a
    * carried-up edit's mask — and stands for the real change the drift keeps
    * away from; `reach` is how far the drift is carried, in these pixels.
    */
  private def matching(
      sourceChannels: IndexedSeq[Array[Float]],
      editedChannels: IndexedSeq[Array[Float]],
      width: Int,
      height: Int,
      changed: Option[Array[Float]],
      reach: Double
  ): Matching = {
    val sourceMeans =
      sourceChannels.map(gaussian(_, width, height, MeanRadius))
    val editedMeans =
      editedChannels.map(gaussian(_, width, height, MeanRadius))
    // The pixels the edit barely moved are the ones that say how its colours
    // relate to the source's; what it did change would skew the fit.
    val firstDifference = largestDifference(editedMeans, sourceMeans)
    val calmLimit = quantile(firstDifference, CalmQuantile) min CalmCap
    val calm = firstDifference.map(_ < calmLimit)
    val enough = calm.count(identity) >= CalmLeast * calm.length
    val fits = sourceChannels.indices.map(c =>
      if (enough) fit(editedChannels(c), sourceChannels(c), calm)
      else (1f, 0f)
    )
    def fitted(values: Array[Float], c: Int): Array[Float] = {
      val (gain, offset) = fits(c)
      values.map(value => clamp(gain * value + offset))
    }
    // Gain and offset are linear, so the mean of the fitted edit is the
    // fitted mean of the edit, up to the clamp.
    val fittedMeans = editedMeans.indices.map(c => fitted(editedMeans(c), c))
    // What the fit leaves is slow: the model lit a wall a little differently
    // from one side of the picture to the other. Read where nothing changed,
    // it is carried across what did by a normalised blur.
    // Not within reach of a real change, though: inside a new garment the
    // patches that happen to match the source would pull it towards the old
    // colour, in pastel blotches.
    val residual = largestDifference(fittedMeans, sourceMeans)
    val nearChange = changed.getOrElse(
      grown(
        residual.map(difference => if (difference > ChangeSeed) 1f else 0f),
        width,
        height,
        reach.toInt
      )
    )
    val quiet = Array.tabulate(residual.length)(index =>
      if (residual(index) < ChangeFloor && nearChange(index) < 0.5f) 1f
      else 0f
    )
    val quietWeight = gaussian(quiet, width, height, reach)
    // And trusted inside a real change only as far as unchanged pixels near by
    // vouch for it: a strap taken off a sleeve is filled at the sleeve's
    // colour — most of its neighbourhood is unchanged — and the middle of a
    // new sweater is left as the model coloured it, since none is: carried
    // there from the rim, the drift was a tint of its own, up to 20 levels. A
    // relit wall has no real change near it and is corrected in full.
    val amongChange = gaussian(nearChange, width, height, reach / 2)
    val drifts = sourceMeans.indices.map { c =>
      val gap = Array.tabulate(quiet.length)(index =>
        (sourceMeans(c)(index) - fittedMeans(c)(index)) * quiet(index)
      )
      val carried = gaussian(gap, width, height, reach)
      Array.tabulate(quiet.length)(index =>
        carried(index) /
          (quietWeight(index) + 1e-3f + DriftTrust * amongChange(index))
      )
    }
    def matchedOf(values: Array[Float], c: Int): Array[Float] =
      Array.tabulate(values.length)(index =>
        clamp(values(index) + drifts(c)(index))
      )
    Matching(
      editedChannels.indices.map(c =>
        matchedOf(fitted(editedChannels(c), c), c)
      ),
      sourceMeans,
      fittedMeans.indices.map(c => matchedOf(fittedMeans(c), c))
    )
  }

  /** The source where `mask` is 0, `matched` where it is 1. */
  private def composited(
      sourceChannels: IndexedSeq[Array[Float]],
      matched: IndexedSeq[Array[Float]],
      mask: Array[Float],
      width: Int,
      height: Int,
      shift: (Int, Int)
  ): Result = {
    val composite = new Array[Int](width * height)
    val shown = new Array[Int](width * height)
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
      shown(index) = 0xff000000 | level << 16 | level << 8 | level
    }
    Result(
      image(composite, width, height),
      image(shown, width, height),
      mask.map(_.toDouble).sum / mask.length,
      shift
    )
  }

  /** How far the model moved the picture: the whole-pixel shift, within
    * `LargestShift` px each way, at which the fine detail of `edited` — its
    * grey minus a 2 px mean, so colour and light do not enter — lines up best
    * with the source's, over one pixel in four. No shift unless one beats it by
    * a fifth: an edit that changed nothing stays where it is. On a flat wall
    * both details are noise and a shift can win by chance: moving a flat wall
    * by a few pixels changes nothing.
    */
  private def shiftOf(
      source: IndexedSeq[Array[Float]],
      edited: IndexedSeq[Array[Float]],
      width: Int,
      height: Int
  ): (Int, Int) =
    if (width <= 4 * LargestShift || height <= 4 * LargestShift) (0, 0)
    else {
      def detail(picture: IndexedSeq[Array[Float]]): Array[Float] = {
        val flat = grey(picture)
        val mean = gaussian(flat, width, height, 2.0)
        Array.tabulate(flat.length)(index => flat(index) - mean(index))
      }
      val from = detail(source)
      val to = detail(edited)
      def agreement(dx: Int, dy: Int): Double = {
        var sum = 0.0
        var y = LargestShift
        while (y < height - LargestShift) {
          var x = LargestShift
          while (x < width - LargestShift) {
            sum += from(y * width + x) * to((y - dy) * width + x - dx)
            x += 2
          }
          y += 2
        }
        sum
      }
      val still = agreement(0, 0)
      val candidates = for {
        dy <- -LargestShift to LargestShift
        dx <- -LargestShift to LargestShift
      } yield ((dx, dy), agreement(dx, dy))
      val (best, agreed) = candidates.maxBy(_._2)
      if (agreed > 0 && agreed > 1.2 * still) best else (0, 0)
    }

  /** `values` moved by `dx`, `dy`, the edge repeated where nothing comes in. */
  private def moved(
      values: Array[Float],
      width: Int,
      height: Int,
      dx: Int,
      dy: Int
  ): Array[Float] =
    if (dx == 0 && dy == 0) values
    else
      Array.tabulate(values.length) { index =>
        val x = (index % width - dx).max(0).min(width - 1)
        val y = (index / width - dy).max(0).min(height - 1)
        values(y * width + x)
      }

  /** The mean of the three channels. */
  private def grey(picture: IndexedSeq[Array[Float]]): Array[Float] =
    Array.tabulate(picture.head.length)(index =>
      (picture(0)(index) + picture(1)(index) + picture(2)(index)) / 3
    )

  /** The local contrast of `values`: their standard deviation under the
    * Gaussian the means are taken with.
    */
  private def contrast(
      values: Array[Float],
      width: Int,
      height: Int
  ): Array[Float] = {
    val mean = gaussian(values, width, height, MeanRadius)
    val square =
      gaussian(values.map(value => value * value), width, height, MeanRadius)
    Array.tabulate(values.length)(index =>
      math.sqrt((square(index) - mean(index) * mean(index)).max(0f)).toFloat
    )
  }

  /** The pixels of `allowed` joined to a seed: those within `JoinReach` px of
    * one already joined, starting from the seeds.
    */
  private def reached(
      seeds: Array[Boolean],
      allowed: Array[Boolean],
      width: Int,
      height: Int
  ): Array[Boolean] = {
    val joined = new Array[Boolean](seeds.length)
    val frontier = mutable.ArrayDeque.empty[Int]
    for (index <- seeds.indices if seeds(index) && allowed(index)) {
      joined(index) = true
      frontier.append(index)
    }
    while (frontier.nonEmpty) {
      val index = frontier.removeHead()
      val x = index % width
      val y = index / width
      var v = (y - JoinReach).max(0)
      while (v <= (y + JoinReach).min(height - 1)) {
        var u = (x - JoinReach).max(0)
        while (u <= (x + JoinReach).min(width - 1)) {
          val next = v * width + u
          if (allowed(next) && !joined(next)) {
            joined(next) = true
            frontier.append(next)
          }
          u += 1
        }
        v += 1
      }
    }
    joined
  }

  /** Every pixel lowered to the smallest value within `radius` of it. */
  private def shrunk(
      values: Array[Float],
      width: Int,
      height: Int,
      radius: Int
  ): Array[Float] =
    grown(values.map(1f - _), width, height, radius).map(1f - _)

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
        var offset = -radius
        while (offset <= radius) {
          val sample =
            if (horizontal) y * width + (x + offset).max(0).min(width - 1)
            else (y + offset).max(0).min(height - 1) * width + x
          if (input(sample) > largest) largest = input(sample)
          offset += 1
        }
        largest
      }
    pass(pass(values, horizontal = true), horizontal = false)
  }
}
