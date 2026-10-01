package drift.runner.diffusion

import java.awt.image.BufferedImage

/** MiniMax H3's latents as the transformer's rows, and how its conditions are
  * drawn and fitted to the canvas.
  */
object MiniMaxH3Conditions {

  /** Latents `[frames, height, width, C]` → the transformer's rows: frame by
    * frame, 2 × 2 patches row by row, each `(c, py, px)` (diffusers'
    * `patchify_video_latents`).
    */
  def patchify(
      latents: Array[Float],
      frames: Int,
      height: Int,
      width: Int,
      channels: Int
  ): Array[Float] = {
    val out = new Array[Float](latents.length)
    latents.indices.foreach { i =>
      val (row, feature) = (i / (4 * channels), i % (4 * channels))
      val (c, py, px) = (feature / 4, feature / 2 % 2, feature % 2)
      val perFrame = (height / 2) * (width / 2)
      val (t, y, x) =
        (row / perFrame, row % perFrame / (width / 2), row % (width / 2))
      out(i) =
        latents(((t * height + 2 * y + py) * width + 2 * x + px) * channels + c)
    }
    out
  }

  /** `patchify`'s inverse. */
  def unpatchify(
      rows: Array[Float],
      frames: Int,
      height: Int,
      width: Int,
      channels: Int
  ): Array[Float] = {
    val out = new Array[Float](rows.length)
    rows.indices.foreach { i =>
      val (row, feature) = (i / (4 * channels), i % (4 * channels))
      val (c, py, px) = (feature / 4, feature / 2 % 2, feature % 2)
      val perFrame = (height / 2) * (width / 2)
      val (t, y, x) =
        (row / perFrame, row % perFrame / (width / 2), row % (width / 2))
      out(((t * height + 2 * y + py) * width + 2 * x + px) * channels + c) =
        rows(i)
    }
    out
  }

  /** Values of a `[C, frames, height, width]` tensor (torch's latent layout, as
    * noise is drawn) as `[frames, height, width, C]`.
    */
  def channelsLast(
      values: Array[Float],
      frames: Int,
      height: Int,
      width: Int,
      channels: Int
  ): Array[Float] = {
    val voxels = frames * height * width
    Array.tabulate(values.length)(i =>
      values((i % channels) * voxels + i / channels)
    )
  }

  /** `level × clean + (1 − level) × noise` (the rectified flow at `t = level`,
    * diffusers' `scale_noise`).
    */
  def mixed(
      clean: Array[Float],
      noise: Array[Float],
      level: Float
  ): Array[Float] =
    Array.tabulate(clean.length)(i => level * clean(i) + (1 - level) * noise(i))

  /** Encoded latents `[frames, height, width, C]` (channels last). */
  final case class Latents(
      values: Array[Float],
      frames: Int,
      height: Int,
      width: Int
  )

  /** Diffusers' draws of `random` for the conditions (fl2va's keyframes,
    * ref2va's image and clip references) then the generated rows: each
    * condition's noise in its latent shape (`randn(condition.shape)`), mixed at
    * `level`, as rows; the video's noise as a `[C, frames, height, width]`
    * tensor, as rows; the audio's rows `[audioRows, audioWidth]`.
    */
  def draws(
      random: TorchRandom,
      conditions: Seq[Latents],
      frames: Int,
      height: Int,
      width: Int,
      channels: Int,
      audioRows: Int,
      audioWidth: Int,
      level: Float
  ): (Seq[Array[Float]], Array[Float], Array[Float]) = {
    val drawn = conditions.map { latents =>
      val (t, h, w) = (latents.frames, latents.height, latents.width)
      val noise =
        channelsLast(random.normal(latents.values.length), t, h, w, channels)
      patchify(mixed(latents.values, noise, level), t, h, w, channels)
    }
    val video = patchify(
      channelsLast(
        random.normal(frames * height * width * channels),
        frames,
        height,
        width,
        channels
      ),
      frames,
      height,
      width,
      channels
    )
    (drawn, video, random.normal(audioRows * audioWidth))
  }

  /** Frames at `rate` onto `target` (diffusers' `_normalize_video_condition`,
    * as ffmpeg's `fps` filter): frame `i` held from slot `⌊i × target / rate +
    * ½⌋` to the next one's, the last to the slot the clip's end rounds to.
    */
  def atRate[A](frames: Seq[A], rate: Double, target: Double): Seq[A] =
    if (rate == target) frames
    else {
      val scale = target / rate
      def slot(i: Int) = math.floor(i * scale + 0.5).toInt
      frames.indices.flatMap(i => Seq.fill(slot(i + 1) - slot(i))(frames(i)))
    }

  /** diffusers' `resolve_canvas_size` (ComfyUI's `adapt_canvas`): the aspect of
    * `width` × `height` with a short edge of `shortEdge`, its area capped at
    * `maximumPixels`, each side rounded to a multiple of `multiple`; `(width,
    * height)`. Aspects beyond 1:4 and 4:1 are refused.
    */
  def canvas(
      width: Int,
      height: Int,
      multiple: Int,
      shortEdge: Int,
      maximumPixels: Int
  ): (Int, Int) = {
    val ratio = width.toDouble / height
    require(
      ratio >= 0.25 && ratio <= 4,
      s"MiniMax H3 takes aspects from 1:4 to 4:1, not $width × $height"
    )
    val (wide, high) =
      if (ratio >= 1) (shortEdge * ratio, shortEdge.toDouble)
      else (shortEdge.toDouble, shortEdge / ratio)
    val scale = math.min(1.0, math.sqrt(maximumPixels / (wide * high)))
    def rounded(side: Double) =
      math.max(multiple, math.rint(side * scale / multiple).toInt * multiple)
    (rounded(wide), rounded(high))
  }

  /** torch's `interpolate(…, align_corners=False)` along one axis without
    * antialiasing: each output's two source taps and the second's weight.
    */
  private def linearTaps(from: Int, to: Int): Array[(Int, Int, Double)] =
    Array.tabulate(to) { i =>
      val source = math.max(0.0, (i + 0.5) * from / to - 0.5)
      val first = math.min(source.toInt, from - 1)
      (first, math.min(first + 1, from - 1), source - first)
    }

  /** ComfyUI's `common_upscale(…, "bilinear", "center")` of `[height, width,
    * channels]` values: the centred window of the target's aspect (its margin
    * rounded half to even, as Python), then torch's bilinear interpolation.
    */
  def bilinearCenter(
      values: Array[Float],
      width: Int,
      height: Int,
      channels: Int,
      toWidth: Int,
      toHeight: Int
  ): Array[Float] = {
    val (oldAspect, newAspect) =
      (width.toDouble / height, toWidth.toDouble / toHeight)
    val (x, y) =
      if (oldAspect > newAspect)
        (math.rint((width - width * (newAspect / oldAspect)) / 2).toInt, 0)
      else if (oldAspect < newAspect)
        (0, math.rint((height - height * (oldAspect / newAspect)) / 2).toInt)
      else (0, 0)
    val (windowWidth, windowHeight) = (width - 2 * x, height - 2 * y)
    val (across, down) =
      (linearTaps(windowWidth, toWidth), linearTaps(windowHeight, toHeight))
    val out = new Array[Float](toWidth * toHeight * channels)
    for {
      row <- 0 until toHeight
      column <- 0 until toWidth
      c <- 0 until channels
    } {
      val (top, bottom, dy) = down(row)
      val (left, right, dx) = across(column)
      def at(r: Int, k: Int) = values(((y + r) * width + x + k) * channels + c)
      val upper = at(top, left) * (1 - dx) + at(top, right) * dx
      val lower = at(bottom, left) * (1 - dx) + at(bottom, right) * dx
      out((row * toWidth + column) * channels + c) =
        (upper * (1 - dy) + lower * dy).toFloat
    }
    out
  }

  /** torch's trilinear `interpolate(…, align_corners=False)` of `[frames,
    * height, width]` values to `[toFrames, toHeight, toWidth]`.
    */
  def trilinear(
      values: Array[Float],
      frames: Int,
      height: Int,
      width: Int,
      toFrames: Int,
      toHeight: Int,
      toWidth: Int
  ): Array[Float] = {
    val (times, rows, columns) = (
      linearTaps(frames, toFrames),
      linearTaps(height, toHeight),
      linearTaps(width, toWidth)
    )
    Array.tabulate(toFrames * toHeight * toWidth) { i =>
      val (t0, t1, dt) = times(i / (toHeight * toWidth))
      val (y0, y1, dy) = rows(i / toWidth % toHeight)
      val (x0, x1, dx) = columns(i % toWidth)
      def at(t: Int, y: Int, x: Int) = values((t * height + y) * width + x)
      def plane(t: Int) = {
        val upper = at(t, y0, x0) * (1 - dx) + at(t, y0, x1) * dx
        val lower = at(t, y1, x0) * (1 - dx) + at(t, y1, x1) * dx
        upper * (1 - dy) + lower * dy
      }
      (plane(t0) * (1 - dt) + plane(t1) * dt).toFloat
    }
  }

  /** The control latents ComfyUI's `MiniMaxH3FunControlPatch` reads, `[frames,
    * height, width, channels]` for the generated video's `latentFrames` ×
    * `latentHeight` × `latentWidth` latents on a `width` × `height` canvas: the
    * control frames (at `fps`, then one a frame of the video, the last held)
    * fitted onto the canvas (`bilinearCenter`) and encoded, `latent` channels;
    * with a mask (white regenerates: its mean > ½, fitted the same way, > ½
    * again), its complement trilinearly onto the latents' grid and the source
    * frames (or black) outside it encoded, after them; the rest zero. `encode`
    * takes frames in [−1, 1] to normalized latents.
    */
  def controlLatents(
      control: VideoControl,
      fps: Int,
      width: Int,
      height: Int,
      latentFrames: Int,
      latentHeight: Int,
      latentWidth: Int,
      latent: Int,
      channels: Int,
      encode: Seq[Array[Float]] => Array[Float]
  ): Array[Float] = {
    val frameCount = math.max((latentFrames - 2) / 5, 0) * 17 + 5
    val voxels = latentFrames * latentHeight * latentWidth
    // each frame's [height, width, depth] values in [0, 1], fitted
    def fitted(
        frames: Seq[BufferedImage],
        depth: Int,
        values: BufferedImage => Array[Float]
    ): Seq[Array[Float]] = {
      val timed = atRate(frames, control.fps, fps)
      (0 until frameCount).map { i =>
        val frame = timed(math.min(i, timed.size - 1))
        bilinearCenter(
          values(frame),
          frame.getWidth,
          frame.getHeight,
          depth,
          width,
          height
        )
      }
    }
    def colors(image: BufferedImage) =
      Images.pixels(image).map(v => (v + 1) / 2)
    def encoded(frames: Seq[Array[Float]]) = {
      val latents = encode(frames.map(_.map(v => v * 2 - 1)))
      require(
        latents.length == voxels * latent,
        s"control latents of ${latents.length} values for $voxels voxels of $latent"
      )
      latents
    }
    val hint = Option.when(control.frames.nonEmpty)(
      encoded(fitted(control.frames, 3, colors))
    )
    val masked = control.mask.map { masks =>
      val visibility = fitted(
        masks,
        1,
        image => {
          val rgb = colors(image)
          Array.tabulate(rgb.length / 3)(i =>
            if (rgb(3 * i) + rgb(3 * i + 1) + rgb(3 * i + 2) > 1.5f) 1f else 0f
          )
        }
      ).map(_.map(m => if (m > 0.5f) 0f else 1f))
      val sources = control.source.map(fitted(_, 3, colors))
      val hidden = visibility.indices.map { f =>
        Array.tabulate(width * height * 3)(i =>
          sources.fold(0f)(_(f)(i)) * visibility(f)(i / 3)
        )
      }
      val grid = trilinear(
        visibility.flatten.toArray,
        frameCount,
        height,
        width,
        latentFrames,
        latentHeight,
        latentWidth
      )
      (grid, encoded(hidden))
    }
    val out = new Array[Float](voxels * channels)
    (0 until voxels).foreach { v =>
      hint.foreach(values =>
        System.arraycopy(values, v * latent, out, v * channels, latent)
      )
      masked.foreach { (grid, values) =>
        out(v * channels + latent) = grid(v)
        System.arraycopy(
          values,
          v * latent,
          out,
          v * channels + latent + 1,
          latent
        )
      }
    }
    out
  }

  /** diffusers' fl2va follower (`MiniMaxH3ResizeStep`): scaled to cover the
    * canvas (sides rounded, at least the canvas'), then cropped at `(size −
    * canvas) / 2`.
    */
  def coverCropped(
      image: BufferedImage,
      width: Int,
      height: Int
  ): BufferedImage = {
    val scale = math.max(
      width.toDouble / image.getWidth,
      height.toDouble / image.getHeight
    )
    val (scaledWidth, scaledHeight) = (
      math.max(width, math.round(image.getWidth * scale).toInt),
      math.max(height, math.round(image.getHeight * scale).toInt)
    )
    val scaled = Images.resized(image, scaledWidth, scaledHeight)
    cropped(
      scaled,
      (scaledWidth - width) / 2,
      (scaledHeight - height) / 2,
      width,
      height
    )
  }

  /** ComfyUI's `common_upscale(…, "center")`: the centred window of the canvas'
    * aspect (its margin rounded), then scaled to the canvas.
    */
  def centerCropped(
      image: BufferedImage,
      width: Int,
      height: Int
  ): BufferedImage = {
    val (oldWidth, oldHeight) = (image.getWidth, image.getHeight)
    val (oldAspect, newAspect) =
      (oldWidth.toDouble / oldHeight, width.toDouble / height)
    val window =
      if (oldAspect > newAspect) {
        val x =
          math.round((oldWidth - oldWidth * (newAspect / oldAspect)) / 2).toInt
        cropped(image, x, 0, oldWidth - 2 * x, oldHeight)
      } else if (oldAspect < newAspect) {
        val y = math
          .round((oldHeight - oldHeight * (oldAspect / newAspect)) / 2)
          .toInt
        cropped(image, 0, y, oldWidth, oldHeight - 2 * y)
      } else image
    Images.resized(window, width, height)
  }

  private def cropped(
      image: BufferedImage,
      left: Int,
      top: Int,
      width: Int,
      height: Int
  ): BufferedImage = {
    val out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val graphics = out.createGraphics()
    try
      graphics.drawImage(
        image.getSubimage(left, top, width, height),
        0,
        0,
        null
      )
    finally graphics.dispose()
    out
  }
}
