package drift.runner.diffusion

import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*

import java.awt.image.BufferedImage
import java.nio.file.*
import scala.collection.mutable

/** How SeedVR2 cuts its work: `batch` frames at a time (`1 + 4n`), each batch
  * starting on the last `overlap` frames of the one before, where the two
  * results are cross-faded; the VAE in spatial tiles of `tile` pixels
  * overlapping by `tileOverlap` at least (multiples of 16).
  */
final case class SeedVr2Options(
    batch: Int,
    overlap: Int,
    tile: Int,
    tileOverlap: Int
) {
  require(
    batch % 4 == 1 && overlap >= 0 && overlap < batch,
    s"batches of $batch frames overlapping by $overlap: 1 + 4n frames, fewer in the overlap"
  )
  require(
    tile % 16 == 0 && tileOverlap % 16 == 0 && tileOverlap < tile,
    s"tiles of $tile pixels overlapping by $tileOverlap: multiples of 16"
  )
}

object SeedVr2Options {

  /** 21 frames a batch, 4 shared with the next; the reference's tiles. */
  val Default: SeedVr2Options = SeedVr2Options(21, 4, 1024, 128)
}

/** SeedVR2 end to end, as the reference runs its one-step models
  * (`numz/ComfyUI-SeedVR2_VideoUpscaler`): frames already at the output's size
  * (the low-quality input resized) are encoded, each latent frame's token is 16
  * channels of noise, the 16 of that latent and a mask at 1, and one Euler step
  * from timestep 1000 gives the restored latent, `noise − velocity`, which is
  * decoded. No prompt and no guidance: the text is the model's fixed positive
  * embedding (a resource of the runner). The colours are left as decoded:
  * matching them to the source is the caller's.
  *
  * Long and large inputs: the frames go in batches (`SeedVr2Options`), every
  * batch from the same noise; the VAE works in tiles (`VaeTiles`), the
  * transformer over the whole frame, its attention being windowed.
  */
final class SeedVr2Pipeline(ops: Ops, diffusionModel: Path, vae: Path)
    extends AutoCloseable {

  private val transformer = SeedVr2.open(ops, diffusionModel)
  private val autoencoder = SeedVr2Vae.open(ops, vae)
  private val channels = SeedVr2Vae.LatentChannels.toInt
  private val patch = SeedVr2Config.Patch

  /** The frames' sides are multiples of this: a token is 2 × 2 latents. */
  val multiple: Int = 8 * patch

  private val text: Tensor = {
    // WeightSource reads files: the resource goes through a temporary one
    val file = Files.createTempFile("seedvr2-text", ".safetensors")
    try {
      val stream = getClass.getResourceAsStream(SeedVr2Pipeline.TextEmbedding)
      require(stream != null, s"no ${SeedVr2Pipeline.TextEmbedding} resource")
      try Files.copy(stream, file, StandardCopyOption.REPLACE_EXISTING)
      finally stream.close()
      val source = WeightSource.open(ops, file)
      try {
        val stored = source("positive")
        val Seq(length, width) = stored.shape.dimensions
        val embedding = ops.allocate(DType.F32, Shape.of(length, width))
        try {
          ops.convert(stored, embedding)
          val out =
            ops.allocate(DType.F32, Shape.of(length, transformer.config.hidden))
          transformer.encodeText(embedding, out)
          out
        } finally ops.release(embedding)
      } finally source.close()
    } finally Files.deleteIfExists(file)
  }

  /** The noise of `frames` frames of `height × width`, as the reference draws
    * it for `seed`: its latent is a permuted view, so torch fills it value by
    * value in the memory's order, `[16, latent frames, H/8, W/8]`; here in
    * `[latent frames, H/8, W/8, 16]`'s.
    */
  def noise(seed: Long, frames: Int, height: Int, width: Int): Array[Float] = {
    val (latentFrames, pixels) =
      (1 + (frames - 1) / 4, height / 8 * (width / 8))
    val stream =
      new TorchRandom(seed).normalOneByOne(channels * latentFrames * pixels)
    Array.tabulate(stream.length) { i =>
      val (place, channel) = (i / channels, i % channels)
      stream(channel * latentFrames * pixels + place)
    }
  }

  private def joined(frames: Seq[Array[Float]]): Array[Float] = {
    val size = frames.head.length
    require(
      frames.size.toLong * size <= Int.MaxValue,
      s"${frames.size} frames of $size values: too many for one batch"
    )
    val all = new Array[Float](frames.size * size)
    frames.zipWithIndex.foreach((frame, t) =>
      System.arraycopy(frame, 0, all, t * size, size)
    )
    all
  }

  /** The VAE's latents of `frames`, `[latent frames, H/8, W/8, 16]`. */
  private def encoded(
      frames: Seq[Array[Float]],
      height: Int,
      width: Int,
      options: SeedVr2Options,
      tick: () => Unit
  ): Array[Float] =
    VaeTiles
      .mapped(
        joined(frames),
        frames.size,
        height,
        width,
        3,
        options.tile,
        options.tileOverlap,
        multiple,
        1,
        8,
        channels,
        tick
      ) { (tile, tileHeight, tileWidth) =>
        val size = tileHeight * tileWidth * 3
        val pixels = frames.indices.map(t =>
          ops.fromFloats(
            Shape.of(tileHeight, tileWidth, 3),
            tile.slice(t * size, (t + 1) * size)
          )
        )
        try {
          val latents = autoencoder.encode(pixels)
          try (joined(latents.map(ops.toFloats)), latents.size)
          finally latents.foreach(ops.release)
        } finally pixels.foreach(ops.release)
      }
      ._1

  /** The frames of the VAE's `latents` (`[count, height, width, 16]`). */
  private def decoded(
      latents: Array[Float],
      count: Int,
      height: Int,
      width: Int,
      options: SeedVr2Options,
      tick: () => Unit
  ): Seq[Array[Float]] = {
    val (pixels, frames) = VaeTiles.mapped(
      latents,
      count,
      height,
      width,
      channels,
      options.tile / 8,
      options.tileOverlap / 8,
      patch,
      8,
      1,
      3,
      tick
    ) { (tile, tileHeight, tileWidth) =>
      val size = tileHeight * tileWidth * channels
      val inputs = (0 until count).map(t =>
        ops.fromFloats(
          Shape.of(tileHeight, tileWidth, channels.toLong),
          tile.slice(t * size, (t + 1) * size)
        )
      )
      try {
        val images = Seq.newBuilder[Array[Float]]
        autoencoder.decode(inputs, image => images += ops.toFloats(image))
        val made = images.result()
        (joined(made), made.size)
      } finally inputs.foreach(ops.release)
    }
    val size = pixels.length / frames
    (0 until frames).map(t => pixels.slice(t * size, (t + 1) * size))
  }

  /** One batch: `frames` (`[height, width, 3]` in [−1, 1] on the host, sides
    * multiples of `multiple`, `1 + 4n` of them) restored from `noise` (as
    * `noise` draws it), unclamped; `stage` is told what starts (`encode`,
    * `restore`, `decode`).
    */
  def restoreBatch(
      frames: Seq[Array[Float]],
      height: Int,
      width: Int,
      noise: Array[Float],
      options: SeedVr2Options = SeedVr2Options.Default,
      stage: String => Unit = _ => (),
      tick: () => Unit = () => ()
  ): Seq[Array[Float]] = {
    require(
      height % multiple == 0 && width % multiple == 0 &&
        frames.size % 4 == 1 &&
        frames.forall(_.length == height * width * 3),
      s"restore: ${frames.size} frames of $height × $width (1 + 4n frames, sides multiples of $multiple)"
    )
    val (latentHeight, latentWidth) = (height / 8, width / 8)
    val pixels = latentHeight.toLong * latentWidth
    val count = pixels.toInt * channels
    val (gridHeight, gridWidth) = (latentHeight / patch, latentWidth / patch)
    val perFrame = gridHeight.toLong * gridWidth
    stage("encode")
    val latents = encoded(frames, height, width, options, tick)
    val latentFrames = latents.length / count
    require(
      noise.length == latents.length,
      s"restore: ${noise.length} noise values for $latentFrames latent frames of $latentHeight × $latentWidth"
    )
    stage("restore")
    val held = mutable.ArrayBuffer.empty[Tensor]
    def hold(tensor: Tensor) = { held += tensor; tensor }
    val restored =
      try {
        val inputs = channels * 2 + 1
        val tokens = hold(
          ops.allocate(
            DType.F32,
            Shape.of(latentFrames * perFrame, inputs.toLong * patch * patch)
          )
        )
        val velocity = hold(
          ops.allocate(
            DType.F32,
            Shape.of(latentFrames * perFrame, channels.toLong * patch * patch)
          )
        )
        val mask = hold(
          ops.fromFloats(Shape.of(pixels, 1), Array.fill(pixels.toInt)(1f))
        )
        val joinedFrame = hold(
          ops.allocate(DType.F32, Shape.of(latentHeight, latentWidth, inputs))
        )
        val latent = hold(
          ops.allocate(
            DType.F32,
            Shape.of(latentHeight, latentWidth, channels.toLong)
          )
        )
        val flat = latent.view(pixels, channels.toLong)
        def frameOf(values: Array[Float], t: Int) =
          ops.fromFloats(
            Shape.of(pixels, channels.toLong),
            values.slice(t * count, (t + 1) * count)
          )
        (0 until latentFrames).foreach { t =>
          val (noised, own) = (frameOf(noise, t), frameOf(latents, t))
          try {
            ops.scale(own, SeedVr2Vae.LatentScale, own)
            ops.concatColumns(
              Seq(noised, own, mask),
              joinedFrame.view(pixels, inputs.toLong)
            )
            ops.packPatches(
              joinedFrame,
              patch,
              tokens.rows(t * perFrame, perFrame)
            )
          } finally Seq(noised, own).foreach(ops.release)
        }
        transformer.velocity(
          tokens,
          text,
          1000f,
          latentFrames,
          gridHeight,
          gridWidth,
          velocity
        )
        // one Euler step from pure noise: the restored latent is noise − velocity
        joined((0 until latentFrames).map { t =>
          ops.unpackPatches(
            velocity.rows(t * perFrame, perFrame),
            gridHeight,
            patch,
            latent
          )
          val noised = frameOf(noise, t)
          try {
            ops.scale(flat, -1f, flat)
            ops.add(noised, flat, flat)
            ops.scale(flat, 1 / SeedVr2Vae.LatentScale, flat)
            ops.toFloats(flat)
          } finally ops.release(noised)
        })
      } finally held.foreach(ops.release)
    tick()
    stage("decode")
    decoded(restored, latentFrames, latentHeight, latentWidth, options, tick)
  }

  /** How many batches `count` frames make. */
  def batches(count: Int, options: SeedVr2Options): Int =
    if (count <= options.batch) 1
    else {
      val step = options.batch - options.overlap
      (count - options.overlap + step - 1) / step
    }

  /** Restores `count` `frames` (as `restoreBatch` takes them), batch by batch,
    * every batch from `seed`'s noise; a last batch short of `1 + 4n` frames is
    * filled with its last frame. The restored frames go to `emit` in order;
    * `progress(batch, batches, stage)` as each batch's stages start.
    */
  def restore(
      frames: Iterator[Array[Float]],
      count: Int,
      height: Int,
      width: Int,
      seed: Long,
      emit: Array[Float] => Unit,
      options: SeedVr2Options = SeedVr2Options.Default,
      progress: (Int, Int, String) => Unit = (_, _, _) => (),
      tick: () => Unit = () => ()
  ): Unit = {
    val total = batches(count, options)
    var carried = Seq.empty[Array[Float]] // the next batch's first frames
    var tail = Seq.empty[Array[Float]] // their results, awaiting the blend
    var batch = 0
    while (frames.hasNext) {
      val input = carried ++ frames.take(options.batch - carried.size).toSeq
      val padded =
        input ++ Seq.fill((4 - (input.size - 1) % 4) % 4)(input.last)
      val out = restoreBatch(
        padded,
        height,
        width,
        noise(seed, padded.size, height, width),
        options,
        progress(batch, total, _),
        tick
      ).take(input.size)
      val weights = SeedVr2Pipeline.fade(tail.size)
      val blended = tail.zip(out).zip(weights).map { case ((before, now), w) =>
        Array.tabulate(now.length)(i => before(i) * w + now(i) * (1 - w))
      }
      val results = blended ++ out.drop(tail.size)
      val keep = if (frames.hasNext) options.overlap else 0
      results.dropRight(keep).foreach(emit)
      tail = results.takeRight(keep)
      carried = input.takeRight(keep)
      batch += 1
    }
  }

  /** `frames` (a picture, or a video's frames) upscaled to `width × height`
    * (even sides): each resized to that size (bicubic) on a black canvas of the
    * next multiples of `multiple`, as the reference pads, restored, and cut
    * back. `progress(done, of)` counts, for every batch, its tiles encoded, the
    * transformer's step and its tiles decoded.
    */
  def upscale(
      frames: Seq[BufferedImage],
      width: Int,
      height: Int,
      seed: Long,
      options: SeedVr2Options = SeedVr2Options.Default,
      progress: (Int, Int) => Unit = (_, _) => ()
  ): Seq[BufferedImage] = {
    require(
      width % 2 == 0 && height % 2 == 0 && width > 0 && height > 0,
      s"$width × $height: even sides"
    )
    def padded(side: Int) = (side + multiple - 1) / multiple * multiple
    val (fullWidth, fullHeight) = (padded(width), padded(height))
    def prepared(frame: BufferedImage): Array[Float] = {
      val canvas =
        new BufferedImage(fullWidth, fullHeight, BufferedImage.TYPE_INT_RGB)
      val graphics = canvas.createGraphics()
      try graphics.drawImage(Images.resized(frame, width, height), 0, 0, null)
      finally graphics.dispose()
      Images.pixels(canvas)
    }
    // a batch's steps: its tiles encoded, the transformer, its tiles decoded
    val tiles = VaeTiles.count(
      fullHeight,
      fullWidth,
      options.tile,
      options.tileOverlap,
      multiple
    )
    val steps = batches(frames.size, options) * (2 * tiles + 1)
    var done = 0
    progress(0, steps)
    val out = Seq.newBuilder[BufferedImage]
    restore(
      frames.iterator.map(prepared),
      frames.size,
      fullHeight,
      fullWidth,
      seed,
      image =>
        out += Images
          .toImage(image, fullWidth, fullHeight)
          .getSubimage(0, 0, width, height),
      options,
      tick = () => {
        done += 1
        progress(done, steps)
      }
    )
    out.result()
  }

  /** The model, as messages name it. */
  def family: String =
    if (transformer.config.swiglu) "SeedVR2 3B" else "SeedVR2 7B"

  def close(): Unit = {
    ops.release(text)
    autoencoder.close()
    transformer.close()
  }
}

object SeedVr2Pipeline {

  /** Whether `diffusionModel` is a SeedVR2 transformer. */
  def holds(ops: Ops, diffusionModel: Path): Boolean = {
    val source = WeightSource.open(ops, diffusionModel)
    try
      source.has("vid_in.proj.weight") && source.has("emb_in.proj_in.weight") &&
        source.has("txt_in.weight")
    finally source.close()
  }

  /** The size asked when none is: four times the source's. */
  val DefaultScale = 4

  /** The model's fixed positive text embedding (`pos_emb.pt` of the reference,
    * `[58, 5120]`), written by `fixtures/tiny_seedvr2.py picture`.
    */
  val TextEmbedding = "/seedvr2/text.safetensors"

  /** The earlier batch's weights over `overlap` shared frames (the reference's
    * `blend_overlapping_frames`): a Hann fall over the middle third from 3
    * frames on, a line from 1 to 0 below.
    */
  def fade(overlap: Int): Seq[Float] =
    if (overlap >= 3)
      (0 until overlap).map { i =>
        val u = math.min(
          1.0,
          math.max(0.0, (i.toDouble / (overlap - 1) - 1.0 / 3) * 3)
        )
        (0.5 + 0.5 * math.cos(math.Pi * u)).toFloat
      }
    else if (overlap == 2) Seq(1f, 0f)
    else Seq.fill(overlap)(1f)
}
