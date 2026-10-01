package drift.runner.diffusion

import drift.runner.vision.*

import java.awt.image.BufferedImage
import java.math.RoundingMode

/** A `ref2va` reference normalized onto MiniMax H3's rates and sizes
  * (diffusers' `MiniMaxH3Ref2VASetupStep`); a soundtrack as its two channels at
  * the audio VAE's rate.
  */
enum MiniMaxH3Normalized {
  case Image(image: BufferedImage)
  case Clip(frames: Seq[BufferedImage], track: Option[Seq[Array[Float]]])
  case Sound(channels: Seq[Array[Float]])

  def soundtrack: Option[Seq[Array[Float]]] = this match {
    case Clip(_, track)  => track
    case Sound(channels) => Some(channels)
    case Image(_)        => None
  }
}

/** A piece of MiniMax H3's presentation before the prompt: a label (tokenized
  * alone), or a vision block (`<|vision_start|>`, one `pad` a token of `image`,
  * `<|vision_end|>`).
  */
enum MiniMaxH3Piece {
  case Label(text: String)
  case Vision(image: PreparedImage, pad: String)
}

/** MiniMax H3's `ref2va` references (diffusers' `MiniMaxH3Ref2VASetupStep` and
  * `MiniMaxH3Ref2VATextEncoderStep`, ComfyUI's `MiniMaxH3ReferenceToVideo` as
  * the second reading): their limits, their normalization and their
  * presentation to the text encoder.
  */
object MiniMaxH3References {

  /** The released checkpoint's limits (diffusers'). */
  val MaximumImages = 9
  val MaximumClips = 3
  val MaximumSounds = 3
  val MaximumReferences = 12

  /** The short edge an image reference is encoded at, down to (ComfyUI's "max";
    * diffusers scales smaller images up to it as well).
    */
  val ImageShortEdge = 2048

  /** The canvas rule a clip reference is put on (the generated video's in
    * diffusers): a short edge of 768, at most 768 × 1344 pixels, sides
    * multiples of 32.
    */
  val CanvasShortEdge = 768
  val CanvasPixels: Int = 768 * 1344
  val Multiple = 32

  /** The rate the text encoder reads a clip at. */
  val ReadingRate = 2.0

  /** Qwen3-VL's video processor's pixel bounds over all of a clip's sampled
    * frames (its `video_preprocessor_config.json`, beside the image bounds the
    * keyframes use).
    */
  val VideoMinimumPixels = 4096
  val VideoMaximumPixels = 25165824

  val ImagePad = "<|image_pad|>"
  val VideoPad = "<|video_pad|>"

  /** Refuses references beyond the released limits, and sounds alone. */
  def validate(references: Seq[Media]): Unit = {
    val images = references.count(_.isInstanceOf[Media.Still])
    val clips = references.count(_.isInstanceOf[Media.Clip])
    val sounds = references.count(_.isInstanceOf[Media.Sound])
    require(
      references.nonEmpty,
      "MiniMax H3's references transformer (ref2va) needs references: load the keyframes one (fl2va) for text alone"
    )
    Seq(
      ("image", images, MaximumImages),
      ("video", clips, MaximumClips),
      ("audio", sounds, MaximumSounds)
    ).foreach { (kind, count, limit) =>
      require(
        count <= limit,
        s"MiniMax H3 takes at most $limit $kind references, not $count"
      )
    }
    require(
      references.size <= MaximumReferences,
      s"MiniMax H3 takes at most $MaximumReferences references, not ${references.size}"
    )
    require(
      sounds < references.size,
      "MiniMax H3 takes an audio reference beside an image or a video, not alone"
    )
  }

  /** `media` normalized for a video of `frames` frames at `fps`: an image down
    * to a short edge of 2048 (sides rounded to multiples of 32), a clip at
    * `fps` (diffusers' slots), cut to the video's frames, on the canvas its own
    * aspect resolves to; a soundtrack in stereo at the audio VAE's
    * `sampleRate`, cut to the video's length at its own rate first.
    */
  def normalized(
      media: Media,
      frames: Int,
      fps: Int,
      sampleRate: Int
  ): MiniMaxH3Normalized = {
    val seconds = frames.toDouble / fps
    media match {
      case Media.Still(image) =>
        val (width, height) = (image.getWidth, image.getHeight)
        require(
          width <= 4 * height && height <= 4 * width,
          s"MiniMax H3 takes reference images from 1:4 to 4:1, not $width × $height"
        )
        val scale =
          math.min(1.0, ImageShortEdge.toDouble / math.min(width, height))
        def rounded(side: Int) =
          math.max(
            Multiple,
            math.rint(side * scale / Multiple).toInt * Multiple
          )
        MiniMaxH3Normalized.Image(
          lanczos(image, rounded(width), rounded(height))
        )
      case Media.Clip(clip, rate, soundtrack) =>
        require(rate > 0, s"a reference video at $rate fps")
        val timed = MiniMaxH3Conditions.atRate(clip, rate, fps).take(frames)
        val (width, height) = MiniMaxH3Conditions.canvas(
          timed.head.getWidth,
          timed.head.getHeight,
          Multiple,
          CanvasShortEdge,
          CanvasPixels
        )
        MiniMaxH3Normalized.Clip(
          timed.map(lanczos(_, width, height)),
          soundtrack.map(stereo(_, seconds, sampleRate))
        )
      case Media.Sound(soundtrack) =>
        MiniMaxH3Normalized.Sound(stereo(soundtrack, seconds, sampleRate))
    }
  }

  /** `image` at `width` × `height` through PIL's LANCZOS, as diffusers and
    * ComfyUI resize references.
    */
  def lanczos(image: BufferedImage, width: Int, height: Int): BufferedImage =
    if (image.getWidth == width && image.getHeight == height) image
    else
      Images.toImage(
        Resampling
          .resampled(
            Resampling.rgbOverWhite(image),
            image.getWidth,
            image.getHeight,
            width,
            height,
            Resampling.Filter.Lanczos
          )
          .map(_ / 127.5f - 1),
        width,
        height
      )

  /** A soundtrack's first `seconds` at its own rate, as two channels (mono
    * twice) resampled to `sampleRate` (torchaudio's).
    */
  def stereo(
      soundtrack: Soundtrack,
      seconds: Double,
      sampleRate: Int
  ): Seq[Array[Float]] = {
    val samples = math.min(
      soundtrack.samples.length / soundtrack.channels,
      (seconds * soundtrack.rate).toInt
    )
    (0 until 2).map { c =>
      val channel = math.min(c, soundtrack.channels - 1)
      SoundResampling.resampled(
        Array.tabulate(samples)(i =>
          soundtrack.samples(i * soundtrack.channels + channel)
        ),
        soundtrack.rate,
        sampleRate
      )
    }
  }

  /** The frames of a clip the video VAE encodes: diffusers snaps its count down
    * to `17n + 5` (at least 22; a shorter clip whole, the VAE padding it).
    */
  def encodedFrames(frames: Int): Int =
    math.min(frames, math.max(1, Math.floorDiv(frames - 5, 17)) * 17 + 5)

  /** The frames of a `frames`-frame clip at `fps` the text encoder reads (one
    * every `fps / 2`), and the time of each vision block: the frames go in
    * pairs (the last repeated when odd), each block at its pair's mean.
    */
  def readFrames(frames: Int, fps: Int): (Seq[Int], Seq[Double]) = {
    val stride = fps / ReadingRate
    val indices = Iterator
      .iterate(0.0)(_ + stride)
      .map(cursor => math.rint(cursor).toInt)
      .takeWhile(_ < frames)
      .toSeq
      .distinct
    require(
      indices.size >= 2,
      s"a reference video is read at ${ReadingRate.toInt} fps in pairs of frames: it needs at least " +
        s"${math.rint(stride).toInt + 1} frames at $fps fps, not $frames"
    )
    val times = indices.indices.map(_ / ReadingRate)
    val padded = times ++ Seq.fill(times.size % 2)(times.last)
    (indices, padded.grouped(2).map(pair => (pair.head + pair.last) / 2).toSeq)
  }

  /** `"<t seconds>"` with Python's `{:.1f}` (half to even). */
  def timestamp(seconds: Double): String =
    s"<${java.math.BigDecimal(seconds).setScale(1, RoundingMode.HALF_EVEN).toPlainString} seconds>"

  /** The vision blocks of a normalized clip: its read frames in pairs, each a
    * two-frame image sized by Qwen3-VL's video `smart_resize` over all of them,
    * and their times.
    */
  def clipBlocks(
      frames: Seq[BufferedImage],
      fps: Int,
      patch: Int,
      merge: Int,
      key: String
  ): Seq[(PreparedImage, Double)] = {
    val (indices, times) = readFrames(frames.size, fps)
    val read = indices.map(frames)
    val (height, width) =
      videoSize(
        read.size,
        read.head.getHeight,
        read.head.getWidth,
        patch * merge
      )
    val sizing = ImageSizing(patch, merge, 1, Int.MaxValue)
    val pixels = read.map { frame =>
      Resampling
        .bicubic(
          Resampling.rgbOverWhite(frame),
          frame.getWidth,
          frame.getHeight,
          width,
          height
        )
        .map(_ / 127.5f - 1)
    }
    val paired = pixels ++ Seq.fill(pixels.size % 2)(pixels.last)
    paired
      .grouped(2)
      .zipWithIndex
      .map { (pair, block) =>
        PreparedImage(
          sizing.framePatches(pair, height, width),
          height / patch,
          width / patch,
          merge,
          s"$key block $block",
          frames = 2
        )
      }
      .toSeq
      .zip(times)
  }

  /** transformers' Qwen3-VL video `smart_resize` of `count` frames of `height`
    * × `width`: sides rounded to the factor, scaled to fit the pixel bounds
    * over all the frames (their count rounded to pairs).
    */
  def videoSize(
      count: Int,
      height: Int,
      width: Int,
      factor: Int
  ): (Int, Int) = {
    def rounded(side: Int) = math.rint(side.toDouble / factor).toInt * factor
    val (h, w) = (rounded(height), rounded(width))
    val pairs = math.rint(count / 2.0).toLong * 2
    val total = count.toDouble * height * width
    if (pairs * h * w > VideoMaximumPixels) {
      val beta = math.sqrt(total / VideoMaximumPixels)
      (
        math.max(factor, math.floor(height / beta / factor).toInt * factor),
        math.max(factor, math.floor(width / beta / factor).toInt * factor)
      )
    } else if (pairs * h * w < VideoMinimumPixels) {
      val beta = math.sqrt(VideoMinimumPixels / total)
      (
        math.ceil(height * beta / factor).toInt * factor,
        math.ceil(width * beta / factor).toInt * factor
      )
    } else (h, w)
  }

  /** The presentation's pieces of `references` (diffusers'
    * `_build_presentation`), numbered per kind in order: an image's
    * `"<Picture i>: "` and vision block (sized by `sizing`); an audio-bearing
    * reference's `"<Audio j>: "` first; a clip's `"<Video k>: "` then, per
    * block, its time and vision block.
    */
  def pieces(
      references: Seq[MiniMaxH3Normalized],
      sizing: ImageSizing,
      fps: Int
  ): Seq[MiniMaxH3Piece] = {
    var (images, clips, sounds) = (0, 0, 0)
    references.zipWithIndex.flatMap { (reference, index) =>
      val sound = reference.soundtrack.toSeq.map { _ =>
        sounds += 1
        MiniMaxH3Piece.Label(s"<Audio $sounds>: ")
      }
      sound ++ (reference match {
        case MiniMaxH3Normalized.Image(image) =>
          images += 1
          Seq(
            MiniMaxH3Piece.Label(s"<Picture $images>: "),
            MiniMaxH3Piece.Vision(
              sizing.prepare(image, s"reference $index"),
              ImagePad
            )
          )
        case MiniMaxH3Normalized.Clip(frames, _) =>
          clips += 1
          MiniMaxH3Piece.Label(s"<Video $clips>: ") +:
            clipBlocks(
              frames,
              fps,
              sizing.patch,
              sizing.merge,
              s"reference $index"
            ).flatMap((block, time) =>
              Seq(
                MiniMaxH3Piece.Label(timestamp(time)),
                MiniMaxH3Piece.Vision(block, VideoPad)
              )
            )
        case MiniMaxH3Normalized.Sound(_) => Nil
      })
    }
  }
}
