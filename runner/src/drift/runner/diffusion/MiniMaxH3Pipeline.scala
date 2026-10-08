package drift.runner.diffusion

import drift.runner.decode.{Prompt, PromptImage}
import drift.runner.models.*
import drift.runner.ops.Ops
import drift.runner.tensor.*
import drift.runner.text.{Tokenizer, TokenizerJson}
import drift.runner.vision.ImageSizing

import java.awt.image.BufferedImage
import java.nio.file.Path
import scala.collection.mutable

/** MiniMax H3, text to video, first/last keyframes and references (diffusers'
  * `t2va`, `fl2va` and `ref2va` modular pipelines) with ComfyUI's guides
  * (`MiniMaxH3AddGuide`) and Fun ControlNet union
  * (`MiniMaxH3FunControlNetApply`).
  *   - The presentation: the prompt tokenized bare, after, per keyframe, a
  *     `"<Picture i>: "` label and a vision block (Qwen3-VL's tower in the text
  *     encoder's file, deepstack included), or the references' labels and
  *     blocks (`MiniMaxH3References.pieces`), through Qwen3-VL-32B's language
  *     model to the residual stream after its 50th layer, projected and refined
  *     once by the transformer. A vision block's rows are modulated as video.
  *   - Keyframes (diffusers' recipe): the init image anchored at the first
  *     frame, the end image at the last (stretched onto the canvas, the end
  *     image cover-cropped when both are given), each encoded by the video VAE
  *     as a draw under seed 42 rounded to F16, then mixed with noise at `t =
  *     0.999`; held through every step at `max(t, 0.999)`.
  *   - Guides (ComfyUI's): an image, a clip (at 24 fps, its first `17k + 5`
  *     frames, else its first frame) and/or a sound (a clip's own, or alone)
  *     held from a frame of the video (negative from the end); the frames
  *     centre-cropped onto the canvas and encoded as the posterior's mean, then
  *     mixed with noise at 0.999 (each guide restarting a generator of the
  *     request's seed, as ComfyUI does); the sound resampled to 32 kHz as
  *     torchaudio does, encoded by the audio VAE, cut where the video's
  *     soundtrack ends, held clean. A clip at frame 0 extends that clip.
  *   - References (diffusers' recipe, the references transformer only): images
  *     down to a 2048 short edge, clips at 24 fps on the canvas of their own
  *     aspect, soundtracks in stereo at 32 kHz, cut to the video's length;
  *     images and clips encoded as keyframes are (a draw under seed 42, F16) on
  *     their own grids, soundtracks as the posterior's mean; packed in the
  *     request's order before the generated rows, each pushing on the rotary
  *     clock the generated rows start from (`MiniMaxH3Layout`); the visual ones
  *     mixed with noise at 0.999, the sounds clean. One pass a step: the
  *     checkpoint is guidance-distilled and diffusers runs no negative prompt.
  *   - The ControlNet (`--control-net`, ComfyUI's patch, whatever the
  *     conditioning): the control video at 24 fps, one frame a frame of the
  *     video (the last held), fitted onto the canvas and encoded (the
  *     posterior's mean); with a mask, its complement on the latents' grid and
  *     the source video outside it encoded too; its blocks run beside the
  *     transformer's over the steps from `start` to `end` (fractions of the
  *     shifted schedule, as ComfyUI's `percent_to_sigma`), in both CFG passes.
  *   - Noise from torch's CPU generator of the request's seed, in diffusers'
  *     order: each keyframe's or visual reference's, the video's as a latent
  *     tensor, the audio's rows; so a seed draws diffusers' noise.
  *   - The video (2 × 2 patches of 24 latent channels) and the audio (40
  *     latents a second on two stereo channels) denoised together, one forward
  *     per step (guidance-distilled), each down its own schedule (`linspace(1,
  *     0)` through a flow shift of 12 for the video, 3 for the audio); LoRAs on
  *     every linear at run time.
  *   - The video latents decoded by the ViT VAE in its released tiling; the
  *     audio latents, one stereo channel after the other, by the audio VAE
  *     (BigVGAN) into 32 kHz when it is given (else the video is silent).
  * A request's frames round up to the `17n + 5` the VAE decodes, its sides up
  * to multiples of 32; the model runs at 24 fps. The keyframes checkpoint
  * (`fl2va`) takes no references, the references one (`ref2va`) none of the
  * keyframes and guides.
  */
final class MiniMaxH3Pipeline(
    ops: Ops,
    diffusionModel: Path,
    vae: Path,
    textEncoder: Path,
    tokenizerFile: Path,
    audioVae: Option[Path],
    controlNet: Option[Path]
) extends VideoPipeline {

  def family: String = "MiniMax H3"
  def fps: Int = 24
  def sizeMultiple: Int = 32
  override def takesInitImage: Boolean = keyframes
  override def takesEndImage: Boolean = keyframes
  override def takesGuides: Boolean = keyframes
  override def takesReferences: Boolean = !keyframes
  override def takesControl: Boolean = transformer.controlWidth.isDefined
  override def takesLoras: Boolean = true
  override def makesSoundtrack: Boolean = true
  override def decodesSoundtrack: Boolean = audioDecoder.isDefined

  def alignedFrames(frames: Int): Int = {
    var aligned = math.max(frames, 5)
    while (aligned % 17 != 5) aligned += 1
    aligned
  }

  override def onCanvas(
      image: BufferedImage,
      width: Int,
      height: Int,
      follower: Boolean
  ): BufferedImage =
    if (follower) MiniMaxH3Conditions.coverCropped(image, width, height)
    else Images.resized(image, width, height)

  // the quantized products' activations pass F16's range
  ops.wideProducts = true
  private val tokenizer: Tokenizer = TokenizerJson.load(tokenizerFile)
  private val encoder = Qwen3.open(ops, textEncoder)
  private val transformer = MiniMaxH3.open(ops, diffusionModel, controlNet)
  private val decoder = MiniMaxH3Vae.open(ops, vae)
  private val audioDecoder = audioVae.map(MiniMaxH3Audio.open(ops, _))
  private val loraFiles = new LoraFiles(ops)

  // the encoders and the vision tower, opened by the first request needing them
  private var videoEncoderOpened = Option.empty[MiniMaxH3VideoEncoder]
  private var audioEncoderOpened = Option.empty[MiniMaxH3AudioEncoder]
  private var towerOpened = Option.empty[QwenVision]
  private def videoEncoder = videoEncoderOpened.getOrElse {
    val opened = MiniMaxH3VideoEncoder.open(ops, vae)
    videoEncoderOpened = Some(opened)
    opened
  }
  private def audioEncoder = audioEncoderOpened.getOrElse {
    val path = audioVae.getOrElse(
      throw new IllegalArgumentException(
        "a sound guide or reference needs MiniMax H3's audio VAE (--audio-vae)"
      )
    )
    val opened = MiniMaxH3AudioEncoder.open(ops, path)
    audioEncoderOpened = Some(opened)
    opened
  }
  private def tower = towerOpened.getOrElse {
    val opened = QwenVision.fromWeights(ops, textEncoder, "visual.")
    towerOpened = Some(opened)
    opened
  }

  private def keyframes =
    transformer.partition == MiniMaxH3Partition.Keyframes

  private val TextLayer = 50
  private val AudioShift = 3.0
  private val AudioLatentsPerSecond = 40
  private val LatentScale = 16

  /** diffusers' `keyframe_encode_seed`, and the conditioning level. */
  private val KeyframeSeed = 42L
  private val ConditionLevel = MiniMaxH3Config.VideoConditionTime

  /** The image sizing of the tower for keyframes and image references
    * (Qwen3-VL's image processor's bounds).
    */
  private def sizing =
    ImageSizing(tower.config.patch, tower.config.merge, 65536, 16777216)

  /** The presentation's text for every step, `[L, hidden]`: `pieces`, then the
    * prompt; and the vision blocks' rows (`(start, count)`).
    */
  private def text(
      prompt: String,
      pieces: Seq[MiniMaxH3Piece]
  ): (Tensor, Seq[(Int, Int)]) = {
    val (presentation, visionRows) =
      MiniMaxH3Pipeline.presentation(tokenizer, pieces, prompt)
    val encoded = MiniMaxH3Pipeline.encoded(
      ops,
      encoder,
      if (presentation.images.isEmpty) None else Some(tower),
      presentation,
      TextLayer
    )
    try {
      val out = ops.allocate(
        DType.F32,
        Shape.of(
          presentation.ids.length.toLong,
          transformer.config.hidden.toLong
        )
      )
      transformer.encodeText(encoded, out)
      (out, visionRows)
    } finally ops.release(encoded)
  }

  /** `linspace(1, 0, steps + 1)` through the flow shift `s`: `s σ / (1 + (s −
    * 1) σ)`.
    */
  private def sigmas(steps: Int, shift: Double): Array[Float] =
    Array.tabulate(steps + 1) { i =>
      val base = 1.0 - i.toDouble / steps
      (shift * base / (1 + (shift - 1) * base)).toFloat
    }

  /** A condition and its rows (the transformer's) on the host. */
  final private case class Held(
      condition: MiniMaxH3Condition,
      video: Option[Array[Float]],
      audio: Option[Array[Float]]
  )

  /** The guides as ComfyUI's `MiniMaxH3AddGuide` holds them, for a video of
    * `frameCount` frames on a `width` × `height` canvas with `audioLatents` a
    * channel.
    */
  private def guides(
      request: VideoRequest,
      width: Int,
      height: Int,
      frameCount: Int,
      audioLatents: Int
  ): Seq[Held] = request.guides.map { guide =>
    val (frames, sound) = guide.media match {
      case Media.Still(image)                 => (Seq(image), None)
      case Media.Clip(clip, rate, soundtrack) =>
        // at the model's rate, then its first 17k + 5 frames (or the first)
        val count = math.max(1, (clip.size * fps / rate).toInt)
        val timed = (0 until count).map(i =>
          clip(math.min(clip.size - 1, (i * rate / fps).toInt))
        )
        val kept = if (timed.size < 5) 1 else (timed.size - 5) / 17 * 17 + 5
        (timed.take(kept), soundtrack)
      case Media.Sound(soundtrack) => (Nil, Some(soundtrack))
    }
    val span = math.max(frames.size, 1)
    val frameIndex =
      if (guide.frameIndex >= 0) guide.frameIndex
      else frameCount + guide.frameIndex
    require(
      frameIndex >= 0 && frameIndex + span <= frameCount,
      if (span == 1)
        s"a guide at frame ${guide.frameIndex} is outside the video's $frameCount frames"
      else
        s"a $span-frame guide clip at frame ${guide.frameIndex} does not fit in the video's $frameCount frames"
    )
    val (h, w) = (height / LatentScale, width / LatentScale)
    val video = Option.when(frames.nonEmpty) {
      val latentChannels = videoEncoder.latentChannels
      val latents = videoEncoder.encode(
        frames.map(frame =>
          Images.pixels(MiniMaxH3Conditions.centerCropped(frame, width, height))
        ),
        height,
        width,
        Posterior.Mean
      )
      val latentFrames = latents.length / (h * w * latentChannels)
      val rows = MiniMaxH3Conditions.patchify(
        latents,
        latentFrames,
        h,
        w,
        latentChannels
      )
      // ComfyUI restarts the request's generator for every guide, in the rows' shape
      val noise = new TorchRandom(request.seed).normal(rows.length)
      (latentFrames, MiniMaxH3Conditions.mixed(rows, noise, ConditionLevel))
    }
    val audio = sound.map { track =>
      val room = math
        .floor(audioLatents - MiniMaxH3Layout.FrameRescale * frameIndex)
        .toInt
      require(
        room >= 1,
        s"a sound guide at frame ${guide.frameIndex} is past the end of the video's soundtrack"
      )
      // one encode a channel (mono twice), cut where the soundtrack ends
      val encoded = soundLatents(
        MiniMaxH3References.stereo(
          track,
          Double.PositiveInfinity,
          MiniMaxH3Audio.SampleRate
        )
      )
      val width = audioEncoder.channels
      val kept = math.min(room, encoded.head.length / width)
      (kept, encoded.flatMap(_.take(kept * width)).toArray)
    }
    Held(
      MiniMaxH3Condition(frameIndex, video.fold(0)(_._1), audio.fold(0)(_._1)),
      video.map(_._2),
      audio.map(_._2)
    )
  }

  /** Each channel's latents (`[latents, channels]`, normalized) of a stereo
    * soundtrack at the audio VAE's rate.
    */
  private def soundLatents(channels: Seq[Array[Float]]): Seq[Array[Float]] =
    channels.map(audioEncoder.encode)

  /** The references of `request` as the transformer reads them, for a
    * `frames`-frame video: normalized, presented, encoded (the visual ones as
    * keyframes are, the sounds as guides' are), laid out.
    */
  final private case class Encoded(
      pieces: Seq[MiniMaxH3Piece],
      layout: Seq[MiniMaxH3Reference],
      visual: Seq[MiniMaxH3Conditions.Latents],
      sounds: Seq[Option[Array[Float]]]
  )

  private def references(request: VideoRequest, frames: Int): Encoded = {
    MiniMaxH3References.validate(request.references)
    val normalized = request.references.map(
      MiniMaxH3References.normalized(_, frames, fps, MiniMaxH3Audio.SampleRate)
    )
    val pieces = MiniMaxH3References.pieces(normalized, sizing, fps)
    val visual = normalized
      .flatMap {
        case MiniMaxH3Normalized.Image(image)  => Seq(Seq(image))
        case MiniMaxH3Normalized.Clip(clip, _) =>
          Seq(clip.take(MiniMaxH3References.encodedFrames(clip.size)))
        case MiniMaxH3Normalized.Sound(_) => Nil
      }
      .map { clip =>
        val (width, height) = (clip.head.getWidth, clip.head.getHeight)
        val values = videoEncoder.encode(
          clip.map(Images.pixels(_)),
          height,
          width,
          Posterior.Sample(KeyframeSeed)
        )
        val (h, w) = (height / LatentScale, width / LatentScale)
        MiniMaxH3Conditions.Latents(
          values,
          values.length / (h * w * videoEncoder.latentChannels),
          h,
          w
        )
      }
    val sounds = normalized.map(
      _.soundtrack.map(channels => soundLatents(channels).flatten.toArray)
    )
    val latentsOf = visual.iterator
    val layout = normalized.zip(sounds).map { (reference, sound) =>
      val audioLatents = sound.fold(0)(
        _.length / (MiniMaxH3Layout.AudioChannels * audioEncoder.channels)
      )
      reference match {
        case MiniMaxH3Normalized.Image(_) =>
          val latents = latentsOf.next()
          MiniMaxH3Reference.Image(latents.height, latents.width)
        case MiniMaxH3Normalized.Clip(_, _) =>
          val latents = latentsOf.next()
          MiniMaxH3Reference.Clip(
            latents.frames,
            latents.height,
            latents.width,
            audioLatents
          )
        case MiniMaxH3Normalized.Sound(_) =>
          MiniMaxH3Reference.Sound(audioLatents)
      }
    }
    Encoded(pieces, layout, visual, sounds)
  }

  /** ComfyUI's `percent_to_sigma` on the video's schedule (flow shift `shift`):
    * 0 the first σ (1), 1 the last (0).
    */
  private def percentSigma(percent: Float, shift: Double): Double =
    if (percent <= 0) 1.0
    else if (percent >= 1) 0.0
    else {
      val base = 1.0 - percent
      shift * base / (1 + (shift - 1) * base)
    }

  def generate(request: VideoRequest, progress: (Int, Int) => Unit): Video = {
    def rounded(side: Int) =
      (side + sizeMultiple - 1) / sizeMultiple * sizeMultiple
    val (width, height) = (rounded(request.width), rounded(request.height))
    val frames = alignedFrames(request.frames)
    val (latentHeight, latentWidth) =
      (height / LatentScale, width / LatentScale)
    val latentFrames = (frames - 5) / 17 * 5 + 2
    val audioLatents =
      math.round(frames.toDouble / fps * AudioLatentsPerSecond).toInt
    val c = transformer.config
    val channels = c.videoWidth / 4
    val (loras, problems) =
      loraFiles.open(request.loras.map(lora => lora.path -> lora.multiplier))
    (problems ++ transformer
      .useLoras(loras)
      .map(target => s"a target of no weight of this file: $target"))
      .foreach(problem => println(s"[WARN] LoRA left unapplied: $problem"))
    val keyframeImages = request.initImage.map(_ -> 0).toSeq ++
      request.endImage.map(_ -> (frames - 1)).toSeq
    if (keyframes)
      require(
        request.references.isEmpty,
        "MiniMax H3's keyframes transformer (fl2va) takes no references: load the references one (ref2va)"
      )
    else
      require(
        keyframeImages.isEmpty && request.guides.isEmpty,
        "MiniMax H3's references transformer (ref2va) takes no keyframes or guides: load the keyframes one (fl2va)"
      )
    require(
      request.control.isEmpty || takesControl,
      "a control video needs MiniMax H3's Fun ControlNet union (--control-net)"
    )
    val referenced = Option.when(!keyframes)(references(request, frames))
    // the references transformer is guidance-distilled: diffusers runs one pass
    if (referenced.isDefined && request.cfgScale != 1f)
      println(
        s"[WARN] MiniMax H3's references transformer runs no negative prompt: CFG ${request.cfgScale} taken as 1"
      )
    val guided = request.cfgScale != 1f && referenced.isEmpty
    val held = mutable.ArrayBuffer.empty[Tensor]
    def keep(tensor: Tensor) = { held += tensor; tensor }
    try {
      val (conditional, visionRows) =
        text(
          request.prompt,
          referenced.fold(keyframeImages.zipWithIndex.flatMap {
            case ((image, _), i) =>
              Seq(
                MiniMaxH3Piece.Label(s"<Picture ${i + 1}>: "),
                MiniMaxH3Piece.Vision(
                  sizing.prepare(image, s"keyframe $i"),
                  MiniMaxH3References.ImagePad
                )
              )
          })(_.pieces)
        )
      keep(conditional)
      val unconditional =
        Option.when(guided)(keep(text(request.negativePrompt, Nil)._1))
      val keyframeLatents = keyframeImages.map { (image, _) =>
        MiniMaxH3Conditions.Latents(
          videoEncoder.encode(
            Seq(Images.pixels(image)),
            height,
            width,
            Posterior.Sample(KeyframeSeed)
          ),
          1,
          latentHeight,
          latentWidth
        )
      }
      val (drawnRows, videoNoise, audioNoise) = MiniMaxH3Conditions.draws(
        new TorchRandom(request.seed),
        keyframeLatents ++ referenced.toSeq.flatMap(_.visual),
        latentFrames,
        latentHeight,
        latentWidth,
        channels,
        MiniMaxH3Layout.AudioChannels * audioLatents,
        c.audioWidth,
        ConditionLevel
      )
      val (keyframeRows, referenceRows) =
        drawnRows.splitAt(keyframeLatents.size)
      val conditions =
        keyframeImages.zip(keyframeRows).map { case ((_, frameIndex), rows) =>
          Held(MiniMaxH3Condition(frameIndex, 1, 0), Some(rows), None)
        } ++ guides(request, width, height, frames, audioLatents)
      def uploaded(video: Option[Array[Float]], audio: Option[Array[Float]]) =
        MiniMaxH3ConditionRows(
          video.map(rows =>
            keep(
              ops.fromFloats(
                Shape.of(rows.length / c.videoWidth, c.videoWidth),
                rows
              )
            )
          ),
          audio.map(rows =>
            keep(
              ops.fromFloats(
                Shape.of(rows.length / c.audioWidth, c.audioWidth),
                rows
              )
            )
          )
        )
      val conditionRows =
        conditions.map(condition => uploaded(condition.video, condition.audio))
      val visualRows = referenceRows.iterator
      val referenceInputs = referenced.toSeq.flatMap(encoded =>
        encoded.layout.zip(encoded.sounds).map { (reference, sound) =>
          uploaded(
            Option.when(reference.videoRows > 0)(visualRows.next()),
            sound
          )
        }
      )
      val layout = MiniMaxH3Layout(
        textTokens = 0,
        frames = latentFrames,
        latentHeight = latentHeight,
        latentWidth = latentWidth,
        audioLatents = audioLatents,
        conditions = conditions.map(_.condition),
        references = referenced.fold(Nil)(_.layout)
      )
      val control = request.control
        .filter(requested =>
          requested.strength != 0 && (requested.frames.nonEmpty || requested.mask.isDefined)
        )
        .map { requested =>
          val controlWidth = transformer.controlWidth.get
          val latents = MiniMaxH3Conditions.controlLatents(
            requested,
            fps,
            width,
            height,
            latentFrames,
            latentHeight,
            latentWidth,
            channels,
            controlWidth / 4,
            clip => videoEncoder.encode(clip, height, width, Posterior.Mean)
          )
          (
            requested,
            MiniMaxH3ControlRows(
              keep(
                ops.fromFloats(
                  Shape.of(layout.videoRows, controlWidth),
                  MiniMaxH3Conditions.patchify(
                    latents,
                    latentFrames,
                    latentHeight,
                    latentWidth,
                    controlWidth / 4
                  )
                )
              ),
              requested.strength
            )
          )
        }
      def laid(prompt: Tensor, vision: Seq[(Int, Int)]) =
        layout.copy(
          textTokens = prompt.shape.dimensions.head.toInt,
          visionRows = vision
        )
      val video = keep(
        ops.fromFloats(Shape.of(layout.videoRows, c.videoWidth), videoNoise)
      )
      val audio = keep(
        ops.fromFloats(Shape.of(layout.audioRows, c.audioWidth), audioNoise)
      )
      val (videoVelocity, audioVelocity) = (
        keep(ops.allocate(DType.F32, video.shape)),
        keep(ops.allocate(DType.F32, audio.shape))
      )
      val others = unconditional.map(_ =>
        (
          keep(ops.allocate(DType.F32, video.shape)),
          keep(ops.allocate(DType.F32, audio.shape))
        )
      )
      val (videoSigmas, audioSigmas) =
        (
          sigmas(request.steps, request.shift),
          sigmas(request.steps, AudioShift)
        )
      progress(0, request.steps)
      (0 until request.steps).foreach { i =>
        val (videoTime, audioTime) = (1 - videoSigmas(i), 1 - audioSigmas(i))
        // ComfyUI's window: σ_end ≤ σ ≤ σ_start
        val controlled = control.collect {
          case (requested, rows)
              if percentSigma(requested.end, request.shift) <= videoSigmas(i) &&
                videoSigmas(i) <= percentSigma(
                  requested.start,
                  request.shift
                ) =>
            rows
        }
        transformer.velocity(
          laid(conditional, visionRows),
          conditional,
          video,
          audio,
          videoTime,
          audioTime,
          videoVelocity,
          audioVelocity,
          conditionRows,
          referenceInputs,
          controlled
        )
        for {
          uncond <- unconditional
          (v, a) <- others
        } {
          transformer.velocity(
            laid(uncond, Nil),
            uncond,
            video,
            audio,
            videoTime,
            audioTime,
            v,
            a,
            conditionRows,
            referenceInputs,
            controlled
          )
          // v = uncond + scale × (cond − uncond)
          Seq(v -> videoVelocity, a -> audioVelocity).foreach {
            (other, velocity) =>
              ops.scale(other, 1 - request.cfgScale, other)
              ops.scale(velocity, request.cfgScale, velocity)
              ops.add(velocity, other, velocity)
          }
        }
        // the velocity points towards the data: x += (σ − σ') v
        ops.scale(
          videoVelocity,
          videoSigmas(i) - videoSigmas(i + 1),
          videoVelocity
        )
        ops.add(video, videoVelocity, video)
        ops.scale(
          audioVelocity,
          audioSigmas(i) - audioSigmas(i + 1),
          audioVelocity
        )
        ops.add(audio, audioVelocity, audio)
        progress(i + 1, request.steps)
      }
      // rows → latents [T, h, w, 24]
      val latents = MiniMaxH3Conditions.unpatchify(
        ops.toFloats(video),
        latentFrames,
        latentHeight,
        latentWidth,
        channels
      )
      val started = System.nanoTime()
      val images = mutable.ArrayBuffer.empty[BufferedImage]
      Images.decoding(s"$latentFrames latent frames")
      decoder.decode(
        latents,
        latentFrames,
        latentHeight,
        latentWidth,
        rgb => images += Images.toImage(rgb.map(_ * 2 - 1), width, height)
      )
      println(
        f"decoded ${images.size} frames in ${(System.nanoTime() - started) / 1e9}%.1f s"
      )
      Video(
        images.toSeq.take(frames),
        fps,
        audioDecoder.map(soundtrack(audio, _))
      )
    } finally held.foreach(ops.release)
  }

  /** The stereo soundtrack of the audio rows (`[2L, width]`, channel-major):
    * each channel decoded alone, interleaved.
    */
  private def soundtrack(audio: Tensor, audioDecoder: MiniMaxH3Audio) = {
    val started = System.nanoTime()
    val rows = ops.toFloats(audio)
    val channels = MiniMaxH3Layout.AudioChannels
    val perChannel = rows.length / channels
    require(
      audio.shape.dimensions.last == audioDecoder.channels,
      s"audio rows of ${audio.shape.dimensions.last} for a decoder of ${audioDecoder.channels} channels"
    )
    val tracks = (0 until channels).map(c =>
      audioDecoder.decode(rows.slice(c * perChannel, (c + 1) * perChannel))
    )
    val samples = Array.tabulate(tracks.head.length * channels)(i =>
      tracks(i % channels)(i / channels)
    )
    println(
      f"decoded ${tracks.head.length.toDouble / audioDecoder.sampleRate}%.2f s of sound in ${(System.nanoTime() - started) / 1e9}%.1f s"
    )
    Soundtrack.fitted(samples, channels, audioDecoder.sampleRate)
  }

  def close(): Unit = {
    loraFiles.close()
    towerOpened.foreach(_.close())
    audioEncoderOpened.foreach(_.close())
    videoEncoderOpened.foreach(_.close())
    audioDecoder.foreach(_.close())
    decoder.close()
    transformer.close()
    encoder.close()
  }
}

object MiniMaxH3Pipeline {

  /** The presentation of `pieces` then `prompt`, each piece tokenized alone as
    * diffusers does (the prompt bare), its vision blocks placed; and the
    * blocks' rows (`(start, count)`: `<|vision_start|>`, the pads,
    * `<|vision_end|>`).
    */
  def presentation(
      tokenizer: Tokenizer,
      pieces: Seq[MiniMaxH3Piece],
      prompt: String
  ): (Prompt, Seq[(Int, Int)]) = {
    def id(token: String) = tokenizer
      .id(token)
      .getOrElse(throw new IllegalStateException(s"no $token in the tokenizer"))
    val ids = Array.newBuilder[Int]
    val placed = Vector.newBuilder[PromptImage]
    var length = 0
    def add(tokens: Seq[Int]): Unit = {
      ids ++= tokens
      length += tokens.size
    }
    pieces.foreach {
      case MiniMaxH3Piece.Label(text) =>
        add(tokenizer.encode(text, addSpecial = false).toSeq)
      case MiniMaxH3Piece.Vision(image, pad) =>
        add(Seq(id("<|vision_start|>")))
        placed += PromptImage(length, image)
        add(Seq.fill(image.tokens)(id(pad)))
        add(Seq(id("<|vision_end|>")))
    }
    add(tokenizer.encode(prompt, addSpecial = false).toSeq)
    require(length > 0, "MiniMax H3 needs a prompt")
    val images = placed.result()
    (
      Prompt(ids.result(), images),
      images.map(image => (image.at - 1, image.image.tokens + 2))
    )
  }

  /** `presentation` through the text encoder (its images through `tower`): the
    * residual stream after `layer`, `[L, hidden]`, which the caller releases.
    */
  def encoded(
      ops: Ops,
      encoder: DenseDecoder,
      tower: => Option[QwenVision],
      presentation: Prompt,
      layer: Int
  ): Tensor = {
    val seen = presentation.images.map(placed => tower.get.encode(placed.image))
    val placed = presentation.images.zip(seen)
    val out = ops.allocate(
      DType.F32,
      Shape.of(presentation.ids.length.toLong, encoder.config.hidden.toLong)
    )
    try {
      encoder.encode(
        presentation.ids,
        Seq(layer),
        out,
        images = Option.when(seen.nonEmpty)(
          SeenImages(
            presentation.positions,
            placed.map((p, v) => GivenRows(p.at, v.tokens)),
            seen.head.deepstack.indices.map(k =>
              placed.map((p, v) => GivenRows(p.at, v.deepstack(k)))
            )
          )
        )
      )
      out
    } catch {
      case error: Throwable =>
        ops.release(out)
        throw error
    } finally seen.foreach(v => (v.tokens +: v.deepstack).foreach(ops.release))
  }
}
