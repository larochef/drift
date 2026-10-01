package drift.runner.models

import drift.runner.diffusion.*
import drift.runner.formats.{FormatException, Safetensors}
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable
import scala.util.matching.Regex

/** MiniMax H3's transformer, read off its weights: `hidden` wide, `blocks`
  * deep, `heads` of `headDimension` (wider together than `hidden`), a SwiGLU of
  * `intermediate`, video rows of `videoWidth` (2 × 2 patches of the latents),
  * audio rows of `audioWidth`, text of `textWidth`, and RoPE over
  * `ropeFrequencies` per axis. The timestep reaches the modulations either
  * through the time MLP (`curveGrid` empty) or, in the pruned files, through a
  * table of `curveGrid` points of `timeWidth` (sd-cpp's "AdaLN curves").
  */
final case class MiniMaxH3Config(
    hidden: Int,
    blocks: Int,
    refinerBlocks: Int,
    heads: Int,
    headDimension: Int,
    intermediate: Int,
    videoWidth: Int,
    audioWidth: Int,
    textWidth: Int,
    ropeFrequencies: Int,
    timeWidth: Int,
    curveGrid: Option[Int]
) {

  /** The rotated pairs of each head: `ropeFrequencies` per axis. */
  def ropePairs: Int = 3 * ropeFrequencies
}

object MiniMaxH3Config {
  val Epsilon = 1e-5f

  /** The timesteps conditions hold at, at least: the video's released
    * augmentation (`t = 0.999`), the audio's clean.
    */
  val VideoConditionTime = 0.999f
  val AudioConditionTime = 1f

  /** The modalities the modulation tables hold a row for, in their order. */
  val Video = 0
  val Text = 1
  val Audio = 2
  val Modalities = 3

  /** diffusers' names of the linears outside the blocks → the original ones. */
  private val DiffusersWholes = Map(
    "proj_in" -> "video_patch_proj",
    "audio_proj_in" -> "audio_patch_proj",
    "context_embedder" -> "condition_proj",
    "time_embedder.linear_1" -> "time_embedder.proj_in",
    "time_embedder.linear_2" -> "time_embedder.proj_out",
    "norm_out.linear" -> "final_layer.adaln_proj.linear",
    "proj_out" -> "final_layer.video_out",
    "audio_proj_out" -> "final_layer.audio_out"
  )

  /** musubi-tuner's (kohya's) flattened names → the original ones, matched
    * whole: H3's own names hold underscores (diffusers' `flattened_modules`).
    */
  private val Kohya: Seq[(Regex, String)] = Seq(
    """lora_unet_blocks_(\d+)_attn_(qkv|out)_proj""".r -> "blocks.$1.attn.$2_proj",
    """lora_unet_blocks_(\d+)_mlp_fc([12])""".r -> "blocks.$1.mlp.fc$2",
    """lora_unet_blocks_(\d+)_adaln_proj_linear""".r ->
      "blocks.$1.adaln_proj.linear",
    """lora_unet_token_refiner_blocks_(\d+)_attn_(qkv|out)_proj""".r ->
      "token_refiner.blocks.$1.attn.$2_proj",
    """lora_unet_token_refiner_blocks_(\d+)_mlp_fc([12])""".r ->
      "token_refiner.blocks.$1.mlp.fc$2",
    """lora_unet_(video|audio)_patch_proj""".r -> "$1_patch_proj",
    """lora_unet_condition_proj""".r -> "condition_proj",
    """lora_unet_time_embedder_proj_(in|out)""".r -> "time_embedder.proj_$1",
    """lora_unet_final_layer_adaln_proj_linear""".r ->
      "final_layer.adaln_proj.linear",
    """lora_unet_final_layer_(video|audio)_out""".r -> "final_layer.$1_out"
  )

  private val DiffusersBlock =
    """(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)""".r
  private val OriginalBlock =
    """(blocks|token_refiner\.blocks)\.(\d+)\.(.+)""".r

  /** The linears a LoRA target updates, by the runner's names for them (a fused
    * weight's parts: `attn.q`, `attn.k`, `attn.v`, `mlp.gate`, `mlp.value`),
    * taking the rows of its `up` in turn. It reads the original names
    * (ai-toolkit's under `diffusion_model.`, ComfyUI's), musubi-tuner's
    * flattened `lora_unet_` ones and diffusers' (whose fused MLP input is
    * [value; gate]). A target naming no linear of H3 gives nothing.
    */
  def placement(target: String): Seq[String] =
    Kohya.collectFirst {
      case (pattern, replacement) if pattern.matches(target) =>
        pattern.replaceAllIn(target, replacement)
    } match {
      case Some(original) => placement(original)
      case None           =>
        target match {
          case DiffusersBlock(stack, n, part) =>
            val block =
              if (stack == "transformer_blocks") s"blocks.$n"
              else s"token_refiner.blocks.$n"
            part match {
              case "attn.to_q"     => Seq(s"$block.attn.q")
              case "attn.to_k"     => Seq(s"$block.attn.k")
              case "attn.to_v"     => Seq(s"$block.attn.v")
              case "attn.to_out.0" => Seq(s"$block.attn.out_proj")
              case "ff.net.0.proj" =>
                Seq(s"$block.mlp.value", s"$block.mlp.gate")
              case "ff.net.2"          => Seq(s"$block.mlp.fc2")
              case "adaln_proj.linear" => Seq(s"$block.adaln_proj.linear")
              case _                   => Nil
            }
          case OriginalBlock(stack, n, part) =>
            val block = s"$stack.$n"
            part match {
              case "attn.qkv_proj" =>
                Seq("q", "k", "v").map(p => s"$block.attn.$p")
              case "mlp.fc1" => Seq(s"$block.mlp.gate", s"$block.mlp.value")
              case "attn.out_proj" | "mlp.fc2" | "adaln_proj.linear" =>
                Seq(s"$block.$part")
              case _ => Nil
            }
          case other =>
            DiffusersWholes
              .get(other)
              .orElse(Some(other).filter(DiffusersWholes.values.toSet))
              .toSeq
        }
    }

  def holds(source: WeightSource): Boolean =
    source.has("video_patch_proj.weight") && source.has(
      "audio_patch_proj.weight"
    )

  def of(source: WeightSource): MiniMaxH3Config = {
    def dimensions(name: String) = source.shape(name).dimensions.map(_.toInt)
    def count(prefix: String) =
      Iterator
        .from(0)
        .takeWhile(i => source.has(s"$prefix.$i.norm1.weight"))
        .size
    val Seq(hidden, videoWidth) = dimensions("video_patch_proj.weight")
    val headDimension = dimensions("blocks.0.attn.q_norm.weight").head
    val curve = Option.when(source.has("adaln_t_table"))(
      dimensions("adaln_t_table")
    )
    MiniMaxH3Config(
      hidden = hidden,
      blocks = count("blocks"),
      refinerBlocks = count("token_refiner.blocks"),
      heads =
        dimensions("blocks.0.attn.qkv_proj.weight").head / (3 * headDimension),
      headDimension = headDimension,
      intermediate = dimensions("blocks.0.mlp.fc1.weight").head / 2,
      videoWidth = videoWidth,
      audioWidth = dimensions("audio_patch_proj.weight").last,
      textWidth = dimensions("condition_proj.weight").last,
      ropeFrequencies = dimensions("rope.inv_freq").head,
      timeWidth = curve.fold(dimensions("time_embedder.proj_out.weight").head)(
        _.last
      ),
      curveGrid = curve.map(_.head)
    )
  }
}

/** Which of MiniMax H3's two independently trained transformers a checkpoint
  * is: `Keyframes` (`t2va` and `fl2va`: text, a first and a last frame, the
  * guides) or `References` (`ref2va`: reference images, videos and sounds). The
  * two share every name and shape (ComfyUI and diffusers leave the choice to
  * the workflow), so the file's name says which it is ("ref2va", "fl2va",
  * "t2va"), else the released weights' fingerprint: the first two values of
  * `final_layer.norm.weight` (BF16 in every published quantization); else
  * keyframes.
  */
enum MiniMaxH3Partition {
  case Keyframes, References
}

object MiniMaxH3Partition {

  private val Fingerprints = Seq(
    (-0.000690460205078125f, 0.01263427734375f) -> Keyframes,
    (-0.01470947265625f, 0.001922607421875f) -> References
  )

  def of(path: Path, source: WeightSource): MiniMaxH3Partition =
    named(path).orElse(fingerprinted(source)).getOrElse {
      println(
        s"[WARN] ${path.getFileName}: neither its name nor its weights say whether it is MiniMax H3's " +
          "keyframes (fl2va) or references (ref2va) transformer; taken as keyframes"
      )
      Keyframes
    }

  /** What the file's name says. */
  def named(path: Path): Option[MiniMaxH3Partition] = {
    val name = path.getFileName.toString.toLowerCase
    if (name.contains("ref2va")) Some(References)
    else if (name.contains("fl2va") || name.contains("t2va")) Some(Keyframes)
    else None
  }

  /** What the released weights' fingerprint says. */
  def fingerprinted(source: WeightSource): Option[MiniMaxH3Partition] = {
    val weight = source("final_layer.norm.weight")
    val values = source.bytes("final_layer.norm.weight")
    val Array(first, second) =
      weight.dtype.decode(values.asSlice(0, weight.dtype.byteSize(2L)), 2)
    Fingerprints.collectFirst {
      case ((a, b), partition)
          if math.abs(first - a) < 1e-6f && math.abs(second - b) < 1e-6f =>
        partition
    }
  }
}

/** A clean block of rows the generated ones attend to, held at pixel frame
  * `frameIndex` of the video: a keyframe's or a guide's `videoFrames` latent
  * frames (on the video's grid) and `audioLatents` stereo audio latents (either
  * may be none). Its rows sit at the conditioning level: the video's at `t =
  * max(t, 0.999)`, the audio's at 1.
  */
final case class MiniMaxH3Condition(
    frameIndex: Int,
    videoFrames: Int,
    audioLatents: Int
)

/** A condition's or a reference's rows as the transformer reads them (its
  * latents, normalized and noised to the conditioning level): `video`
  * `[videoFrames × rowsPerFrame, videoWidth]` patchified as the generated rows,
  * `audio` `[2 × audioLatents, audioWidth]` channel-major.
  */
final case class MiniMaxH3ConditionRows(
    video: Option[Tensor],
    audio: Option[Tensor]
)

/** A control as the ControlNet reads it: `rows` `[videoRows, controlWidth]`,
  * its latents patchified as the generated rows, added at `strength`.
  */
final case class MiniMaxH3ControlRows(rows: Tensor, strength: Float)

/** A reference block of a `ref2va` request (diffusers'
  * `build_ref2va_packed_sequence`, ComfyUI's `PackedLayout` refs), on its own
  * latent grid, held at the conditioning level as conditions are. The blocks
  * share one rotary clock, which each pushes on by the time it spans; the
  * generated rows start where the last one leaves it.
  */
enum MiniMaxH3Reference {

  /** An image: one latent frame, one slot of the clock. */
  case Image(latentHeight: Int, latentWidth: Int)

  /** A clip: `latentFrames` latent frames and its soundtrack's `soundLatents`
    * (0 without one), the audio rows first, both from the same time, the audio
    * pinned to the clip's own width axis; the longer of the two spans.
    */
  case Clip(
      latentFrames: Int,
      latentHeight: Int,
      latentWidth: Int,
      soundLatents: Int
  )

  /** A sound alone, pinned to the generated video's width axis. */
  case Sound(soundLatents: Int)

  def videoFrames: Int = this match {
    case Image(_, _)           => 1
    case Clip(frames, _, _, _) => frames
    case Sound(_)              => 0
  }

  def audioLatents: Int = this match {
    case Clip(_, _, _, latents) => latents
    case Sound(latents)         => latents
    case Image(_, _)            => 0
  }

  /** Its latent grid (`(height, width)`), none for a sound. */
  def grid: Option[(Int, Int)] = this match {
    case Image(height, width)      => Some((height, width))
    case Clip(_, height, width, _) => Some((height, width))
    case Sound(_)                  => None
  }

  def videoRows: Int =
    grid.fold(0)((height, width) => videoFrames * (height / 2) * (width / 2))

  def audioRows: Int = MiniMaxH3Layout.AudioChannels * audioLatents

  /** The time it takes on the rotary clock. */
  def span: Double = this match {
    case Image(_, _)                 => 1.0
    case Sound(latents)              => latents.toDouble
    case Clip(frames, _, _, latents) =>
      math.max(latents.toDouble, MiniMaxH3Layout.videoSpan(frames))
  }
}

/** Where the rows of a request sit in MiniMax H3's packed sequence (diffusers'
  * `MiniMaxH3PrepareLayoutStep` and `MiniMaxH3Ref2VAPrepareLayoutStep`,
  * ComfyUI's `PackedLayout`): `[text | conditions | references | audio |
  * video]`, each condition its video rows then its audio rows, each reference
  * as `MiniMaxH3Reference` packs it; audio channel-major over its two stereo
  * channels, video frame-major in 2 × 2 patches. `visionRows` are the text rows
  * of vision blocks (a keyframe's or a reference's `<|vision_start|>`, pads and
  * `<|vision_end|>`, as `(start, count)`), which the transformer modulates as
  * video. `positions` gives each row's (t, h, w) rotary coordinates: text on
  * the time axis at its index; the references from there on, one after the
  * other (`MiniMaxH3Reference.span`); the generated media from where they end
  * (`origin`: a latent frame spans 5/3 × (1, 4, 4, 4, 4)), a condition from 5/3
  * × its frame index past it; the spatial axes centred and scaled to each
  * grid's aspect; audio pinned to its width axis' two ends.
  */
final case class MiniMaxH3Layout(
    textTokens: Int,
    frames: Int,
    latentHeight: Int,
    latentWidth: Int,
    audioLatents: Int,
    conditions: Seq[MiniMaxH3Condition] = Nil,
    visionRows: Seq[(Int, Int)] = Nil,
    references: Seq[MiniMaxH3Reference] = Nil
) {
  val audioRows: Int = MiniMaxH3Layout.AudioChannels * audioLatents
  val rowsPerFrame: Int = (latentHeight / 2) * (latentWidth / 2)
  val videoRows: Int = frames * rowsPerFrame

  /** Each condition's first video row and first audio row. */
  val conditionStarts: Seq[(Int, Int)] =
    conditions
      .scanLeft(textTokens) { (start, condition) =>
        start + condition.videoFrames * rowsPerFrame +
          MiniMaxH3Layout.AudioChannels * condition.audioLatents
      }
      .zip(conditions)
      .map((start, condition) =>
        (start, start + condition.videoFrames * rowsPerFrame)
      )
  private val referencesStart: Int =
    conditions.lastOption.fold(textTokens)(last =>
      conditionStarts.last._2 + MiniMaxH3Layout.AudioChannels * last.audioLatents
    )

  /** Each reference's first video row and first audio row (a clip's audio
    * before its video).
    */
  val referenceStarts: Seq[(Int, Int)] =
    references
      .scanLeft(referencesStart)((start, reference) =>
        start + reference.audioRows + reference.videoRows
      )
      .zip(references)
      .map((start, reference) => (start + reference.audioRows, start))
  val audioStart: Int = referencesStart + references
    .map(reference => reference.audioRows + reference.videoRows)
    .sum
  val videoStart: Int = audioStart + audioRows
  val rows: Int = videoStart + videoRows

  /** The rotary time the generated rows start at: past the text and every
    * reference, summed in order.
    */
  val origin: Double = references.foldLeft(textTokens.toDouble)(_ + _.span)

  /** `[rows × 3]`: (t, h, w) of each row. */
  def positions: Array[Double] = {
    val out = new Array[Double](3 * rows)
    def put(row: Int, t: Double, h: Double, w: Double): Unit = {
      out(3 * row) = t
      out(3 * row + 1) = h
      out(3 * row + 2) = w
    }
    (0 until textTokens).foreach(i => put(i, i, 0, 0))
    def axes(height: Int, width: Int) = {
      val sqrtArea = math.sqrt(height.toDouble * width)
      (
        MiniMaxH3Layout.spatialAxis(height, sqrtArea),
        MiniMaxH3Layout.spatialAxis(width, sqrtArea)
      )
    }
    val (heights, widths) = axes(latentHeight, latentWidth)
    def audio(
        start: Int,
        latents: Int,
        origin: Double,
        widths: Seq[Double]
    ): Unit =
      (0 until 2 * latents).foreach { i =>
        val (channel, latent) = (i / latents, i % latents)
        put(
          start + i,
          origin + latent,
          0,
          if (channel == 0) widths.head else widths.last
        )
      }
    def video(
        start: Int,
        count: Int,
        origin: Double,
        heights: Seq[Double],
        widths: Seq[Double]
    ): Unit = {
      val perFrame = heights.size * widths.size
      var time = origin
      (0 until count).foreach { frame =>
        for {
          (h, y) <- heights.zipWithIndex
          (w, x) <- widths.zipWithIndex
        } put(start + frame * perFrame + y * widths.length + x, time, h, w)
        time += MiniMaxH3Layout.FrameRescale * MiniMaxH3Layout.FrameSpans(
          frame % MiniMaxH3Layout.FrameSpans.length
        )
      }
    }
    conditions.zip(conditionStarts).foreach {
      case (condition, (videoAt, audioAt)) =>
        val at = origin + MiniMaxH3Layout.FrameRescale * condition.frameIndex
        video(videoAt, condition.videoFrames, at, heights, widths)
        audio(audioAt, condition.audioLatents, at, widths)
    }
    var clock = textTokens.toDouble
    references.zip(referenceStarts).foreach {
      case (reference, (videoAt, audioAt)) =>
        reference.grid match {
          case Some((height, width)) =>
            val (ownHeights, ownWidths) = axes(height, width)
            video(videoAt, reference.videoFrames, clock, ownHeights, ownWidths)
            audio(audioAt, reference.audioLatents, clock, ownWidths)
          case None =>
            audio(audioAt, reference.audioLatents, clock, widths)
        }
        clock += reference.span
    }
    audio(audioStart, audioLatents, origin, widths)
    video(videoStart, frames, origin, heights, widths)
    out
  }
}

object MiniMaxH3Layout {
  val AudioChannels = 2
  val FrameRescale: Double = 5.0 / 3.0
  val FrameSpans: Seq[Int] = Seq(1, 4, 4, 4, 4)

  /** One aspect-normalized spatial axis: `dim / 2` coordinates centred on the
    * unit interval, × 32 (numpy's `linspace(endpoint=False)`).
    */
  def spatialAxis(dim: Int, sqrtArea: Double): Seq[Double] = {
    val ratio = dim / sqrtArea
    val left = (1 - ratio) / 2
    val count = dim / 2
    (0 until count).map(i => (left + i * (ratio / count)) * 32)
  }

  /** The time `frames` latent frames span, summed one after the other (as
    * diffusers advances the references' clock).
    */
  def videoSpan(frames: Int): Double =
    (0 until frames).foldLeft(0.0)((sum, frame) =>
      sum + FrameRescale * FrameSpans(frame % FrameSpans.length)
    )
}

/** MiniMax H3's joint video and audio transformer (diffusers'
  * `MiniMaxH3Transformer3DModel`, sd-cpp's `minimax_h3.hpp`), from the original
  * names (the GGUFs', ComfyUI's).
  *   - The text: projected to the model's width and refined by plain blocks (no
  *     modulation, no RoPE). It depends on the prompt only: `encodeText` once
  *     per video.
  *   - One stack of blocks over the whole packed sequence (`MiniMaxH3Layout`),
  *     full attention with q/k RMS norms and a rotate-half RoPE over the heads'
  *     first `2 × ropePairs` values, a SwiGLU MLP (`fc1` holds [gate; value]).
  *     Every row is modulated by its (timestep, modality) row of each block's
  *     AdaLN table: the video and the text at the video's timestep, the audio
  *     at its own.
  *   - Timesteps are `t = 1 − σ` in [0, 1]; the velocity points towards the
  *     data (`x₀ = x + σ v`).
  *   - With `controlNet`, the Fun ControlNet union (`MiniMaxH3ControlNet`)
  *     beside the blocks, for the velocities given a control.
  * The quantized products run in BF16 (`Ops.wideProducts`), whose range the MLP
  * needs.
  */
final class MiniMaxH3 private (
    ops: Ops,
    source: WeightSource,
    path: Path,
    controlNet: Option[(WeightSource, Map[String, String])]
) extends AutoCloseable {

  val config: MiniMaxH3Config = MiniMaxH3Config.of(source)
  private val c = config

  /** The transformer this checkpoint is: keyframes or references. */
  val partition: MiniMaxH3Partition = MiniMaxH3Partition.of(path, source)
  final private case class Affine(weight: Tensor, bias: Tensor)

  /** A weight file's layers, as the blocks read them: the transformer's or its
    * ControlNet's.
    */
  final private class Layers(val source: WeightSource) {
    val weights = new HybridWeights(ops, source, gguf = true)
    def floats(name: String, count: Long): Tensor =
      weights.floats(name, name, Shape.of(count))
    def affine(prefix: String, outputs: Long): Affine =
      Affine(source(s"$prefix.weight"), floats(s"$prefix.bias", outputs))
  }

  private val own = new Layers(source)
  private val weights = own.weights
  private val epsilon = MiniMaxH3Config.Epsilon

  private def floats(name: String, count: Long): Tensor =
    own.floats(name, count)

  private def affineLayer(prefix: String, outputs: Long): Affine =
    own.affine(prefix, outputs)

  final private class Block(val prefix: String, layers: Layers) {
    private val inner = c.heads.toLong * c.headDimension
    private val source = layers.source
    val norm1: Tensor = layers.floats(s"$prefix.norm1.weight", c.hidden)
    val norm2: Tensor = layers.floats(s"$prefix.norm2.weight", c.hidden)
    private val qkv = source(s"$prefix.attn.qkv_proj.weight")
    val q: Tensor = qkv.rows(0, inner)
    val k: Tensor = qkv.rows(inner, inner)
    val v: Tensor = qkv.rows(2 * inner, inner)
    val qNorm: Tensor =
      layers.floats(s"$prefix.attn.q_norm.weight", c.headDimension)
    val kNorm: Tensor =
      layers.floats(s"$prefix.attn.k_norm.weight", c.headDimension)
    val o: Tensor = source(s"$prefix.attn.out_proj.weight")
    private val fc1 = source(s"$prefix.mlp.fc1.weight")
    val gate: Tensor = fc1.rows(0, c.intermediate)
    val value: Tensor = fc1.rows(c.intermediate, c.intermediate)
    val down: Tensor = source(s"$prefix.mlp.fc2.weight")

    /** Its linears by their LoRA sites. */
    def sites: Seq[(String, Tensor)] = Seq(
      "attn.q" -> q,
      "attn.k" -> k,
      "attn.v" -> v,
      "attn.out_proj" -> o,
      "mlp.gate" -> gate,
      "mlp.value" -> value,
      "mlp.fc2" -> down
    ).map((part, weight) => s"$prefix.$part" -> weight)
  }

  /** A modulation projection from the time embedding: `rows` of `outputs` per
    * timestep. The full files' through the GPU from the time MLP's output
    * (after a SiLU); the pruned files' (K = the table's width, too narrow for
    * the kernels) on the host from the table's point.
    */
  final private class Modulation(
      val prefix: String,
      outputs: Long,
      layers: Layers
  ) {
    val layer: Affine = layers.affine(prefix, outputs)
    private lazy val host: (Array[Float], Array[Float]) =
      (
        layers.weights.hostFloats(s"$prefix.weight"),
        layers.weights.hostFloats(s"$prefix.bias")
      )

    /** `[timesteps, outputs]` for `embedded` (`[timesteps, timeWidth]`). */
    def apply(embedded: TimeEmbedding, out: Tensor): Unit = embedded match {
      case TimeEmbedding.Device(values) =>
        linear(values, prefix, out)
        ops.addRow(out, layer.bias, out)
      case TimeEmbedding.Host(values) =>
        val (weight, bias) = host
        val width = c.timeWidth
        val result = Array.tabulate(values.length * outputs.toInt) { i =>
          val (row, column) = (i / outputs.toInt, i % outputs.toInt)
          var sum = bias(column).toDouble
          var j = 0
          while (j < width) {
            sum += weight(column * width + j).toDouble * values(row)(j)
            j += 1
          }
          sum.toFloat
        }
        // the LoRAs' updates, scale × up · (down · v)
        hostLoras.getOrElse(prefix, Nil).foreach { update =>
          values.indices.foreach { row =>
            val reduced = Array.tabulate(update.rank) { r =>
              var sum = 0.0
              var j = 0
              while (j < width) {
                sum += update.down(r * width + j).toDouble * values(row)(j)
                j += 1
              }
              sum
            }
            var column = 0
            while (column < outputs) {
              var sum = 0.0
              var r = 0
              while (r < update.rank) {
                sum += update.up(column * update.rank + r) * reduced(r)
                r += 1
              }
              val at = row * outputs.toInt + column
              result(at) = (result(at) + update.scale * sum).toFloat
              column += 1
            }
          }
        }
        val uploaded = ops.fromFloats(out.shape, result)
        try ops.copy(uploaded, out)
        finally ops.release(uploaded)
    }
  }

  private enum TimeEmbedding {
    case Device(values: Tensor)
    case Host(values: Seq[Array[Float]])
  }

  private val videoIn = affineLayer("video_patch_proj", c.hidden)
  private val audioIn = affineLayer("audio_patch_proj", c.hidden)
  private val textIn = affineLayer("condition_proj", c.hidden)
  private val refiner =
    (0 until c.refinerBlocks).map(i =>
      new Block(s"token_refiner.blocks.$i", own)
    )
  private val refinerNorm = floats("token_refiner.final_norm.weight", c.hidden)
  private val blocks = (0 until c.blocks).map(i =>
    (
      new Block(s"blocks.$i", own),
      new Modulation(
        s"blocks.$i.adaln_proj.linear",
        6L * c.hidden * MiniMaxH3Config.Modalities,
        own
      )
    )
  )
  private val finalNorm = floats("final_layer.norm.weight", c.hidden)
  private val finalModulation =
    new Modulation("final_layer.adaln_proj.linear", 2L * c.hidden, own)
  private val videoHead = affineLayer("final_layer.video_out", c.videoWidth)
  private val audioHead = affineLayer("final_layer.audio_out", c.audioWidth)
  private val timeIn = Option.when(c.curveGrid.isEmpty)(
    affineLayer("time_embedder.proj_in", c.hidden)
  )
  private val timeOut = Option.when(c.curveGrid.isEmpty)(
    affineLayer("time_embedder.proj_out", c.timeWidth)
  )
  private lazy val curveTable: Array[Float] =
    weights.hostFloats("adaln_t_table")
  private val inverseFrequencies: Array[Float] =
    weights.hostFloats("rope.inv_freq")

  // ---- the Fun ControlNet union --------------------------------------------------

  /** MiniMax H3's Fun ControlNet union (VideoX-Fun's, as ComfyUI's
    * `MiniMaxH3FunControl` and `MiniMaxH3FunControlPatch` run it): a block of
    * the transformer's shape beside each of its blocks at `places` (the file's
    * `control_blocks_places`, else every tenth from the first as ComfyUI
    * assumes), each modulated by its own projection of the same time embedding.
    * `projectIn` reads a control row (2 × 2 patches of the control's
    * `width / 4` latent channels), `before` joins it to the stream before the
    * first block, each block's `after` projects its output into the
    * transformer's stream.
    */
  final private class ControlNet(
      layers: Layers,
      metadata: Map[String, String]
  ) {
    private val source = layers.source
    private val count = Iterator
      .from(0)
      .takeWhile(i => source.has(s"control_blocks.$i.after_proj.weight"))
      .size
    Seq(
      "control_proj_in.weight",
      "control_blocks.0.adaln_proj.linear.weight",
      "control_blocks.0.before_proj.weight",
      "control_blocks.0.attn.qkv_proj.weight",
      "control_blocks.0.attn.q_norm.weight",
      "control_blocks.0.mlp.fc1.weight"
    ).filterNot(source.has).foreach { name =>
      throw new FormatException(
        s"the ControlNet is no MiniMax H3 Fun ControlNet union (no $name)"
      )
    }
    val places: Seq[Int] = metadata
      .get("control_blocks_places")
      .fold((0 until count).map(_ * 10))(
        ujson.read(_).arr.map(_.num.toInt).toSeq
      )
    require(
      places.size == count && places.headOption.contains(0) &&
        places == places.distinct.sorted && places.last < c.blocks,
      s"the ControlNet's blocks sit at $places: ${count} blocks from the first of the transformer's ${c.blocks}"
    )
    // the pruned files' AdaLN table needs the ControlNet's own (`adaln_basis`)
    private val timeColumns =
      source.shape("control_blocks.0.adaln_proj.linear.weight").last
    if (
      timeColumns != c.timeWidth || metadata
        .get("minimax_h3_fun_controlnet")
        .contains("adaln_basis") != c.curveGrid.isDefined
    )
      throw new FormatException(
        s"the ControlNet's AdaLN reads $timeColumns-wide time embeddings, the transformer's are ${c.timeWidth}: " +
          "the two use different AdaLN forms (the pruned files' curve table or the time MLP)"
      )
    val width: Int =
      source.shape("control_proj_in.weight").last.toInt
    // the kernels' products take rows in whole blocks of 32: the projection
    // padded with zero columns, the rows with zeros to match
    val paddedWidth: Int = (width + 31) / 32 * 32
    val projectIn: Affine = {
      val weight = layers.weights.hostFloats("control_proj_in.weight")
      Affine(
        layers.weights.keep(
          ops.fromFloats(
            Shape.of(c.hidden, paddedWidth),
            Array.tabulate(c.hidden * paddedWidth) { i =>
              val (row, column) = (i / paddedWidth, i % paddedWidth)
              if (column < width) weight(row * width + column) else 0f
            }
          )
        ),
        layers.floats("control_proj_in.bias", c.hidden)
      )
    }
    val before: Affine =
      layers.affine("control_blocks.0.before_proj", c.hidden)
    val blocks: Seq[(Block, Modulation, Affine)] = (0 until count).map { i =>
      val prefix = s"control_blocks.$i"
      (
        new Block(prefix, layers),
        new Modulation(
          s"$prefix.adaln_proj.linear",
          6L * c.hidden * MiniMaxH3Config.Modalities,
          layers
        ),
        layers.affine(s"$prefix.after_proj", c.hidden)
      )
    }

    /** Every linear by its site, for `linear`. */
    def sites: Seq[(String, Tensor)] =
      Seq(
        "control_proj_in" -> projectIn.weight,
        "control_blocks.0.before_proj" -> before.weight
      ) ++ blocks.zipWithIndex.flatMap { case ((block, modulation, after), i) =>
        block.sites ++ Seq(
          modulation.prefix -> modulation.layer.weight,
          s"control_blocks.$i.after_proj" -> after.weight
        )
      }

    def release(): Unit = {
      layers.weights.release()
      source.close()
    }
  }

  private val control =
    controlNet.map((source, metadata) =>
      new ControlNet(new Layers(source), metadata)
    )

  /** The control rows' width the loaded ControlNet reads (2 × 2 patches of
    * `width / 4` latent channels), none without one.
    */
  def controlWidth: Option[Int] = control.map(_.width)

  // ---- LoRAs ---------------------------------------------------------------------

  /** Every linear by its LoRA site (the original names, a fused weight's parts
    * each their own).
    */
  private val siteWeights: Map[String, Tensor] =
    (refiner.flatMap(_.sites) ++ blocks.flatMap(_._1.sites) ++
      (blocks.map(_._2) :+ finalModulation).map(m =>
        m.prefix -> m.layer.weight
      ) ++
      Seq(
        "video_patch_proj" -> videoIn.weight,
        "audio_patch_proj" -> audioIn.weight,
        "condition_proj" -> textIn.weight,
        "final_layer.video_out" -> videoHead.weight,
        "final_layer.audio_out" -> audioHead.weight
      ) ++ timeIn.map("time_embedder.proj_in" -> _.weight) ++
      timeOut.map("time_embedder.proj_out" -> _.weight) ++
      control.toSeq.flatMap(_.sites)).toMap

  private val siteShapes = MiniMaxH3.siteShapes(source)
  private val updates = new LoraUpdates(ops)

  /** A LoRA update of a modulation computed on the host (the pruned files'):
    * `down` `[rank, timeWidth]`, `up` `[outputs, rank]`, F32.
    */
  final private case class HostUpdate(
      down: Array[Float],
      up: Array[Float],
      rank: Int,
      scale: Float
  )
  private var hostLoras = Map.empty[String, Seq[HostUpdate]]

  private def hostFloats(tensor: Tensor): Array[Float] =
    if (tensor.dtype == DType.F32) ops.toFloats(tensor)
    else {
      val copy = ops.allocate(DType.F32, tensor.shape)
      try {
        ops.convert(tensor, copy)
        ops.toFloats(copy)
      } finally ops.release(copy)
    }

  /** Makes `loras` (each at its multiplier) the active set, replacing the last;
    * they read the names `MiniMaxH3Config.placement` does, a fused weight's
    * update split by the rows of its `up`. Returns the targets left unapplied:
    * those naming no weight of this file (a pruned file has no time MLP) or a
    * weight of another shape (a LoRA of the full files' modulations on a pruned
    * one). The prompt's text depends on the LoRAs (the refiner's): encode it
    * after.
    */
  def useLoras(loras: Seq[(Lora, Float)]): Seq[String] = {
    val placed = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map { (target, pair) =>
        (
          target,
          pair.copy(scale = pair.scale * multiplier),
          MiniMaxH3.sitesOf(target, pair, siteShapes)
        )
      }
    )
    val bySite = placed.flatMap { (_, pair, sites) =>
      val rows = sites.map(siteWeights(_).shape.dimensions.head)
      sites.zip(rows.scanLeft(0L)(_ + _)).zip(rows).map {
        case ((site, first), count) =>
          site -> pair.copy(up = pair.up.rows(first, count))
      }
    }
    val (onHost, onDevice) = bySite.partition((site, _) =>
      c.curveGrid.isDefined && site.endsWith("adaln_proj.linear")
    )
    updates.use(onDevice.groupMap(_._1)(_._2))
    hostLoras = onHost.groupMap(_._1)((_, pair) =>
      HostUpdate(
        hostFloats(pair.down),
        hostFloats(pair.up),
        pair.rank,
        pair.scale
      )
    )
    placed.collect { case (target, _, Nil) => target }.distinct
  }

  /** `out = x · weightᵀ` for the linear at `site`, and its LoRA updates. */
  private def linear(x: Tensor, site: String, out: Tensor): Unit = {
    ops.linear(x, siteWeights(site), out)
    updates(x, site, out)
  }

  /** `out = x · weightᵀ + bias` for the layer at `site`, and its LoRA updates.
    */
  private def affine(
      x: Tensor,
      site: String,
      layer: Affine,
      out: Tensor
  ): Unit = {
    linear(x, site, out)
    ops.addRow(out, layer.bias, out)
  }

  private val attention = Attention(
    (1 / math.sqrt(c.headDimension)).toFloat,
    causal = false,
    None,
    None,
    None
  )

  /** Flat buffers for `tokens` rows, grown when too small. */
  final private class Buffers(val tokens: Int) {
    private val held = mutable.ArrayBuffer.empty[Tensor]
    private def flat(dtype: DType, elements: Long): Tensor = {
      val tensor = ops.allocate(dtype, Shape.of(elements))
      held += tensor
      tensor
    }
    private val inner = c.heads.toLong * c.headDimension
    private val rows = tokens.toLong
    val x: Tensor = flat(DType.F32, rows * c.hidden)
    val normed: Tensor = flat(DType.F32, rows * math.max(c.hidden, c.textWidth))
    val q: Tensor = flat(DType.F32, rows * inner)
    val k: Tensor = flat(DType.F32, rows * inner)
    val v: Tensor = flat(DType.F32, rows * inner)
    val rotatedQueries: Tensor = flat(DType.F32, rows * inner)
    val rotatedKeys: Tensor = flat(DType.F32, rows * inner)
    val attended: Tensor = flat(DType.F32, rows * inner)
    val projected: Tensor = flat(DType.F32, rows * c.hidden)
    val gate: Tensor = flat(DType.F32, rows * c.intermediate)
    val value: Tensor = flat(DType.F32, rows * c.intermediate)
    val cosines: Tensor = flat(DType.F32, rows * c.ropePairs)
    val sines: Tensor = flat(DType.F32, rows * c.ropePairs)
    // the ControlNet's: the stream before the first block, its own, its output
    lazy val pristine: Tensor = flat(DType.F32, rows * c.hidden)
    lazy val controlStream: Tensor = flat(DType.F32, rows * c.hidden)
    lazy val controlOut: Tensor = flat(DType.F32, rows * c.hidden)
    val cache: KvCache =
      ops.allocateCache(1, (tokens + 15) / 16 * 16, c.heads, c.headDimension)
    val pageTable: Tensor = {
      val tensor = ops.fromInts(Shape.of(1), Array(0))
      held += tensor
      tensor
    }
    def release(): Unit = {
      held.foreach(ops.release)
      ops.release(cache.keys)
      ops.release(cache.values)
    }
  }

  private var buffers = Option.empty[Buffers]
  private def buffersFor(tokens: Int): Buffers =
    buffers.filter(_.tokens >= tokens).getOrElse {
      buffers.foreach(_.release())
      val created = new Buffers(tokens)
      buffers = Some(created)
      created
    }

  /** A modulated span of rows: `[start, start + count)` at its table row. */
  final private case class Span(start: Int, count: Int, row: Int)

  /** A block on `x` (`[rows, hidden]`, in place). With `modulation` (the
    * block's projection, `[timesteps × modalities, 6 × hidden]`) each span is
    * modulated and gated by its row; `rotate` turns the queries and keys.
    */
  private def block(
      b: Block,
      x: Tensor,
      modulation: Option[(Tensor, Seq[Span])],
      rotate: Boolean,
      w: Buffers
  ): Unit = {
    val rows = x.shape.dimensions.head
    val (heads, d) = (c.heads.toLong, c.headDimension.toLong)
    val f = c.hidden.toLong
    def chunk(row: Int, index: Int)(table: Tensor) =
      table
        .view(table.shape.elementCount / (6 * f), 6 * f)
        .rows(row, 1)
        .view(6, f)
        .rows(index, 1)
        .view(f)
    def modulate(normed: Tensor, shift: Int, scale: Int): Unit =
      modulation.foreach { (table, spans) =>
        spans.foreach { span =>
          val part = normed.rows(span.start, span.count)
          ops.modulate(
            part,
            chunk(span.row, scale)(table),
            chunk(span.row, shift)(table),
            part
          )
        }
      }
    def residual(y: Tensor, gate: Int): Unit = modulation match {
      case Some((table, spans)) =>
        spans.foreach(span =>
          ops.gatedAdd(
            x.rows(span.start, span.count),
            y.rows(span.start, span.count),
            chunk(span.row, gate)(table)
          )
        )
      case None => ops.add(x, y, x)
    }
    val normed = w.normed.prefix(rows, f)
    ops.rmsNorm(x, b.norm1, epsilon, 0f, normed)
    modulate(normed, 0, 1)
    val inner = heads * d
    val (q, k, v) =
      (
        w.q.prefix(rows, inner),
        w.k.prefix(rows, inner),
        w.v.prefix(rows, inner)
      )
    ops.linears(normed, Seq(b.q, b.k, b.v), Seq(q, k, v))
    Seq("q" -> q, "k" -> k, "v" -> v).foreach((part, y) =>
      updates(normed, s"${b.prefix}.attn.$part", y)
    )
    ops.rmsNorm(
      q.view(rows * heads, d),
      b.qNorm,
      epsilon,
      0f,
      q.view(rows * heads, d)
    )
    ops.rmsNorm(
      k.view(rows * heads, d),
      b.kNorm,
      epsilon,
      0f,
      k.view(rows * heads, d)
    )
    val (queries, keys) =
      if (rotate) {
        val (rq, rk) = (
          w.rotatedQueries.prefix(rows, heads, d),
          w.rotatedKeys.prefix(rows, heads, d)
        )
        val (cosines, sines) = (
          w.cosines.prefix(rows, c.ropePairs),
          w.sines.prefix(rows, c.ropePairs)
        )
        ops.ropeTable(q.view(rows, heads, d), cosines, sines, rq, halves = true)
        ops.ropeTable(k.view(rows, heads, d), cosines, sines, rk, halves = true)
        (rq, rk)
      } else (q.view(rows, heads, d), k.view(rows, heads, d))
    ops.cacheWrite(keys, v.view(rows, heads, d), w.cache, w.pageTable, 0)
    val attended = w.attended.prefix(rows, heads, d)
    ops.attention(
      queries,
      w.cache,
      w.pageTable,
      0,
      rows.toInt,
      attention,
      attended
    )
    val projected = w.projected.prefix(rows, f)
    linear(attended.view(rows, inner), s"${b.prefix}.attn.out_proj", projected)
    residual(projected, 2)
    ops.rmsNorm(x, b.norm2, epsilon, 0f, normed)
    modulate(normed, 3, 4)
    val (gate, value) = (
      w.gate.prefix(rows, c.intermediate),
      w.value.prefix(rows, c.intermediate)
    )
    ops.linears(normed, Seq(b.gate, b.value), Seq(gate, value))
    updates(normed, s"${b.prefix}.mlp.gate", gate)
    updates(normed, s"${b.prefix}.mlp.value", value)
    ops.gated(Activation.Silu, gate, value, gate)
    linear(gate, s"${b.prefix}.mlp.fc2", projected)
    residual(projected, 5)
  }

  /** The ControlNet's block `index` after the transformer's at its place
    * (ComfyUI's `MiniMaxH3FunControlPatch.after_block`): the first one starts
    * the ControlNet's stream from the one the transformer's first block read,
    * every video row replaced by the projection of its control row (none for
    * the conditions' and references' rows: the bias alone), through `before`
    * and added back; each block runs over that stream with the transformer's
    * spans and rotary angles and its own modulation, and its output, projected
    * by `after` and silent on every audio row, goes into the transformer's
    * stream at the control's strength.
    */
  private def controlStep(
      net: ControlNet,
      index: Int,
      input: MiniMaxH3ControlRows,
      layout: MiniMaxH3Layout,
      stream: Tensor,
      embedded: TimeEmbedding,
      table: Tensor,
      spans: Seq[Span],
      w: Buffers
  ): Unit = {
    val (tokens, f) = (layout.rows.toLong, c.hidden.toLong)
    val controlled = w.controlStream.prefix(tokens, f)
    val out = w.controlOut.prefix(tokens, f)
    if (index == 0) {
      val pristine = w.pristine.prefix(tokens, f)
      ops.copy(pristine, controlled)
      val held = layout.conditions.zip(layout.conditionStarts).map {
        case (condition, (videoAt, _)) =>
          (videoAt, condition.videoFrames * layout.rowsPerFrame)
      } ++ layout.references.zip(layout.referenceStarts).map {
        case (reference, (videoAt, _)) => (videoAt, reference.videoRows)
      }
      held.filter(_._2 > 0).foreach { (start, count) =>
        val part = controlled.rows(start, count)
        ops.scale(part, 0f, part)
        ops.addRow(part, net.projectIn.bias, part)
      }
      val rows = layout.videoRows.toLong
      val padded =
        if (net.paddedWidth == net.width) input.rows
        else {
          val zeros = ops.fromFloats(
            Shape.of(rows, net.paddedWidth - net.width),
            new Array[Float]((rows * (net.paddedWidth - net.width)).toInt)
          )
          val out = ops.allocate(DType.F32, Shape.of(rows, net.paddedWidth))
          try ops.concatColumns(Seq(input.rows, zeros), out)
          finally ops.release(zeros)
          out
        }
      try
        affine(
          padded,
          "control_proj_in",
          net.projectIn,
          controlled.rows(layout.videoStart, layout.videoRows)
        )
      finally if (padded ne input.rows) ops.release(padded)
      affine(controlled, "control_blocks.0.before_proj", net.before, out)
      ops.add(out, pristine, controlled)
    }
    val (b, modulation, after) = net.blocks(index)
    modulation(embedded, table)
    block(b, controlled, Some(table -> spans), rotate = true, w)
    affine(controlled, s"control_blocks.$index.after_proj", after, out)
    val audio = layout.conditions.zip(layout.conditionStarts).map {
      case (condition, (_, audioAt)) =>
        (audioAt, MiniMaxH3Layout.AudioChannels * condition.audioLatents)
    } ++ layout.references.zip(layout.referenceStarts).map {
      case (reference, (_, audioAt)) => (audioAt, reference.audioRows)
    } :+ (layout.audioStart, layout.audioRows)
    audio.filter(_._2 > 0).foreach { (start, count) =>
      val part = out.rows(start, count)
      ops.scale(part, 0f, part)
    }
    ops.scale(out, input.strength, out)
    ops.add(stream, out, stream)
  }

  /** The prompt's text for every step: the encoder's `[L, textWidth]` projected
    * and refined into `out` (`[L, hidden]`).
    */
  def encodeText(text: Tensor, out: Tensor): Unit = {
    val tokens = text.shape.dimensions.head
    require(
      text.shape == Shape.of(tokens, c.textWidth) &&
        out.shape == Shape.of(tokens, c.hidden),
      s"encodeText: text ${text.shape}, out ${out.shape}"
    )
    val w = buffersFor(tokens.toInt)
    affine(text, "condition_proj", textIn, out)
    refiner.foreach(block(_, out, None, rotate = false, w))
    val normed = w.normed.prefix(tokens, c.hidden)
    ops.rmsNorm(out, refinerNorm, epsilon, 0f, normed)
    ops.copy(normed, out)
  }

  /** The time embeddings of `timesteps` (each in [0, 1]). */
  private def embed(timesteps: Seq[Float], small: mutable.Buffer[Tensor]) =
    c.curveGrid match {
      case Some(grid) =>
        val width = c.timeWidth
        TimeEmbedding.Host(timesteps.map { t =>
          val position = math.min(math.max(t, 0f), 1f) * (grid - 1)
          val index = math.min(math.floor(position).toInt, grid - 2)
          val fraction = position - index
          Array.tabulate(width) { j =>
            val lower = curveTable(index * width + j)
            val upper = curveTable((index + 1) * width + j)
            lower + (upper - lower) * fraction
          }
        })
      case None =>
        // cosines first, over frequencies exp(−ln(10⁴) i / half), t unscaled
        val half = timeIn.get.weight.shape.last.toInt / 2
        val sinusoid = timesteps.flatMap { t =>
          (0 until 2 * half).map { i =>
            val frequency = math.exp(-math.log(10000) * (i % half) / half)
            val angle = t * frequency
            (if (i < half) math.cos(angle) else math.sin(angle)).toFloat
          }
        }.toArray
        val rows = timesteps.size.toLong
        def keep(tensor: Tensor) = { small += tensor; tensor }
        val embedded = keep(ops.fromFloats(Shape.of(rows, 2L * half), sinusoid))
        val inner = keep(
          ops.allocate(DType.F32, Shape.of(rows, c.hidden.toLong))
        )
        val time = keep(
          ops.allocate(DType.F32, Shape.of(rows, c.timeWidth.toLong))
        )
        affine(embedded, "time_embedder.proj_in", timeIn.get, inner)
        ops.activation(Activation.Silu, inner, inner)
        affine(inner, "time_embedder.proj_out", timeOut.get, time)
        // every modulation reads it through a SiLU
        ops.activation(Activation.Silu, time, time)
        TimeEmbedding.Device(time)
    }

  /** The velocities of a sequence laid out as `layout`: `videoRows`
    * (`[videoRows, videoWidth]`, the patchified latents) at `videoTime`,
    * `audioRows` (`[audioRows, audioWidth]`, channel-major) at `audioTime`,
    * given the prompt's `text` (`[L, hidden]` from `encodeText`), the rows of
    * `layout.conditions` and of `layout.references` (in order) and a `control`
    * for the ControlNet, into `videoOut` and `audioOut` (like their inputs).
    * Text rows run at the video's timestep, as text or, the vision blocks', as
    * video; conditions and references at their level.
    */
  def velocity(
      layout: MiniMaxH3Layout,
      text: Tensor,
      videoRows: Tensor,
      audioRows: Tensor,
      videoTime: Float,
      audioTime: Float,
      videoOut: Tensor,
      audioOut: Tensor,
      conditions: Seq[MiniMaxH3ConditionRows] = Nil,
      references: Seq[MiniMaxH3ConditionRows] = Nil,
      control: Option[MiniMaxH3ControlRows] = None
  ): Unit = {
    val (length, audio, video) =
      (layout.textTokens, layout.audioRows, layout.videoRows)
    require(
      text.shape == Shape.of(length, c.hidden) &&
        videoRows.shape == Shape.of(video, c.videoWidth) &&
        audioRows.shape == Shape.of(audio, c.audioWidth) &&
        videoOut.shape == videoRows.shape && audioOut.shape == audioRows.shape,
      s"velocity: text ${text.shape}, video ${videoRows.shape}, audio ${audioRows.shape} for $layout"
    )
    require(
      conditions.size == layout.conditions.size &&
        conditions.zip(layout.conditions).forall { (supplied, laid) =>
          supplied.video.map(_.shape) == Option.when(laid.videoFrames > 0)(
            Shape.of(laid.videoFrames * layout.rowsPerFrame, c.videoWidth)
          ) && supplied.audio
            .map(_.shape) == Option.when(laid.audioLatents > 0)(
            Shape.of(2 * laid.audioLatents, c.audioWidth)
          )
        },
      s"velocity: condition rows ${conditions
          .map(r => (r.video.map(_.shape), r.audio.map(_.shape)))} for $layout"
    )
    require(
      references.size == layout.references.size &&
        references.zip(layout.references).forall { (supplied, laid) =>
          supplied.video.map(_.shape) == Option.when(laid.videoRows > 0)(
            Shape.of(laid.videoRows, c.videoWidth)
          ) && supplied.audio.map(_.shape) == Option.when(laid.audioRows > 0)(
            Shape.of(laid.audioRows, c.audioWidth)
          )
        },
      s"velocity: reference rows ${references
          .map(r => (r.video.map(_.shape), r.audio.map(_.shape)))} for $layout"
    )
    val controlled = control.map { input =>
      val net = this.control.getOrElse(
        throw new IllegalArgumentException(
          "velocity: a control without a ControlNet"
        )
      )
      require(
        input.rows.shape == Shape.of(video, net.width),
        s"velocity: control rows ${input.rows.shape} for $video rows of ${net.width}"
      )
      (net, input)
    }
    val tokens = layout.rows
    val w = buffersFor(tokens)
    val f = c.hidden.toLong
    val small = mutable.ArrayBuffer.empty[Tensor]
    try {
      val conditionVideoTime =
        math.max(videoTime, MiniMaxH3Config.VideoConditionTime)
      val conditionAudioTime =
        math.max(audioTime, MiniMaxH3Config.AudioConditionTime)
      val timesteps = (Seq(videoTime, audioTime) ++
        Option.when(
          layout.conditions.exists(_.videoFrames > 0) ||
            layout.references.exists(_.videoRows > 0)
        )(conditionVideoTime) ++
        Option.when(
          layout.conditions.exists(_.audioLatents > 0) ||
            layout.references.exists(_.audioRows > 0)
        )(conditionAudioTime)).distinct
      val (videoRow, audioRow) =
        (timesteps.indexOf(videoTime), timesteps.indexOf(audioTime))
      def row(timestep: Int, modality: Int) =
        timestep * MiniMaxH3Config.Modalities + modality
      // text in runs: the vision blocks' rows as video
      val textSpans = {
        val vision = layout.visionRows.sorted
        val cuts = (0 +: vision.flatMap((start, count) =>
          Seq(start, start + count)
        ) :+ length).distinct
        cuts
          .sliding(2)
          .collect {
            case Seq(from, until) if until > from =>
              val isVision = vision
                .exists((start, count) => from >= start && from < start + count)
              Span(
                from,
                until - from,
                row(
                  videoRow,
                  if (isVision) MiniMaxH3Config.Video else MiniMaxH3Config.Text
                )
              )
          }
          .toSeq
      }
      val conditionSpans =
        layout.conditions.zip(layout.conditionStarts).flatMap {
          case (condition, (videoAt, audioAt)) =>
            Seq(
              Span(
                videoAt,
                condition.videoFrames * layout.rowsPerFrame,
                row(
                  timesteps.indexOf(conditionVideoTime),
                  MiniMaxH3Config.Video
                )
              ),
              Span(
                audioAt,
                2 * condition.audioLatents,
                row(
                  timesteps.indexOf(conditionAudioTime),
                  MiniMaxH3Config.Audio
                )
              )
            ).filter(_.count > 0)
        }
      val referenceSpans =
        layout.references.zip(layout.referenceStarts).flatMap {
          case (reference, (videoAt, audioAt)) =>
            Seq(
              Span(
                audioAt,
                reference.audioRows,
                row(
                  timesteps.indexOf(conditionAudioTime),
                  MiniMaxH3Config.Audio
                )
              ),
              Span(
                videoAt,
                reference.videoRows,
                row(
                  timesteps.indexOf(conditionVideoTime),
                  MiniMaxH3Config.Video
                )
              )
            ).filter(_.count > 0)
        }
      val spans = textSpans ++ conditionSpans ++ referenceSpans ++ Seq(
        Span(layout.audioStart, audio, row(audioRow, MiniMaxH3Config.Audio)),
        Span(layout.videoStart, video, row(videoRow, MiniMaxH3Config.Video))
      )
      val embedded = embed(timesteps, small)
      // the packed sequence
      val stream = w.x.prefix(tokens, f)
      ops.copy(text, stream.rows(0, length))
      (conditions.zip(layout.conditionStarts) ++
        references.zip(layout.referenceStarts)).foreach {
        case (supplied, (videoAt, audioAt)) =>
          supplied.video.foreach(rows =>
            affine(
              rows,
              "video_patch_proj",
              videoIn,
              stream.rows(videoAt, rows.shape.dimensions.head)
            )
          )
          supplied.audio.foreach(rows =>
            affine(
              rows,
              "audio_patch_proj",
              audioIn,
              stream.rows(audioAt, rows.shape.dimensions.head)
            )
          )
      }
      affine(
        audioRows,
        "audio_patch_proj",
        audioIn,
        stream.rows(layout.audioStart, audio)
      )
      affine(
        videoRows,
        "video_patch_proj",
        videoIn,
        stream.rows(layout.videoStart, video)
      )
      // the rotary angles: (t, h, w) × each axis' frequencies
      val positions = layout.positions
      val pairs = c.ropePairs
      val angles = Array.tabulate(tokens * pairs) { i =>
        val (token, pair) = (i / pairs, i % pairs)
        val (axis, frequency) =
          (pair / c.ropeFrequencies, pair % c.ropeFrequencies)
        positions(3 * token + axis) * inverseFrequencies(frequency).toDouble
      }
      val cosines = ops.fromFloats(
        Shape.of(tokens, pairs),
        angles.map(a => math.cos(a).toFloat)
      )
      small += cosines
      val sines = ops.fromFloats(
        Shape.of(tokens, pairs),
        angles.map(a => math.sin(a).toFloat)
      )
      small += sines
      ops.copy(cosines, w.cosines.prefix(tokens, pairs))
      ops.copy(sines, w.sines.prefix(tokens, pairs))
      val table = ops.allocate(
        DType.F32,
        Shape.of(timesteps.size.toLong, 6 * f * MiniMaxH3Config.Modalities)
      )
      small += table
      blocks.zipWithIndex.foreach { case ((b, modulation), i) =>
        // ComfyUI's patch keeps the stream the first block reads
        if (i == 0 && controlled.isDefined)
          ops.copy(stream, w.pristine.prefix(tokens, f))
        modulation(embedded, table)
        block(b, stream, Some(table -> spans), rotate = true, w)
        controlled.foreach { (net, input) =>
          val index = net.places.indexOf(i)
          if (index >= 0)
            controlStep(
              net,
              index,
              input,
              layout,
              stream,
              embedded,
              table,
              spans,
              w
            )
        }
      }
      // the final norm, shift and scale per timestep, then each head
      val last = ops.allocate(DType.F32, Shape.of(timesteps.size.toLong, 2 * f))
      small += last
      finalModulation(embedded, last)
      val normed = w.normed.prefix(tokens, f)
      ops.rmsNorm(stream, finalNorm, epsilon, 0f, normed)
      def head(
          start: Int,
          count: Int,
          timestep: Int,
          site: String,
          layer: Affine,
          out: Tensor
      ) = {
        val part = normed.rows(start, count)
        val row = last.rows(timestep, 1).view(2, f)
        ops.modulate(part, row.rows(1, 1).view(f), row.rows(0, 1).view(f), part)
        affine(part, site, layer, out)
      }
      head(
        layout.videoStart,
        video,
        videoRow,
        "final_layer.video_out",
        videoHead,
        videoOut
      )
      head(
        layout.audioStart,
        audio,
        audioRow,
        "final_layer.audio_out",
        audioHead,
        audioOut
      )
    } finally small.foreach(ops.release)
  }

  def close(): Unit = {
    updates.close()
    buffers.foreach(_.release())
    control.foreach(_.release())
    weights.release()
    source.close()
  }
}

object MiniMaxH3 {

  /** The `[rows, columns]` of every LoRA site of `source`'s transformer, from
    * its shapes alone (`MiniMaxH3Config.placement`'s names).
    */
  def siteShapes(source: WeightSource): Map[String, (Long, Long)] = {
    val c = MiniMaxH3Config.of(source)
    def shape(name: String) = {
      val Seq(rows, columns) = source.shape(s"$name.weight").dimensions
      (rows, columns)
    }
    val inner = c.heads.toLong * c.headDimension
    def block(prefix: String) = Seq(
      "attn.q" -> (inner, c.hidden.toLong),
      "attn.k" -> (inner, c.hidden.toLong),
      "attn.v" -> (inner, c.hidden.toLong),
      "attn.out_proj" -> shape(s"$prefix.attn.out_proj"),
      "mlp.gate" -> (c.intermediate.toLong, c.hidden.toLong),
      "mlp.value" -> (c.intermediate.toLong, c.hidden.toLong),
      "mlp.fc2" -> shape(s"$prefix.mlp.fc2")
    ).map((part, dimensions) => s"$prefix.$part" -> dimensions)
    val wholes = (0 until c.blocks).map(i => s"blocks.$i.adaln_proj.linear") ++
      Seq(
        "final_layer.adaln_proj.linear",
        "video_patch_proj",
        "audio_patch_proj",
        "condition_proj",
        "final_layer.video_out",
        "final_layer.audio_out"
      ) ++ Option
        .when(c.curveGrid.isEmpty)(
          Seq("time_embedder.proj_in", "time_embedder.proj_out")
        )
        .toSeq
        .flatten
    ((0 until c.refinerBlocks).flatMap(i =>
      block(s"token_refiner.blocks.$i")
    ) ++
      (0 until c.blocks).flatMap(i => block(s"blocks.$i")) ++
      wholes.map(name => name -> shape(name))).toMap
  }

  /** The sites `pair` (for `target`) updates, the rows of its `up` in turn;
    * empty when it names no weight of `shapes` or one of another shape.
    */
  def sitesOf(
      target: String,
      pair: LoraPair,
      shapes: Map[String, (Long, Long)]
  ): Seq[String] = {
    val sites = MiniMaxH3Config.placement(target)
    val fits = sites.nonEmpty && sites.forall(shapes.contains) &&
      sites.map(shapes(_)._1).sum == pair.up.shape.dimensions.head &&
      sites.forall(shapes(_)._2 == pair.down.shape.last)
    if (fits) sites else Nil
  }

  /** The transformer of `path`, and the Fun ControlNet union of `controlNet`
    * beside it when given.
    */
  def open(
      ops: Ops,
      path: Path,
      controlNet: Option[Path] = None
  ): MiniMaxH3 = {
    val source = WeightSource.open(ops, path)
    val control = controlNet.map(file => WeightSource.open(ops, file))
    try {
      if (!MiniMaxH3Config.holds(source))
        throw new FormatException(
          s"$path is no MiniMax H3 checkpoint (no video_patch_proj.weight)"
        )
      new MiniMaxH3(
        ops,
        source,
        path,
        control
          .zip(controlNet)
          .map((weights, file) => weights -> metadata(file))
      )
    } catch {
      case error: Throwable =>
        control.foreach(_.close())
        source.close()
        throw error
    }
  }

  /** A safetensors file's `__metadata__` (where the ControlNet records its
    * blocks' places and AdaLN form), none for a GGUF.
    */
  private def metadata(path: Path): Map[String, String] =
    if (path.getFileName.toString.endsWith(".gguf")) Map.empty
    else {
      val mapped = new MappedFile(path)
      try Safetensors.read(mapped.segment, path.toString).metadata
      finally mapped.close()
    }
}
