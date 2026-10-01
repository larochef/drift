package drift.runner.models

import drift.runner.diffusion.*
import drift.runner.formats.FormatException
import drift.runner.ops.*
import drift.runner.state.KvCache
import drift.runner.tensor.*

import java.nio.file.Path
import scala.collection.mutable

/** An LTX 2 transformer's sizes, read off its weights: the video stream
  * `hidden` wide in `heads`, the audio stream `audioHidden` in `audioHeads`,
  * the cross-modal attentions `crossHidden` in `audioHeads`, latent channels in
  * and out.
  */
final case class Ltx2Config(
    hidden: Int,
    heads: Int,
    audioHidden: Int,
    audioHeads: Int,
    crossHidden: Int,
    blocks: Int,
    videoChannels: Int,
    audioChannels: Int
) {
  def head: Int = hidden / heads
  def audioHead: Int = audioHidden / audioHeads
  def crossHead: Int = crossHidden / audioHeads
}

object Ltx2Config {
  def holds(source: WeightSource): Boolean =
    source.has("patchify_proj.weight") && source.has(
      "audio_patchify_proj.weight"
    ) &&
      source.has("transformer_blocks.0.audio_to_video_attn.to_q.weight")

  def of(source: WeightSource): Ltx2Config = {
    def dimensions(name: String) = source.shape(name).dimensions.map(_.toInt)
    val Seq(hidden, videoChannels) = dimensions("patchify_proj.weight")
    val Seq(audioHidden, audioChannels) = dimensions(
      "audio_patchify_proj.weight"
    )
    Ltx2Config(
      hidden = hidden,
      heads =
        dimensions("transformer_blocks.0.attn1.to_gate_logits.weight").head,
      audioHidden = audioHidden,
      audioHeads = dimensions(
        "transformer_blocks.0.audio_attn1.to_gate_logits.weight"
      ).head,
      crossHidden =
        dimensions("transformer_blocks.0.audio_to_video_attn.to_q.weight").head,
      blocks = Iterator
        .from(0)
        .takeWhile(i => source.has(s"transformer_blocks.$i.attn1.to_q.weight"))
        .size,
      videoChannels = videoChannels,
      audioChannels = audioChannels
    )
  }
}

/** Where an LTX 2 request's tokens sit in time and space, as the RoPE reads
  * them (diffusers' `LTX2AudioVideoRotaryPosEmbed`): each video latent the
  * middle of its pixel span (a latent frame 8 frames, the first causal, over
  * `fps`; 32 × 32 pixels), each audio latent the middle of its span in seconds
  * (4 mel frames of 160 samples at 16 kHz, the first causal). `appended` are
  * the positions of the keyframe tokens after the video's
  * (`Ltx2Layout.keyframe`).
  */
final case class Ltx2Layout(
    frames: Int,
    height: Int,
    width: Int,
    audioFrames: Int,
    fps: Double,
    appended: IndexedSeq[Array[Double]] = Vector.empty
) {

  /** The generated video's tokens, before the appended ones. */
  def generatedTokens: Int = frames * height * width

  def videoTokens: Int = generatedTokens + appended.size

  /** Per video token, its (t, y, x) middles in seconds and pixels. */
  def videoPositions: Array[Array[Double]] =
    Array.tabulate(generatedTokens) { i =>
      val (f, y, x) = (i / (height * width), i / width % height, i % width)
      def time(frame: Int) = math.max(0.0, frame * 8 + 1 - 8) / fps
      Array(
        (time(f) + time(f + 1)) / 2,
        (y * 32 + (y + 1) * 32) / 2.0,
        (x * 32 + (x + 1) * 32) / 2.0
      )
    } ++ appended

  /** Per audio token, its middle in seconds. */
  def audioPositions: Array[Double] =
    Array.tabulate(audioFrames) { f =>
      def time(frame: Int) = math.max(0.0, frame * 4 + 1 - 4) * 160 / 16000
      (time(f) + time(f + 1)) / 2
    }
}

object Ltx2Layout {

  /** The positions of a keyframe's tokens appended after the video's
    * (diffusers' `_prepare_keyframe_coords`, ComfyUI's `LTXVAddGuide`): its
    * `frames` latent frames of `height` × `width` from pixel frame `index`,
    * each spanning 8 frames, or the one frame `[index, index + 1)` when the
    * keyframe is a single pixel frame; in seconds over `fps`.
    */
  def keyframe(
      frames: Int,
      height: Int,
      width: Int,
      index: Int,
      single: Boolean,
      fps: Double
  ): IndexedSeq[Array[Double]] =
    (0 until frames * height * width).map { i =>
      val (f, y, x) = (i / (height * width), i / width % height, i % width)
      val (start, end) =
        if (single) (index.toDouble, index + 1.0)
        else (f * 8.0 + index, (f + 1) * 8.0 + index)
      Array(
        (start / fps + end / fps) / 2,
        (y * 32 + (y + 1) * 32) / 2.0,
        (x * 32 + (x + 1) * 32) / 2.0
      )
    }
}

/** LTX 2.5's joint audio and video transformer (diffusers'
  * `LTX2VideoTransformer3DModel`, the official single file's names): two
  * streams through each block,
  *   - self-attention in each (the split RoPE over their positions),
  *   - cross-attention to the prompt's connector rows, the query modulated and
  *     gated by the timestep, the rows themselves by the prompt modulation,
  *   - audio-to-video and video-to-audio attention (a temporal RoPE on both
  *     sides), each side modulated by the cross-attention tables,
  *   - a GELU (tanh) MLP;
  * every attention gated per head by 2σ(a projection of its input), q/k RMS
  * norms across the heads, the stream norms without weights. One timestep for
  * every token (σ × 1000), but the video tokens held by a condition (diffusers'
  * `conditioning_mask` at 1) take 0 in the video's own AdaLN (the blocks' and
  * the head's); the prompt's, the cross-modal ones and the audio's keep σ (the
  * cross timestep). The output is the flow velocity.
  *
  * LoRAs apply at run time on every linear (`useLoras`), the text connectors'
  * too when the file holds them (`connectors`).
  */
final class Ltx2 private (ops: Ops, source: WeightSource)
    extends AutoCloseable {

  val config: Ltx2Config = Ltx2Config.of(source)
  private val c = config
  private val weights = new HybridWeights(ops, source, gguf = true)
  private val Epsilon = 1e-6f
  private val updates = new LoraUpdates(ops)

  /** Every linear's weight by its site (its name in the file). */
  private val siteWeights = mutable.Map.empty[String, Tensor]

  private def floats(name: String, count: Long) =
    weights.floats(name, name, Shape.of(count))
  private def table(name: String, rows: Int, width: Int) =
    weights.floats(name, name, Shape.of(rows, width))

  /** A linear layer at `site` (its name in the file, a LoRA's target). */
  final private case class Affine(site: String, weight: Tensor, bias: Tensor)
  private def linearLayer(prefix: String, outputs: Long) = {
    val weight = source(s"$prefix.weight")
    siteWeights(prefix) = weight
    Affine(
      prefix,
      weight,
      if (source.has(s"$prefix.bias")) floats(s"$prefix.bias", outputs)
      else
        weights.keep(
          ops.fromFloats(Shape.of(outputs), new Array[Float](outputs.toInt))
        )
    )
  }
  private def affineLayer(prefix: String, outputs: Long) =
    linearLayer(prefix, outputs)

  /** An attention's projections: q from `inner`-wide queries, k and v, the
    * output back to `outputs`, the gate's logits per head.
    */
  final private class AttentionLayer(
      prefix: String,
      inner: Int,
      outputs: Int,
      heads: Int
  ) {
    val q: Affine = affineLayer(s"$prefix.to_q", inner)
    val k: Affine = affineLayer(s"$prefix.to_k", inner)
    val v: Affine = affineLayer(s"$prefix.to_v", inner)
    val o: Affine = affineLayer(s"$prefix.to_out.0", outputs)
    val gate: Affine = affineLayer(s"$prefix.to_gate_logits", heads)
    val qNorm: Tensor = floats(s"$prefix.q_norm.weight", inner)
    val kNorm: Tensor = floats(s"$prefix.k_norm.weight", inner)
  }

  /** An AdaLN-single: the timestep's sinusoid through its MLP (`emb`), and
    * `rows` vectors from it.
    */
  final private class AdaLn(prefix: String, width: Int, rows: Int) {
    val first: Affine =
      affineLayer(s"$prefix.emb.timestep_embedder.linear_1", width)
    val second: Affine =
      affineLayer(s"$prefix.emb.timestep_embedder.linear_2", width)
    val out: Affine = affineLayer(s"$prefix.linear", rows.toLong * width)
    val size: Int = rows * width
  }

  final private class Block(i: Int) {
    private val at = s"transformer_blocks.$i"
    val attention =
      new AttentionLayer(s"$at.attn1", c.hidden, c.hidden, c.heads)
    val audioAttention = new AttentionLayer(
      s"$at.audio_attn1",
      c.audioHidden,
      c.audioHidden,
      c.audioHeads
    )
    val text = new AttentionLayer(s"$at.attn2", c.hidden, c.hidden, c.heads)
    val audioText = new AttentionLayer(
      s"$at.audio_attn2",
      c.audioHidden,
      c.audioHidden,
      c.audioHeads
    )
    val audioToVideo = new AttentionLayer(
      s"$at.audio_to_video_attn",
      c.crossHidden,
      c.hidden,
      c.audioHeads
    )
    val videoToAudio = new AttentionLayer(
      s"$at.video_to_audio_attn",
      c.crossHidden,
      c.audioHidden,
      c.audioHeads
    )
    val up: Affine = linearLayer(s"$at.ff.net.0.proj", 4L * c.hidden)
    val down: Affine = linearLayer(s"$at.ff.net.2", c.hidden)
    val audioUp: Affine =
      linearLayer(s"$at.audio_ff.net.0.proj", 4L * c.audioHidden)
    val audioDown: Affine = linearLayer(s"$at.audio_ff.net.2", c.audioHidden)
    val modulation: Tensor = table(s"$at.scale_shift_table", 9, c.hidden)
    val audioModulation: Tensor =
      table(s"$at.audio_scale_shift_table", 9, c.audioHidden)
    val prompt: Tensor = table(s"$at.prompt_scale_shift_table", 2, c.hidden)
    val audioPrompt: Tensor =
      table(s"$at.audio_prompt_scale_shift_table", 2, c.audioHidden)
    val cross: Tensor =
      table(s"$at.scale_shift_table_a2v_ca_video", 5, c.hidden)
    val audioCross: Tensor =
      table(s"$at.scale_shift_table_a2v_ca_audio", 5, c.audioHidden)
  }

  private val videoIn = affineLayer("patchify_proj", c.hidden)
  private val audioIn = affineLayer("audio_patchify_proj", c.audioHidden)
  private val videoOut = affineLayer("proj_out", c.videoChannels)
  private val audioOut = affineLayer("audio_proj_out", c.audioChannels)
  private val finalModulation = table("scale_shift_table", 2, c.hidden)
  private val audioFinalModulation =
    table("audio_scale_shift_table", 2, c.audioHidden)
  private val time = new AdaLn("adaln_single", c.hidden, 9)
  private val audioTime = new AdaLn("audio_adaln_single", c.audioHidden, 9)
  private val promptTime = new AdaLn("prompt_adaln_single", c.hidden, 2)
  private val audioPromptTime =
    new AdaLn("audio_prompt_adaln_single", c.audioHidden, 2)
  private val crossTime =
    new AdaLn("av_ca_video_scale_shift_adaln_single", c.hidden, 4)
  private val audioCrossTime =
    new AdaLn("av_ca_audio_scale_shift_adaln_single", c.audioHidden, 4)
  private val crossGate = new AdaLn("av_ca_a2v_gate_adaln_single", c.hidden, 1)
  private val audioCrossGate =
    new AdaLn("av_ca_v2a_gate_adaln_single", c.audioHidden, 1)
  private val blocks = (0 until c.blocks).map(new Block(_))
  private val videoOnes =
    weights.keep(ops.fromFloats(Shape.of(c.hidden), Array.fill(c.hidden)(1f)))
  private val audioOnes = weights.keep(
    ops.fromFloats(Shape.of(c.audioHidden), Array.fill(c.audioHidden)(1f))
  )

  /** The prompt's text connectors (the video's and the audio's), when the file
    * holds them: they share its LoRAs.
    */
  val connectors: Option[(LtxConnector, LtxConnector)] =
    Option.when(source.has("video_embeddings_connector.learnable_registers"))(
      (
        LtxConnector(ops, source, "video_embeddings_connector", Some(updates)),
        LtxConnector(ops, source, "audio_embeddings_connector", Some(updates))
      )
    )

  private def affine(x: Tensor, layer: Affine, out: Tensor): Unit = {
    ops.linear(x, layer.weight, out)
    updates(x, layer.site, out)
    ops.addRow(out, layer.bias, out)
  }

  // ---- LoRAs ---------------------------------------------------------------------

  private def allSites: Map[String, Tensor] =
    siteWeights.toMap ++ connectors.toSeq.flatMap((v, a) => v.sites ++ a.sites)

  /** Makes `loras` (each at its multiplier) the active set, replacing the last;
    * they read the original names (ComfyUI's, the official trainer's) and
    * diffusers' (`Ltx2.siteOf`). Returns the targets that matched nothing or a
    * weight of another shape, left unapplied. Text encoded before depends on
    * the LoRAs of its time (the connectors).
    */
  def useLoras(loras: Seq[(Lora, Float)]): Seq[String] = {
    val sites = allSites
    val placed = loras.flatMap((lora, multiplier) =>
      lora.pairs.toSeq.map { (target, pair) =>
        (
          target,
          Ltx2.placement(target, pair, sites.get(_).map(_.shape)),
          pair.copy(scale = pair.scale * multiplier)
        )
      }
    )
    updates.use(
      placed
        .collect { case (_, Some(site), pair) => site -> pair }
        .groupMap(_._1)(_._2)
    )
    placed.collect { case (target, None, _) => target }.distinct
  }

  /** The split RoPE's per-head tables over `positions` (per token, one value
    * per axis), each axis normalized by its `maxima` and turned by `width / 2`
    * frequencies shared out over the axes (frequency-major), the missing ones
    * at the front turning by nothing: `[tokens, heads, width / heads / 2]`.
    */
  private def ropeTables(
      positions: Array[Array[Double]],
      maxima: Seq[Double],
      width: Int,
      heads: Int
  ): (Tensor, Tensor) = {
    val axes = maxima.size
    val pairs = width / 2
    val steps = width / (2 * axes)
    val padding = pairs - steps * axes
    val frequencies = Array.tabulate(steps)(j =>
      (math.pow(
        10000.0,
        if (steps > 1) j.toDouble / (steps - 1) else 0.0
      ) * math.Pi / 2).toFloat.toDouble
    )
    val tokens = positions.length
    val cos = new Array[Float](tokens * pairs)
    val sin = new Array[Float](tokens * pairs)
    (0 until tokens).foreach { t =>
      (0 until pairs).foreach { m =>
        val at = t * pairs + m
        if (m < padding) { cos(at) = 1f; sin(at) = 0f }
        else {
          val index = m - padding
          val (j, axis) = (index / axes, index % axes)
          val grid = (positions(t)(axis) / maxima(axis)).toFloat.toDouble
          val angle = ((grid * 2 - 1).toFloat * frequencies(j).toFloat).toDouble
          cos(at) = math.cos(angle).toFloat
          sin(at) = math.sin(angle).toFloat
        }
      }
    }
    val shape = Shape.of(tokens, heads, pairs / heads)
    (ops.fromFloats(shape, cos), ops.fromFloats(shape, sin))
  }

  /** The timestep's sinusoid (cosines first, 256 wide). */
  private def sinusoid(timestep: Float): Array[Float] = {
    val half = 128
    Array.tabulate(2 * half) { i =>
      val frequency = math.exp(-math.log(10000) * (i % half) / half)
      val angle = timestep * frequency
      (if (i < half) math.cos(angle) else math.sin(angle)).toFloat
    }
  }

  /** Velocities of one step: `video` (`[frames × h × w + appended, channels]`)
    * and `audio` (`[audioFrames, channels]`) at `timestep` (σ × 1000), given
    * the prompt's connector rows `text` (`[L, hidden]`) and `audioText` (`[L,
    * audioHidden]`), into `videoVelocity` and `audioVelocity`. The video tokens
    * in the `conditioned` ranges (start, count) are held by conditions, at
    * timestep 0 in the video's AdaLN. `isolated` skips the audio-to-video and
    * video-to-audio attentions (diffusers' `isolate_modalities`): each stream
    * alone.
    */
  def velocity(
      layout: Ltx2Layout,
      video: Tensor,
      audio: Tensor,
      text: Tensor,
      audioText: Tensor,
      timestep: Float,
      videoVelocity: Tensor,
      audioVelocity: Tensor,
      isolated: Boolean = false,
      conditioned: Seq[(Long, Long)] = Nil
  ): Unit = {
    val (nv, na, nt) = (
      layout.videoTokens.toLong,
      layout.audioFrames.toLong,
      text.shape.dimensions.head
    )
    require(
      video.shape == Shape.of(nv, c.videoChannels) && audio.shape == Shape
        .of(na, c.audioChannels) &&
        text.shape == Shape.of(nt, c.hidden) && audioText.shape == Shape
          .of(nt, c.audioHidden) &&
        videoVelocity.shape == video.shape && audioVelocity.shape == audio.shape,
      s"velocity: video ${video.shape}, audio ${audio.shape}, text ${text.shape}, ${audioText.shape} for $layout"
    )
    val held = mutable.ArrayBuffer.empty[Tensor]
    def keep(tensor: Tensor) = { held += tensor; tensor }
    def allocate(rows: Long, columns: Long) = keep(
      ops.allocate(DType.F32, Shape.of(rows, columns))
    )
    val caches = mutable.Map.empty[(Long, Int, Int), KvCache]
    def cache(keys: Long, heads: Int, head: Int) =
      caches.getOrElseUpdate(
        (keys, heads, head), {
          val made =
            ops.allocateCache(1, (keys.toInt + 15) / 16 * 16, heads, head)
          held ++= Seq(made.keys, made.values)
          made
        }
      )
    try {
      val pageTable = keep(ops.fromInts(Shape.of(1), Array(0)))
      // the video tokens' runs by timestep: (start, count, 0 for σ, 1 for the
      // conditioned ones' 0)
      val runs = {
        val sorted = conditioned.filter(_._2 > 0).sortBy(_._1)
        val (gaps, end) =
          sorted.foldLeft((Vector.empty[(Long, Long, Int)], 0L)) {
            case ((acc, at), (start, count)) =>
              require(
                start >= at && start + count <= nv,
                s"conditioned: $conditioned"
              )
              (
                acc ++ Option.when(start > at)((at, start - at, 0)) :+
                  (start, count, 1),
                start + count
              )
          }
        gaps ++ Option.when(end < nv)((end, nv - end, 0))
      }
      val timesteps =
        if (conditioned.exists(_._2 > 0)) Seq(timestep, 0f) else Seq(timestep)
      // the timestep's embeddings, and each AdaLN-single's vectors
      val embeddings =
        timesteps.map(t => keep(ops.fromFloats(Shape.of(1, 256), sinusoid(t))))
      val embedded = embeddings.head
      def adaLn(
          layer: AdaLn,
          width: Int,
          embedded: Tensor = embedded
      ): (Tensor, Tensor) = {
        val inner = allocate(1, width)
        val emb = allocate(1, width)
        affine(embedded, layer.first, inner)
        ops.activation(Activation.Silu, inner, inner)
        affine(inner, layer.second, emb)
        ops.activation(Activation.Silu, emb, inner)
        val out = allocate(1, layer.size)
        affine(inner, layer.out, out)
        (out, emb)
      }
      // the video's own, per timestep
      val (videoTimes, videoEmbeddeds) =
        embeddings.map(e => adaLn(time, c.hidden, e)).unzip
      val (audioTimeOut, audioEmbedded) = adaLn(audioTime, c.audioHidden)
      val (promptOut, _) = adaLn(promptTime, c.hidden)
      val (audioPromptOut, _) = adaLn(audioPromptTime, c.audioHidden)
      val (crossOut, _) = adaLn(crossTime, c.hidden)
      val (audioCrossOut, _) = adaLn(audioCrossTime, c.audioHidden)
      val (crossGateOut, _) = adaLn(crossGate, c.hidden)
      val (audioCrossGateOut, _) = adaLn(audioCrossGate, c.audioHidden)
      // the RoPE tables: each stream's own, and the cross-modal ones (time
      // alone, over the cross-attention's heads)
      val videoPositions = layout.videoPositions
      val audioPositions = layout.audioPositions.map(Array(_))
      val (videoCos, videoSin) =
        ropeTables(videoPositions, Seq(20.0, 2048.0, 2048.0), c.hidden, c.heads)
      val (audioCos, audioSin) =
        ropeTables(audioPositions, Seq(20.0), c.audioHidden, c.audioHeads)
      val (crossVideoCos, crossVideoSin) =
        ropeTables(
          videoPositions.map(p => Array(p(0))),
          Seq(20.0),
          c.crossHidden,
          c.audioHeads
        )
      val (crossAudioCos, crossAudioSin) =
        ropeTables(audioPositions, Seq(20.0), c.crossHidden, c.audioHeads)
      held ++= Seq(
        videoCos,
        videoSin,
        audioCos,
        audioSin,
        crossVideoCos,
        crossVideoSin,
        crossAudioCos,
        crossAudioSin
      )
      // the streams
      val x = allocate(nv, c.hidden)
      val a = allocate(na, c.audioHidden)
      affine(video, videoIn, x)
      affine(audio, audioIn, a)
      // scratch
      val widest = math.max(c.hidden, c.audioHidden).toLong
      val rows = math.max(math.max(nv, na), nt)
      val (normed, normedAudio, queriesIn, keysIn) =
        (
          allocate(nv, c.hidden),
          allocate(na, c.audioHidden),
          allocate(rows, widest),
          allocate(rows, widest)
        )
      val (q, k, v, rq, rk, attended, gated, logits, projected) = (
        allocate(rows, widest),
        allocate(rows, widest),
        allocate(rows, widest),
        allocate(rows, widest),
        allocate(rows, widest),
        allocate(rows, widest),
        allocate(rows, widest),
        allocate(rows, math.max(c.heads, c.audioHeads).toLong),
        allocate(rows, widest)
      )
      val up = allocate(math.max(nv, na), 4 * widest)
      def row(t: Tensor, i: Int) = t.rows(i, 1).view(t.shape.dimensions.last)

      /** A gated attention: queries from `queryIn` (`[nq, ·]`), keys and values
        * from `keyIn` (`[nk, ·]`), `heads` of `head`, the RoPE tables when
        * given; the result projected into `out`.
        */
      def attend(
          layer: AttentionLayer,
          queryIn: Tensor,
          keyIn: Tensor,
          heads: Int,
          head: Int,
          queryRope: Option[(Tensor, Tensor)],
          keyRope: Option[(Tensor, Tensor)],
          out: Tensor
      ): Unit = {
        val (nq, nk) =
          (queryIn.shape.dimensions.head, keyIn.shape.dimensions.head)
        val inner = heads.toLong * head
        val (qs, ks, vs) =
          (q.prefix(nq, inner), k.prefix(nk, inner), v.prefix(nk, inner))
        affine(queryIn, layer.q, qs)
        affine(keyIn, layer.k, ks)
        affine(keyIn, layer.v, vs)
        ops.rmsNorm(qs, layer.qNorm, Epsilon, 0f, qs)
        ops.rmsNorm(ks, layer.kNorm, Epsilon, 0f, ks)
        val queries = queryRope.fold(qs.view(nq, heads, head)) { (cos, sin) =>
          val target = rq.prefix(nq, heads, head)
          ops.ropeTable(
            qs.view(nq, heads, head),
            cos,
            sin,
            target,
            halves = true
          )
          target
        }
        val keys = keyRope.fold(ks.view(nk, heads, head)) { (cos, sin) =>
          val target = rk.prefix(nk, heads, head)
          ops.ropeTable(
            ks.view(nk, heads, head),
            cos,
            sin,
            target,
            halves = true
          )
          target
        }
        val kv = cache(nk, heads, head)
        ops.cacheWrite(keys, vs.view(nk, heads, head), kv, pageTable, 0)
        val result = attended.prefix(nq, heads, head)
        ops.attention(
          queries,
          kv,
          pageTable,
          0,
          nk.toInt,
          drift.runner.ops.Attention(
            (1 / math.sqrt(head)).toFloat,
            causal = false,
            None,
            None,
            None
          ),
          result
        )
        val gates = logits.prefix(nq, heads.toLong)
        affine(queryIn, layer.gate, gates)
        ops.activation(Activation.Sigmoid, gates, gates)
        ops.scale(gates, 2f, gates)
        val weighted = gated.prefix(nq, inner)
        ops.zero(weighted)
        ops.rowGatedAdd(
          weighted.view(nq * heads, head),
          result.view(nq * heads, head),
          gates.view(nq * heads)
        )
        affine(weighted, layer.o, out)
      }

      def mlp(
          input: Tensor,
          upLayer: Affine,
          downLayer: Affine,
          out: Tensor
      ): Unit = {
        val n = input.shape.dimensions.head
        val inner = up.prefix(n, upLayer.bias.shape.elementCount)
        affine(input, upLayer, inner)
        ops.activation(Activation.GeluTanh, inner, inner)
        affine(inner, downLayer, out)
      }

      val videoRope = Some((videoCos, videoSin))
      val audioRope = Some((audioCos, audioSin))
      val (f, fa) = (c.hidden.toLong, c.audioHidden.toLong)
      blocks.foreach { b =>
        val videoMods = videoTimes.map { videoTime =>
          val mods = keep(ops.allocate(DType.F32, Shape.of(9, f)))
          ops.add(b.modulation, videoTime.view(9, f), mods)
          mods
        }
        // the video's modulation and gates, run by run
        def modulateVideo(x: Tensor, scale: Int, shift: Int): Unit =
          runs.foreach { (start, count, k) =>
            val part = x.rows(start, count)
            ops.modulate(
              part,
              row(videoMods(k), scale),
              row(videoMods(k), shift),
              part
            )
          }
        def gateVideo(y: Tensor, gate: Int): Unit =
          runs.foreach((start, count, k) =>
            ops.gatedAdd(
              x.rows(start, count),
              y.rows(start, count),
              row(videoMods(k), gate)
            )
          )
        val audioMods = keep(ops.allocate(DType.F32, Shape.of(9, fa)))
        ops.add(b.audioModulation, audioTimeOut.view(9, fa), audioMods)
        // self-attention
        ops.rmsNorm(x, videoOnes, Epsilon, 0f, normed)
        modulateVideo(normed, 1, 0)
        val out = projected.prefix(nv, f)
        attend(
          b.attention,
          normed,
          normed,
          c.heads,
          c.head,
          videoRope,
          videoRope,
          out
        )
        gateVideo(out, 2)
        ops.rmsNorm(a, audioOnes, Epsilon, 0f, normedAudio)
        ops.modulate(
          normedAudio,
          row(audioMods, 1),
          row(audioMods, 0),
          normedAudio
        )
        val audioOutput = projected.prefix(na, fa)
        attend(
          b.audioAttention,
          normedAudio,
          normedAudio,
          c.audioHeads,
          c.audioHead,
          audioRope,
          audioRope,
          audioOutput
        )
        ops.gatedAdd(a, audioOutput, row(audioMods, 2))
        // the prompt: the query modulated, the rows by the prompt modulation
        val promptVectors = keep(ops.allocate(DType.F32, Shape.of(2, f)))
        ops.add(b.prompt, promptOut.view(2, f), promptVectors)
        val textRows = keysIn.prefix(nt, f)
        ops.modulate(
          text,
          row(promptVectors, 1),
          row(promptVectors, 0),
          textRows
        )
        ops.rmsNorm(x, videoOnes, Epsilon, 0f, normed)
        modulateVideo(normed, 7, 6)
        attend(b.text, normed, textRows, c.heads, c.head, None, None, out)
        gateVideo(out, 8)
        val audioPromptVectors = keep(ops.allocate(DType.F32, Shape.of(2, fa)))
        ops.add(b.audioPrompt, audioPromptOut.view(2, fa), audioPromptVectors)
        val audioTextRows = keysIn.prefix(nt, fa)
        ops.modulate(
          audioText,
          row(audioPromptVectors, 1),
          row(audioPromptVectors, 0),
          audioTextRows
        )
        ops.rmsNorm(a, audioOnes, Epsilon, 0f, normedAudio)
        ops.modulate(
          normedAudio,
          row(audioMods, 7),
          row(audioMods, 6),
          normedAudio
        )
        attend(
          b.audioText,
          normedAudio,
          audioTextRows,
          c.audioHeads,
          c.audioHead,
          None,
          None,
          audioOutput
        )
        ops.gatedAdd(a, audioOutput, row(audioMods, 8))
        // audio to video and video to audio, from the streams as they are now;
        // none when the modalities are isolated (the modality guidance's pass)
        if (!isolated) {
          val crossVideo = keep(ops.allocate(DType.F32, Shape.of(5, f)))
          ops.add(
            b.cross.rows(0, 4),
            crossOut.view(4, f),
            crossVideo.rows(0, 4)
          )
          ops.add(
            b.cross.rows(4, 1),
            crossGateOut.view(1, f),
            crossVideo.rows(4, 1)
          )
          val crossAudio = keep(ops.allocate(DType.F32, Shape.of(5, fa)))
          ops.add(
            b.audioCross.rows(0, 4),
            audioCrossOut.view(4, fa),
            crossAudio.rows(0, 4)
          )
          ops.add(
            b.audioCross.rows(4, 1),
            audioCrossGateOut.view(1, fa),
            crossAudio.rows(4, 1)
          )
          ops.rmsNorm(x, videoOnes, Epsilon, 0f, normed)
          ops.rmsNorm(a, audioOnes, Epsilon, 0f, normedAudio)
          val (videoQuery, audioKey) =
            (queriesIn.prefix(nv, f), keysIn.prefix(na, fa))
          // a2v: (scale 0, shift 1) on both sides
          ops.modulate(
            normed,
            row(crossVideo, 0),
            row(crossVideo, 1),
            videoQuery
          )
          ops.modulate(
            normedAudio,
            row(crossAudio, 0),
            row(crossAudio, 1),
            audioKey
          )
          attend(
            b.audioToVideo,
            videoQuery,
            audioKey,
            c.audioHeads,
            c.crossHead,
            Some((crossVideoCos, crossVideoSin)),
            Some((crossAudioCos, crossAudioSin)),
            out
          )
          ops.gatedAdd(x, out, row(crossVideo, 4))
          // v2a: (scale 2, shift 3) on both sides
          val (audioQuery, videoKey) =
            (queriesIn.prefix(na, fa), keysIn.prefix(nv, f))
          ops.modulate(
            normedAudio,
            row(crossAudio, 2),
            row(crossAudio, 3),
            audioQuery
          )
          ops.modulate(normed, row(crossVideo, 2), row(crossVideo, 3), videoKey)
          attend(
            b.videoToAudio,
            audioQuery,
            videoKey,
            c.audioHeads,
            c.crossHead,
            Some((crossAudioCos, crossAudioSin)),
            Some((crossVideoCos, crossVideoSin)),
            audioOutput
          )
          ops.gatedAdd(a, audioOutput, row(crossAudio, 4))
          Seq(crossVideo, crossAudio).foreach { t =>
            held -= t
            ops.release(t)
          }
        }
        // MLPs
        ops.rmsNorm(x, videoOnes, Epsilon, 0f, normed)
        modulateVideo(normed, 4, 3)
        mlp(normed, b.up, b.down, out)
        gateVideo(out, 5)
        ops.rmsNorm(a, audioOnes, Epsilon, 0f, normedAudio)
        ops.modulate(
          normedAudio,
          row(audioMods, 4),
          row(audioMods, 3),
          normedAudio
        )
        mlp(normedAudio, b.audioUp, b.audioDown, audioOutput)
        ops.gatedAdd(a, audioOutput, row(audioMods, 5))
        (videoMods ++ Seq(
          audioMods,
          promptVectors,
          audioPromptVectors
        )).foreach { t =>
          held -= t
          ops.release(t)
        }
      }
      // the heads: shift and scale from the tables and the time embedding
      // (per run of timesteps: `runs` over `embs`)
      def finish(
          stream: Tensor,
          width: Long,
          modulation: Tensor,
          embs: Seq[Tensor],
          runs: Seq[(Long, Long, Int)],
          layer: Affine,
          target: Tensor
      ) = {
        val vectors = embs.map { emb =>
          val made = allocate(2, width)
          ops.addRow(modulation, emb.view(width), made)
          made
        }
        val norm = if (width == f) normed else normedAudio
        ops.layerNorm(stream, None, None, Epsilon, norm)
        runs.foreach { (start, count, k) =>
          val part = norm.rows(start, count)
          ops.modulate(part, row(vectors(k), 1), row(vectors(k), 0), part)
        }
        affine(norm, layer, target)
      }
      finish(
        x,
        f,
        finalModulation,
        videoEmbeddeds,
        runs,
        videoOut,
        videoVelocity
      )
      finish(
        a,
        fa,
        audioFinalModulation,
        Seq(audioEmbedded),
        Seq((0L, na, 0)),
        audioOut,
        audioVelocity
      )
    } finally held.foreach(ops.release)
  }

  def close(): Unit = {
    updates.close()
    connectors.foreach { (video, audio) =>
      video.release()
      audio.release()
    }
    weights.release()
    source.close()
  }
}

object Ltx2 {

  /** diffusers' names (the LoRAs trained with it) → the original ones, each a
    * whole name or its first parts.
    */
  private val DiffusersNames = Seq(
    "^proj_in(?=\\.|$)" -> "patchify_proj",
    "^audio_proj_in(?=\\.|$)" -> "audio_patchify_proj",
    "^time_embed(?=\\.|$)" -> "adaln_single",
    "^audio_time_embed(?=\\.|$)" -> "audio_adaln_single",
    "^prompt_adaln(?=\\.|$)" -> "prompt_adaln_single",
    "^audio_prompt_adaln(?=\\.|$)" -> "audio_prompt_adaln_single",
    "^av_cross_attn_video_scale_shift(?=\\.|$)" -> "av_ca_video_scale_shift_adaln_single",
    "^av_cross_attn_video_a2v_gate(?=\\.|$)" -> "av_ca_a2v_gate_adaln_single",
    "^av_cross_attn_audio_scale_shift(?=\\.|$)" -> "av_ca_audio_scale_shift_adaln_single",
    "^av_cross_attn_audio_v2a_gate(?=\\.|$)" -> "av_ca_v2a_gate_adaln_single",
    "^connectors\\.video_connector\\.transformer_blocks\\." ->
      "video_embeddings_connector.transformer_1d_blocks.",
    "^connectors\\.audio_connector\\.transformer_blocks\\." ->
      "audio_embeddings_connector.transformer_1d_blocks."
  ).map((pattern, replacement) => pattern.r -> replacement)

  /** The site a LoRA `target` names, in the original naming. */
  def siteOf(target: String): String =
    DiffusersNames.foldLeft(target)((name, rename) =>
      rename._1.replaceFirstIn(name, rename._2)
    )

  /** Where `pair` (the update of `target`) applies, given the model's sites'
    * shapes: its site when the site exists and its weight's shape is the
    * update's.
    */
  def placement(
      target: String,
      pair: LoraPair,
      shape: String => Option[Shape]
  ): Option[String] = {
    val site = siteOf(target)
    shape(site)
      .filter(weight =>
        weight.dimensions.head == pair.up.shape.dimensions.head &&
          weight.last == pair.down.shape.last
      )
      .map(_ => site)
  }

  def open(ops: Ops, path: Path): Ltx2 = {
    val source = WeightSource.open(ops, path)
    try {
      if (!Ltx2Config.holds(source))
        throw new FormatException(
          s"$path is no LTX 2 audio and video transformer"
        )
      new Ltx2(ops, source)
    } catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }

  /** The transformer file's text connectors alone, without LoRAs (the text
    * checks'); the pipeline takes `connectors` of the whole model.
    */
  def connectors(
      ops: Ops,
      path: Path
  ): (WeightSource, LtxConnector, LtxConnector) = {
    val source = WeightSource.open(ops, path)
    try
      (
        source,
        LtxConnector(ops, source, "video_embeddings_connector"),
        LtxConnector(ops, source, "audio_embeddings_connector")
      )
    catch {
      case error: Throwable =>
        source.close()
        throw error
    }
  }
}
