package drift.runner

import drift.runner.decode.{Prompt, PromptImage}
import drift.runner.diffusion.{
  Lora,
  MiniMaxH3Conditions,
  SoundResampling,
  TorchRandom
}
import drift.runner.models.{
  MiniMaxH3,
  MiniMaxH3AudioEncoder,
  MiniMaxH3Condition,
  MiniMaxH3ConditionRows,
  MiniMaxH3Layout,
  MiniMaxH3Vae,
  MiniMaxH3VideoEncoder,
  GivenRows,
  Posterior,
  Qwen3,
  QwenVision,
  SeenImages
}
import drift.runner.vision.{ImageSizing, PreparedImage}
import drift.runner.formats.SafetensorsModel
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape}

/** The golden tiny MiniMax H3 (`fixtures/tiny_diffusion.py`) on a backend: the
  * transformer's velocities over the t2va layout the runner builds, and the
  * video VAE's frames, compared with diffusers'.
  */
object TinyMiniMaxH3Case {

  /** The largest difference between the runner's (t, h, w) of each row and
    * diffusers' layout's.
    */
  def layoutError(): Double =
    Fixtures.withSafetensors("tiny/minimax_h3/expected.safetensors") { golden =>
      val expected = golden("positions").decode()
      val actual = layout(golden).positions
      expected.indices.map(i => math.abs(actual(i) - expected(i))).max
    }

  private def layout(golden: SafetensorsModel) = {
    val Seq(text, frames, height, width, audio) =
      golden("shape").decode().map(_.toInt).toSeq
    MiniMaxH3Layout(text, frames, height, width, audio)
  }

  /** The worst error of the video and the audio velocities, each relative to
    * its largest magnitude.
    */
  def velocityError(ops: Ops): Double = {
    val model =
      MiniMaxH3.open(ops, Fixtures.path("tiny/minimax_h3/model.safetensors"))
    try
      Fixtures.withSafetensors("tiny/minimax_h3/expected.safetensors") {
        golden =>
          val c = model.config
          val laid = layout(golden)
          val encoded = ops.fromFloats(
            Shape.of(laid.textTokens, c.textWidth),
            golden("text").decode()
          )
          val text =
            ops.allocate(DType.F32, Shape.of(laid.textTokens, c.hidden))
          model.encodeText(encoded, text)
          val video = ops.fromFloats(
            Shape.of(laid.videoRows, c.videoWidth),
            golden("video_rows").decode()
          )
          val audio = ops.fromFloats(
            Shape.of(laid.audioRows, c.audioWidth),
            golden("audio_rows").decode()
          )
          val (videoOut, audioOut) = (
            ops.allocate(DType.F32, video.shape),
            ops.allocate(DType.F32, audio.shape)
          )
          val Seq(videoTime, audioTime) = golden("timesteps").decode().toSeq
          model.velocity(
            laid,
            text,
            video,
            audio,
            videoTime,
            audioTime,
            videoOut,
            audioOut
          )
          def error(actual: Array[Float], expected: Array[Float]) = {
            val scale = expected.map(math.abs).max.toDouble
            expected.indices
              .map(i => math.abs(actual(i) - expected(i)))
              .max / scale
          }
          math.max(
            error(ops.toFloats(videoOut), golden("video").decode()),
            error(ops.toFloats(audioOut), golden("audio").decode())
          )
      }
    finally model.close()
  }

  /** The worst error of the velocities with the fixture's LoRA (three namings)
    * at its multiplier, on the full file or the pruned one, and the targets
    * left unapplied.
    */
  def loraError(ops: Ops, pruned: Boolean): (Double, Seq[String]) = {
    val folder = "tiny/minimax_h3_lora"
    val model = MiniMaxH3.open(
      ops,
      Fixtures.path(
        s"$folder/${if (pruned) "pruned" else "model"}.safetensors"
      )
    )
    val lora = Lora.open(ops, Fixtures.path(s"$folder/lora.safetensors"))
    try
      Fixtures.withSafetensors(s"$folder/expected.safetensors") { golden =>
        val c = model.config
        val unapplied =
          model.useLoras(Seq(lora -> golden("multiplier").decode().head))
        val laid = MiniMaxH3Layout(5, 3, 4, 8, 3)
        val encoded = ops.fromFloats(
          Shape.of(laid.textTokens, c.textWidth),
          golden("text").decode()
        )
        val text = ops.allocate(DType.F32, Shape.of(laid.textTokens, c.hidden))
        model.encodeText(encoded, text)
        val video = ops.fromFloats(
          Shape.of(laid.videoRows, c.videoWidth),
          golden("video_rows").decode()
        )
        val audio = ops.fromFloats(
          Shape.of(laid.audioRows, c.audioWidth),
          golden("audio_rows").decode()
        )
        val (videoOut, audioOut) = (
          ops.allocate(DType.F32, video.shape),
          ops.allocate(DType.F32, audio.shape)
        )
        val Seq(videoTime, audioTime) = golden("timesteps").decode().toSeq
        model.velocity(
          laid,
          text,
          video,
          audio,
          videoTime,
          audioTime,
          videoOut,
          audioOut
        )
        val name = if (pruned) "pruned" else "full"
        (
          math.max(
            relativeError(
              ops.toFloats(videoOut),
              golden(s"${name}_video").decode()
            ),
            relativeError(
              ops.toFloats(audioOut),
              golden(s"${name}_audio").decode()
            )
          ),
          unapplied
        )
      }
    finally {
      lora.close()
      model.close()
    }
  }

  /** The largest difference between `TorchRandom`'s draws and `torch.randn`'s
    * (seeds 0, 42 and 12345; sizes 5, 16, 100 and 1000 in turn).
    */
  def randomError(): Double =
    Fixtures.withSafetensors("tiny/minimax_h3_randn/expected.safetensors") {
      golden =>
        Seq(0L, 42L, 12345L).map { seed =>
          val random = new TorchRandom(seed)
          val drawn = Seq(5, 16, 100, 1000).flatMap(random.normal(_)).toArray
          val expected = golden(s"seed_$seed").decode()
          expected.indices
            .map(i => math.abs(drawn(i) - expected(i)).toDouble)
            .max
        }.max
    }

  /** The video encoder's errors against diffusers', each relative to the
    * largest latent: one frame in four tiles as the mean and as the keyframe
    * draw (seed 42, F16), a 22-frame clip as the mean.
    */
  def videoEncoderErrors(ops: Ops): Seq[Double] = {
    val folder = "tiny/minimax_h3_video_encoder"
    val encoder = MiniMaxH3VideoEncoder.open(
      ops,
      Fixtures.path(s"$folder/model.safetensors"),
      tilePixels = 64,
      tileOverlap = 16,
      groups = 8
    )
    try
      Fixtures.withSafetensors(s"$folder/expected.safetensors") { golden =>
        def frames(name: String, count: Int) =
          golden(name)
            .decode()
            .grouped(golden(name).decode().length / count)
            .toSeq
        Seq(
          relativeError(
            encoder.encode(frames("frame", 1), 96, 112, Posterior.Mean),
            golden("frame_mean").decode()
          ),
          relativeError(
            encoder.encode(frames("frame", 1), 96, 112, Posterior.Sample(42)),
            golden("frame_sample").decode()
          ),
          relativeError(
            encoder.encode(frames("clip", 22), 32, 48, Posterior.Mean),
            golden("clip_mean").decode()
          )
        )
      }
    finally encoder.close()
  }

  /** The fl2va case's errors: the layout's positions (absolute), the seed's
    * draws (absolute: the keyframes' mixed rows, the video's, the audio's),
    * then one step's video and audio velocities (relative), all against
    * diffusers'.
    */
  def fl2vaErrors(ops: Ops): Seq[Double] =
    Fixtures.withSafetensors("tiny/minimax_h3_fl2va/expected.safetensors") {
      golden =>
        val tags = golden("tags").decode().map(_.toInt)
        val visionRows = tags.indices
          .filter(i => tags(i) == 0 && (i == 0 || tags(i - 1) != 0))
          .map(start =>
            (start, tags.indices.drop(start).takeWhile(tags(_) == 0).size)
          )
        val layout = MiniMaxH3Layout(
          tags.length,
          7,
          4,
          8,
          4,
          Seq(MiniMaxH3Condition(0, 1, 0), MiniMaxH3Condition(21, 1, 0)),
          visionRows
        )
        val expectedPositions = golden("positions").decode()
        val positions = layout.positions
        val positionError =
          expectedPositions.indices
            .map(i => math.abs(positions(i) - expectedPositions(i)))
            .max
        def largest(actual: Array[Float], expected: Array[Float]) =
          expected.indices
            .map(i => math.abs(actual(i) - expected(i)).toDouble)
            .max
        val keyframes = golden("keyframes").decode().grouped(4 * 8 * 8).toSeq
        val (conditionRows, videoRows, audioRows) = MiniMaxH3Conditions.draws(
          new TorchRandom(42),
          keyframes.map(MiniMaxH3Conditions.Latents(_, 1, 4, 8)),
          7,
          4,
          8,
          8,
          8,
          32,
          0.999f
        )
        val drawErrors = Seq(
          largest(
            conditionRows.flatten.toArray,
            golden("condition_rows").decode()
          ),
          largest(videoRows, golden("video_rows").decode()),
          largest(audioRows, golden("audio_rows").decode())
        )
        val model =
          MiniMaxH3.open(
            ops,
            Fixtures.path("tiny/minimax_h3_fl2va/model.safetensors")
          )
        try {
          val c = model.config
          val encoded =
            ops.fromFloats(
              Shape.of(tags.length, c.textWidth),
              golden("text").decode()
            )
          val text = ops.allocate(DType.F32, Shape.of(tags.length, c.hidden))
          model.encodeText(encoded, text)
          val rows = golden("condition_rows").decode()
          val perCondition = rows.length / 2
          val conditions = (0 until 2).map(i =>
            MiniMaxH3ConditionRows(
              Some(
                ops.fromFloats(
                  Shape.of(perCondition / c.videoWidth, c.videoWidth),
                  rows.slice(i * perCondition, (i + 1) * perCondition)
                )
              ),
              None
            )
          )
          val video =
            ops.fromFloats(
              Shape.of(layout.videoRows, c.videoWidth),
              golden("video_rows").decode()
            )
          val audio =
            ops.fromFloats(
              Shape.of(layout.audioRows, c.audioWidth),
              golden("audio_rows").decode()
            )
          val (videoOut, audioOut) =
            (
              ops.allocate(DType.F32, video.shape),
              ops.allocate(DType.F32, audio.shape)
            )
          val Seq(videoTime, audioTime) = golden("timesteps").decode().toSeq
          model.velocity(
            layout,
            text,
            video,
            audio,
            videoTime,
            audioTime,
            videoOut,
            audioOut,
            conditions
          )
          Seq(positionError) ++ drawErrors ++ Seq(
            relativeError(ops.toFloats(videoOut), golden("video").decode()),
            relativeError(ops.toFloats(audioOut), golden("audio").decode())
          )
        } finally model.close()
    }

  /** The guides case's errors against ComfyUI's: the layout's positions and the
    * guides' mixed rows (absolute), then one step's video and audio velocities
    * (relative).
    */
  def guidesErrors(ops: Ops): Seq[Double] =
    Fixtures.withSafetensors("tiny/minimax_h3_guides/expected.safetensors") {
      golden =>
        val layout = MiniMaxH3Layout(
          5,
          7,
          4,
          8,
          37,
          Seq(
            MiniMaxH3Condition(0, 2, 5),
            MiniMaxH3Condition(10, 1, 0),
            MiniMaxH3Condition(21, 0, 3)
          )
        )
        val expectedPositions = golden("positions").decode()
        val positions = layout.positions
        val positionError =
          expectedPositions.indices
            .map(i => math.abs(positions(i) - expectedPositions(i)))
            .max
        val width = 32
        val clean = golden("clean_video_conditions").decode()
        val (first, second) = clean.splitAt(2 * 8 * width)
        val mixed = Seq(first, second)
          .flatMap(rows =>
            MiniMaxH3Conditions
              .mixed(rows, new TorchRandom(7).normal(rows.length), 0.999f)
          )
          .toArray
        val expectedMixed = golden("video_conditions").decode()
        val mixError =
          expectedMixed.indices
            .map(i => math.abs(mixed(i) - expectedMixed(i)).toDouble)
            .max
        val model =
          MiniMaxH3.open(
            ops,
            Fixtures.path("tiny/minimax_h3_guides/model.safetensors")
          )
        try {
          val c = model.config
          val encoded =
            ops.fromFloats(Shape.of(5, c.textWidth), golden("text").decode())
          val text = ops.allocate(DType.F32, Shape.of(5, c.hidden))
          model.encodeText(encoded, text)
          val (videoFirst, videoSecond) = expectedMixed.splitAt(16 * width)
          val (audioFirst, audioThird) =
            golden("audio_conditions").decode().splitAt(10 * width)
          def rows(values: Array[Float]) =
            Some(ops.fromFloats(Shape.of(values.length / width, width), values))
          val conditions = Seq(
            MiniMaxH3ConditionRows(rows(videoFirst), rows(audioFirst)),
            MiniMaxH3ConditionRows(rows(videoSecond), None),
            MiniMaxH3ConditionRows(None, rows(audioThird))
          )
          val video =
            ops.fromFloats(
              Shape.of(layout.videoRows, c.videoWidth),
              golden("video_rows").decode()
            )
          val audio =
            ops.fromFloats(
              Shape.of(layout.audioRows, c.audioWidth),
              golden("audio_rows").decode()
            )
          val (videoOut, audioOut) =
            (
              ops.allocate(DType.F32, video.shape),
              ops.allocate(DType.F32, audio.shape)
            )
          val Seq(videoTime, audioTime) = golden("timesteps").decode().toSeq
          model.velocity(
            layout,
            text,
            video,
            audio,
            videoTime,
            audioTime,
            videoOut,
            audioOut,
            conditions
          )
          Seq(
            positionError,
            mixError,
            relativeError(ops.toFloats(videoOut), golden("video").decode()),
            relativeError(ops.toFloats(audioOut), golden("audio").decode())
          )
        } finally model.close()
    }

  /** The audio encoder's error against diffusers' (both channels), relative to
    * the largest latent.
    */
  def audioEncoderError(ops: Ops): Double = {
    val folder = "tiny/minimax_h3_audio_encoder"
    val encoder = MiniMaxH3AudioEncoder.open(
      ops,
      Fixtures.path(s"$folder/model.safetensors"),
      heads = 2
    )
    try
      Fixtures.withSafetensors(s"$folder/expected.safetensors") { golden =>
        val waveform = golden("waveform").decode()
        val channels = waveform.grouped(waveform.length / 2).toSeq
        relativeError(
          channels.flatMap(encoder.encode).toArray,
          golden("latents").decode()
        )
      }
    finally encoder.close()
  }

  /** The fl2va presentation through a config-less Qwen3-VL (MiniMax H3's text
    * encoder shrunk: `model.` and `visual.`): two keyframes after their labels,
    * then the prompt; the last layer's residual stream against transformers',
    * relative to its largest.
    */
  def presentationError(ops: Ops): Double = {
    val folder = "tiny/minimax_h3_presentation"
    val path = Fixtures.path(s"$folder/model.safetensors")
    Fixtures.withSafetensors(s"$folder/expected.safetensors") { golden =>
      val ids = golden("ids").decode().map(_.toInt)
      val sizing = ImageSizing.qwen(16, 2)
      val images = Seq((64, 96), (32, 64)).zipWithIndex.map {
        case ((h, w), i) =>
          PreparedImage(
            sizing
              .patches(golden(s"image_$i").decode().map(_ / 127.5f - 1f), h, w),
            h / 16,
            w / 16,
            2,
            s"keyframe $i"
          )
      }
      val starts = ids.indices.filter(i => ids(i) == 300 && ids(i - 1) == 301)
      val prompt = Prompt(ids, starts.zip(images).map(PromptImage(_, _)))
      val tower = QwenVision.fromWeights(ops, path, "visual.", Some(Seq(0, 1)))
      val encoder = Qwen3.open(ops, path)
      val seen = images.map(tower.encode)
      val out = ops.allocate(DType.F32, Shape.of(ids.length, 64))
      try {
        encoder.encode(
          ids,
          Seq(encoder.config.layers),
          out,
          images = Some(
            SeenImages(
              prompt.positions,
              starts.zip(seen).map((at, v) => GivenRows(at, v.tokens)),
              seen.head.deepstack.indices.map(k =>
                starts.zip(seen).map((at, v) => GivenRows(at, v.deepstack(k)))
              )
            )
          )
        )
        relativeError(ops.toFloats(out), golden("hidden").decode())
      } finally {
        ops.release(out)
        seen.foreach(v => (v.tokens +: v.deepstack).foreach(ops.release))
        encoder.close()
        tower.close()
      }
    }
  }

  /** `SoundResampling` against torchaudio's `resample`, from 48 kHz and from
    * 44.1 kHz to 32 kHz: the largest difference, relative to the largest
    * sample.
    */
  def resamplingError(): Double =
    Fixtures.withSafetensors(
      "tiny/minimax_h3_resampling/expected.safetensors"
    ) { golden =>
      val waveform = golden("waveform").decode()
      Seq(48000, 44100).map { rate =>
        relativeError(
          SoundResampling.resampled(waveform, rate, 32000),
          golden(s"from_$rate").decode()
        )
      }.max
    }

  /** The largest difference relative to the largest magnitude. */
  def relativeError(actual: Array[Float], expected: Array[Float]): Double = {
    require(
      actual.length == expected.length,
      s"${actual.length} values, not ${expected.length}"
    )
    val scale = expected.map(math.abs).max.toDouble
    expected.indices.map(i => math.abs(actual(i) - expected(i))).max / scale
  }

  /** The worst error of the decoded frames (in [0, 1]): two temporal chunks,
    * four spatial tiles each.
    */
  def framesError(ops: Ops): Double = {
    val decoder = MiniMaxH3Vae.open(
      ops,
      Fixtures.path("tiny/minimax_h3_vae/model.safetensors"),
      tilePixels = 64,
      tileOverlap = 16
    )
    try
      Fixtures.withSafetensors("tiny/minimax_h3_vae/expected.safetensors") {
        golden =>
          val latents = golden("latents").decode()
          val expected = golden("frames").decode()
          val frames = scala.collection.mutable.ArrayBuffer.empty[Array[Float]]
          decoder.decode(latents, 12, 6, 7, frames += _)
          val frameValues = 96 * 112 * 3
          assert(
            frames.size == 39 && decoder.frameCount(12) == 39,
            s"${frames.size} frames, not 39"
          )
          frames.zipWithIndex.map { (frame, f) =>
            frame.indices
              .map(i => math.abs(frame(i) - expected(f * frameValues + i)))
              .max
              .toDouble
          }.max
      }
    finally decoder.close()
  }
}
