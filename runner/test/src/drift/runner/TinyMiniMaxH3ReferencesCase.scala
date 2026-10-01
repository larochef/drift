package drift.runner

import drift.runner.decode.Prompt
import drift.runner.diffusion.{
  Images,
  MiniMaxH3Conditions,
  MiniMaxH3Normalized,
  MiniMaxH3Pipeline,
  MiniMaxH3References,
  TorchRandom,
  VideoControl
}
import drift.runner.models.{
  MiniMaxH3,
  MiniMaxH3Condition,
  MiniMaxH3ConditionRows,
  MiniMaxH3ControlRows,
  MiniMaxH3Layout,
  MiniMaxH3Reference,
  Qwen3,
  QwenVision
}
import drift.runner.ops.Ops
import drift.runner.tensor.{DType, Shape, Tensor}
import drift.runner.text.TokenizerJson
import drift.runner.vision.ImageSizing
import TinyMiniMaxH3Case.relativeError

import java.awt.image.BufferedImage
import scala.collection.mutable

/** The golden tiny MiniMax H3 references and ControlNet cases
  * (`fixtures/tiny_minimax_h3_references.py`) on a backend: the ref2va
  * presentation, layout, draws and step against diffusers', a step under the
  * Fun ControlNet union against ComfyUI's patch.
  */
object TinyMiniMaxH3ReferencesCase {

  /** `[count, height, width, 3]` values in 0–255 as images. */
  private def images(
      values: Array[Float],
      width: Int,
      height: Int
  ): Seq[BufferedImage] =
    values
      .grouped(width * height * 3)
      .map(frame => Images.toImage(frame.map(_ / 127.5f - 1), width, height))
      .toSeq

  /** The vision blocks' rows of a presentation's tags (0 a vision row). */
  private def visionRows(tags: Array[Int]): Seq[(Int, Int)] =
    tags.indices
      .filter(i => tags(i) == 0 && (i == 0 || tags(i - 1) != 0))
      .map(start =>
        (start, tags.indices.drop(start).takeWhile(tags(_) == 0).size)
      )

  /** The ref2va presentation's errors: the token ids that differ from
    * diffusers' (with Qwen3's tokenizer), the vision rows that differ from its
    * tags, the tiny Qwen3-VL's hidden state (relative), and the video
    * `smart_resize` shapes that differ from transformers'.
    */
  def presentationErrors(ops: Ops): (Int, Int, Double, Int) = {
    val folder = "tiny/minimax_h3_ref2va_presentation"
    val path = Fixtures.path(s"$folder/model.safetensors")
    Fixtures.withSafetensors(s"$folder/expected.safetensors") { golden =>
      val prompt = golden.files.head.metadata("prompt")
      val silence = Seq.fill(2)(new Array[Float](3200))
      val normalized = Seq(
        MiniMaxH3Normalized.Clip(
          images(golden("clip").decode(), 96, 64),
          Some(silence)
        ),
        MiniMaxH3Normalized.Image(
          images(golden("image").decode(), 64, 64).head
        ),
        MiniMaxH3Normalized.Sound(silence),
        MiniMaxH3Normalized.Clip(
          images(golden("still_clip").decode(), 64, 32),
          None
        )
      )
      val tokenizer = TokenizerJson.load(TokenizerFiles.json("Qwen/Qwen3-4B"))
      val pieces = MiniMaxH3References.pieces(
        normalized,
        ImageSizing(16, 2, 1, Int.MaxValue),
        24
      )
      val (presentation, rows) =
        MiniMaxH3Pipeline.presentation(tokenizer, pieces, prompt)
      val expected = golden("ids").decode().map(_.toInt)
      val idErrors =
        math.abs(expected.length - presentation.ids.length) +
          expected.indices.count(i =>
            i >= presentation.ids.length || presentation.ids(i) != expected(i)
          )
      val tags = golden("tags").decode().map(_.toInt)
      val rowErrors = (visionRows(tags).toSet diff rows.toSet).size +
        (rows.toSet diff visionRows(tags).toSet).size
      // the tiny Qwen3-VL's vocabulary
      val special = Seq(
        "<|image_pad|>" -> 300,
        "<|vision_start|>" -> 301,
        "<|vision_end|>" -> 302,
        "<|video_pad|>" -> 303
      ).map((token, tiny) => tokenizer.id(token).get -> tiny).toMap
      val tiny = Prompt(
        presentation.ids.map(id => special.getOrElse(id, id % 300)),
        presentation.images
      )
      val tower = QwenVision.fromWeights(ops, path, "visual.", Some(Seq(0, 1)))
      val encoder = Qwen3.open(ops, path)
      val hiddenError =
        try {
          val hidden = MiniMaxH3Pipeline.encoded(
            ops,
            encoder,
            Some(tower),
            tiny,
            encoder.config.layers
          )
          try relativeError(ops.toFloats(hidden), golden("hidden").decode())
          finally ops.release(hidden)
        } finally {
          encoder.close()
          tower.close()
        }
      val shapes = golden("shapes").decode().map(_.toInt).grouped(3).toSeq
      val resized = golden("resized").decode().map(_.toInt).grouped(2).toSeq
      val sizeErrors = shapes.zip(resized).count {
        case (Array(count, height, width), Array(h, w)) =>
          MiniMaxH3References.videoSize(count, height, width, 32) != (h, w)
        case _ => true
      }
      (idErrors, rowErrors, hiddenError, sizeErrors)
    }
  }

  /** The ref2va normalization's errors against diffusers': the frames whose
    * resampling onto 24 fps differs, the canvases that differ, and the worst
    * difference of PIL's LANCZOS up and down (in levels of 255).
    */
  def normalizationErrors(): (Int, Int, Double) =
    Fixtures.withSafetensors(
      "tiny/minimax_h3_ref2va_normalization/expected.safetensors"
    ) { golden =>
      val timing = Seq((30, 37), (12, 10), (25, 50), (24, 9)).map {
        (rate, count) =>
          val expected = golden(s"timed_$rate").decode().map(_.toInt).toSeq
          val actual =
            MiniMaxH3Conditions.atRate(0 until count, rate, 24).take(22)
          expected.size.max(actual.size) -
            expected.zip(actual).count(_ == _)
      }.sum
      val aspects = golden("aspects").decode().map(_.toInt).grouped(2).toSeq
      val canvases =
        golden("canvases").decode().map(_.toInt).grouped(2).toSeq
      val canvasErrors = aspects.zip(canvases).count {
        case (Array(width, height), Array(w, h)) =>
          MiniMaxH3Conditions.canvas(
            width,
            height,
            MiniMaxH3References.Multiple,
            MiniMaxH3References.CanvasShortEdge,
            MiniMaxH3References.CanvasPixels
          ) != (w, h)
        case _ => true
      }
      val image = images(golden("image").decode(), 50, 30).head
      val resized = Seq(("up", 64, 96), ("down", 24, 16)).map {
        (name, width, height) =>
          largest(
            Images
              .pixels(MiniMaxH3References.lanczos(image, width, height))
              .map(v => (v + 1) * 127.5f),
            golden(name).decode()
          )
      }.max
      (timing, canvasErrors, resized)
    }

  /** Uploads `rows` as `[rows / width, width]`. */
  private def uploaded(ops: Ops, rows: Array[Float], width: Int): Tensor =
    ops.fromFloats(Shape.of(rows.length / width, width), rows)

  /** The ref2va case's errors against diffusers': the layout's positions, the
    * draws (absolute), then one step's video and audio velocities (relative).
    */
  def ref2vaErrors(ops: Ops): Seq[Double] =
    Fixtures.withSafetensors(
      "tiny/minimax_h3_ref2va_layout/expected.safetensors"
    ) { golden =>
      val tags = golden("tags").decode().map(_.toInt)
      val references = Seq(
        MiniMaxH3Reference.Clip(7, 4, 6, 9),
        MiniMaxH3Reference.Image(6, 4),
        MiniMaxH3Reference.Sound(5),
        MiniMaxH3Reference.Clip(2, 4, 8, 0)
      )
      val layout = MiniMaxH3Layout(
        tags.length,
        2,
        4,
        8,
        6,
        Nil,
        visionRows(tags),
        references
      )
      val expectedPositions = golden("positions").decode()
      val positions = layout.positions
      val positionError = expectedPositions.indices
        .map(i => math.abs(positions(i) - expectedPositions(i)))
        .max
      def largest(actual: Array[Float], expected: Array[Float]) =
        expected.indices
          .map(i => math.abs(actual(i) - expected(i)).toDouble)
          .max
      val visual = Seq((7, 4, 6), (1, 6, 4), (2, 4, 8)).zipWithIndex.map {
        case ((t, h, w), i) =>
          MiniMaxH3Conditions.Latents(golden(s"visual_$i").decode(), t, h, w)
      }
      val (drawn, videoRows, audioRows) = MiniMaxH3Conditions.draws(
        new TorchRandom(42),
        visual,
        2,
        4,
        8,
        8,
        12,
        32,
        0.999f
      )
      val drawErrors = Seq(
        largest(drawn.flatten.toArray, golden("condition_rows").decode()),
        largest(videoRows, golden("video_rows").decode()),
        largest(audioRows, golden("audio_rows").decode())
      )
      val model = MiniMaxH3.open(
        ops,
        Fixtures.path("tiny/minimax_h3_ref2va_layout/model.safetensors")
      )
      try {
        val c = model.config
        val text = ops.allocate(DType.F32, Shape.of(tags.length, c.hidden))
        model.encodeText(
          uploaded(ops, golden("text").decode(), c.textWidth),
          text
        )
        val Seq(clip, image, still) = drawn.map(uploaded(ops, _, c.videoWidth))
        val Seq(clipSound, sound) = (0 until 2).map(i =>
          uploaded(ops, golden(s"sound_$i").decode(), c.audioWidth)
        )
        val rows = Seq(
          MiniMaxH3ConditionRows(Some(clip), Some(clipSound)),
          MiniMaxH3ConditionRows(Some(image), None),
          MiniMaxH3ConditionRows(None, Some(sound)),
          MiniMaxH3ConditionRows(Some(still), None)
        )
        val video = uploaded(ops, videoRows, c.videoWidth)
        val audio = uploaded(ops, audioRows, c.audioWidth)
        val (videoOut, audioOut) = (
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
          Nil,
          rows
        )
        Seq(positionError) ++ drawErrors ++ Seq(
          relativeError(ops.toFloats(videoOut), golden("video").decode()),
          relativeError(ops.toFloats(audioOut), golden("audio").decode())
        )
      } finally model.close()
    }

  /** The ControlNet case's errors against ComfyUI's, without then with the
    * mask: the frames encoderInputs to the encoder (absolute, in [0, 1]), the
    * control latents (absolute), one step's video and audio velocities
    * (relative).
    */
  def controlErrors(ops: Ops): Seq[Seq[Double]] = {
    val folder = "tiny/minimax_h3_control"
    Fixtures.withSafetensors(s"$folder/expected.safetensors") { golden =>
      val model = MiniMaxH3.open(
        ops,
        Fixtures.path(s"$folder/model.safetensors"),
        Some(Fixtures.path(s"$folder/controlnet.safetensors"))
      )
      try {
        val c = model.config
        val tags = Array(1, 1, 0, 0, 0, 0, 1, 1, 1)
        val layout = MiniMaxH3Layout(
          tags.length,
          7,
          4,
          8,
          4,
          Seq(MiniMaxH3Condition(0, 1, 0)),
          visionRows(tags)
        )
        val text = ops.allocate(DType.F32, Shape.of(tags.length, c.hidden))
        model.encodeText(
          uploaded(ops, golden("text").decode(), c.textWidth),
          text
        )
        val keyframe = MiniMaxH3ConditionRows(
          Some(uploaded(ops, golden("keyframe_rows").decode(), c.videoWidth)),
          None
        )
        val video = uploaded(ops, golden("video_rows").decode(), c.videoWidth)
        val audio = uploaded(ops, golden("audio_rows").decode(), c.audioWidth)
        val Seq(videoTime, audioTime) = golden("timesteps").decode().toSeq
        val controlFrames = images(golden("control_frames").decode(), 72, 40)
        val masks = images(
          golden("mask_frames").decode().flatMap(Array.fill(3)(_)),
          96,
          48
        )
        val sources = images(golden("source_frames").decode(), 96, 48)
        val controlWidth = model.controlWidth.get
        Seq(false, true).map { masked =>
          val name = if (masked) "masked" else "plain"
          val encoderInputs = mutable.ArrayBuffer.empty[Seq[Array[Float]]]
          val control = VideoControl(
            controlFrames,
            24,
            0.7f,
            0f,
            1f,
            Option.when(masked)(masks),
            Option.when(masked)(sources)
          )
          val latents = MiniMaxH3Conditions.controlLatents(
            control,
            24,
            128,
            64,
            7,
            4,
            8,
            8,
            controlWidth / 4,
            frames => {
              encoderInputs += frames
              golden(s"encoded_${encoderInputs.size - 1}").decode()
            }
          )
          val inputError = largest(
            encoderInputs(if (masked) 1 else 0).flatten
              .map(v => (v + 1) / 2)
              .toArray,
            golden(s"${name}_input").decode()
          )
          val expectedLatent = golden(s"${name}_latent").decode()
          val used = expectedLatent.length / (7 * 4 * 8)
          val latentError = math.max(
            largest(
              latents.grouped(controlWidth / 4).flatMap(_.take(used)).toArray,
              expectedLatent
            ),
            latents
              .grouped(controlWidth / 4)
              .flatMap(_.drop(used))
              .map(math.abs)
              .maxOption
              .getOrElse(0f)
              .toDouble
          )
          val rows =
            MiniMaxH3Conditions.patchify(latents, 7, 4, 8, controlWidth / 4)
          val (videoOut, audioOut) = (
            ops.allocate(DType.F32, video.shape),
            ops.allocate(DType.F32, audio.shape)
          )
          model.velocity(
            layout,
            text,
            video,
            audio,
            videoTime,
            audioTime,
            videoOut,
            audioOut,
            Seq(keyframe),
            Nil,
            Some(MiniMaxH3ControlRows(uploaded(ops, rows, controlWidth), 0.7f))
          )
          Seq(
            inputError,
            latentError,
            relativeError(
              ops.toFloats(videoOut),
              golden(s"${name}_video").decode()
            ),
            relativeError(
              ops.toFloats(audioOut),
              golden(s"${name}_audio").decode()
            )
          )
        }
      } finally model.close()
    }
  }

  private def largest(actual: Array[Float], expected: Array[Float]): Double = {
    require(
      actual.length == expected.length,
      s"${actual.length} values, not ${expected.length}"
    )
    expected.indices.map(i => math.abs(actual(i) - expected(i)).toDouble).max
  }
}
