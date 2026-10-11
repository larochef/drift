package drift.backend.nativejobs

import drift.protocol.*
import drift.protocol.Requests.given
import drift.backend.nativejobs.ServerRequests.given
import drift.shared.*

import utest.*

import com.github.plokhotnyuk.jsoniter_scala.core.{
  readFromString,
  writeToString
}

/** A request to a server names every field: one left out is the server's launch
  * flag, not the default the form showed.
  */
object ServerRequestsTests extends TestSuite {
  val tests = Tests {
    test("an image request writes the fields equal to their defaults") {
      val json = writeToString(ImageGenerationParameters(prompt = "x"))
      assert(
        json.contains(""""width":512"""),
        json.contains(""""height":512"""),
        json.contains(""""sample_steps":20"""),
        json.contains(""""txt_cfg":7.0"""),
        json.contains(""""batch_count":1"""),
        !json.contains("hires"),
        !json.contains("vae_tiling_params")
      )
    }
    test("a job's step preview and warnings are read, as sd-server's") {
      val job = readFromString[NativeJob](
        """{"id":"a","kind":"img_gen","status":"generating","created":1,
          |"queue_position":0,"preview":{"pass":2,"step":3,"total_steps":8,
          |"b64_json":"/9j/AAAA"},"warnings":["LoRA left unapplied: x"]}""".stripMargin
      )
      val bare = readFromString[NativeJob](
        """{"id":"a","kind":"img_gen","status":"generating","created":1,
          |"queue_position":0}""".stripMargin
      )
      assert(
        job.preview.contains(NativeJobPreview(2, 3, 8, "/9j/AAAA")),
        // the runner's JPEG and sd-server's PNG, told apart by their start
        job.preview.exists(_.mimeType == "image/jpeg"),
        NativeJobPreview(1, 1, 8, "iVBORw0KGgo").mimeType == "image/png",
        // a video's, on the runner: a GIF ("GIF89a")
        NativeJobPreview(2, 1, 4, "R0lGODlhAAEA").mimeType == "image/gif",
        job.warnings == List("LoRA left unapplied: x"),
        bare.preview.isEmpty && bare.warnings.isEmpty
      )
    }
    test("an image request asks for its previews only when told to") {
      val plain = writeToString(ImageGenerationParameters(prompt = "x"))
      val asked = writeToString(
        ImageGenerationParameters(prompt = "x", preview = Some("proj"))
      )
      assert(!plain.contains("preview"), asked.contains(""""preview":"proj""""))
    }
    test("a video request asks for its previews only when told to") {
      val plain = writeToString(VideoGenerationParameters(prompt = "x"))
      val asked = writeToString(
        VideoGenerationParameters(
          prompt = "x",
          preview = Some(StepPreview.Mode)
        )
      )
      assert(
        !plain.contains("preview"),
        asked.contains(""""preview":"proj""""),
        readFromString[VideoGenerationParameters](asked).preview
          .contains("proj")
      )
    }
    test("the preferences say whether previews are shown, and start shown") {
      val hidden = GenerationPreferences(stepPreview = false)
      assert(
        GenerationPreferences.Initial.stepPreview,
        writeToString(hidden) == """{"stepPreview":false}""",
        readFromString[GenerationPreferences](writeToString(hidden)) == hidden
      )
    }
    test("a video request does too") {
      val json = writeToString(VideoGenerationParameters(prompt = "x"))
      assert(
        json.contains(""""width":512"""),
        json.contains(""""video_frames":33"""),
        json.contains(""""fps":16"""),
        json.contains(""""sample_steps":20"""),
        !json.contains("high_noise_sample_params")
      )
    }
    test("what drift sends for an image is a request the protocol reads") {
      val sent = ImageGenerationParameters(
        prompt = "a",
        negativePrompt = "b",
        clipSkip = 2,
        width = 768,
        height = 1024,
        strength = 0.4,
        seed = 7,
        batchCount = 3,
        initImage = Some("init"),
        refImages = List("one", "two"),
        autoResizeRefImage = Some(false),
        maskImage = Some("mask"),
        sampleParams = SampleParameters(
          scheduler = Some("simple"),
          sampleMethod = Some("euler"),
          sampleSteps = 9,
          eta = Some(0.5),
          shiftedTimestep = 3,
          customSigmas = List(1.0, 0.5),
          flowShift = Some(3.0),
          guidance = GuidanceParameters(
            txtCfg = 4.0,
            imgCfg = Some(2.0),
            distilledGuidance = 2.5,
            slg = Some(SlgParameters(List(7), 0.1, 0.3, 1.5))
          )
        ),
        lora = List(LoraSelection("x.safetensors", 0.8, isHighNoise = true)),
        hires = Some(HiresParameters(enabled = true)),
        vaeTilingParams = Some(VaeTilingParameters(enabled = true)),
        outputFormat = "jpg",
        outputCompression = 90,
        preview = Some("proj")
      )
      // Read by the types the runner reads it with: a field drift writes and
      // the protocol lacks fails here, not in a job.
      val read = readFromString[ImageJobRequest](writeToString(sent))
      assert(
        read.prompt.contains("a"),
        read.width.contains(768),
        read.seed.contains(7L),
        read.refImages.contains(List("one", "two")),
        read.sampleParams.flatMap(_.sampleSteps).contains(9),
        read.sampleParams.map(_.sigmas).contains(List(1.0, 0.5)),
        read.sampleParams.flatMap(_.guidance).flatMap(_.txtCfg).contains(4.0),
        read.lora.exists(_.head.isHighNoise.contains(true)),
        read.hires.flatMap(_.enabled).contains(true),
        read.preview.contains("proj")
      )
    }
    test("what drift sends for a video is one too") {
      val sent = VideoGenerationParameters(
        prompt = "a",
        moeBoundary = Some(0.9),
        vaceStrength = Some(1.0),
        initImage = Some("init"),
        endImage = Some("end"),
        controlFrames = List("frame"),
        references = List("reference"),
        guides = List(VideoGuide("guide", 8)),
        controlVideo = Some("control"),
        controlStrength = Some(0.5),
        controlStart = Some(0.1),
        controlEnd = Some(0.9),
        controlMask = Some("mask"),
        sourceVideo = Some("source"),
        highNoiseSampleParams = Some(SampleParameters(sampleSteps = 4)),
        lora = List(LoraSelection("x.safetensors")),
        vaeTilingParams = Some(VaeTilingParameters()),
        preview = Some("proj")
      )
      val read = readFromString[VideoJobRequest](writeToString(sent))
      assert(
        read.videoFrames.contains(33),
        read.guides.exists(_.head.frameIndex.contains(8)),
        read.highNoiseSampleParams.flatMap(_.sampleSteps).contains(4),
        read.controlMask.contains("mask")
      )
    }
    test("a request naming a field the protocol lacks is refused by name") {
      val problem =
        try {
          readFromString[ImageJobRequest]("""{"prompt":"a","wdith":64}"""); ""
        } catch { case error: Exception => error.getMessage }
      assert(problem.contains("wdith"))
    }
    test("a request leaves out what it leaves to the server") {
      val read = readFromString[UpscaleJobRequest](
        writeToString(UpscaleJobRequest.picture("picture", 2, -1))
      )
      assert(read.source.contains("picture"), read.scale.contains(2))
      assert(read.width.isEmpty, read.batch.isEmpty)
      assert(!writeToString(read).contains("width"))
    }
  }
}
