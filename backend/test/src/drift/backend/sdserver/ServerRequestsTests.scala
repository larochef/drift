package drift.backend.sdserver

import drift.backend.sdserver.ServerRequests.given
import drift.shared.*

import utest.*

import com.github.plokhotnyuk.jsoniter_scala.core.{
  readFromString,
  writeToString
}

/** A request to a server names every field (`bugs/29`): one left out is the
  * server's launch flag, not the default the form showed.
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
  }
}
