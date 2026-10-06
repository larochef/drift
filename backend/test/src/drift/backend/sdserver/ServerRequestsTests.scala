package drift.backend.sdserver

import drift.backend.sdserver.ServerRequests.given
import drift.shared.*

import utest.*

import com.github.plokhotnyuk.jsoniter_scala.core.writeToString

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
