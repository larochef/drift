package drift.shared

import com.github.plokhotnyuk.jsoniter_scala.core.*
import utest.*

/** What a paused job is written as (`specs/40-pause-and-resume.md`).
  *
  * Worth testing because the record is only read after a drift restart: a
  * request that does not survive the round trip is a job whose tiles are on
  * disk and can never be picked up, and nothing says so until then.
  */
object PausedJobTests extends TestSuite {

  private def roundTrip(work: PausedWork): PausedJob = {
    val paused = PausedJob(
      id = "g1-1001",
      kind = "pid",
      sourceDate = "2026-09-22",
      sourceFileName = "g1-0.png",
      work = work,
      tiles = 169,
      startedAt = 1758573600000L
    )
    readFromString[PausedJob](writeToString(paused))
  }

  val tests = Tests {

    test("a PiD job keeps the seed it drew") {
      val work = PausedWork.Pid(
        PidUpscaleRequest(
          runConfigurationId = "pid-flux2",
          width = Some(16384),
          height = Some(16384),
          prompt = "a beach",
          steps = 4,
          seed = 424242
        )
      )
      val back = roundTrip(work)
      assert(back.tiles == 169)
      back.work match {
        case PausedWork.Pid(request) =>
          assert(request.seed == 424242, request.width == Some(16384))
        case other => assert(false)
      }
    }

    test("a redraw keeps its geometry, its template and its context") {
      val work = PausedWork.Redraw(
        RedrawRequest(
          runConfigurationId = "krea2-snofs",
          strength = 0.4,
          instructions = "keep it calm",
          templateId = Some("redraw-women-portrait"),
          seed = 7,
          tileSize = 1280,
          contextMargin = 128,
          region = Some(ImageRegion(10, 20, 300, 400))
        )
      )
      roundTrip(work).work match {
        case PausedWork.Redraw(request) =>
          assert(request.contextMargin == 128)
          assert(request.region == Some(ImageRegion(10, 20, 300, 400)))
          assert(request.templateId == Some("redraw-women-portrait"))
        case other => assert(false)
      }
    }

    test("an edit keeps its instruction") {
      val work = PausedWork.Edit(
        EditRequest(
          runConfigurationId = "flux2-snofs",
          instructions = "the bikini top is red",
          seed = 777
        )
      )
      roundTrip(work).work match {
        case PausedWork.Edit(request) =>
          assert(request.instructions == "the bikini top is red")
          assert(request.seed == 777)
        case other => assert(false)
      }
    }
  }
}
