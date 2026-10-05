package drift.shared

import utest.*

/** Reading the assistant's answer for a redraw (`specs/52-auto-redraw.md`).
  *
  * Worth testing because the answer is a model's: it may come after thinking
  * aloud, fenced, with a tile missing or a strength out of range, and each of
  * those must end as a plan with a note rather than as a failed redraw.
  */
object RedrawPlanTests extends TestSuite {

  private val tiles = List(
    "A1" -> ImageRegion(0, 0, 1280, 1280),
    "B1" -> ImageRegion(1024, 0, 1280, 1280)
  )

  private def plan(reply: String) =
    RedrawPlan.parse(reply, tiles, 2304, 1280, 12.0)

  val tests = Tests {
    test("an answer in order becomes the tiles' settings and the repairs") {
      val Right(read) = plan(
        """{"kind": "photograph",
          | "cells": {"A1": {"holds": ["sky"], "prompt": "an even blue sky", "strength": 0.0},
          |           "B1": {"holds": ["face", "eyes"], "prompt": "skin with pores", "strength": 0.3}},
          | "repairs": [{"defect": "eyes glow", "fix": "natural eyes", "box": [0.5, 0.25, 0.75, 0.5]}]}""".stripMargin
      ): @unchecked
      assert(read.kind == "photograph")
      assert(read.tiles.map(_.name) == List("A1", "B1"))
      assert(read.tiles.map(_.settings.strength) == List(0.0, 0.3))
      assert(read.tiles(1).settings.region == ImageRegion(1024, 0, 1280, 1280))
      assert(read.tiles(1).holds == List("face", "eyes"))
      assert(
        read.repairs == List(
          PlannedRepair(
            "eyes glow",
            "natural eyes",
            ImageRegion(1152, 320, 576, 320)
          )
        )
      )
      assert(read.notes.isEmpty)
    }

    test("the last object is read, after thinking aloud and inside a fence") {
      val Right(read) = plan(
        """I see {braces} in my notes.
          |```json
          |{"cells": {"A1": {"prompt": "a sign reading \"{OPEN}\"", "strength": 0.4},
          |           "B1": {"prompt": "", "strength": 0.4}}}
          |```""".stripMargin
      ): @unchecked
      assert(read.tiles.head.settings.prompt == "a sign reading \"{OPEN}\"")
      assert(read.repairs.isEmpty)
    }

    test("a tile left out keeps the default, and the plan says so") {
      val Right(read) = plan(
        """{"cells": {"A1": {"prompt": "sand", "strength": 0.5}, "C9": {"strength": 0.2}}}"""
      ): @unchecked
      assert(read.tiles(1).settings.strength == RedrawPlan.DefaultStrength)
      assert(read.tiles(1).settings.prompt == "")
      assert(read.notes.exists(_.contains("left out B1")))
      assert(read.notes.exists(_.contains("C9")))
    }

    test("a strength out of range is brought back in") {
      val Right(read) = plan(
        """{"cells": {"A1": {"strength": 0.95}, "B1": {"strength": -1}}}"""
      ): @unchecked
      assert(
        read.tiles.map(_.settings.strength) == List(RedrawPlan.MaxStrength, 0.0)
      )
      assert(read.notes.exists(_.contains("A1, B1")))
    }

    test("a repair without a usable box is dropped, not guessed") {
      val Right(read) = plan(
        """{"cells": {"A1": {"strength": 0.4}, "B1": {"strength": 0.4}},
          | "repairs": [{"defect": "hand", "fix": "five fingers", "box": [0.6, 0.2, 0.4, 0.5]},
          |             {"defect": "sign", "fix": "legible", "box": [0.1, 0.1]},
          |             {"defect": "eye", "fix": "an eye", "box": [0, 0, 0.5, 0.5]}]}""".stripMargin
      ): @unchecked
      assert(read.repairs.map(_.defect) == List("eye"))
      assert(read.notes.exists(_.contains("2 proposed repairs")))
    }

    test("an answer with no JSON object is refused by name") {
      assert(plan("I cannot see the picture.").isLeft)
    }

    test("tiles are named by column letter and row number") {
      val rows = List(
        List(ImageRegion(0, 0, 8, 8), ImageRegion(8, 0, 8, 8)),
        List(ImageRegion(0, 8, 8, 8), ImageRegion(8, 8, 8, 8))
      )
      assert(RedrawPlan.named(rows).map(_._1) == List("A1", "B1", "A2", "B2"))
    }
  }
}
