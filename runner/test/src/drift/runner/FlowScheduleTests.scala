package drift.runner

import utest.*

import drift.runner.diffusion.FlowSchedule

/** The flow-matching schedule against diffusers'
  * `FlowMatchEulerDiscreteScheduler` (μ = 1.15, Krea 2 at 1024²; Qwen Image
  * 2.1's configuration).
  */
object FlowScheduleTests extends TestSuite {
  val tests = Tests {
    def close(actual: Seq[Float], expected: Seq[Double]) =
      actual
        .zip(expected)
        .forall((a, e) => math.abs(a - e) < 1e-5) &&
        actual.size == expected.size
    test("img2img starts where diffusers' get_timesteps does") {
      // int(steps × strength) first: 4 steps at 0.4 run one, not two
      assert(FlowSchedule.firstStep(4, 0.4f) == 3)
      assert(FlowSchedule.firstStep(4, 0.5f) == 2)
      assert(FlowSchedule.firstStep(10, 0.75f) == 3)
      assert(FlowSchedule.firstStep(4, 1f) == 0)
      assert(FlowSchedule.firstStep(4, 0.1f) == 4)
    }
    test("4 and 8 steps at μ 1.15, as diffusers") {
      assert(
        close(
          FlowSchedule.sigmas(4, 1.15),
          Seq(1.0, 0.904531, 0.759511, 0.512844, 0)
        )
      )
      assert(
        close(
          FlowSchedule.sigmas(8, 1.15),
          Seq(1.0, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844,
            0.310901, 0)
        )
      )
    }
    test("FLUX.2's μ at 1024² in 4 steps, as diffusers' Klein pipeline") {
      val shift = FlowSchedule.flux2Shift(64 * 64, 4)
      assert(
        close(
          FlowSchedule.sigmas(4, shift),
          Seq(1.0, 0.96738, 0.90814, 0.76720, 0)
        )
      )
    }
    test(
      "Qwen Image 2.1 at 1024² in 8 steps, stretched to 0.02, as diffusers"
    ) {
      val sigmas = FlowSchedule.stretched(
        FlowSchedule.sigmas(8, FlowSchedule.qwenImage21Shift(64 * 64)),
        0.02
      )
      assert(
        close(
          sigmas,
          Seq(1.0, 0.91602, 0.82005, 0.70929, 0.58007, 0.42735, 0.24405, 0.02,
            0)
        )
      )
    }
    test("custom sigmas: a turbo LoRA's levels, the final 0 added") {
      val turbo = Seq(1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
      assert(FlowSchedule.custom(turbo).exists(close(_, turbo :+ 0.0)))
      assert(FlowSchedule.custom(turbo :+ 0.0).exists(_.size == 7))
      assert(FlowSchedule.custom(Seq(1.0, 0.5, 0.5)).isLeft)
      assert(FlowSchedule.custom(Seq(1.5, 0.5)).isLeft)
      assert(FlowSchedule.custom(Seq(0.0)).isLeft)
      assert(FlowSchedule.custom(Nil).isLeft)
    }
  }
}
