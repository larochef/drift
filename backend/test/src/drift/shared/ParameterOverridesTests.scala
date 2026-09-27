package drift.shared

import utest.*

/** What a stored record is allowed to carry (`ParameterOverrides`,
  * `specs/16-parameter-resolution.md`).
  *
  * Worth testing because the rule is subtractive: every case below is one where
  * the wrong answer either keeps a no-op that reads like a decision, or drops
  * something that was holding a lower layer's value off a command line. The
  * second kind is silent until a generation comes out wrong.
  */
object ParameterOverridesTests extends TestSuite {

  private def architecture(
      id: String,
      familyId: String,
      defaults: Map[String, String]
  ) = Architecture(
    id = id,
    label = id,
    tool = RuntimeTool.SdCpp,
    checkpoints =
      List(CheckpointRef("Diffusion", familyId, "--diffusion-model")),
    defaultParameters = defaults,
    sizeMultiple = 16,
    modelKind = None,
    runners = List(RuntimeEngine.SdCpp)
  )

  private val ernie = architecture(
    "ernie-image",
    "ernie-diffusion",
    Map("--attn-scale" -> "0.0078125", "--steps" -> "8", "--diffusion-fa" -> "")
  )

  private def model(
      parameters: Map[String, String] = Map.empty,
      removed: List[String] = Nil,
      familyId: String = "ernie-diffusion"
  ) = Model(
    id = "big-love",
    familyId = familyId,
    label = "Big Love",
    source = Local("/models/big-love.safetensors"),
    format = "safetensors",
    parameters = parameters,
    removedParameters = removed
  )

  private def configuration(
      overrides: Map[String, String] = Map.empty,
      removed: List[String] = Nil
  ) = RunConfiguration(
    id = "ernie",
    label = "Ernie",
    architectureId = "ernie-image",
    assignments = Map("Diffusion" -> "big-love"),
    overriddenParameters = overrides,
    removedParameters = removed,
    createdAt = 0L,
    lastUsedAt = 0L,
    runner = RuntimeEngine.SdCpp
  )

  val tests = Tests {

    test("a model") {

      test("drops a parameter repeating the architecture's default") {
        // The one that started this: an override of `--attn-scale` with the
        // value the architecture already gives, which no form could tell from
        // a removal (François, 2026-09-20).
        val pruned = ParameterOverrides
          .prunedModel(model(Map("--attn-scale" -> "0.0078125")), List(ernie))
        assert(pruned.parameters.isEmpty)
      }

      test("keeps a parameter that differs") {
        val pruned = ParameterOverrides
          .prunedModel(model(Map("--steps" -> "4")), List(ernie))
        assert(pruned.parameters == Map("--steps" -> "4"))
      }

      test("drops a valueless flag the architecture already passes") {
        val pruned = ParameterOverrides
          .prunedModel(model(Map("--diffusion-fa" -> "")), List(ernie))
        assert(pruned.parameters.isEmpty)
      }

      test("keeps a flag no architecture sets") {
        val pruned = ParameterOverrides
          .prunedModel(model(Map("--clip-on-cpu" -> "")), List(ernie))
        assert(pruned.parameters == Map("--clip-on-cpu" -> ""))
      }

      test("keeps a value another architecture of the family disagrees with") {
        // The reason this is not pruned against the architecture whose card
        // the model was edited from: the same file fills a slot in both.
        val turbo = architecture(
          "ernie-turbo",
          "ernie-diffusion",
          Map("--steps" -> "30")
        )
        val pruned = ParameterOverrides
          .prunedModel(model(Map("--steps" -> "8")), List(ernie, turbo))
        assert(pruned.parameters == Map("--steps" -> "8"))
      }

      test("ignores an architecture of another family") {
        val other =
          architecture("flux", "flux-diffusion", Map("--steps" -> "8"))
        val pruned = ParameterOverrides
          .prunedModel(model(Map("--steps" -> "8")), List(other))
        assert(pruned.parameters == Map("--steps" -> "8"))
      }

      test("keeps everything when nothing can assign it") {
        val orphan = model(Map("--steps" -> "8"), familyId = "nobody")
        val pruned = ParameterOverrides.prunedModel(orphan, List(ernie))
        assert(pruned == orphan)
      }

      test("keeps a removal of a flag an architecture sets") {
        val pruned = ParameterOverrides
          .prunedModel(model(removed = List("--attn-scale")), List(ernie))
        assert(pruned.removedParameters == List("--attn-scale"))
      }

      test("drops a removal of a flag nothing sets") {
        val pruned = ParameterOverrides
          .prunedModel(model(removed = List("--vae-tiling")), List(ernie))
        assert(pruned.removedParameters.isEmpty)
      }

      test("a flag both set and removed keeps only the removal") {
        val pruned = ParameterOverrides.prunedModel(
          model(Map("--attn-scale" -> "0.5"), List("--attn-scale")),
          List(ernie)
        )
        assert(pruned.parameters.isEmpty)
        assert(pruned.removedParameters == List("--attn-scale"))
      }
    }

    test("a run configuration") {

      test("drops an override repeating its architecture") {
        val pruned = ParameterOverrides.prunedConfiguration(
          configuration(Map("--steps" -> "8")),
          Some(ernie),
          List(model())
        )
        assert(pruned.overriddenParameters.isEmpty)
      }

      test("drops an override repeating its assigned model") {
        val pruned = ParameterOverrides.prunedConfiguration(
          configuration(Map("--steps" -> "4")),
          Some(ernie),
          List(model(Map("--steps" -> "4")))
        )
        assert(pruned.overriddenParameters.isEmpty)
      }

      test("keeps an override the model moved away from") {
        // The architecture says 8, the model says 4, the configuration says 8
        // again: that is a decision, not an echo.
        val pruned = ParameterOverrides.prunedConfiguration(
          configuration(Map("--steps" -> "8")),
          Some(ernie),
          List(model(Map("--steps" -> "4")))
        )
        assert(pruned.overriddenParameters == Map("--steps" -> "8"))
      }

      test("keeps an override of a flag its model removed") {
        val pruned = ParameterOverrides.prunedConfiguration(
          configuration(Map("--attn-scale" -> "0.0078125")),
          Some(ernie),
          List(model(removed = List("--attn-scale")))
        )
        assert(
          pruned.overriddenParameters == Map("--attn-scale" -> "0.0078125")
        )
      }

      test("keeps a removal of something below, drops one of nothing") {
        val pruned = ParameterOverrides.prunedConfiguration(
          configuration(removed = List("--diffusion-fa", "--vae-tiling")),
          Some(ernie),
          List(model())
        )
        assert(pruned.removedParameters == List("--diffusion-fa"))
      }

      test("keeps what an unknown architecture cannot judge") {
        val typed = configuration(Map("--steps" -> "8"))
        assert(
          ParameterOverrides.prunedConfiguration(typed, None, Nil) == typed
        )
      }
    }

    test("the runtime layer never prunes") {
      // A runtime is picked at launch and changes under a configuration that
      // was never edited: an override matching one build's default is still a
      // decision about every other build, so `prunedConfiguration` resolves
      // the layers below without a rule.
      val pruned = ParameterOverrides.prunedConfiguration(
        configuration(Map("--threads" -> "8")),
        Some(ernie),
        List(model())
      )
      assert(pruned.overriddenParameters == Map("--threads" -> "8"))
    }
  }
}
