package drift.shared

import utest.*

import com.github.plokhotnyuk.jsoniter_scala.core.readFromStream

/** How the layers of `specs/16-parameter-resolution.md` fold, removals
  * included.
  *
  * Worth testing because the preview and the launcher both go through this one
  * function: a layer applied in the wrong order is invisible until a command
  * line is wrong, and a removal that silently removes nothing is worse — the
  * form says the flag is gone and sd-cpp still gets it.
  */
object ParameterResolutionTests extends TestSuite {

  private val architecture = Architecture(
    id = "test",
    label = "Test",
    tool = RuntimeTool.SdCpp,
    checkpoints =
      List(CheckpointRef("Diffusion", "diffusion", "--diffusion-model")),
    defaultParameters = Map("--steps" -> "30", "--diffusion-fa" -> ""),
    sizeMultiple = 16,
    modelKind = None,
    runners = List(RuntimeEngine.SdCpp)
  )

  private def model(
      parameters: Map[String, String] = Map.empty,
      removed: List[String] = Nil
  ) = Model(
    id = "checkpoint",
    familyId = "diffusion",
    label = "Checkpoint",
    source = Local("/models/checkpoint.gguf"),
    format = "gguf",
    parameters = parameters,
    removedParameters = removed
  )

  private def configuration(
      overrides: Map[String, String] = Map.empty,
      removed: List[String] = Nil
  ) = RunConfiguration(
    id = "run",
    label = "Run",
    architectureId = "test",
    assignments = Map("Diffusion" -> "checkpoint"),
    overriddenParameters = overrides,
    removedParameters = removed,
    createdAt = 0L,
    lastUsedAt = 0L,
    runner = RuntimeEngine.SdCpp
  )

  private val vulkan = RuntimeRule(
    tool = RuntimeTool.SdCpp,
    backend = RuntimeBackend.Vulkan,
    engine = RuntimeEngine.SdCpp,
    unsupported = List(
      UnsupportedFlag("--diffusion-fa", "the Vulkan build cannot take it")
    )
  )

  private def resolve(
      configuration: RunConfiguration,
      model: Model,
      rule: Option[RuntimeRule] = None
  ) =
    CommandLine
      .resolveParameters(configuration, architecture, List(model), rule)

  private def inherited(
      configuration: RunConfiguration,
      model: Model,
      rule: Option[RuntimeRule] = None
  ) =
    CommandLine
      .inheritedParameters(configuration, architecture, List(model), rule)

  val tests = Tests {

    test("a model removes one of its architecture's defaults") {
      val resolved =
        resolve(configuration(), model(removed = List("--diffusion-fa")))
      assert(resolved.arguments == List("--steps" -> "30"))
      assert(
        resolved.notes == List(
          ResolutionNote.Removed(
            "--diffusion-fa",
            ParameterLayer.Architecture,
            ParameterLayer.Model
          )
        )
      )
    }

    test("a configuration removes what the model set") {
      val resolved = resolve(
        configuration(removed = List("--steps")),
        model(Map("--steps" -> "4"))
      )
      assert(resolved.arguments == List("--diffusion-fa" -> ""))
      assert(
        resolved.notes == List(
          ResolutionNote.Removed(
            "--steps",
            ParameterLayer.Model,
            ParameterLayer.Configuration
          )
        )
      )
    }

    test("a layer above a removal sets the flag again") {
      val resolved = resolve(
        configuration(overrides = Map("--diffusion-fa" -> "")),
        model(removed = List("--diffusion-fa"))
      )
      assert(resolved.arguments.contains("--diffusion-fa" -> ""))
      // The removal still happened below, and is still worth saying: it is
      // why the architecture's value is not the one on the command line.
      assert(resolved.notes.size == 1)
    }

    test("removing a flag nobody set says nothing") {
      val resolved =
        resolve(configuration(removed = List("--cfg-scale")), model())
      assert(
        resolved.arguments ==
          List("--diffusion-fa" -> "", "--steps" -> "30")
      )
      assert(resolved.notes.isEmpty)
    }

    test("a removed flag is not also reported as vetoed") {
      val resolved = resolve(
        configuration(removed = List("--diffusion-fa")),
        model(),
        Some(vulkan)
      )
      assert(resolved.arguments == List("--steps" -> "30"))
      assert(
        resolved.notes == List(
          ResolutionNote.Removed(
            "--diffusion-fa",
            ParameterLayer.Architecture,
            ParameterLayer.Configuration
          )
        )
      )
    }

    // What the parameter forms show: the flags already set below the layer
    // being edited, each with the layer that asked for it. A wrong view here
    // is what sent François to override `--attn-scale` with the value it
    // already had (2026-09-20) — the chip could not say whether the number
    // was still the architecture's or already his model's.
    test("the view of what is set below") {

      test("names the architecture for a default nothing overrides") {
        val view = inherited(configuration(), model())
        assert(
          view == List(
            ResolvedParameter(
              "--diffusion-fa",
              "",
              ParameterLayer.Architecture
            ),
            ResolvedParameter("--steps", "30", ParameterLayer.Architecture)
          )
        )
      }

      test("names the model even when its value is the architecture's") {
        val view = inherited(configuration(), model(Map("--steps" -> "30")))
        assert(
          view.find(_.flag == "--steps").map(_.layer) ==
            Some(ParameterLayer.Model)
        )
      }

      test("drops what the model removed") {
        val view =
          inherited(configuration(), model(removed = List("--diffusion-fa")))
        assert(view.map(_.flag) == List("--steps"))
      }

      test("ignores the configuration's own overrides") {
        // The configuration is the layer being edited: its rows are in the
        // editor, not in what it inherits — otherwise every override would
        // also appear as something to override.
        val view = inherited(
          configuration(
            overrides = Map("--steps" -> "8"),
            removed = List("--diffusion-fa")
          ),
          model()
        )
        assert(
          view == List(
            ResolvedParameter(
              "--diffusion-fa",
              "",
              ParameterLayer.Architecture
            ),
            ResolvedParameter("--steps", "30", ParameterLayer.Architecture)
          )
        )
      }

      test("names the runtime for a flag only the build wants") {
        val rule = vulkan.copy(defaults = Map("--threads" -> "8"))
        val view = inherited(configuration(), model(), Some(rule))
        assert(
          view.find(_.flag == "--threads").map(_.layer) ==
            Some(ParameterLayer.Runtime)
        )
        // And a vetoed flag is not offered: nothing can make the build take it.
        assert(!view.exists(_.flag == "--diffusion-fa"))
      }
    }

    test("a rule is the one of the runtime's engine") {
      def rocm(tag: String) = Runtime(
        id = tag,
        label = tag,
        tool = RuntimeTool.SdCpp,
        backend = RuntimeBackend.Rocm,
        releaseTag = tag,
        installedAt = "/nowhere",
        modelKinds = None
      )
      val shipped = readFromStream[List[RuntimeRule]](
        getClass.getResourceAsStream("/reference/runtime-rules.json")
      )
      // sd-cpp's ROCm build takes --attn-scale; the runner, on the same tool
      // and backend, drops it with its reason, and takes --guidance (FLUX.2
      // [dev]'s distilled guidance)
      assert(
        RuntimeRule.forRuntime(rocm("master-919-19bbbca"), shipped).isEmpty
      )
      val runner = RuntimeRule.forRuntime(rocm(Runtime.DriftRunnerTag), shipped)
      assert(runner.exists(_.engine == RuntimeEngine.DriftRunner))
      val resolved = resolve(
        configuration(overrides =
          Map("--guidance" -> "3.0", "--attn-scale" -> "0.5")
        ),
        model(),
        runner
      )
      assert(resolved.arguments.contains("--guidance" -> "3.0"))
      assert(!resolved.arguments.exists(_._1 == "--attn-scale"))
      assert(
        resolved.notes.collect { case ResolutionNote.Dropped(flag, _, _) =>
          flag
        } == List("--attn-scale")
      )
    }

    test("a veto still drops what nobody removed") {
      val resolved = resolve(configuration(), model(), Some(vulkan))
      assert(resolved.arguments == List("--steps" -> "30"))
      assert(
        resolved.notes.map(_.message) == List(
          "--diffusion-fa — set by the architecture, dropped: " +
            "the Vulkan build cannot take it"
        )
      )
    }
  }
}
