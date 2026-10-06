package drift.shared

import utest.*

/** A slot that is one runner's (`bugs/36`): required and passed there, absent
  * everywhere else — the preview and the launcher read the same function.
  */
object SlotsPerRunnerTests extends TestSuite {

  private val architecture = Architecture(
    id = "video",
    label = "Video",
    tool = RuntimeTool.SdCpp,
    checkpoints = List(
      CheckpointRef(
        "diffusion",
        "diffusion",
        "--diffusion-model",
        runners = Nil
      ),
      CheckpointRef(
        "tokenizer",
        "tokenizer",
        "--tokenizer",
        runners = List(RuntimeEngine.DriftRunner)
      )
    ),
    defaultParameters = Map.empty,
    sizeMultiple = 16,
    modelKind = None,
    runners = List(RuntimeEngine.SdCpp, RuntimeEngine.DriftRunner)
  )

  private def model(id: String) = Model(
    id = id,
    familyId = id,
    label = id,
    source = Local(s"/models/$id"),
    format = "gguf",
    parameters = Map.empty,
    removedParameters = Nil
  )

  private val models = List(model("diffusion"), model("tokenizer"))

  private def configuration(runner: RuntimeEngine, tokenizer: Boolean) =
    RunConfiguration(
      id = "run",
      label = "Run",
      architectureId = "video",
      assignments = Map("diffusion" -> "diffusion") ++
        Option.when(tokenizer)("tokenizer" -> "tokenizer"),
      overriddenParameters = Map.empty,
      removedParameters = Nil,
      createdAt = 0L,
      lastUsedAt = 0L,
      runner = runner
    )

  private def argv(configuration: RunConfiguration) =
    CommandLine.argv(configuration, List(architecture), models, _.id)

  val tests = Tests {
    test("empty on its runner, the slot blocks the launch") {
      val blockers = CommandLine.blockers(
        configuration(RuntimeEngine.DriftRunner, tokenizer = false),
        List(architecture),
        models
      )
      assert(
        blockers == List(
          LaunchBlocker.UnassignedCheckpoint("tokenizer", "tokenizer")
        )
      )
    }
    test("empty on another runner, it does not") {
      val blockers = CommandLine.blockers(
        configuration(RuntimeEngine.SdCpp, tokenizer = false),
        List(architecture),
        models
      )
      assert(blockers.isEmpty)
    }
    test("assigned, it is passed to its runner") {
      val arguments =
        argv(configuration(RuntimeEngine.DriftRunner, tokenizer = true))
      assert(
        arguments.exists(_.containsSlice(List("--tokenizer", "tokenizer")))
      )
    }
    test("assigned, it is not passed to another runner") {
      val arguments = argv(configuration(RuntimeEngine.SdCpp, tokenizer = true))
      assert(arguments.exists(!_.contains("--tokenizer")))
    }
  }
}
