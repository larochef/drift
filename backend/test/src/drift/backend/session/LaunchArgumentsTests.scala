package drift.backend.session

import drift.backend.cache.ModelCache
import drift.backend.runtime.LaunchRuntime
import drift.backend.storage.{Collections, StorageService}
import drift.shared.*

import java.nio.file.{Files, Paths}

import utest.*

/** A run configuration resolved for a launch (`LaunchArguments.launch`): what
  * stops it is said by name before anything is started, and what launches names
  * the files on disk.
  */
object LaunchArgumentsTests extends TestSuite {

  private def fresh() = {
    val storage = StorageService(Files.createTempDirectory("launch-config"))
    storage.init
    val cache = ModelCache(
      Files.createTempDirectory("launch-cache"),
      Files.createTempDirectory("launch-hub")
    )
    val arguments =
      LaunchArguments(storage, cache, Files.createTempDirectory("launch-loras"))
    (storage, arguments)
  }

  private val runtime = LaunchRuntime(
    Runtime(
      id = "test-runtime",
      label = "test runtime",
      tool = RuntimeTool.SdCpp,
      backend = RuntimeBackend.Vulkan,
      releaseTag = "master-1-abcdef0",
      rocmVersion = None,
      gfxTarget = None,
      theRockVersion = None,
      installedAt = "/nowhere",
      theRockPath = None,
      adopted = false,
      tracksLatest = false,
      valid = true,
      validationError = None,
      reportedVersion = None,
      createdAt = 0L,
      modelKinds = None
    ),
    Paths.get("/nowhere/sd-server"),
    Map.empty
  )

  private def starter(storage: StorageService) =
    storage.get(Collections.runConfigurations, "starter-mage-flow-turbo").get

  val tests = Tests {
    test("models that are not on disk stop the launch, each named") {
      val (storage, arguments) = fresh()
      val configuration = starter(storage)
      val blockers = arguments.blockers(configuration)
      assert(blockers.nonEmpty)
      val refused = arguments.launch(configuration, runtime, Some(7860))
      val reason = refused.left.toOption.get
      assert(reason.startsWith("cannot launch: "))
      assert(blockers.forall(blocker => reason.contains(blocker.message)))
    }
    test("an architecture that does not exist stops the launch") {
      val (storage, arguments) = fresh()
      val configuration = starter(storage).copy(architectureId = "no-such")
      assert(arguments.architectureOf(configuration).isEmpty)
      val reason =
        arguments.launch(configuration, runtime, None).left.toOption.get
      assert(reason.startsWith("cannot launch: "), reason.contains("no-such"))
    }
    test("a run configuration that does not exist is said so") {
      val (_, arguments) = fresh()
      assert(
        arguments.resolve("nope", runtime) ==
          Left("run configuration 'nope' does not exist")
      )
    }
    test("with every model on disk it launches on their paths") {
      val (storage, arguments) = fresh()
      val configuration = starter(storage)
      val files = configuration.assignments.values.toList.distinct.map { id =>
        val model = storage.get(Collections.models, id).get
        val file = Files.createTempFile(s"$id-", ".safetensors")
        Files.write(file, Array[Byte](1))
        storage.save(
          Collections.models,
          id,
          model.copy(source = Local(file.toString))
        )
        file.toString
      }
      assert(arguments.blockers(configuration).isEmpty)
      val launch =
        arguments.launch(configuration, runtime, Some(7860)).toOption.get
      assert(launch.architecture.id == configuration.architectureId)
      assert(files.forall(launch.arguments.contains))
      assert(launch.arguments.contains("7860"))
      // a one-shot run listens nowhere
      val once = arguments.launch(configuration, runtime, None).toOption.get
      assert(!once.arguments.contains("7860"))
      assert(
        arguments.resolve(configuration.id, runtime).map(_._3) ==
          Right(once.arguments)
      )
    }
  }
}
