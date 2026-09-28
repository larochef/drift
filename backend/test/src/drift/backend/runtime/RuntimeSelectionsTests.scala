package drift.backend.runtime

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.Files
import java.util.concurrent.{CountDownLatch, Executors, TimeUnit}

import utest.*

/** Two runtimes validating at once — the drift runner's pair — each set their
  * tool's default; both must stick (`RuntimeSelections.update`).
  */
object RuntimeSelectionsTests extends TestSuite {

  val tests = Tests {
    test("concurrent defaults both stick") {
      (1 to 20).foreach { _ =>
        val selections =
          RuntimeSelections(StorageService(Files.createTempDirectory("sel")))
        val start = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        List(
          RuntimeTool.SdCpp -> "drift-runner-images",
          RuntimeTool.LlamaCpp -> "drift-runner"
        ).foreach { (tool, id) =>
          pool.submit(() => {
            start.await()
            selections.update(current =>
              if (current.defaultFor(tool).isEmpty)
                current.withDefault(tool, Some(id))
              else current
            )
          })
        }
        start.countDown()
        pool.shutdown()
        pool.awaitTermination(10, TimeUnit.SECONDS)
        val selection = selections.current
        assert(
          selection.defaultRuntimeId.contains("drift-runner-images"),
          selection.defaultAssistantRuntimeId.contains("drift-runner")
        )
      }
    }
  }
}
