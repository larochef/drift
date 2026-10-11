package drift.backend.storage

import drift.protocol.ImageServerFlags
import drift.shared.*

import java.nio.file.Files

import utest.*

/** The reference data seeds as written: a record the codecs refuse is only
  * logged at startup, and its whole file is then missing from a fresh install.
  */
object ReferenceDataTests extends TestSuite {

  val tests = Tests {
    test("every built-in architecture, model and starter is seeded") {
      val storage = StorageService(Files.createTempDirectory("reference"))
      storage.init
      val architectures =
        storage.list[Architecture]("architectures").map(_.id).toSet
      val models = storage.list[Model]("models")
      val starters = storage.list[RunConfiguration]("run-configurations")
      assert(
        Set(
          "krea2",
          "mage-flow",
          "mage-flow-edit",
          "nucleus-image",
          "llada-image",
          "llada-image-turbo",
          "grn"
        ).subsetOf(architectures)
      )
      assert(models.exists(_.id == "nucleus-image-bf16"))
      assert(starters.exists(_.id == "starter-llada-image-turbo"))
      assert(starters.exists(_.id == "starter-grn-2b"))
      assert(starters.exists(_.id == "starter-mage-flow"))
      // Turbo is a checkpoint of Mage-Flow, with its own steps and CFG
      assert(
        models
          .find(_.id == "mage-flow-turbo-bf16")
          .exists(turbo =>
            turbo.familyId == "mage-flow-diffusion" &&
              turbo.parameters.get("--steps").contains("4") &&
              turbo.parameters.get("--cfg-scale").contains("1.0")
          )
      )
      // a starter names an architecture and models that exist
      val known = models.map(_.id).toSet
      val broken = starters.filter(starter =>
        !architectures(starter.architectureId) ||
          !starter.assignments.values.forall(known)
      )
      assert(broken.isEmpty)
    }
    test("what drift launches its runner with are flags the runner takes") {
      val storage = StorageService(Files.createTempDirectory("reference"))
      storage.init
      val models = storage.list[Model]("models")
      // what the runner's rule drops before a launch
      val vetoed = storage.runtimeRules
        .filter(rule =>
          rule.tool == RuntimeTool.SdCpp &&
            rule.engine == RuntimeEngine.DriftRunner
        )
        .flatMap(_.unsupported.map(_.flag))
        .toSet
      val refused = for {
        architecture <- storage.list[Architecture]("architectures")
        if architecture.tool == RuntimeTool.SdCpp &&
          architecture.runners.contains(RuntimeEngine.DriftRunner)
        slots = architecture.checkpoints.filter(slot =>
          slot.runners.isEmpty ||
            slot.runners.contains(RuntimeEngine.DriftRunner)
        )
        families = slots.map(_.familyId).toSet
        flag <- slots.map(_.flag) ++
          architecture.defaultParameters.keys ++
          models
            .filter(model => families(model.familyId))
            .flatMap(_.parameters.keys) ++
          // what every launch adds (`CommandLine.resolve`)
          List("--lora-model-dir", "--listen-ip", "--listen-port")
        if !vetoed(flag) && !ImageServerFlags.accepts(flag)
      } yield s"${architecture.id}: $flag"
      assert(refused.distinct.isEmpty)
    }
  }
}
