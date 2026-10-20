package drift.backend.storage

import drift.protocol.{ImageServerFlags, RunnerModelKinds}
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
        storage.list(Collections.architectures).map(_.id).toSet
      val models = storage.list(Collections.models)
      val starters = storage.list(Collections.runConfigurations)
      assert(
        Set(
          "krea2",
          "mage-flow",
          "mage-flow-edit",
          "nucleus-image",
          "llada-image",
          "llada-image-turbo",
          "grn",
          "flux.3-image-klein",
          "flux-3-prompt-upsampler"
        ).subsetOf(architectures)
      )
      assert(models.exists(_.id == "nucleus-image-bf16"))
      assert(starters.exists(_.id == "starter-llada-image-turbo"))
      assert(starters.exists(_.id == "starter-grn-2b"))
      assert(starters.exists(_.id == "starter-mage-flow"))
      assert(starters.exists(_.id == "starter-flux3-image-klein"))
      // the caption writers read the whole reply as the prompt
      val templates = storage.list(Collections.promptTemplates).map(_.id).toSet
      assert(
        Set("flux-3-caption-writer", "flux-3-upsampler").subsetOf(templates)
      )
      // FLUX 3 Image Klein's base is a checkpoint of the same architecture,
      // with its own steps and CFG
      assert(
        models
          .find(_.id == "flux-3-image-klein-base-q8-0")
          .exists(base =>
            base.familyId == "flux.3-image-klein-diffusion" &&
              base.parameters.get("--steps").contains("50") &&
              base.parameters.get("--cfg-scale").contains("4.0")
          )
      )
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
      val models = storage.list(Collections.models)
      // what the runner's rule drops before a launch
      val vetoed = storage.runtimeRules
        .filter(rule =>
          rule.tool == RuntimeTool.SdCpp &&
            rule.engine == RuntimeEngine.DriftRunner
        )
        .flatMap(_.unsupported.map(_.flag))
        .toSet
      val refused = for {
        architecture <- storage.list(Collections.architectures)
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
    test("an architecture the runner is offered for is of a kind it runs") {
      val storage = StorageService(Files.createTempDirectory("reference"))
      storage.init
      val unknown = for {
        architecture <- storage.list(Collections.architectures)
        if architecture.runners.contains(RuntimeEngine.DriftRunner)
        kinds = architecture.tool match {
          case RuntimeTool.SdCpp    => RunnerModelKinds.Images
          case RuntimeTool.LlamaCpp => RunnerModelKinds.Chat
        }
        if !architecture.modelKind.exists(kinds.contains)
      } yield s"${architecture.id}: ${architecture.modelKind.getOrElse("no kind")}"
      assert(unknown.isEmpty)
    }
    test("what the reference data names exists") {
      val storage = StorageService(Files.createTempDirectory("reference"))
      storage.init
      val architectures = storage.list(Collections.architectures)
      val families =
        architectures.flatMap(_.checkpoints.map(_.familyId)).toSet
      // a built-in model is a checkpoint of some architecture's slot
      val orphans = storage
        .list(Collections.models)
        .filter(model => model.builtIn && !families(model.familyId))
        .map(model => s"${model.id}: ${model.familyId}")
      assert(orphans.isEmpty)
      // a starter fills slots its architecture has, with models of the
      // slot's family
      val models =
        storage.list(Collections.models).map(model => model.id -> model).toMap
      val misfiled = for {
        starter <- storage.list(Collections.runConfigurations)
        architecture <- architectures
          .find(_.id == starter.architectureId)
          .toList
        (slot, modelId) <- starter.assignments
        problem <- architecture.checkpoints.find(_.name == slot) match {
          case None             => Some(s"no slot '$slot'")
          case Some(checkpoint) =>
            models.get(modelId) match {
              case None => Some(s"no model '$modelId'")
              case Some(model) if model.familyId != checkpoint.familyId =>
                Some(
                  s"'$modelId' is of ${model.familyId}, not ${checkpoint.familyId}"
                )
              case _ => None
            }
        }
      } yield s"${starter.id}: $problem"
      assert(misfiled.isEmpty)
    }
  }
}
