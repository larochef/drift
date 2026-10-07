package drift.backend.storage

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
          "mage-flow-turbo",
          "mage-flow-edit-turbo",
          "nucleus-image",
          "llada-image",
          "llada-image-turbo",
          "grn"
        ).subsetOf(architectures)
      )
      assert(models.exists(_.id == "nucleus-image-bf16"))
      assert(starters.exists(_.id == "starter-llada-image-turbo"))
      assert(starters.exists(_.id == "starter-grn-2b"))
      // a starter names an architecture and models that exist
      val known = models.map(_.id).toSet
      val broken = starters.filter(starter =>
        !architectures(starter.architectureId) ||
          !starter.assignments.values.forall(known)
      )
      assert(broken.isEmpty)
    }
  }
}
