package drift.backend.routes

import drift.backend.storage.StorageService
import drift.shared.*

import java.nio.file.Files

import utest.*

/** What the generic update of an entity stores and refuses. */
object EntityUpdateTests extends TestSuite {

  private def token(id: String, label: String): AuthToken =
    AuthToken(id, label, AuthProvider.Civitai, "secret", createdAt = 1L)

  /** A token stands for any entity; one labelled "shipped" for a built-in. */
  private def update(storage: StorageService)(id: String, sent: AuthToken) =
    updated[AuthToken](
      storage,
      "auth-tokens",
      _.id,
      _.label == "shipped",
      (_, incoming) => incoming
    )(id, sent)

  val tests = Tests {
    test("an update is stored under the id it names") {
      val storage = StorageService(Files.createTempDirectory("drift-update"))
      storage.save("auth-tokens", "mine", token("mine", "before"))
      val answer = update(storage)("mine", token("mine", "after"))
      assert(answer == Right(Some(token("mine", "after"))))
      assert(
        storage.get[AuthToken]("auth-tokens", "mine").exists(_.label == "after")
      )
    }
    test("a body naming another id is refused and nothing is written") {
      val storage = StorageService(Files.createTempDirectory("drift-update"))
      storage.save("auth-tokens", "mine", token("mine", "before"))
      val answer = update(storage)("mine", token("other", "after"))
      assert(answer.isLeft)
      assert(
        storage
          .get[AuthToken]("auth-tokens", "mine")
          .exists(_.label == "before")
      )
      assert(!storage.exists("auth-tokens", "other"))
    }
    test("a built-in is answered as stored, whatever is sent") {
      val storage = StorageService(Files.createTempDirectory("drift-update"))
      storage.save("auth-tokens", "theirs", token("theirs", "shipped"))
      val answer = update(storage)("theirs", token("theirs", "mine now"))
      assert(answer == Right(Some(token("theirs", "shipped"))))
      assert(
        storage
          .get[AuthToken]("auth-tokens", "theirs")
          .exists(_.label == "shipped")
      )
    }
    test("an entity that does not exist is not created by an update") {
      val storage = StorageService(Files.createTempDirectory("drift-update"))
      assert(update(storage)("absent", token("absent", "new")) == Right(None))
      assert(!storage.exists("auth-tokens", "absent"))
    }
  }
}
