package drift.backend.cache

import java.nio.file.Files

import drift.shared.HuggingFace
import utest.*

/** Where `ModelCache` finds a HuggingFace file: the shared cache's snapshots,
  * or drift's own tree, where a file the shared cache could not take lands.
  */
object ModelCacheTests extends TestSuite {
  val tests = Tests {
    val source = HuggingFace("HiDream-ai/HiDream-O1-Image-Dev", "tokenizer.json")
    test("a file in drift's own tree is present") {
      val drift = Files.createTempDirectory("drift")
      val cache = ModelCache(drift, Files.createTempDirectory("hub"))
      assert(cache.resolve(source) == CacheEntry.Absent)
      val file = drift
        .resolve("models/huggingface/models--HiDream-ai--HiDream-O1-Image-Dev")
        .resolve("tokenizer.json")
      Files.createDirectories(file.getParent)
      Files.writeString(file, "{}")
      assert(cache.resolve(source) == CacheEntry.Present(file, 2))
    }
    test("a snapshot in the shared cache is present") {
      val hub = Files.createTempDirectory("hub")
      val cache = ModelCache(Files.createTempDirectory("drift"), hub)
      val file = hub
        .resolve("models--HiDream-ai--HiDream-O1-Image-Dev/snapshots/abc")
        .resolve("tokenizer.json")
      Files.createDirectories(file.getParent)
      Files.writeString(file, "{}")
      assert(cache.resolve(source) == CacheEntry.Present(file, 2))
    }
  }
}
