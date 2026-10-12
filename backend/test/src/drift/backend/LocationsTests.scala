package drift.backend

import java.nio.file.Paths

import utest.*

/** Where drift keeps its files (`Locations`), and how the page is told
  * (`FileLocations`): a relocated drift names the folders it really uses.
  */
object LocationsTests extends TestSuite {

  private val home = Paths.get("/home/someone")

  val tests = Tests {
    test("with nothing set, the conventional folders, written from the home") {
      val shown = Locations.fromEnvironment(Map.empty, home).shown(home)
      assert(
        shown.configuration == "~/.config/drift",
        shown.authTokens == "~/.config/drift/auth-tokens",
        shown.models == "~/.cache/drift/models",
        shown.modelScope == "~/.cache/drift/modelscope",
        shown.loras == "~/.cache/drift/loras",
        shown.upscalers == "~/.cache/drift/upscale",
        shown.runtimes == "~/.cache/drift/runtimes",
        shown.outputs == "~/.local/share/drift/outputs",
        shown.huggingFace == "~/.cache/huggingface/hub"
      )
    }
    test("a relocated folder is shown where it is") {
      val shown = Locations
        .fromEnvironment(
          Map(
            "DRIFT_CACHE_ROOT" -> "/mnt/big/drift",
            "DRIFT_CONFIG_DIR" -> "/home/someone/other/config",
            "DRIFT_APP_NAME" -> "drift-test"
          ),
          home
        )
        .shown(home)
      assert(
        shown.loras == "/mnt/big/drift/loras",
        shown.runtimes == "/mnt/big/drift/runtimes",
        shown.configuration == "~/other/config",
        shown.outputs == "~/.local/share/drift-test/outputs"
      )
    }
  }
}
