package drift.backend.runtime

import java.nio.file.*

import utest.*

/** Which gfx target a first sd-cpp install pairs ROCm for (`GpuDetection`,
  * `specs/46-starter-configurations.md`).
  */
object GpuDetectionTests extends TestSuite {

  private def nodes(versions: Int*): Path = {
    val root = Files.createTempDirectory("kfd-nodes")
    versions.zipWithIndex.foreach { (version, index) =>
      val node = Files.createDirectories(root.resolve(index.toString))
      Files.writeString(
        node.resolve("properties"),
        s"cpu_cores_count 0\ngfx_target_version $version\nvendor_id 4098\n"
      )
    }
    root
  }

  val tests = Tests {
    test("names") {
      assert(GpuDetection.gfxName(110501) == "gfx1151")
      assert(GpuDetection.gfxName(110000) == "gfx1100")
      assert(GpuDetection.gfxName(90010) == "gfx90a")
    }
    test("the CPU node is skipped") {
      assert(GpuDetection.amdGfxTarget(nodes(0, 110501)).contains("gfx1151"))
    }
    test("no AMD GPU") {
      assert(GpuDetection.amdGfxTarget(nodes(0)).isEmpty)
      assert(GpuDetection.amdGfxTarget(Paths.get("/nonexistent")).isEmpty)
    }
  }
}
