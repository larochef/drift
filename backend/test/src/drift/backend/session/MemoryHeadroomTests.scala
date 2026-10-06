package drift.backend.session

import utest.*

/** The memory left after a load (`specs/07-launch-and-supervision.md`), on the
  * numbers of 2026-10-05: LTX 2.5 alone, then with a chat model beside it.
  */
object MemoryHeadroomTests extends TestSuite {

  private def meminfo(free: Long, cached: Long, mapped: Long) =
    s"""MemTotal:       127396012 kB
       |MemFree:        $free kB
       |MemAvailable:   56814436 kB
       |Buffers:              16 kB
       |Cached:         $cached kB
       |Mapped:         $mapped kB
       |Shmem:            394888 kB
       |""".stripMargin

  val tests = Tests {

    test("mapped weight files are not memory left") {
      val both = MemoryHeadroom.parse(meminfo(707556, 57392868, 56587168)).get
      assert(both.freeBytes < 2L * 1000 * 1000 * 1000)
      val warning = both.warning(List("LTX 2.5", "Qwen 27B"), 10).get
      assert(warning.contains("LTX 2.5 and Qwen 27B all loaded"))
      assert(warning.contains("Stop one of them"))
      assert(both.warning(List("LTX 2.5"), 10).exists(!_.contains("Stop one")))
    }

    test("a model with room around it warns of nothing") {
      val alone =
        MemoryHeadroom.parse(meminfo(23068672, 45088768, 44000000)).get
      assert(alone.warning(List("LTX 2.5"), 10).isEmpty)
    }

    test("anything else than a meminfo is no reading") {
      assert(MemoryHeadroom.parse("").isEmpty)
    }
  }
}
