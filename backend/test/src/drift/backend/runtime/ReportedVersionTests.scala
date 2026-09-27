package drift.backend.runtime

import drift.shared.RuntimeTool

import utest.*

/** What the runtime row's version chip shows for a validated build
  * (`RuntimeManager.reportedVersion`, `specs/06-sdcpp-runtime.md`).
  *
  * Worth testing because it reads output drift does not control and cannot pin:
  * llama.cpp moved its banner off the first line at b11060, and every build
  * older than that has to keep reading exactly as it did.
  */
object ReportedVersionTests extends TestSuite {

  private def llama(output: String) =
    RuntimeManager.reportedVersion(RuntimeTool.LlamaCpp, output)

  private def sdCpp(output: String) =
    RuntimeManager.reportedVersion(RuntimeTool.SdCpp, output)

  val tests = Tests {

    test("llama.cpp b11060 logs a line before its banner") {
      val output =
        """0.00.000.283 I srv  llama_server: initializing ...
          |version: 0.4.1-dev (build 11060, commit 426090367)
          |built with GNU 11.4.0 for Linux x86_64""".stripMargin
      assert(llama(output) == "0.4.1-dev (b11060, 426090367)")
    }

    test("the same banner as the first line reads the same") {
      val output =
        """version: 0.4.0-dev (build 10850, commit f114f91f9)
          |built with cc (GCC) 11.4.0 for x86_64""".stripMargin
      assert(llama(output) == "0.4.0-dev (b10850, f114f91f9)")
    }

    test("a banner drift does not parse is kept as printed") {
      // The build-number-only banner of older llama.cpp releases.
      assert(
        llama("version: 4589 (a94e6ff8)\nbuilt with cc") ==
          "version: 4589 (a94e6ff8)"
      )
    }

    test("sd-cpp prints version unknown, so the commit is the identity") {
      val output =
        """stable-diffusion.cpp version unknown, commit 6b3edaa
          |usage: sd-server [arguments]""".stripMargin
      assert(sdCpp(output) == "commit 6b3edaa")
    }

    test("a build that does print a version keeps both") {
      assert(
        sdCpp("stable-diffusion.cpp version 0.1.0, commit 6b3edaa") ==
          "0.1.0 (6b3edaa)"
      )
    }

    test("nothing recognisable falls back to the first line") {
      assert(
        llama("  \nsomething else entirely\nand more") ==
          "something else entirely"
      )
    }
  }
}
