package drift.backend.routes

import drift.backend.routes.ByteRanges.Request
import utest.*

/** The `Range` headers a browser sends for videos and large images, and the
  * ones this server serves whole.
  */
object ByteRangesTests extends TestSuite {

  val tests = Tests {
    test("an open range runs to the end") {
      assert(ByteRanges.of(Some("bytes=0-"), 100) == Request.Part(0, 99))
      assert(ByteRanges.of(Some("bytes=40-"), 100) == Request.Part(40, 99))
    }
    test("a closed range stops at its last byte, or the file's") {
      assert(ByteRanges.of(Some("bytes=10-19"), 100) == Request.Part(10, 19))
      assert(ByteRanges.of(Some("bytes=90-500"), 100) == Request.Part(90, 99))
    }
    test("a suffix takes the last bytes") {
      assert(ByteRanges.of(Some("bytes=-10"), 100) == Request.Part(90, 99))
      assert(ByteRanges.of(Some("bytes=-500"), 100) == Request.Part(0, 99))
    }
    test("a range past the end cannot be served") {
      assert(ByteRanges.of(Some("bytes=100-"), 100) == Request.Unsatisfiable)
      assert(ByteRanges.of(Some("bytes=-0"), 100) == Request.Unsatisfiable)
      assert(ByteRanges.of(Some("bytes=0-"), 0) == Request.Unsatisfiable)
    }
    test("no header, several ranges or bad syntax get the whole file") {
      assert(ByteRanges.of(None, 100) == Request.Whole)
      assert(ByteRanges.of(Some("bytes=0-9,20-29"), 100) == Request.Whole)
      assert(ByteRanges.of(Some("items=0-9"), 100) == Request.Whole)
      assert(ByteRanges.of(Some("bytes=-"), 100) == Request.Whole)
      assert(ByteRanges.of(Some("bytes=20-10"), 100) == Request.Whole)
    }
  }
}
