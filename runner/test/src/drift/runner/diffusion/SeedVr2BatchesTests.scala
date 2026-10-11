package drift.runner.diffusion

import utest.*

/** SeedVR2's batches over a video (`SeedVr2Pipeline.batched`): read as they are
  * asked for, so a video of any length goes through.
  */
object SeedVr2BatchesTests extends TestSuite {

  private val options = SeedVr2Options.Default

  private def frames(count: Int) =
    Iterator.tabulate(count)(index => Array(index.toFloat))

  val tests = Tests {

    test("every frame once and in order") {
      val sizes = Seq.newBuilder[Int]
      val out = SeedVr2Pipeline
        .batched(frames(50), options) { (batch, _) =>
          sizes += batch.size
          batch
        }
        .map(_.head)
        .toSeq
      assert(out == Seq.tabulate(50)(_.toFloat))
      // 21, then 4 carried and 17 new, then 4 and 12 filled up to 1 + 4n
      assert(sizes.result() == Seq(21, 21, 17))
    }

    test("a batch is restored when its first frame is asked for") {
      var read = 0
      var restored = 0
      val source = frames(200).map { frame =>
        read += 1
        frame
      }
      val out = SeedVr2Pipeline.batched(source, options) { (batch, _) =>
        restored += 1
        batch
      }
      assert(read == 0 && restored == 0)
      out.next()
      assert(read == 21 && restored == 1)
      (0 until 16).foreach(_ => out.next())
      assert(read == 21 && restored == 1)
      out.next()
      assert(read == 38 && restored == 2)
    }

    test("the shared frames fade from one batch to the next") {
      val out = SeedVr2Pipeline
        .batched(frames(30), options)((batch, index) =>
          batch.map(frame => Array(frame.head + 100 * index))
        )
        .map(_.head)
        .toSeq
      assert(out.size == 30)
      assert(out.take(17) == Seq.tabulate(17)(_.toFloat))
      // frames 17 to 20 are both batches': the first's, then the second's
      // (four frames: no room for the fall between, `fade`)
      assert(out.slice(17, 21) == Seq(17f, 18f, 119f, 120f))
      assert(out.drop(21) == Seq.tabulate(9)(index => 121f + index))
    }
  }
}
