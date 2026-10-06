package drift.shared

import utest.*

/** How an upscale cuts its target once the tile size and the grid's shift are
  * the user's (`bugs/45`): whatever is asked, the tiles cover the target, stay
  * inside it, and each remains an exact crop of the source.
  */
object UpscaleTilingTests extends TestSuite {

  private def covers(tiling: UpscaleTiling): Boolean = {
    val tiles = tiling.rows.flatten
    def axis(spans: List[(Int, Int)], length: Int) = {
      val sorted = spans.distinct.sortBy(_._1)
      sorted.head._1 == 0 &&
      sorted.map((start, size) => start + size).max == length &&
      sorted.zip(sorted.tail).forall { case ((start, size), (next, _)) =>
        next <= start + size
      }
    }
    axis(tiles.map(tile => (tile.x, tile.width)), tiling.width) &&
    axis(tiles.map(tile => (tile.y, tile.height)), tiling.height)
  }

  val tests = Tests {

    test("no tile asked for and no shift is the cut it always was") {
      val tiling = SeedVr2UpscaleRequest.tilingFor((4096, 4096), 4)
      assert(tiling.withTile(None).shifted(0, 0) == tiling)
      assert(
        SeedVr2UpscaleRequest.tilesFor((4096, 4096), 4) == tiling.rows
      )
      assert(tiling.rows.flatten.size == 16)
    }

    test("a smaller tile is more tiles, kept to the multiple and the bounds") {
      val tiling = SeedVr2UpscaleRequest.tilingFor((4096, 4096), 2)
      assert(tiling.rows.flatten.size == 4)
      val smaller = tiling.withTile(Some(2000))
      assert(smaller.tile == 2016)
      assert(smaller.rows.flatten.size > 4)
      assert(covers(smaller))
      // never past what the runtime takes, never under the minimum
      assert(tiling.withTile(Some(100000)).tile == tiling.tile)
      assert(tiling.withTile(Some(64)).tile == UpscaleTiling.MinimumTile)
    }

    test("a shifted grid still covers the target with exact crops") {
      for {
        (tiling, scale) <- List(
          SeedVr2UpscaleRequest.tilingFor((4096, 3072), 2) -> 2,
          SeedVr2UpscaleRequest.tilingFor((2048, 2048), 4) -> 4,
          PidUpscaleRequest
            .tilingFor((8192, 8192), PidUpscaleRequest.RunnerMaxTile) -> 4,
          PidUpscaleRequest.tilingFor((4096, 6144)) -> 4
        )
        shift <- List(1, 70, 333, 1000, 2500, 5000)
      } {
        val shifted = tiling.shifted(shift, shift * 2)
        assert(shifted.offsetX % tiling.multiple == 0)
        assert(covers(shifted))
        shifted.rows.flatten.foreach { tile =>
          assert(tile.x % scale == 0, tile.y % scale == 0)
          assert(
            tile.width % tiling.multiple == 0,
            tile.height % tiling.multiple == 0
          )
          assert(tile.width <= tiling.tile, tile.height <= tiling.tile)
        }
      }
    }
  }
}
