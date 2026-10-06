package drift.runner.models

/** A VAE's work cut in spatial tiles, for any model whose whole frame does not
  * fit: frames on the host, channels-last, cut in overlapping tiles on the
  * model's grid, each tile through the model on its own, the results blended
  * over the overlaps (the cut is diffusers' `_split_tiles`). Time is the
  * model's: a tile is all its frames, and may come back as more or fewer.
  */
object VaeTiles {

  /** Tiles over `length` (diffusers' `_split_tiles`): starts, the tile length,
    * and the overlaps between neighbours — `overlap` at least, more where the
    * tiles would run past the end — every boundary on a grid of `step`.
    */
  def split(
      length: Int,
      tile: Int,
      overlap: Int,
      step: Int
  ): (Seq[Int], Int, Seq[Int]) =
    if (tile >= length) (Seq(0), length, Nil)
    else {
      var count = (length + tile - 1) / tile
      while (tile * count - overlap * (count - 1) - length < 0)
        count += 1
      val overlaps = Array.fill(count - 1)(overlap)
      val remaining = tile * count - overlaps.sum - length
      (0 until remaining / step).foreach(i => overlaps(i % (count - 1)) += step)
      val starts =
        overlaps.scanLeft(0)((start, overlap) => start + tile - overlap)
      (starts.toSeq, tile, overlaps.toSeq)
    }

  /** Tiles (each `[frames, tileHeight, tileWidth, channels]`) into one
    * `[frames, height, width, channels]`: every tile weighs 1 where it is alone
    * and ramps over the overlap it shares with each neighbour, on both axes at
    * once, and a pixel is the weighted mean of the tiles that reach it.
    *
    * Not diffusers' `_stitch_tiles`, which blends a tile with the one above and
    * the one to its left *as made*: the left neighbour's top rows were never
    * blended with what is above them, so across the whole overlap between two
    * columns the picture steps at the row where a tile starts — a hard
    * horizontal line, the wider the overlap the longer, and these overlaps grow
    * to half a tile (`bugs/42`: y = 1360 of a 2048 × 3072 SeedVR2 picture).
    */
  def stitch(
      tiles: Seq[Seq[Array[Float]]],
      frames: Int,
      tileHeight: Int,
      tileWidth: Int,
      rowOverlaps: Seq[Int],
      columnOverlaps: Seq[Int],
      height: Int,
      width: Int,
      channels: Int
  ): Array[Float] = {
    // A tile's weights along one axis: up over the overlap before it, down
    // over the one after — two neighbours' ramps add up to 1.
    def ramps(index: Int, length: Int, overlaps: Seq[Int]): Array[Float] = {
      val before = if (index > 0) math.min(overlaps(index - 1), length) else 0
      val after =
        if (index < overlaps.size) math.min(overlaps(index), length) else 0
      Array.tabulate(length) { position =>
        val up = if (position < before) (position + 0.5f) / before else 1f
        val down =
          if (position >= length - after) (length - position - 0.5f) / after
          else 1f
        math.min(up, down)
      }
    }
    val tops =
      rowOverlaps.scanLeft(0)((top, overlap) => top + tileHeight - overlap)
    val lefts =
      columnOverlaps.scanLeft(0)((left, overlap) => left + tileWidth - overlap)
    val result = new Array[Float](frames * height * width * channels)
    val total = new Array[Float](height * width)
    tiles.indices.foreach { i =>
      val down = ramps(i, tileHeight, rowOverlaps)
      tiles(i).indices.foreach { j =>
        val across = ramps(j, tileWidth, columnOverlaps)
        val tile = tiles(i)(j)
        var y = 0
        while (y < tileHeight) {
          var x = 0
          while (x < tileWidth) {
            val weight = down(y) * across(x)
            val place = (tops(i) + y) * width + lefts(j) + x
            total(place) += weight
            var f = 0
            while (f < frames) {
              val from = ((f * tileHeight + y) * tileWidth + x) * channels
              val to = (f * height * width + place) * channels
              var c = 0
              while (c < channels) {
                result(to + c) += tile(from + c) * weight
                c += 1
              }
              f += 1
            }
            x += 1
          }
          y += 1
        }
      }
    }
    var index = 0
    while (index < result.length) {
      val weight = total(index / channels % (height * width))
      // a lone tile's pixel stays the tile's own, to the bit
      if (weight != 1f) result(index) /= weight
      index += 1
    }
    result
  }

  /** The same tiles stitched as diffusers' `_stitch_tiles` does: each blended
    * with the one above and the one to its left (both as made), over the
    * overlap between them, then cut by its own overlaps below and to the right.
    * It steps where `stitch` does not (`bugs/42`); MiniMax H3's VAE keeps it,
    * being checked against its reference's frames to the pixel
    * (`TinyMiniMaxH3Case`) and its video not yet looked at with the other.
    */
  def stitchAsDiffusers(
      tiles: Seq[Seq[Array[Float]]],
      frames: Int,
      tileHeight: Int,
      tileWidth: Int,
      rowOverlaps: Seq[Int],
      columnOverlaps: Seq[Int],
      height: Int,
      width: Int,
      channels: Int
  ): Array[Float] = {
    val result = new Array[Float](frames * height * width * channels)
    var top = 0
    tiles.indices.foreach { i =>
      var left = 0
      val keptHeight =
        if (i < tiles.size - 1) tileHeight - rowOverlaps(i) else tileHeight
      tiles(i).indices.foreach { j =>
        val keptWidth =
          if (j < tiles(i).size - 1) tileWidth - columnOverlaps(j)
          else tileWidth
        val tile = tiles(i)(j).clone()
        def at(frame: Int, y: Int, x: Int) =
          ((frame * tileHeight + y) * tileWidth + x) * channels
        if (i > 0) {
          val above = tiles(i - 1)(j)
          val extent = math.min(rowOverlaps(i - 1), tileHeight)
          for {
            f <- 0 until frames
            y <- 0 until extent
            x <- 0 until tileWidth
            c <- 0 until channels
          } {
            val weight = y.toFloat / extent
            tile(at(f, y, x) + c) =
              above(at(f, tileHeight - extent + y, x) + c) *
                (1 - weight) + tile(at(f, y, x) + c) * weight
          }
        }
        if (j > 0) {
          val leftTile = tiles(i)(j - 1)
          val extent = math.min(columnOverlaps(j - 1), tileWidth)
          for {
            f <- 0 until frames
            y <- 0 until tileHeight
            x <- 0 until extent
            c <- 0 until channels
          } {
            val weight = x.toFloat / extent
            tile(at(f, y, x) + c) =
              leftTile(at(f, y, tileWidth - extent + x) + c) *
                (1 - weight) + tile(at(f, y, x) + c) * weight
          }
        }
        for {
          f <- 0 until frames
          y <- 0 until keptHeight
        } System.arraycopy(
          tile,
          at(f, y, 0),
          result,
          ((f * height + top + y) * width + left) * channels,
          keptWidth * channels
        )
        left += keptWidth
      }
      top += keptHeight
    }
    result
  }

  /** How many tiles `mapped` cuts frames of `height × width` in. */
  def count(height: Int, width: Int, tile: Int, overlap: Int, step: Int): Int =
    split(height, tile, overlap, step)._1.size *
      split(width, tile, overlap, step)._1.size

  /** `values` (`[frames, height, width, channels]`) through `each`, tile by
    * tile. Tiles are `tile` long at most and overlap by `overlap` at least, on
    * a grid of `step` (all three in `values`' pixels; `height`, `width` and
    * `tile` multiples of `step`). `each(tile, tileHeight, tileWidth)` returns
    * its frames and their count, `up / down` times the tile's size (`step × up
    * / down` whole) with `outChannels` channels: `(1, 8)` for an encoder at
    * 1/8, `(8, 1)` for its decoder. The result is the stitched frames and their
    * count.
    */
  def mapped(
      values: Array[Float],
      frames: Int,
      height: Int,
      width: Int,
      channels: Int,
      tile: Int,
      overlap: Int,
      step: Int,
      up: Int,
      down: Int,
      outChannels: Int,
      /** Called after each tile, for a progress bar. */
      tick: () => Unit = () => ()
  )(
      each: (Array[Float], Int, Int) => (Array[Float], Int)
  ): (Array[Float], Int) = {
    require(
      height % step == 0 && width % step == 0 && tile % step == 0 &&
        step * up % down == 0,
      s"tiles of $tile on a grid of $step over $height × $width, scaled by $up / $down"
    )
    val (rowStarts, tileHeight, rowOverlaps) =
      split(height, tile, overlap, step)
    val (columnStarts, tileWidth, columnOverlaps) =
      split(width, tile, overlap, step)
    def scaled(length: Int) = length * up / down
    var made = 0
    val tiles = rowStarts.map { top =>
      columnStarts.map { left =>
        val cut = new Array[Float](frames * tileHeight * tileWidth * channels)
        for {
          t <- 0 until frames
          y <- 0 until tileHeight
        } System.arraycopy(
          values,
          ((t * height + top + y) * width + left) * channels,
          cut,
          (t * tileHeight + y) * tileWidth * channels,
          tileWidth * channels
        )
        val (result, count) = each(cut, tileHeight, tileWidth)
        require(
          result.length == count * scaled(tileHeight) * scaled(
            tileWidth
          ) * outChannels,
          s"a tile of ${result.length} values for $count frames of ${scaled(tileHeight)} × ${scaled(tileWidth)} × $outChannels"
        )
        made = count
        tick()
        result
      }
    }
    (
      stitch(
        tiles,
        made,
        scaled(tileHeight),
        scaled(tileWidth),
        rowOverlaps.map(scaled),
        columnOverlaps.map(scaled),
        scaled(height),
        scaled(width),
        outChannels
      ),
      made
    )
  }
}
