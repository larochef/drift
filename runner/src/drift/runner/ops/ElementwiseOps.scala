package drift.runner.ops

import drift.runner.tensor.*

/** Elementwise arithmetic: sums, products, biases, and the modulations and
  * gated residuals of diffusion blocks.
  */
trait ElementwiseOps {

  /** The largest magnitude among `x`'s values (NaNs skipped), on the host. */
  def maxAbs(x: Tensor): Float

  /** `out = a + b`, elementwise. */
  def add(a: Tensor, b: Tensor, out: Tensor): Unit

  /** `out = a × b`, elementwise. */
  def mul(a: Tensor, b: Tensor, out: Tensor): Unit

  /** `out = x × factor`. */
  def scale(x: Tensor, factor: Float, out: Tensor): Unit

  /** `out = x + row`, `row` added to every row of `x` (a bias). */
  def addRow(x: Tensor, row: Tensor, out: Tensor): Unit

  /** AdaLN's modulation: `out = x × (1 + scale) + shift` per row of `x`,
    * `scale` and `shift` F32 `[cols]`.
    */
  def modulate(x: Tensor, scale: Tensor, shift: Tensor, out: Tensor): Unit

  /** A gated residual: `x += gate × y`, `gate` F32 `[cols]` for every row. */
  def gatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit

  /** AdaLN from a table of chunks per row: `x`, `out` `[rows, C]`, `table`
    * `[rows, chunks × C]`; `out = x × (1 + table chunk scale) + table chunk
    * shift`.
    */
  def modulateChunks(
      x: Tensor,
      table: Tensor,
      shift: Int,
      scale: Int,
      out: Tensor
  ): Unit

  /** `x += table chunk gate × y`, the tables as `modulateChunks`'. */
  def gatedAddChunk(x: Tensor, y: Tensor, table: Tensor, gate: Int): Unit

  /** `x[r] += gate[r] × y[r]`: one gate per row (`gate` `[rows]`). */
  def rowGatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit
}

/** The rules of `ElementwiseOps`.
  */
private[ops] trait ElementwiseChecks extends CommonChecks {

  /** Returns `(rows, C, chunks)`. */
  def checkChunks(
      operation: String,
      x: Tensor,
      table: Tensor,
      out: Tensor,
      indices: Int*
  ): (Long, Int, Int) = {
    checkElementwise(operation, x, out)
    requireF32(operation, table)
    require(x.shape.rank == 2, s"$operation takes [rows, C], not ${x.shape}")
    val Seq(rows, channels) = x.shape.dimensions
    val width = table.shape.last
    require(
      table.shape.rank == 2 && table.shape.dimensions.head == rows &&
        width % channels == 0 &&
        indices.forall(i => i >= 0 && i < width / channels),
      s"$operation: table ${table.shape} for ${x.shape}, chunks $indices"
    )
    (rows, channels.toInt, (width / channels).toInt)
  }

  /** Returns `(rows, cols)`. */
  def checkRowGatedAdd(x: Tensor, y: Tensor, gate: Tensor): (Long, Int) = {
    checkElementwise("rowGatedAdd", x, y)
    requireF32("rowGatedAdd", gate)
    require(
      x.shape.rank == 2,
      s"rowGatedAdd takes [rows, cols], not ${x.shape}"
    )
    val Seq(rows, cols) = x.shape.dimensions
    require(
      gate.shape.elementCount == rows,
      s"rowGatedAdd: gate ${gate.shape} for ${x.shape}"
    )
    (rows, cols.toInt)
  }

  def checkAddRow(x: Tensor, row: Tensor, out: Tensor): Unit = {
    checkElementwise("addRow", x, out)
    requireF32("addRow", row)
    require(
      row.shape == Shape.of(x.shape.last),
      s"addRow: a row of ${row.shape} for rows of ${x.shape.last}"
    )
  }
}
