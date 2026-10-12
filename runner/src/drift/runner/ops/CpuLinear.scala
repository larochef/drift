package drift.runner.ops

import drift.runner.tensor.*

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*

/** The reference backend's linear layers and embedding tables.
  */
private[ops] trait CpuLinear extends CpuBase {

  def embedding(table: Tensor, ids: Tensor, out: Tensor): Unit = {
    Ops.checkEmbedding(table, ids, out)
    val (rows, target, columns) =
      (segment(table), segment(out), table.shape.last.toInt)
    val rowBytes = table.dtype.byteSize(columns.toLong)
    val index = segment(ids)
    (0L until ids.shape.elementCount).foreach { t =>
      val row = index.getAtIndex(JAVA_INT, t).toLong
      require(
        row >= 0 && row < table.shape.dimensions.head,
        s"token id $row outside the table"
      )
      val values =
        if (ConvRot.stored(table.dtype))
          // the BF16 weight the codes hold
          rotatedRows(table, row, 1).map(value =>
            java.lang.Float
              .intBitsToFloat((CpuOps.toBfloat16(value) & 0xffff) << 16)
          )
        else table.dtype.decode(rows.asSlice(row * rowBytes, rowBytes), columns)
      MemorySegment.copy(
        values,
        0,
        target,
        JAVA_FLOAT,
        t * columns * 4,
        columns
      )
    }
  }

  def linear(x: Tensor, weight: Tensor, out: Tensor): Unit = {
    val (m, n, k) = Ops.checkLinear(x, weight, out)
    val (input, weights, target) = (segment(x), segment(weight), segment(out))
    val rowBytes = weight.dtype.byteSize(k)
    var row = 0L
    while (row < n) {
      val decoded =
        if (ConvRot.stored(weight.dtype)) rotatedRows(weight, row, 1)
        else
          weight.dtype
            .decode(weights.asSlice(row * rowBytes, rowBytes), k.toInt)
      var sample = 0L
      while (sample < m) {
        var sum = 0.0
        var column = 0
        while (column < k) {
          sum += decoded(column).toDouble * input.getAtIndex(
            JAVA_FLOAT,
            sample * k + column
          )
          column += 1
        }
        target.setAtIndex(JAVA_FLOAT, sample * n + row, sum.toFloat)
        sample += 1
      }
      row += 1
    }
  }
}
