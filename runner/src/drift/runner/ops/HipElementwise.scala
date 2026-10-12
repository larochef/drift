package drift.runner.ops

import drift.runner.native.KernelArgument.*
import drift.runner.tensor.*

import java.lang.foreign.ValueLayout.*

/** The GPU backend's elementwise arithmetic.
  */
private[ops] trait HipElementwise extends HipBase {

  def maxAbs(x: Tensor): Float = {
    Ops.requireF32("maxAbs", x)
    hip.zero(pointer(argmaxIndex), 4)
    val count = x.shape.elementCount
    launch(
      kernel(normKernels, "max_abs_f32"),
      math.max(1L, math.min(1024L, (count + 255) / 256)),
      256,
      Pointer(pointer(x)),
      Pointer(pointer(argmaxIndex)),
      I64(count)
    )
    hip.copy(intStaging, pointer(argmaxIndex), 4)
    java.lang.Float.intBitsToFloat(intStaging.get(JAVA_INT, 0))
  }

  def add(a: Tensor, b: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("add", a, b, out)
    elementwise(
      "add_f32",
      out.shape.elementCount,
      Pointer(pointer(a)),
      Pointer(pointer(b)),
      Pointer(pointer(out)),
      I64(out.shape.elementCount)
    )
  }

  def mul(a: Tensor, b: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("mul", a, b, out)
    elementwise(
      "mul_f32",
      out.shape.elementCount,
      Pointer(pointer(a)),
      Pointer(pointer(b)),
      Pointer(pointer(out)),
      I64(out.shape.elementCount)
    )
  }

  def scale(x: Tensor, factor: Float, out: Tensor): Unit = {
    Ops.checkElementwise("scale", x, out)
    elementwise(
      "scale_f32",
      out.shape.elementCount,
      Pointer(pointer(x)),
      F32(factor),
      Pointer(pointer(out)),
      I64(out.shape.elementCount)
    )
  }

  def addRow(x: Tensor, row: Tensor, out: Tensor): Unit = {
    Ops.checkAddRow(x, row, out)
    elementwise(
      "add_row_f32",
      out.shape.elementCount,
      Pointer(pointer(x)),
      Pointer(pointer(row)),
      Pointer(pointer(out)),
      I64(out.shape.elementCount),
      I32(x.shape.last.toInt)
    )
  }

  def modulate(x: Tensor, scale: Tensor, shift: Tensor, out: Tensor): Unit = {
    Ops.checkRowWise("modulate", x, out, scale, shift)
    elementwise(
      "modulate_f32",
      x.shape.elementCount,
      Pointer(pointer(x)),
      Pointer(pointer(scale)),
      Pointer(pointer(shift)),
      Pointer(pointer(out)),
      I64(x.shape.elementCount),
      I32(x.shape.last.toInt)
    )
  }

  def gatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit = {
    Ops.checkRowWise("gatedAdd", x, y, gate)
    elementwise(
      "gated_add_f32",
      x.shape.elementCount,
      Pointer(pointer(x)),
      Pointer(pointer(y)),
      Pointer(pointer(gate)),
      I64(x.shape.elementCount),
      I32(x.shape.last.toInt)
    )
  }

  def modulateChunks(
      x: Tensor,
      table: Tensor,
      shift: Int,
      scale: Int,
      out: Tensor
  ): Unit = {
    val (rows, channels, chunks) =
      Ops.checkChunks("modulateChunks", x, table, out, shift, scale)
    launch(
      kernel(elementwiseKernels, "modulate_chunks_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(table)),
      Pointer(pointer(out)),
      I64(rows),
      I32(channels),
      I32(chunks),
      I32(shift),
      I32(scale)
    )
  }

  def gatedAddChunk(x: Tensor, y: Tensor, table: Tensor, gate: Int): Unit = {
    val (rows, channels, chunks) =
      Ops.checkChunks("gatedAddChunk", x, table, y, gate)
    launch(
      kernel(elementwiseKernels, "gated_add_chunk_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(y)),
      Pointer(pointer(table)),
      I64(rows),
      I32(channels),
      I32(chunks),
      I32(gate)
    )
  }

  def rowGatedAdd(x: Tensor, y: Tensor, gate: Tensor): Unit = {
    val (rows, cols) = Ops.checkRowGatedAdd(x, y, gate)
    launch(
      kernel(elementwiseKernels, "row_gated_add_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(pointer(y)),
      Pointer(pointer(gate)),
      I64(rows),
      I32(cols)
    )
  }
}
