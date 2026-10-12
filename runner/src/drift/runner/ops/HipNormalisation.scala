package drift.runner.ops

import drift.runner.native.*
import drift.runner.native.KernelArgument.*
import drift.runner.tensor.*

import java.lang.foreign.*

/** The GPU backend's normalisations and activations; the norms write their
  * result quantized too when an int8 product may read it.
  */
private[ops] trait HipNormalisation extends HipBase {

  def scaledActivation(
      kind: Activation,
      inputScale: Float,
      x: Tensor,
      out: Tensor
  ): Unit = {
    Ops.checkElementwise("activation", x, out)
    elementwise(
      "activation_f32",
      out.shape.elementCount,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I64(out.shape.elementCount),
      I32(kind.code),
      F32(inputScale)
    )
  }

  def gated(kind: Activation, gate: Tensor, up: Tensor, out: Tensor): Unit = {
    Ops.checkElementwise("gated", gate, up, out)
    val (target, quantized) = quantizedTarget(out, out.shape.last)
    elementwise(
      "gated_f32",
      out.shape.elementCount,
      (Seq(
        Pointer(pointer(gate)),
        Pointer(pointer(up)),
        Pointer(pointer(out)),
        I64(out.shape.elementCount),
        I32(kind.code)
      ) ++ quantized)*
    )
    wroteQuantized(target)
  }

  /** One workgroup of 256 per row. */
  private def rowWise(
      name: String,
      rows: Long,
      arguments: KernelArgument*
  ): Unit =
    launch(kernel(normKernels, name), rows, 256, arguments*)

  def rmsNorm(
      x: Tensor,
      weight: Tensor,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit = {
    val rows = Ops.checkRowWise("rmsNorm", x, out, weight)
    // narrow rows (a pixel's channels), millions of them: a thread each
    if (x.shape.last <= 32)
      return launch(
        kernel(normKernels, "rms_norm_narrow_f32"),
        (rows + 255) / 256,
        256,
        Pointer(pointer(x)),
        Pointer(pointer(weight)),
        Pointer(pointer(out)),
        I64(rows),
        I32(x.shape.last.toInt),
        F32(epsilon),
        F32(weightOffset)
      )
    normRows(x, None, weight, epsilon, weightOffset, out, rows)
  }

  /** rms_norm_f32 over `rows`, `y` added to x first when given, the result
    * quantized too when int8 products may read it. A long row (the residual
    * stream) is latency-bound: more threads, for every row count, so a row's
    * sum does not depend on how many rows come with it (a verification's tokens
    * norm as a decode step's token does).
    */
  private def normRows(
      x: Tensor,
      y: Option[Tensor],
      weight: Tensor,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor,
      rows: Long
  ): Unit = {
    val cols = x.shape.last
    val (target, quantized) = quantizedTarget(out, cols)
    launch(
      kernel(normKernels, "rms_norm_f32"),
      rows,
      if (cols >= 1024) 1024 else 256,
      (Seq(
        Pointer(pointer(x)),
        Pointer(y.fold(MemorySegment.NULL)(pointer)),
        Pointer(pointer(weight)),
        Pointer(pointer(out)),
        I32(cols.toInt),
        F32(epsilon),
        F32(weightOffset),
        I32(1)
      ) ++ quantized)*
    )
    wroteQuantized(target)
  }

  override def addRmsNorm(x: Tensor, y: Tensor, norm: RowNorm): Unit = {
    Ops.checkElementwise("addRmsNorm", x, y)
    val rows = Ops.checkRowWise("addRmsNorm", x, norm.out, norm.weight)
    if (x.shape.last <= 32) super.addRmsNorm(x, y, norm)
    else
      normRows(
        x,
        Some(y),
        norm.weight,
        norm.epsilon,
        norm.weightOffset,
        norm.out,
        rows
      )
  }

  def groupRmsNorm(
      x: Tensor,
      weight: Tensor,
      groups: Int,
      epsilon: Float,
      weightOffset: Float,
      out: Tensor
  ): Unit = {
    val rows = Ops.checkRowWise("groupRmsNorm", x, out, weight)
    require(x.shape.last % groups == 0, s"${x.shape} in $groups groups")
    rowWise(
      "rms_norm_f32",
      rows * groups,
      Pointer(pointer(x)),
      Pointer(MemorySegment.NULL),
      Pointer(pointer(weight)),
      Pointer(pointer(out)),
      I32((x.shape.last / groups).toInt),
      F32(epsilon),
      F32(weightOffset),
      I32(groups),
      Pointer(MemorySegment.NULL),
      Pointer(MemorySegment.NULL),
      Pointer(MemorySegment.NULL)
    )
  }

  /** Rows each block of a group norm's partial sums covers. */
  private val GroupRowsPerBlock = 1024

  def groupNorm(
      x: Tensor,
      groups: Int,
      weight: Tensor,
      bias: Tensor,
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val (rows, channels) = Ops.checkGroupNorm(x, groups, weight, bias, out)
    val blocks = ((rows + GroupRowsPerBlock - 1) / GroupRowsPerBlock).toInt
    val partials = scratch(0, 4L * groups * blocks)
    val stats = scratch(4L * groups * blocks, 8L * groups)
    Seq(MemorySegment.NULL, stats).zipWithIndex.foreach { (center, mode) =>
      launch(
        kernel(normKernels, "group_partials_f32"),
        groups.toLong * blocks,
        256,
        Pointer(pointer(x)),
        Pointer(center),
        Pointer(partials),
        I32(rows.toInt),
        I32(channels),
        I32(groups),
        I32(blocks),
        I32(GroupRowsPerBlock)
      )
      launch(
        kernel(normKernels, "group_finish_f32"),
        groups,
        256,
        Pointer(partials),
        Pointer(stats),
        I32(groups),
        I32(blocks),
        I64(rows * (channels / groups)),
        F32(epsilon),
        I32(mode)
      )
    }
    launch(
      kernel(normKernels, "group_apply_f32"),
      (x.shape.elementCount + 255) / 256,
      256,
      Pointer(pointer(x)),
      Pointer(stats),
      Pointer(pointer(weight)),
      Pointer(pointer(bias)),
      Pointer(pointer(out)),
      I64(x.shape.elementCount),
      I32(channels),
      I32(groups)
    )
  }

  def layerNorm(
      x: Tensor,
      weight: Option[Tensor],
      bias: Option[Tensor],
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val rows = Ops.checkRowWise("layerNorm", x, out, (weight ++ bias).toSeq*)
    def optional(tensor: Option[Tensor]) = Pointer(
      tensor.fold(MemorySegment.NULL)(pointer)
    )
    rowWise(
      "layer_norm_f32",
      rows,
      Pointer(pointer(x)),
      optional(weight),
      optional(bias),
      Pointer(pointer(out)),
      I32(x.shape.last.toInt),
      F32(epsilon)
    )
  }

  def softmax(x: Tensor, scale: Float, out: Tensor): Unit = {
    val rows = Ops.checkRowWise("softmax", x, out)
    rowWise(
      "softmax_f32",
      rows,
      Pointer(pointer(x)),
      Pointer(pointer(out)),
      I32(x.shape.last.toInt),
      F32(scale)
    )
  }

  def gatedRmsNorm(
      x: Tensor,
      weight: Tensor,
      gate: Tensor,
      gateActivation: Activation,
      epsilon: Float,
      out: Tensor
  ): Unit = {
    val rows = Ops.checkRowWise("gatedRmsNorm", x, out, weight)
    Ops.requireSameShape("gatedRmsNorm", x, gate)
    require(
      Set(Activation.Silu, Activation.Sigmoid).contains(gateActivation),
      s"gatedRmsNorm gates by SiLU or sigmoid, not $gateActivation"
    )
    // the heads' rows side by side are a product's input rows
    val (target, quantized) = quantizedTarget(out, out.shape.last)
    rowWise(
      "gated_rms_norm_f32",
      rows,
      (Seq(
        Pointer(pointer(x)),
        Pointer(pointer(weight)),
        Pointer(pointer(gate)),
        Pointer(pointer(out)),
        I32(x.shape.last.toInt),
        F32(epsilon),
        I32(if (gateActivation == Activation.Sigmoid) 1 else 0)
      ) ++ quantized)*
    )
    wroteQuantized(target)
  }
}
