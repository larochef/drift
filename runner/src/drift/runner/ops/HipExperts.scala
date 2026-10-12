package drift.runner.ops

import drift.runner.native.KernelArgument.*
import drift.runner.tensor.*

import java.lang.foreign.*

/** The GPU backend's mixtures of experts.
  */
private[ops] trait HipExperts extends HipLinear {

  /** From this many slots, the experts' products group the slots by expert, so
    * that an expert's rows are read once per batch of up to 8 of its slots, not
    * once per slot: a prompt's tokens share experts. A verification's few slots
    * gain nothing from it (`specs/42`, step 10).
    */
  val GroupedExpertsMinimumSlots = 64

  def route(logits: Tensor, ids: Tensor, weights: Tensor): Unit = {
    val (tokens, experts, k) = Ops.checkRoute(logits, ids, weights)
    launch(
      kernel(linearAttentionKernels, "route_topk"),
      (tokens + 7L) / 8,
      256,
      Pointer(pointer(logits)),
      Pointer(pointer(ids)),
      Pointer(pointer(weights)),
      I32(tokens),
      I32(experts),
      I32(k)
    )
  }

  /** Counts the workgroups of matvec_rows_f32_route that finished; the last
    * leaves it at 0.
    */
  private lazy val finishedRouters = {
    val counter = allocate(DType.I32, Shape.of(1))
    zero(counter)
    counter
  }

  override def routeLinear(
      x: Tensor,
      router: Tensor,
      logits: Tensor,
      ids: Tensor,
      weights: Tensor
  ): Unit = {
    val (m, n, k) = Ops.checkLinear(x, router, logits)
    val (_, _, used) = Ops.checkRoute(logits, ids, weights)
    // an F32 router of few rows, a verification's vectors at most: the last
    // workgroup routes, one wave per vector
    if (
      router.dtype == DType.F32 && n <= SplitMaximumRows && k % 4 == 0 &&
      m <= MaxMatVecRows
    )
      launch(
        kernel(matvecKernels, "matvec_rows_f32_route"),
        (n + 3) / 4, // F32_ROWS rows a workgroup
        256,
        Pointer(pointer(router)),
        Pointer(pointer(x)),
        Pointer(pointer(logits)),
        I32(n.toInt),
        I32(k.toInt),
        I32(m.toInt),
        Pointer(pointer(ids)),
        Pointer(pointer(weights)),
        I32(used),
        Pointer(pointer(finishedRouters))
      )
    else super.routeLinear(x, router, logits, ids, weights)
  }

  def expertsLinear(
      x: Tensor,
      weight: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit = {
    val (slots, n, k, divisor) = Ops.checkExperts(x, weight, ids, out)
    require(
      integerX
        .contains(weight.dtype) && weight.dtype.byteSize(k.toLong) % 4 == 0,
      s"no expert kernel for ${weight.dtype} rows of $k"
    )
    lazy val directGroup = directRows(k)
    if (
      slots >= GroupedExpertsMinimumSlots && matrixExperts(weight.dtype, n, k)
    )
      expertsMatrix(x, weight, None, ids, out, slots, n, k, divisor)
    else if (slots >= GroupedExpertsMinimumSlots) {
      val group = rowsPerWave(weight.dtype, n, k)
      val (order, batches, maxBatches) =
        groupSlots(ids, slots, weight.shape.dimensions.head.toInt, 8)
      launch(
        kernel(
          matvecKernels,
          s"matvec_experts_grouped_${weight.dtype.name.toLowerCase}"
        ),
        (maxBatches.toLong * (n / group) + 7) / 8,
        256,
        Pointer(pointer(weight)),
        I64(weight.dtype.byteSize(n.toLong * k)),
        Pointer(pointer(ids)),
        Pointer(order),
        Pointer(batches),
        Pointer(pointer(x)),
        I32(divisor),
        Pointer(pointer(out)),
        I32(n),
        I32(k),
        I32(maxBatches),
        I32(group)
      )
    } else if (direct(weight.dtype, k) && n % (directGroup * 2) == 0)
      launch(
        kernel(
          matvecKernels,
          s"matvec_experts_direct_${weight.dtype.name.toLowerCase}"
        ),
        (slots.toLong * n / (directGroup * 2) + 7) / 8,
        256,
        Pointer(pointer(weight)),
        I64(weight.dtype.byteSize(n.toLong * k)),
        Pointer(pointer(ids)),
        Pointer(pointer(x)),
        I32(divisor),
        Pointer(pointer(out)),
        I32(n),
        I32(k),
        I32(slots),
        I32(directGroup)
      )
    else expertsLinearStaged(x, weight, ids, out, slots, n, k, divisor)
  }

  private def expertsLinearStaged(
      x: Tensor,
      weight: Tensor,
      ids: Tensor,
      out: Tensor,
      slots: Int,
      n: Int,
      k: Int,
      divisor: Int
  ): Unit = {
    val group = rowsPerWave(weight.dtype, n, k)
    launch(
      kernel(
        matvecKernels,
        s"matvec_experts${narrow(weight.dtype, k.toLong * group)}_${weight.dtype.name.toLowerCase}"
      ),
      (slots.toLong * n / group + 7) / 8,
      256,
      Pointer(pointer(weight)),
      I64(weight.dtype.byteSize(n.toLong * k)),
      Pointer(pointer(ids)),
      Pointer(pointer(x)),
      I32(divisor),
      Pointer(pointer(out)),
      I32(n),
      I32(k),
      I32(slots),
      I32(group)
    )
  }

  /** The slots of `ids` grouped by expert (`group_slots`) in scratch memory:
    * their order, and batches of up to 8 slots of one expert, room for as many
    * batches as there can be.
    */
  /** A prompt's ROCmFP4 experts on the matrix units (`matvec.hip`'s
    * `experts_matrix`): slots grouped by expert in batches of 16, tiles of 16
    * rows.
    */
  private def matrixExperts(dtype: DType, n: Int, k: Int): Boolean =
    (dtype == RocmFp4.Fast || dtype == RocmFp4.Dual) &&
      k % DirectBlockElements == 0 && n % 16 == 0

  private def expertsMatrix(
      x: Tensor,
      gate: Tensor,
      up: Option[Tensor],
      ids: Tensor,
      out: Tensor,
      slots: Int,
      n: Int,
      k: Int,
      divisor: Int
  ): Unit = {
    val (order, batches, maxBatches) =
      groupSlots(ids, slots, gate.shape.dimensions.head.toInt, 16)
    launch(
      kernel(
        matvecKernels,
        s"matvec_experts_matrix_${gate.dtype.name.toLowerCase}"
      ),
      (maxBatches.toLong * (n / 16) + 7) / 8,
      256,
      Pointer(pointer(gate)),
      Pointer(up.fold(MemorySegment.NULL)(pointer)),
      I64(gate.dtype.byteSize(n.toLong * k)),
      Pointer(pointer(ids)),
      Pointer(order),
      Pointer(batches),
      Pointer(pointer(x)),
      I32(divisor),
      Pointer(pointer(out)),
      I32(n),
      I32(k),
      I32(maxBatches)
    )
  }

  private def groupSlots(
      ids: Tensor,
      slots: Int,
      experts: Int,
      batchSize: Int
  ): (MemorySegment, MemorySegment, Int) = {
    require(experts <= 1024, s"$experts experts; grouping takes 1024 at most")
    val maxBatches = slots / batchSize + math.min(experts, slots)
    val order = scratch(0, 4L * slots + 8L * maxBatches)
    val batches = offset(order, 4L * slots)
    launch(
      kernel(matvecKernels, "group_slots"),
      1,
      1024,
      Pointer(pointer(ids)),
      I32(slots),
      I32(experts),
      Pointer(order),
      Pointer(batches),
      I32(maxBatches),
      I32(batchSize)
    )
    (order, batches, maxBatches)
  }

  def moeCombine(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Option[Tensor],
      out: Tensor
  ): Unit = {
    Ops.requireF32(
      "moeCombine",
      (Seq(expertOut, weights, out) ++ residual ++ shared.toSeq.flatMap(e =>
        Seq(e.output, e.input, e.router)
      ))*
    )
    val Seq(tokens, hidden) = out.shape.dimensions
    val k = weights.shape.elementCount / tokens
    def optional(tensor: SharedExpert => Tensor) = Pointer(
      shared.fold(MemorySegment.NULL)(e => pointer(tensor(e)))
    )
    launch(
      kernel(linearAttentionKernels, "moe_combine"),
      tokens * 8, // SLICES in the kernel
      256,
      Pointer(pointer(expertOut)),
      Pointer(pointer(weights)),
      optional(_.output),
      optional(_.input),
      optional(_.router),
      Pointer(residual.fold(MemorySegment.NULL)(pointer)),
      Pointer(pointer(out)),
      I32(k.toInt),
      I32(hidden.toInt)
    )
  }

  /** matvec.hip's `moe` counters: zero between launches. */
  private lazy val moeCounters = {
    val counters = allocate(DType.I32, Shape.of(512))
    zero(counters)
    counters
  }

  /** ROCmFP4 experts of a decode step: the routing, then the rest in one launch
    * (`matvec.hip`'s `moe_rocmfp4`), which computes every part as the separate
    * launches do, where these would take the same kernels. A verification's
    * tokens take the separate launches, which measured faster for them.
    */
  override def expertOutputs(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations
  ): Unit =
    if (!mixtureInOneLaunch(x, weights, into))
      super.expertOutputs(x, weights, into)
    else mixture(x, weights, into, None)

  override def mixtureNorm(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations,
      residual: Tensor,
      norm: RowNorm
  ): Unit =
    if (!mixtureInOneLaunch(x, weights, into) || residual.shape.last < 1024)
      super.mixtureNorm(x, weights, into, residual, norm)
    else {
      Ops.requireF32("mixtureNorm", residual, norm.out, weights.sharedRouter)
      Ops.checkRowWise("mixtureNorm", residual, norm.out, norm.weight)
      mixture(x, weights, into, Some((residual, norm)))
    }

  /** Whether `moe_rocmfp4` computes these experts as the separate launches
    * would.
    */
  private def mixtureInOneLaunch(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations
  ): Boolean = {
    val dtype = weights.gate.dtype
    val hidden = weights.router.shape.last
    val intermediate = weights.gate.shape.dimensions(1)
    (dtype == RocmFp4.Fast || dtype == RocmFp4.Dual) &&
    Seq(
      weights.up,
      weights.down,
      weights.sharedGate,
      weights.sharedUp,
      weights.sharedDown
    )
      .forall(_.dtype == dtype) &&
    x.shape.dimensions.head == 1 &&
    hidden % DirectBlockElements == 0 && intermediate % DirectBlockElements == 0 &&
    intermediate <= MoeSharedValues &&
    hidden % (Int8Rows * directRows(intermediate.toInt)) == 0 &&
    rocmFp4Int8(dtype, intermediate) && hidden > SplitMaximumRows &&
    weights.sharedGate.shape == Shape.of(1, intermediate, hidden) &&
    weights.sharedDown.shape == Shape.of(hidden, intermediate)
  }

  /** The routing, then `moe_rocmfp4`: the experts' outputs, and with a residual
    * and a norm their sum and its norm.
    */
  private def mixture(
      x: Tensor,
      weights: ExpertProjections,
      into: ExpertActivations,
      sum: Option[(Tensor, RowNorm)]
  ): Unit = {
    val ExpertProjections(
      router,
      gate,
      up,
      down,
      sharedGate,
      sharedUp,
      sharedDown,
      sharedRouter
    ) =
      weights
    val hidden = router.shape.last
    val intermediate = gate.shape.dimensions(1)
    val tokens = x.shape.dimensions.head
    val k = into.routeWeights.shape.last
    val slots = tokens * k
    val dtype = gate.dtype
    // rows of the down projections walked at once, as the separate launches do
    val group = directRows(intermediate.toInt)
    val ids = into.ids.view(slots)
    Ops.checkExperts(x, gate, ids, into.gated)
    Ops.checkExperts(into.gated, down, ids, into.expertOut)
    Ops.checkExperts(x, sharedGate, into.sharedIds, into.sharedGated)
    Ops.checkLinear(into.sharedGated, sharedDown, into.sharedOut)
    routeLinear(x, router, into.logits, into.ids, into.routeWeights)
    val pairs = intermediate / 2
    def workgroups(waves: Long) = (waves + 7) / 8
    // with a sum: a workgroup per token for its shared expert's gate, and
    // four (SUM_WAVES) for the sum itself
    val grid = sum.fold(0L)(_ => 5 * tokens) + workgroups(tokens * pairs) +
      workgroups(hidden / (Int8Rows * group)) + workgroups(slots * pairs) +
      workgroups(slots * hidden / (2 * group))
    val (target, quantized) = sum.fold(
      (Option.empty[QuantizedValues], Seq.fill(3)(Pointer(MemorySegment.NULL)))
    )((_, norm) => quantizedTarget(norm.out, hidden))
    def summing(part: ((Tensor, RowNorm)) => Tensor) =
      Pointer(sum.fold(MemorySegment.NULL)(s => pointer(part(s))))
    launch(
      kernel(matvecKernels, s"moe_group${group}_${dtype.name.toLowerCase}"),
      grid,
      256,
      (Seq(
        Pointer(pointer(x)),
        Pointer(pointer(into.ids)),
        I32(hidden.toInt),
        I32(tokens.toInt),
        I32(k.toInt),
        Pointer(pointer(gate)),
        Pointer(pointer(up)),
        I64(dtype.byteSize(intermediate * hidden)),
        Pointer(pointer(into.gated)),
        I32(intermediate.toInt),
        Pointer(pointer(sharedGate)),
        Pointer(pointer(sharedUp)),
        Pointer(pointer(into.sharedGated)),
        Pointer(pointer(down)),
        I64(dtype.byteSize(hidden * intermediate)),
        Pointer(pointer(into.expertOut)),
        Pointer(pointer(sharedDown)),
        Pointer(pointer(into.sharedOut)),
        Pointer(pointer(into.routeWeights)),
        Pointer(pointer(sharedRouter)),
        Pointer(offset(pointer(moeCounters), 4L * MoeCounters)),
        summing(_._1),
        summing(_._2.weight),
        summing(_._2.out),
        F32(sum.fold(0f)(_._2.epsilon)),
        F32(sum.fold(0f)(_._2.weightOffset))
      ) ++ quantized :+ Pointer(pointer(moeCounters)))*
    )
    wroteQuantized(target)
  }

  /** `moe_rocmfp4`'s counters, then its tokens' shared expert gates and sums'
    * totals.
    */
  private val MoeCounters = 96

  /** `moe_rocmfp4`'s LDS for the shared expert's quantized input. */
  private val MoeSharedValues = 8192

  override def moeCombineNorm(
      expertOut: Tensor,
      weights: Tensor,
      shared: Option[SharedExpert],
      residual: Tensor,
      norm: RowNorm
  ): Unit = {
    Ops.requireF32(
      "moeCombineNorm",
      (Seq(expertOut, weights, residual, norm.out) ++ shared.toSeq.flatMap(e =>
        Seq(e.output, e.input, e.router)
      ))*
    )
    val rows =
      Ops.checkRowWise("moeCombineNorm", residual, norm.out, norm.weight)
    val hidden = residual.shape.last
    val k = weights.shape.elementCount / rows
    if (hidden < 1024)
      super.moeCombineNorm(expertOut, weights, shared, residual, norm)
    else {
      def optional(tensor: SharedExpert => Tensor) = Pointer(
        shared.fold(MemorySegment.NULL)(e => pointer(tensor(e)))
      )
      val (target, quantized) = quantizedTarget(norm.out, hidden)
      launch(
        kernel(normKernels, "moe_combine_norm"),
        rows,
        1024,
        (Seq(
          Pointer(pointer(expertOut)),
          Pointer(pointer(weights)),
          optional(_.output),
          optional(_.input),
          optional(_.router),
          Pointer(pointer(residual)),
          I32(k.toInt),
          I32(hidden.toInt),
          Pointer(pointer(norm.weight)),
          Pointer(pointer(norm.out)),
          F32(norm.epsilon),
          F32(norm.weightOffset)
        ) ++ quantized)*
      )
      wroteQuantized(target)
    }
  }

  def expertsGatedLinear(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor
  ): Unit = {
    Ops.requireSameShape("expertsGatedLinear", gate, up)
    require(
      gate.dtype == up.dtype,
      s"expertsGatedLinear: gate ${gate.dtype}, up ${up.dtype}"
    )
    val (slots, n, k, divisor) = Ops.checkExperts(x, gate, ids, out)
    require(
      integerX.contains(gate.dtype) && gate.dtype.byteSize(k.toLong) % 4 == 0,
      s"no expert kernel for ${gate.dtype} rows of $k"
    )
    val directly = direct(gate.dtype, k) && n % 2 == 0
    if (slots >= GroupedExpertsMinimumSlots && matrixExperts(gate.dtype, n, k))
      expertsMatrix(x, gate, Some(up), ids, out, slots, n, k, divisor)
    else if (slots >= GroupedExpertsMinimumSlots) {
      val (order, batches, maxBatches) =
        groupSlots(ids, slots, gate.shape.dimensions.head.toInt, 8)
      launch(
        kernel(
          matvecKernels,
          s"matvec_experts_gated_grouped_${gate.dtype.name.toLowerCase}"
        ),
        (maxBatches.toLong * n + 7) / 8,
        256,
        Pointer(pointer(gate)),
        Pointer(pointer(up)),
        I64(gate.dtype.byteSize(n.toLong * k)),
        Pointer(pointer(ids)),
        Pointer(order),
        Pointer(batches),
        Pointer(pointer(x)),
        I32(divisor),
        Pointer(pointer(out)),
        I32(n),
        I32(k),
        I32(maxBatches)
      )
    } else if (directly)
      gatedDirect(x, gate, up, ids, out, slots, n, k, divisor, None)
    else
      launch(
        kernel(
          matvecKernels,
          s"matvec_experts_gated_${gate.dtype.name.toLowerCase}"
        ),
        (slots.toLong * n + 7) / 8,
        256,
        Pointer(pointer(gate)),
        Pointer(pointer(up)),
        I64(gate.dtype.byteSize(n.toLong * k)),
        Pointer(pointer(ids)),
        Pointer(pointer(x)),
        I32(divisor),
        Pointer(pointer(out)),
        I32(n),
        I32(k),
        I32(slots)
      )
  }

  /** matvec_experts_gated_direct: the routed slots, then the shared expert's
    * gate, up and output (`[1, N, K]`, `[tokens, N]`) when given.
    */
  private def gatedDirect(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor,
      slots: Int,
      n: Int,
      k: Int,
      divisor: Int,
      shared: Option[(Tensor, Tensor, Tensor)]
  ): Unit = {
    val sharedSlots = shared.fold(0L)(_._3.shape.dimensions.head)
    def sharedPointer(part: ((Tensor, Tensor, Tensor)) => Tensor) =
      Pointer(shared.fold(MemorySegment.NULL)(s => pointer(part(s))))
    launch(
      kernel(
        matvecKernels,
        s"matvec_experts_gated_direct_${gate.dtype.name.toLowerCase}"
      ),
      ((slots + sharedSlots) * n / 2 + 7) / 8,
      256,
      Pointer(pointer(gate)),
      Pointer(pointer(up)),
      I64(gate.dtype.byteSize(n.toLong * k)),
      Pointer(pointer(ids)),
      Pointer(pointer(x)),
      I32(divisor),
      Pointer(pointer(out)),
      I32(n),
      I32(k),
      I32(slots),
      sharedPointer(_._1),
      sharedPointer(_._2),
      sharedPointer(_._3),
      I32(sharedSlots.toInt)
    )
  }

  override def expertsGatedLinearWithShared(
      x: Tensor,
      gate: Tensor,
      up: Tensor,
      ids: Tensor,
      out: Tensor,
      sharedGate: Tensor,
      sharedUp: Tensor,
      sharedIds: Tensor,
      sharedOut: Tensor
  ): Unit = {
    val (slots, n, k, divisor) = Ops.checkExperts(x, gate, ids, out)
    val (sharedSlots, sharedN, _, sharedDivisor) =
      Ops.checkExperts(x, sharedGate, sharedIds, sharedOut)
    val fused = slots < GroupedExpertsMinimumSlots && direct(gate.dtype, k) &&
      n % 2 == 0 && gate.shape == up.shape && gate.dtype == up.dtype &&
      sharedGate.dtype == gate.dtype && sharedUp.dtype == gate.dtype &&
      sharedGate.shape == sharedUp.shape && sharedN == n && sharedDivisor == 1
    if (!fused)
      super.expertsGatedLinearWithShared(
        x,
        gate,
        up,
        ids,
        out,
        sharedGate,
        sharedUp,
        sharedIds,
        sharedOut
      )
    else
      gatedDirect(
        x,
        gate,
        up,
        ids,
        out,
        slots,
        n,
        k,
        divisor,
        Some((sharedGate, sharedUp, sharedOut))
      )
  }
}
