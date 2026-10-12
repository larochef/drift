package drift.runner.ops

import drift.runner.native.HipRuntime

/** The GPU backend: each operation launches kernels of `runner/kernels` on the
  * default stream. Every allocation is freed when the backend closes.
  */
final class HipOps(
    protected val hip: HipRuntime,
    protected val inputs: MatVecInputs
) extends HipBase
    with HipMemory
    with HipElementwise
    with HipNormalisation
    with HipLinear
    with HipExperts
    with HipAttention
    with HipConvolution
    with HipLayout
    with HipSequence
