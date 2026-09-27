package drift.runner.state

import drift.runner.tensor.{DType, Tensor}

/** A paged key-value cache (`specs/42`, Chat requirements): pages of `pageSize`
  * tokens, `pageSize` a multiple of 16. Keys are
  * `[pages, pageSize, kvHeads, headDimension]`; values are transposed within a
  * page, `[pages, kvHeads, headDimension, pageSize]`, so that the attention
  * kernel reads both as contiguous runs. A sequence's page table (I32) maps
  * `position / pageSize` to a page; `PageAllocator` hands pages out.
  */
final case class KvCache(keys: Tensor, values: Tensor, pageSize: Int) {
  require(
    pageSize % 16 == 0,
    s"pages of $pageSize tokens: not a multiple of 16"
  )
  require(
    keys.dtype == DType.F16 && values.dtype == DType.F16,
    "the cache is F16"
  )

  val Seq(pages, _, kvHeads, headDimension) = keys.shape.dimensions.map(_.toInt)
}
