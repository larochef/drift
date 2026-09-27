#pragma once
#include <hip/hip_runtime.h>

// Across the block: each wave's lanes by shuffles, then the waves' results
// through `partial` (one slot per wave).
__device__ inline float block_sum(float value, float* partial) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) value += __shfl_xor(value, offset);
  if (threadIdx.x % warpSize == 0) partial[threadIdx.x / warpSize] = value;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < blockDim.x / warpSize; ++w) total += partial[w];
  __syncthreads();  // partial is reused by the next reduction
  return total;
}

__device__ inline float block_max(float value, float* partial) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) value = fmaxf(value, __shfl_xor(value, offset));
  if (threadIdx.x % warpSize == 0) partial[threadIdx.x / warpSize] = value;
  __syncthreads();
  float total = -INFINITY;
  for (int w = 0; w < blockDim.x / warpSize; ++w) total = fmaxf(total, partial[w]);
  __syncthreads();
  return total;
}
