#pragma once
#include <hip/hip_runtime.h>

// The mixture of experts' output (moe_combine, moe_combine_norm):
// out[t] = residual[t] + sum_j weights[t*k + j] * expert_out[t*k + j]
//          + sigmoid(x[t] . shared_router) * shared[t]

// The shared expert's gate: its logit reduced by the first 256 threads, in the
// same order whatever the block's size (every thread reaches the barriers).
__device__ inline float shared_expert_gate(const float* x, const float* shared_router, int hidden,
                                           float* partial /* [256] */) {
  if (threadIdx.x < 256) {
    float dot = 0.0f;
    for (int c = threadIdx.x; c < hidden; c += 256) dot += x[c] * shared_router[c];
    partial[threadIdx.x] = dot;
  }
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  return 1.0f / (1.0f + expf(-partial[0]));
}

// Element c of token t's output.
__device__ inline float combined(const float* expert_out, const float* weights, const float* shared, float gate,
                                 const float* residual, long long t, int c, int k, int hidden) {
  float sum = residual ? residual[t * hidden + c] : 0.0f;
  for (int j = 0; j < k; ++j) sum += weights[t * k + j] * expert_out[(t * k + j) * hidden + c];
  if (shared) sum += gate * shared[t * hidden + c];
  return sum;
}
