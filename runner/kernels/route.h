#pragma once
#include <hip/hip_runtime.h>

// Top-k routing of one token by one wave: the k largest of `experts` logits
// (the smallest expert first among equals), their softmax among themselves as
// weights, which is transformers' softmax over all experts, top-k,
// renormalized. `load(i)` reads logit i.
template <typename Load>
__device__ inline void route_token(Load load, int* ids, float* weights, int experts, int k) {
  const int lane = threadIdx.x % 32;
  float v[16];  // experts <= 512
  const int per_lane = (experts + 31) / 32;
  for (int i = 0; i < 16; ++i) v[i] = (i < per_lane && lane + 32 * i < experts) ? load(lane + 32 * i) : -INFINITY;
  float chosen[16];  // k <= 16
  for (int s = 0; s < k; ++s) {
    float best = -INFINITY;
    int best_expert = 1 << 30;
    for (int i = 0; i < 16; ++i)
      if (v[i] > best) {
        best = v[i];
        best_expert = lane + 32 * i;
      }
    for (int offset = 16; offset > 0; offset /= 2) {
      const float other = __shfl_xor(best, offset, 32);
      const int other_expert = __shfl_xor(best_expert, offset, 32);
      if (other > best || (other == best && other_expert < best_expert)) {
        best = other;
        best_expert = other_expert;
      }
    }
    chosen[s] = best;
    if (lane == 0) ids[s] = best_expert;
    for (int i = 0; i < 16; ++i)
      if (lane + 32 * i == best_expert) v[i] = -INFINITY;
  }
  if (lane == 0) {
    float sum = 0.0f;
    for (int s = 0; s < k; ++s) sum += expf(chosen[s] - chosen[0]);
    for (int s = 0; s < k; ++s) weights[s] = expf(chosen[s] - chosen[0]) / sum;
  }
}

