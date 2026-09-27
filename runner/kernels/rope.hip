#include <hip/hip_runtime.h>

#include "rope.h"

// Rotary position embedding over x [tokens, heads, head_dim] (rope.h).
extern "C" __global__ void rope_f32(const float* x, const int* positions, float* out, int tokens, int heads,
                                    int head_dim, int rotary, float theta, int layout, int sections, int s0,
                                    int s1, int s2, int s3) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long values = (long long)tokens * heads * head_dim;
  if (index >= values) return;
  const int d = index % head_dim;
  const int t = index / ((long long)heads * head_dim);
  out[index] = rotated(x + index - d, d, positions, t, tokens, rotary, theta, layout, sections, s0, s1, s2, s3);
}

// Rotary embedding from precomputed angles: pair m of each head, values
// (2m, 2m + 1), turns by the angle whose cosine and sine are cosines[t][m] and
// sines[t][m], for m < pairs; the head's other values are copied.
extern "C" __global__ void rope_table_f32(const float* x, const float* cosines, const float* sines, float* out,
                                          int tokens, int heads, int head_dim, int pairs) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)tokens * heads * head_dim) return;
  const int d = index % head_dim;
  const long long t = index / ((long long)heads * head_dim);
  const int m = d / 2;
  if (m >= pairs) {
    out[index] = x[index];
    return;
  }
  const float c = cosines[t * pairs + m], s = sines[t * pairs + m];
  const float a = x[index - d % 2], b = x[index - d % 2 + 1];
  out[index] = d % 2 == 0 ? a * c - b * s : a * s + b * c;
}
