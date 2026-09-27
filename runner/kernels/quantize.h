#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

// x quantized per 32 values, as the int8 products meet it (matvec.hip): one
// wave holds a block, lane l its value l, and every lane of the wave calls.
// xq = round(x / xd), xd = max|x| / 127, xs = xd * sum(xq). quantize_x runs it
// on its own; the kernels that write a product's input (the norms) run it on
// the values they write, so the product needs no launch of its own.
__device__ inline void quantize_block(float value, long long block, int8_t* xq, float* xd, float* xs) {
  const int lane = threadIdx.x % 32;
  float amax = fabsf(value);
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
  const float d = amax / 127.0f;
  const int q = d == 0.0f ? 0 : int(roundf(value / d));
  xq[block * 32 + lane] = int8_t(q);
  int sum = q;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset, 32);
  if (lane == 0) {
    xd[block] = d;
    xs[block] = d * float(sum);
  }
}

// Where the norms write x quantized: codes, scales and sums, null for none.
struct Quantized {
  int8_t* xq;
  float* xd;
  float* xs;
};
