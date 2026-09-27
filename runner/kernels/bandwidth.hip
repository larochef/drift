#include <hip/hip_runtime.h>

// Reads n4 float4 values and writes one partial sum per block: a pure streaming
// read, to measure how fast the GPU reads a given kind of memory.
extern "C" __global__ void read_sum_f32x4(const float4* x, long long n4, float* partials) {
  __shared__ float partial[256];
  const long long stride = (long long)gridDim.x * blockDim.x;
  float sum = 0.0f;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 v = x[i];
    sum += v.x + v.y + v.z + v.w;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = partial[0];
}
