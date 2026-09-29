#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "quantize.h"

// Elementwise kernels over n contiguous values, as grid-stride loops so any
// grid covers any n. The activation codes are Activation.code in Ops.scala.

#define FOR_EACH(i, n)                                                                       \
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n);             \
       i += (long long)gridDim.x * blockDim.x)

extern "C" __global__ void add_f32(const float* a, const float* b, float* out, long long n) {
  FOR_EACH(i, n) out[i] = a[i] + b[i];
}

extern "C" __global__ void mul_f32(const float* a, const float* b, float* out, long long n) {
  FOR_EACH(i, n) out[i] = a[i] * b[i];
}

extern "C" __global__ void scale_f32(const float* x, float factor, float* out, long long n) {
  FOR_EACH(i, n) out[i] = x[i] * factor;
}

// out = x + row, row repeated over every row of `cols` values (a bias).
extern "C" __global__ void add_row_f32(const float* x, const float* row, float* out, long long n, int cols) {
  FOR_EACH(i, n) out[i] = x[i] + row[i % cols];
}

__device__ inline float activate(float v, int kind) {
  switch (kind) {
    case 0: return v / (1.0f + expf(-v));  // SiLU
    case 1: {                              // GELU, tanh approximation
      const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
      return 0.5f * v * (1.0f + tanhf(inner));
    }
    case 2: return 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));  // GELU, erf
    default: return 1.0f / (1.0f + expf(-v));                           // sigmoid
  }
}

// out = act(scale * x)
extern "C" __global__ void activation_f32(const float* x, float* out, long long n, int kind, float scale) {
  FOR_EACH(i, n) out[i] = activate(scale * x[i], kind);
}

// out = act(gate) * up: SwiGLU with SiLU, GeGLU with a GELU. With xq, out is
// quantized too (quantize.h: whole waves of consecutive values, n a multiple
// of 32), for the int8 product it feeds.
extern "C" __global__ void gated_f32(const float* gate, const float* up, float* out, long long n, int kind,
                                     int8_t* xq, float* xd, float* xs) {
  FOR_EACH(i, n) {
    const float value = activate(gate[i], kind) * up[i];
    out[i] = value;
    if (xq) quantize_block(value, i / 32, xq, xd, xs);
  }
}

// Storage conversions; codes are Convert.code in HipOps.scala.
extern "C" __global__ void convert(const void* in, void* out, long long n, int kind) {
  FOR_EACH(i, n) {
    switch (kind) {
      case 0: reinterpret_cast<__half*>(out)[i] = __float2half(reinterpret_cast<const float*>(in)[i]); break;
      case 1: reinterpret_cast<__hip_bfloat16*>(out)[i] = __float2bfloat16(reinterpret_cast<const float*>(in)[i]); break;
      case 2: reinterpret_cast<float*>(out)[i] = __half2float(reinterpret_cast<const __half*>(in)[i]); break;
      case 4: {  // fp8 E4M3 "fn": bias 7, no infinities, NaN at S.1111.111
        const unsigned char code = reinterpret_cast<const unsigned char*>(in)[i];
        const int exponent = (code >> 3) & 0xf, mantissa = code & 0x7;
        const float magnitude = exponent == 0xf && mantissa == 0x7 ? __builtin_nanf("")
                                : exponent == 0 ? ldexpf((float)mantissa, -9)
                                                : ldexpf((float)(8 + mantissa), exponent - 10);
        reinterpret_cast<float*>(out)[i] = (code & 0x80) ? -magnitude : magnitude;
        break;
      }
      default: reinterpret_cast<float*>(out)[i] = __bfloat162float(reinterpret_cast<const __hip_bfloat16*>(in)[i]); break;
    }
  }
}

// Rows ids[i] of an F32 table of any width: one thread per value.
extern "C" __global__ void gather_rows_f32(const float* table, const int* ids, float* out, int count, int cols) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)count * cols) return;
  out[index] = table[(long long)ids[index / cols] * cols + index % cols];
}

// AdaLN's modulation, x [n / cols][cols]: out = x * (1 + scale) + shift, the
// same scale and shift vectors for every row.
extern "C" __global__ void modulate_f32(const float* x, const float* scale, const float* shift, float* out,
                                        long long n, int cols) {
  FOR_EACH(i, n) {
    const int c = i % cols;
    out[i] = x[i] * (1.0f + scale[c]) + shift[c];
  }
}

// A gated residual: x += gate * y, the same gate vector for every row.
extern "C" __global__ void gated_add_f32(float* x, const float* y, const float* gate, long long n, int cols) {
  FOR_EACH(i, n) x[i] += gate[i % cols] * y[i];
}

// AdaLN from a table of chunks per row: x and out [rows][C], table
// [rows][chunks * C] (chunk k of row r at r * chunks * C + k * C);
// out = x * (1 + chunk `scale`) + chunk `shift`.
extern "C" __global__ void modulate_chunks_f32(const float* x, const float* table, float* out, long long rows,
                                               int channels, int chunks, int shift, int scale) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * channels) return;
  const long long r = index / channels;
  const int c = index % channels;
  const float* row = table + r * chunks * channels;
  out[index] = x[index] * (1.0f + row[scale * channels + c]) + row[shift * channels + c];
}

// x += table[r][gate chunk][c] * y, the same layout.
extern "C" __global__ void gated_add_chunk_f32(float* x, const float* y, const float* table, long long rows,
                                               int channels, int chunks, int gate) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * channels) return;
  const long long r = index / channels;
  x[index] += table[r * chunks * channels + gate * channels + index % channels] * y[index];
}

// x[r][c] += gate[r] * y[r][c]: one gate per row.
extern "C" __global__ void row_gated_add_f32(float* x, const float* y, const float* gate, long long rows,
                                             int cols) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * cols) return;
  x[index] += gate[index / cols] * y[index];
}
