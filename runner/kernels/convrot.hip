#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

// ConvRot's rotated linears (ConvRot.scala), decoded to BF16 for the GEMM path
// from the codes where the file holds them: nothing is copied.

#define GROUP 256

// fp8 E4M3 "fn": bias 7, no infinities, NaN at S.1111.111 (elementwise.hip).
__device__ inline float fp8_e4m3(unsigned char code) {
  const int exponent = (code >> 3) & 0xf, mantissa = code & 0x7;
  const float magnitude = exponent == 0xf && mantissa == 0x7 ? __builtin_nanf("")
                          : exponent == 0 ? ldexpf((float)mantissa, -9)
                                          : ldexpf((float)(8 + mantissa), exponent - 10);
  return (code & 0x80) ? -magnitude : magnitude;
}

// The rotation's sums in the order ConvRot.rotate takes them, a product never
// fused into one: the decode is the CPU's bit for bit.
#pragma clang fp contract(off)

// One factor of the rotation on four values of v, `s` apart from `i`.
#define BUTTERFLY(v, i, s)                                                                   \
  {                                                                                          \
    const float a = v[i], b = v[(i) + (s)], c = v[(i) + 2 * (s)], d = v[(i) + 3 * (s)];      \
    const float half = (a + b + c + d) * 0.5f;                                               \
    v[i] = half - d;                                                                         \
    v[(i) + (s)] = half - c;                                                                 \
    v[(i) + 2 * (s)] = half - b;                                                             \
    v[(i) + 3 * (s)] = half - a;                                                             \
  }

// Two factors on sixteen values in registers: those 1 apart, then those 4 apart.
#define ROTATE16(v)                                                                          \
  BUTTERFLY(v, 0, 1) BUTTERFLY(v, 4, 1) BUTTERFLY(v, 8, 1) BUTTERFLY(v, 12, 1)               \
  BUTTERFLY(v, 0, 4) BUTTERFLY(v, 1, 4) BUTTERFLY(v, 2, 4) BUTTERFLY(v, 3, 4)

// W as BF16, row-major, for the GEMM path: a workgroup sixteen groups of 256
// columns, sixteen threads a group. A thread rotates sixteen neighbours (the
// factors 1 and 4 apart), leaves them in shared memory, then rotates the
// sixteen values 16 apart that the group's threads left (the factors 16 and
// 64 apart) and writes them. PREPARE declares what a thread's sixteen values
// share, from `element`, the first of them; VALUE(i) is value i, scaled.
#define DEQUANTIZE(PREPARE, VALUE)                                                           \
  __shared__ float shared[16 * GROUP];                                                       \
  const int group = threadIdx.x / 16, slot = threadIdx.x % 16;                               \
  for (long long first = (long long)blockIdx.x * 16; first < blocks;                         \
       first += (long long)gridDim.x * 16) {                                                 \
    const long long block = first + group;                                                   \
    float v[16];                                                                             \
    if (block < blocks) {                                                                    \
      const long long element = block * GROUP + slot * 16;                                   \
      PREPARE                                                                                \
      _Pragma("unroll") for (int i = 0; i < 16; ++i) v[i] = VALUE(i);                        \
      ROTATE16(v)                                                                            \
      _Pragma("unroll") for (int i = 0; i < 16; ++i) shared[group * GROUP + slot * 16 + i] = v[i]; \
    }                                                                                        \
    __syncthreads();                                                                         \
    if (block < blocks) {                                                                    \
      _Pragma("unroll") for (int i = 0; i < 16; ++i) v[i] = shared[group * GROUP + i * 16 + slot]; \
      ROTATE16(v)                                                                            \
      _Pragma("unroll") for (int i = 0; i < 16; ++i)                                         \
        out[block * GROUP + i * 16 + slot] = __float2bfloat16(v[i]);                         \
    }                                                                                        \
    __syncthreads();                                                                         \
  }

// int8_tensorwise: codes [blocks × 256 / cols][cols] as stored, at any
// alignment, times one scale a row or one for all.
#define I8_PREPARE const float scale = scales[scale_count == 1 ? 0 : element / cols];
#define I8_VALUE(i) ((float)codes[element + (i)] * scale)

extern "C" __global__ void __launch_bounds__(GROUP)
    convrot_dequantize_bf16_i8(const int8_t* codes, const float* scales, int scale_count,
                               __hip_bfloat16* out, long long blocks, int cols) {
  DEQUANTIZE(I8_PREPARE, I8_VALUE)
}

// asym_w4a8_int8: two 4-bit codes a byte (the even column in the low nibble),
// a code being a level of the codebook times its group's relative scale (fp8,
// one a group_size columns, a multiple of 16), rounded to the int8 the 4-bit
// kernel multiplies by, times one scale a row.
#define W4A8_PREPARE                                                                         \
  const long long row = element / cols;                                                      \
  const float relative_scale =                                                               \
      fp8_e4m3(relative[row * (cols / group_size) + (element % cols) / group_size]);         \
  const float scale = channel[row];                                                          \
  const uint8_t* pairs = codes + element / 2;
#define W4A8_VALUE(i)                                                                        \
  (fmaxf(-127.0f, fminf(127.0f, rintf(codebook[(pairs[(i) / 2] >> (4 * ((i) & 1))) & 0xf] * relative_scale))) * scale)

extern "C" __global__ void __launch_bounds__(GROUP)
    convrot_dequantize_bf16_w4a8(const uint8_t* codes, const float* codebook, const uint8_t* relative,
                                 const float* channel, __hip_bfloat16* out, long long blocks, int cols,
                                 int group_size) {
  DEQUANTIZE(W4A8_PREPARE, W4A8_VALUE)
}
