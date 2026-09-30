#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "combine.h"
#include "quantize.h"
#include "route.h"

// y[v, row] = sum_k W[row, k] * x[v, k] for up to 8 vectors v, W row-major in
// any storage type, y float; and W dequantized to f16 for the GEMM path.
//
// One wave (32 lanes) per row. The wave walks its row in trips of about 4 KB:
// it copies the trip's blocks into its own LDS slice with coalesced
// non-temporal dword loads (the block sizes, 17 to 210 bytes, are rarely dword
// aligned, so lanes cannot load their own block directly), then each lane
// decodes its 32 weights of a block (LANES lanes per block) once and meets
// every vector with them. Rows must be a multiple of 4 bytes long, which every
// trip start then is too.
//
// Two kinds of x:
// - Integer types (Q4_0, Q8_0, Q4_K, Q6_K, ROCmFP4) take x quantized per 32
//   values by quantize_x: int8 codes xq, a scale xd and xs = xd * sum(xq).
//   Their weights become int8 and meet xq four at a time in v_dot4 (sudot4),
//   as llama.cpp's MMVQ does. The result is exactly sum_k w_k * x̂_k, x̂ the
//   quantized x: its error is at most sum_k |w_k| * xd/2, which the tests check.
// - Dense types (F32, F16, BF16) take x as float.
//
// The decoders repeat runner/src/drift/runner/tensor/*.scala.

constexpr int WAVE = 32;
constexpr int WAVES_PER_BLOCK = 8;
constexpr int MAX_VECTORS = 8;

// N words of LDS starting at any byte `offset`: N + 1 aligned reads joined by
// v_alignbyte. The staged slice has one spare word for the last read.
template <int N>
__device__ inline void words_at(const uint32_t* staged, int offset, uint32_t* out) {
  const uint32_t* w = staged + (offset >> 2);
  const int shift = offset & 3;
  uint32_t current = w[0];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const uint32_t next = w[i + 1];
    out[i] = __builtin_amdgcn_alignbyte(next, current, shift);
    current = next;
  }
}
__device__ inline int byte_at(const uint32_t* staged, int offset) {
  return reinterpret_cast<const uint8_t*>(staged)[offset];
}
__device__ inline float half_at(const uint32_t* staged, int offset) {  // offset even
  return __half2float(*reinterpret_cast<const __half*>(reinterpret_cast<const uint8_t*>(staged) + offset));
}

// sum of the four products of unsigned (u) or signed (s) bytes, plus acc.
__device__ inline int dot_us(uint32_t u, int s, int acc) {
  return __builtin_amdgcn_sudot4(false, int(u), true, s, acc, false);
}
__device__ inline int dot_ss(uint32_t a, int s, int acc) {
  return __builtin_amdgcn_sudot4(true, int(a), true, s, acc, false);
}
__device__ inline int unsigned_byte(uint32_t word, int i) { return int((word >> (8 * i)) & 0xFF); }
__device__ inline int signed_byte(uint32_t word, int i) { return int(int8_t(word >> (8 * i))); }

// The quantized x of one 32-value block.
struct XBlock {
  const int* q;  // eight words of four int8 codes
  float d;       // scale
  float s;       // d * sum of the codes
};

// ---- integer types ------------------------------------------------------------------
// decode(staged, at, part) gives the lane's 32 weights as eight words of bytes
// and their scales; dot meets them with one quantized x block; value(j) expands
// weight j to a float, for dequantization.

struct Q8_0 {
  static constexpr int BYTES = 34, ELEMS = 32, LANES = 1;
  struct Part { uint32_t q[8]; float d; };
  __device__ static Part decode(const uint32_t* staged, int at, int) {
    Part p;
    words_at<8>(staged, at + 2, p.q);
    p.d = half_at(staged, at);
    return p;
  }
  __device__ static float dot(const Part& p, const XBlock& x) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum = dot_ss(p.q[i], x.q[i], sum);
    return p.d * x.d * float(sum);
  }
  __device__ static float value(const Part& p, int j) { return p.d * float(signed_byte(p.q[j / 4], j % 4)); }
};

struct Q4_0 {
  static constexpr int BYTES = 18, ELEMS = 32, LANES = 1;
  struct Part { uint32_t q[8]; float d; };  // nibbles 0..15 as bytes, elements in order
  __device__ static Part decode(const uint32_t* staged, int at, int) {
    Part p;
    uint32_t raw[4];
    words_at<4>(staged, at + 2, raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p.q[i] = raw[i] & 0x0F0F0F0F;
      p.q[i + 4] = (raw[i] >> 4) & 0x0F0F0F0F;
    }
    p.d = half_at(staged, at);
    return p;
  }
  // sum (nibble - 8) x̂ = xd * sum nibble xq - 8 xs
  __device__ static float dot(const Part& p, const XBlock& x) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum = dot_us(p.q[i], x.q[i], sum);
    return p.d * (x.d * float(sum) - 8.0f * x.s);
  }
  __device__ static float value(const Part& p, int j) {
    return p.d * float(unsigned_byte(p.q[j / 4], j % 4) - 8);
  }
};

// Q5_1: d, m, 32 fifth bits, then nibbles as Q4_0's; d q + m. Bit j of qh is
// element j's fifth bit.
struct Q5_1 {
  static constexpr int BYTES = 24, ELEMS = 32, LANES = 1;
  struct Part { uint32_t q[8]; float d, m; };  // five-bit codes as bytes, elements in order
  // four bits of qh to bit 4 of four bytes
  __device__ static uint32_t fifth_bits(uint32_t bits) {
    return ((bits & 1) << 4) | ((bits & 2) << 11) | ((bits & 4) << 18) | ((bits & 8) << 25);
  }
  __device__ static Part decode(const uint32_t* staged, int at, int) {
    Part p;
    uint32_t raw[4], qh[1];
    words_at<4>(staged, at + 8, raw);
    words_at<1>(staged, at + 4, qh);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p.q[i] = (raw[i] & 0x0F0F0F0F) | fifth_bits((qh[0] >> (4 * i)) & 0xF);
      p.q[i + 4] = ((raw[i] >> 4) & 0x0F0F0F0F) | fifth_bits((qh[0] >> (16 + 4 * i)) & 0xF);
    }
    p.d = half_at(staged, at);
    p.m = half_at(staged, at + 2);
    return p;
  }
  // sum (d q + m) x̂ = d xd sum q xq + m xs
  __device__ static float dot(const Part& p, const XBlock& x) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum = dot_us(p.q[i], x.q[i], sum);
    return p.d * x.d * float(sum) + p.m * x.s;
  }
  __device__ static float value(const Part& p, int j) { return p.d * float(unsigned_byte(p.q[j / 4], j % 4)) + p.m; }
};

// IQ4_NL's and IQ4_XS's table of 16 values, four nibbles (one per byte) at a
// time: v_perm picks entries 0–7 and 8–15 by the low three bits, a third
// v_perm between the two by the fourth bit.
__device__ inline uint32_t iq4_values(uint32_t nibbles) {
  const uint32_t index = nibbles & 0x07070707;
  const uint32_t low = __builtin_amdgcn_perm(0xF6EADDCFu, 0xBFAD9881u, index);   // -127 -104 -83 -65 | -49 -35 -22 -10
  const uint32_t high = __builtin_amdgcn_perm(0x71594535u, 0x26190D01u, index);  // 1 13 25 38 | 53 69 89 113
  const uint32_t select = 0x03020100u | ((nibbles & 0x08080808) >> 1);
  return __builtin_amdgcn_perm(high, low, select);
}

// A run of 32 table values from 16 bytes of nibbles (low nibbles, then high),
// scaled by d.
struct Iq4Run {
  struct Part { uint32_t q[8]; float d; };  // signed table values as bytes, elements in order
  __device__ static void values(const uint32_t* staged, int at, Part& p) {
    uint32_t raw[4];
    words_at<4>(staged, at, raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p.q[i] = iq4_values(raw[i] & 0x0F0F0F0F);
      p.q[i + 4] = iq4_values((raw[i] >> 4) & 0x0F0F0F0F);
    }
  }
  __device__ static float dot(const Part& p, const XBlock& x) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum = dot_ss(p.q[i], x.q[i], sum);
    return p.d * x.d * float(sum);
  }
  __device__ static float value(const Part& p, int j) { return p.d * float(signed_byte(p.q[j / 4], j % 4)); }
};

// IQ4_NL: d, then one run.
struct IQ4_NL : Iq4Run {
  static constexpr int BYTES = 18, ELEMS = 32, LANES = 1;
  __device__ static Part decode(const uint32_t* staged, int at, int) {
    Part p;
    values(staged, at + 2, p);
    p.d = half_at(staged, at);
    return p;
  }
};

// IQ4_XS: d, scales_h (u16), scales_l[4], eight runs; run c scaled by
// d (ls - 32), ls of six bits (four from scales_l, two from scales_h).
struct IQ4_XS : Iq4Run {
  static constexpr int BYTES = 136, ELEMS = 256, LANES = 8;
  __device__ static Part decode(const uint32_t* staged, int at, int c) {
    Part p;
    values(staged, at + 8 + 16 * c, p);
    const int low = (byte_at(staged, at + 4 + c / 2) >> (4 * (c % 2))) & 0xF;
    const int high = ((byte_at(staged, at + 2 + c / 4) >> (2 * (c % 4))) & 3);
    p.d = half_at(staged, at) * float((low | (high << 4)) - 32);
    return p;
  }
};

// (8 + mantissa) 2^(exponent - 11) built as a float's bits, mantissa 2^-10 when
// the exponent is 0.
__device__ inline float rocmfp4_half_scale(int e) {
  if (e > 0x7e) return 0.0f;
  const int exponent = e >> 3, mantissa = e & 7;
  return exponent == 0 ? float(mantissa) * 0x1p-10f
                       : __uint_as_float(uint32_t(exponent + 119) << 23 | uint32_t(mantissa) << 20);
}

// Four nibbles (one per byte) to four signed Codebook10 bytes: v_perm picks the
// magnitudes 0 1 2 3 | 4 6 8 10 and their negations by the low three bits, and
// a third v_perm picks between the two by the fourth bit (llama.cpp's
// get_int_from_table_16).
__device__ inline uint32_t rocmfp4_codes(uint32_t nibbles) {
  const uint32_t index = nibbles & 0x07070707;
  const uint32_t positive = __builtin_amdgcn_perm(0x0A080604u, 0x03020100u, index);
  const uint32_t negative = __builtin_amdgcn_perm(0xF6F8FAFCu, 0xFDFEFF00u, index);
  const uint32_t select = 0x03020100u | ((nibbles & 0x08080808) >> 1);
  return __builtin_amdgcn_perm(negative, positive, select);
}

template <int SCALES>
struct RocmFp4 {
  static constexpr int BYTES = 16 + SCALES, ELEMS = 32, LANES = 1;
  struct Part { uint32_t q[8]; float low, high; };  // signed codes, elements in order
  __device__ static Part decode(const uint32_t* staged, int at, int) {
    Part p;
    uint32_t raw[4];
    words_at<4>(staged, at, raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p.q[i] = rocmfp4_codes(raw[i] & 0x0F0F0F0F);
      p.q[i + 4] = rocmfp4_codes((raw[i] >> 4) & 0x0F0F0F0F);
    }
    p.low = rocmfp4_half_scale(byte_at(staged, at + 16));
    p.high = rocmfp4_half_scale(byte_at(staged, at + 16 + SCALES - 1));
    return p;
  }
  __device__ static float dot(const Part& p, const XBlock& x) {
    int low = 0, high = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      low = dot_ss(p.q[i], x.q[i], low);
      high = dot_ss(p.q[i + 4], x.q[i + 4], high);
    }
    return x.d * (float(low) * p.low + float(high) * p.high);
  }
  __device__ static float value(const Part& p, int j) {
    return float(signed_byte(p.q[j / 4], j % 4)) * (j < 16 ? p.low : p.high);
  }
};

// K-quants: 256 elements, eight lanes per block, part p = elements 32p .. 32p+31.

__device__ inline void scale_min_k4(const uint32_t* staged, int at, int j, int& sc, int& m) {
  auto s = [&](int i) { return byte_at(staged, at + i); };
  if (j < 4) {
    sc = s(j) & 63;
    m = s(j + 4) & 63;
  } else {
    sc = (s(j + 4) & 0xF) | ((s(j - 4) >> 2) & 0x30);
    m = (s(j + 4) >> 4) | ((s(j) >> 2) & 0x30);
  }
}

struct Q4_K {
  static constexpr int BYTES = 144, ELEMS = 256, LANES = 8;
  struct Part { uint32_t q[8]; float dl, ml; };  // nibbles as bytes
  // part c: the low or high nibbles of 32 bytes; d sc sum q x̂ - dmin m sum x̂
  __device__ static Part decode(const uint32_t* staged, int at, int c) {
    Part p;
    int sc, m;
    scale_min_k4(staged, at + 4, c, sc, m);
    uint32_t raw[8];
    words_at<8>(staged, at + 16 + (c / 2) * 32, raw);
    const int shift = 4 * (c % 2);
#pragma unroll
    for (int i = 0; i < 8; ++i) p.q[i] = (raw[i] >> shift) & 0x0F0F0F0F;
    p.dl = half_at(staged, at) * float(sc);
    p.ml = half_at(staged, at + 2) * float(m);
    return p;
  }
  __device__ static float dot(const Part& p, const XBlock& x) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum = dot_us(p.q[i], x.q[i], sum);
    return p.dl * x.d * float(sum) - p.ml * x.s;
  }
  __device__ static float value(const Part& p, int j) {
    return p.dl * float(unsigned_byte(p.q[j / 4], j % 4)) - p.ml;
  }
};

// Q5_K: Q4_K's layout with qh[32] before the nibbles; part c's fifth bits
// are bit c of each qh byte.
struct Q5_K : Q4_K {
  static constexpr int BYTES = 176;
  __device__ static Part decode(const uint32_t* staged, int at, int c) {
    Part p;
    int sc, m;
    scale_min_k4(staged, at + 4, c, sc, m);
    uint32_t raw[8], qh[8];
    words_at<8>(staged, at + 48 + (c / 2) * 32, raw);
    words_at<8>(staged, at + 16, qh);
    const int shift = 4 * (c % 2);
#pragma unroll
    for (int i = 0; i < 8; ++i) p.q[i] = ((raw[i] >> shift) & 0x0F0F0F0F) | (((qh[i] >> c) & 0x01010101) << 4);
    p.dl = half_at(staged, at) * float(sc);
    p.ml = half_at(staged, at + 2) * float(m);
    return p;
  }
};

struct Q6_K {
  static constexpr int BYTES = 210, ELEMS = 256, LANES = 8;
  // six-bit codes 0..63 as bytes, two sub-blocks of 16 with their own scale;
  // sum (q - 32) x̂ = xd * (sum q xq - 32 sum xq) per sub-block
  struct Part { uint32_t q[8]; float d0, d1; };
  __device__ static Part decode(const uint32_t* staged, int at, int part) {
    Part p;
    const int half = part / 4, plane = part % 4;  // 32 part = 128 half + 32 plane
    uint32_t low[8], high[8];
    words_at<8>(staged, at + half * 64 + (plane % 2) * 32, low);
    words_at<8>(staged, at + 128 + half * 32, high);
    const int low_shift = 4 * (plane / 2), high_shift = 2 * plane;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      p.q[i] = ((low[i] >> low_shift) & 0x0F0F0F0F) | (((high[i] >> high_shift) & 0x03030303) << 4);
    const float d = half_at(staged, at + 208);
    p.d0 = d * float(int8_t(byte_at(staged, at + 192 + 2 * part)));
    p.d1 = d * float(int8_t(byte_at(staged, at + 193 + 2 * part)));
    return p;
  }
  __device__ static float dot(const Part& p, const XBlock& x) {
    int sums[2] = {0, 0}, xsums[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      sums[i / 4] = dot_us(p.q[i], x.q[i], sums[i / 4]);
      xsums[i / 4] = dot_us(0x01010101u, x.q[i], xsums[i / 4]);
    }
    return x.d * (p.d0 * float(sums[0] - 32 * xsums[0]) + p.d1 * float(sums[1] - 32 * xsums[1]));
  }
  __device__ static float value(const Part& p, int j) {
    return (j < 16 ? p.d0 : p.d1) * float(unsigned_byte(p.q[j / 4], j % 4) - 32);
  }
};

// ---- dense types: 32 elements per lane, x as float ----------------------------------

enum class DenseKind { F32, F16, BF16 };

template <DenseKind KIND>
struct Dense {
  static constexpr int ELEMENT_BYTES = KIND == DenseKind::F32 ? 4 : 2;
  static constexpr int BYTES = 32 * ELEMENT_BYTES, ELEMS = 32, LANES = 1, WORDS = BYTES / 4;
  struct Part { uint32_t w[WORDS]; };
  __device__ static Part decode(const uint32_t* staged, int at, int) {
    Part p;
    words_at<WORDS>(staged, at, p.w);
    return p;
  }
  __device__ static float value(const Part& p, int j) {
    if constexpr (KIND == DenseKind::F32) return __uint_as_float(p.w[j]);
    const uint32_t bits = (p.w[j / 2] >> (16 * (j % 2))) & 0xFFFF;
    if constexpr (KIND == DenseKind::F16) return __half2float(__ushort_as_half(uint16_t(bits)));
    return __uint_as_float(bits << 16);
  }
  __device__ static float dot(const Part& p, const float* x) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) sum += value(p, j) * x[j];
    return sum;
  }
};

// ---- x quantized per 32 values ---------------------------------------------------

// One wave per 32-value block (quantize.h).
extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)
    quantize_x(const float* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xd,
               float* __restrict__ xs, int blocks) {
  const int block = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (block >= blocks) return;
  quantize_block(x[(long long)block * 32 + threadIdx.x % WAVE], block, xq, xd, xs);
}

// ---- walking a row ------------------------------------------------------------------

// A trip stages about TRIP bytes per wave, several chunks at once, so each lane
// has several loads in flight. Wide trips (4 KB, 32 KB of LDS for 8 waves) suit
// long rows; narrow ones (1 KB) suit short rows, where the LDS a wide trip
// reserves would cap how many workgroups a compute unit holds and latency
// would dominate. `visit(part, element)` gets each decoded part and the index of
// its first element.
template <typename T, int TRIP, typename Visit>
__device__ inline void walk_blocks(const uint8_t* __restrict__ weight, int row, int cols, int first_block,
                                   int end_block, Visit visit);

template <typename T, int TRIP, typename Visit>
__device__ inline void walk_row(const uint8_t* __restrict__ weight, int row, int cols, Visit visit) {
  walk_blocks<T, TRIP>(weight, row, cols, 0, cols / T::ELEMS, visit);
}

// walk_row over blocks [first_block, end_block) of the row only: a split row.
template <typename T, int TRIP, typename Visit>
__device__ inline void walk_blocks(const uint8_t* __restrict__ weight, int row, int cols, int first_block,
                                   int end_block, Visit visit) {
  constexpr int CHUNK_BLOCKS = WAVE / T::LANES;
  constexpr int CHUNK_BYTES = CHUNK_BLOCKS * T::BYTES;
  constexpr int CHUNKS = CHUNK_BYTES >= TRIP ? 1 : TRIP / CHUNK_BYTES;
  constexpr int TRIP_BLOCKS = CHUNKS * CHUNK_BLOCKS;
  constexpr int TRIP_WORDS = CHUNKS * CHUNK_BYTES / 4;
  constexpr int LANE_WORDS = (TRIP_WORDS + 1 + WAVE - 1) / WAVE;  // + a skew's word
  __shared__ uint32_t staged[WAVES_PER_BLOCK][TRIP_WORDS + 2];  // + a skew's word + words_at's spare

  const int wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
  const int blocks = end_block;
  const long long row_bytes = (long long)(cols / T::ELEMS) * T::BYTES;
  const uint32_t* row_words = reinterpret_cast<const uint32_t*>(weight + row * row_bytes);
  const int block_in_chunk = lane / T::LANES, part = lane % T::LANES;

  for (int first = first_block; first < blocks; first += TRIP_BLOCKS) {
    // a split's first block may start mid-word: stage from the word before it
    const int skew = (first * T::BYTES) % 4;
    const int base = (first * T::BYTES) / 4;
    const int words = (skew + min(TRIP_BLOCKS, blocks - first) * T::BYTES + 3) / 4;
    uint32_t loaded[LANE_WORDS];
#pragma unroll
    for (int i = 0; i < LANE_WORDS; ++i) {
      const int w = lane + i * WAVE;
      if (w < words) loaded[i] = row_words[base + w];
    }
    __builtin_amdgcn_wave_barrier();  // the previous trip is read by now
#pragma unroll
    for (int i = 0; i < LANE_WORDS; ++i) {
      const int w = lane + i * WAVE;
      if (w < words) staged[wave][w] = loaded[i];
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int in_trip = c * CHUNK_BLOCKS + block_in_chunk;
      const int block = first + in_trip;
      if (block < blocks) {
        visit(T::decode(staged[wave], skew + in_trip * T::BYTES, part),
              block * T::ELEMS + part * (T::ELEMS / T::LANES));
      }
    }
  }
}

// Sums across the wave; lane 0 writes each vector's result.
__device__ inline void write_sums(float* sums, float* y, int rows, int row, int vectors) {
#pragma unroll
  for (int v = 0; v < MAX_VECTORS; ++v) {
    if (v < vectors) {
      float sum = sums[v];
#pragma unroll
      for (int offset = WAVE / 2; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset, WAVE);
      if (threadIdx.x % WAVE == 0) y[(long long)v * rows + row] = sum;
    }
  }
}

template <typename T, int TRIP>
__device__ void matvec_int(const uint8_t* weight, const int8_t* xq, const float* xd, const float* xs,
                           float* y, int rows, int cols, int vectors) {
  const int row = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (row >= rows) return;  // whole waves only: no block-wide barrier
  const int x_blocks = cols / 32;
  float sums[MAX_VECTORS] = {};
  walk_row<T, TRIP>(weight, row, cols, [&](const typename T::Part& p, int element) {
    const int block = element / 32;
#pragma unroll
    for (int v = 0; v < MAX_VECTORS; ++v) {
      if (v < vectors) {
        const int i = v * x_blocks + block;
        sums[v] += T::dot(p, XBlock{reinterpret_cast<const int*>(xq + (long long)i * 32), xd[i], xs[i]});
      }
    }
  });
  write_sums(sums, y, rows, row, vectors);
}

template <typename T, int TRIP>
__device__ void matvec_float(const uint8_t* weight, const float* x, float* y, int rows, int cols,
                             int vectors) {
  const int row = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (row >= rows) return;
  float sums[MAX_VECTORS] = {};
  walk_row<T, TRIP>(weight, row, cols, [&](const typename T::Part& p, int element) {
#pragma unroll
    for (int v = 0; v < MAX_VECTORS; ++v) {
      if (v < vectors) sums[v] += T::dot(p, x + (long long)v * cols + element);
    }
  });
  write_sums(sums, y, rows, row, vectors);
}

// Integer types with x kept as float: the weights decoded to floats in
// registers (T::value), so the only rounding is the float sums'.
template <typename T, int TRIP>
__device__ __forceinline__ void matvec_values_row(const uint8_t* weight, const float* x, float* y, int row, int rows, int cols,
                                  int vectors) {
  float sums[MAX_VECTORS] = {};
  walk_row<T, TRIP>(weight, row, cols, [&](const typename T::Part& p, int element) {
    float w[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) w[j] = T::value(p, j);
#pragma unroll
    for (int v = 0; v < MAX_VECTORS; ++v) {
      if (v < vectors) {
        const float* xv = x + (long long)v * cols + element;
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) sum += w[j] * xv[j];
        sums[v] += sum;
      }
    }
  });
  write_sums(sums, y, rows, row, vectors);
}

template <typename T, int TRIP>
__device__ void matvec_values(const uint8_t* weight, const float* x, float* y, int rows, int cols, int vectors) {
  const int row = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (row >= rows) return;
  matvec_values_row<T, TRIP>(weight, x, y, row, rows, cols, vectors);
}

// Up to four matrices of one type and width over the same x, in one launch:
// their rows are numbered one after the other and each wave finds its matrix.
// A layer's input projections then pay one launch, ramp and tail, not four.
template <typename T, int TRIP>
__device__ void matvec_values_multi(const uint8_t* w0, float* y0, int r0, const uint8_t* w1, float* y1, int r1,
                                    const uint8_t* w2, float* y2, int r2, const uint8_t* w3, float* y3, int r3,
                                    const float* x, int cols, int vectors) {
  int row = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  const uint8_t* weight = w0;
  float* y = y0;
  int rows = r0;
  if (row >= r0) {
    row -= r0;
    weight = w1, y = y1, rows = r1;
    if (row >= r1) {
      row -= r1;
      weight = w2, y = y2, rows = r2;
      if (row >= r2) {
        row -= r2;
        weight = w3, y = y3, rows = r3;
        if (row >= r3) return;
      }
    }
  }
  matvec_values_row<T, TRIP>(weight, x, y, row, rows, cols, vectors);
}

// A mixture of experts' products: slot s multiplies expert ids[s] (of a
// [experts][rows][cols] weight, `expert_bytes` apart) with x row s / divisor
// (one x per token for gate and up, one per slot for down), into y [slots][rows].
// Float x, the weights decoded in registers (T::value). One wave per (slot,
// `group` rows): rows shorter than a chunk (the down products' rows of a few
// hundred bytes) would leave lanes idle, so `group` consecutive rows are walked
// as one row of group × cols, and lane l's sums belong to row l / (WAVE / group).
template <typename T, int TRIP>
__device__ void matvec_experts(const uint8_t* weight, long long expert_bytes, const int* ids, const float* x,
                               int divisor, float* y, int rows, int cols, int slots, int group) {
  const long long index = (long long)blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  const int groups = rows / group;
  if (index >= (long long)slots * groups) return;
  const int slot = index / groups, rows_group = index % groups;
  const float* xs = x + (long long)(slot / divisor) * cols;
  float sum = 0.0f;
  walk_row<T, TRIP>(weight + ids[slot] * expert_bytes, rows_group, cols * group,
                    [&](const typename T::Part& p, int element) {
                      const float* xe = xs + element % cols;
                      float partial = 0.0f;
#pragma unroll
                      for (int j = 0; j < 32; ++j) partial += T::value(p, j) * xe[j];
                      sum += partial;
                    });
  const int width = WAVE / group, lane = threadIdx.x % WAVE;
  for (int offset = width / 2; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset, WAVE);
  if (lane % width == 0) y[(long long)slot * rows + rows_group * group + lane / width] = sum;
}

// A split row: the workgroup's eight waves share one row, each over an eighth
// of its blocks, their sums reduced in LDS. For matrices of few rows (routers,
// small projections), which one wave per row would leave most of the GPU idle.
// Float x, vector blockIdx.y.
template <typename T, int TRIP>
__device__ void matvec_split(const uint8_t* weight, const float* x, float* y, int rows, int cols) {
  __shared__ float partial[WAVES_PER_BLOCK];
  const int row = blockIdx.x, wave = threadIdx.x / WAVE;
  x += (long long)blockIdx.y * cols;
  y += (long long)blockIdx.y * rows;
  const int blocks = cols / T::ELEMS;
  const int first = (long long)blocks * wave / WAVES_PER_BLOCK, end = (long long)blocks * (wave + 1) / WAVES_PER_BLOCK;
  float sum = 0.0f;
  walk_blocks<T, TRIP>(weight, row, cols, first, end, [&](const typename T::Part& p, int element) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) s += T::value(p, j) * x[element + j];
    sum += s;
  });
#pragma unroll
  for (int offset = WAVE / 2; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset, WAVE);
  if (threadIdx.x % WAVE == 0) partial[wave] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int w = 0; w < WAVES_PER_BLOCK; ++w) total += partial[w];
    y[row] = total;
  }
}

// Experts' gate and up at once: slot s's wave takes row r of expert ids[s]'s
// gate and of its up (same shapes, `expert_bytes` apart) and writes
// silu(gate) * up — SwiGLU without its intermediate launches.
template <typename T, int TRIP>
__device__ void matvec_experts_gated(const uint8_t* gate, const uint8_t* up, long long expert_bytes, const int* ids,
                                     const float* x, int divisor, float* y, int rows, int cols, int slots) {
  const long long index = (long long)blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (index >= (long long)slots * rows) return;
  const int slot = index / rows, row = index % rows;
  const float* xs = x + (long long)(slot / divisor) * cols;
  const long long expert = ids[slot] * expert_bytes;
  float sums[2] = {0.0f, 0.0f};
  for (int which = 0; which < 2; ++which)
    walk_row<T, TRIP>((which == 0 ? gate : up) + expert, row, cols, [&](const typename T::Part& p, int element) {
      float s = 0.0f;
#pragma unroll
      for (int j = 0; j < 32; ++j) s += T::value(p, j) * xs[element + j];
      sums[which] += s;
    });
#pragma unroll
  for (int offset = WAVE / 2; offset > 0; offset /= 2) {
    sums[0] += __shfl_xor(sums[0], offset, WAVE);
    sums[1] += __shfl_xor(sums[1], offset, WAVE);
  }
  if (threadIdx.x % WAVE == 0) y[(long long)slot * rows + row] = sums[0] / (1.0f + expf(-sums[0])) * sums[1];
}

// ---- experts grouped by expert, for a prompt's many tokens -------------------------------
// Per slot, a prompt's thousands of slots would read their experts' rows once
// each; grouped, the slots of one expert share a wave, which reads a row once
// for up to GROUP_TOKENS of them.

constexpr int GROUP_TOKENS = MAX_VECTORS, MAX_EXPERTS = 1024;

// One workgroup: `order` lists the slots expert by expert, and `batches`
// ([max_batches][2]: first index in `order`, count) cuts each expert's run into
// batches of up to `batch_size` (GROUP_TOKENS, or 16 for the matrix units'
// tiles); the batches past the last count 0. The order
// within an expert varies, but each slot's sums do not depend on it.
extern "C" __global__ void __launch_bounds__(1024)
    group_slots(const int* ids, int slots, int experts, int* order, int* batches, int max_batches, int batch_size) {
  __shared__ int count[MAX_EXPERTS], first[MAX_EXPERTS], first_batch[MAX_EXPERTS], filled[MAX_EXPERTS];
  __shared__ int total_batches;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) count[e] = filled[e] = 0;
  __syncthreads();
  for (int s = threadIdx.x; s < slots; s += blockDim.x) atomicAdd(&count[ids[s]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int at = 0, batch = 0;
    for (int e = 0; e < experts; ++e) {
      first[e] = at;
      first_batch[e] = batch;
      at += count[e];
      batch += (count[e] + batch_size - 1) / batch_size;
    }
    total_batches = batch;
  }
  __syncthreads();
  for (int s = threadIdx.x; s < slots; s += blockDim.x) {
    const int e = ids[s];
    order[first[e] + atomicAdd(&filled[e], 1)] = s;
  }
  for (int e = threadIdx.x; e < experts; e += blockDim.x)
    for (int i = 0; i * batch_size < count[e]; ++i) {
      batches[2 * (first_batch[e] + i)] = first[e] + i * batch_size;
      batches[2 * (first_batch[e] + i) + 1] = min(batch_size, count[e] - i * batch_size);
    }
  for (int b = total_batches + threadIdx.x; b < max_batches; b += blockDim.x) batches[2 * b + 1] = 0;
}

// A batch's slots, and its expert's weight.
struct Batch {
  int count;
  int slots[GROUP_TOKENS];
  long long expert;
};

__device__ inline Batch batch_at(const int* ids, const int* order, const int* batches, int batch,
                                 long long expert_bytes) {
  Batch b;
  const int start = batches[2 * batch];
  b.count = batches[2 * batch + 1];
#pragma unroll
  for (int v = 0; v < GROUP_TOKENS; ++v) b.slots[v] = v < b.count ? order[start + v] : 0;
  b.expert = b.count > 0 ? ids[b.slots[0]] * expert_bytes : 0;
  return b;
}

// matvec_experts over batches: one wave per (batch, `group` rows).
template <typename T, int TRIP>
__device__ void matvec_experts_grouped(const uint8_t* weight, long long expert_bytes, const int* ids,
                                       const int* order, const int* batches, const float* x, int divisor,
                                       float* y, int rows, int cols, int max_batches, int group) {
  const long long index = (long long)blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  const int groups = rows / group;
  if (index >= (long long)max_batches * groups) return;
  const int rows_group = index % groups;
  const Batch b = batch_at(ids, order, batches, index / groups, expert_bytes);
  if (b.count == 0) return;
  float sums[GROUP_TOKENS] = {};
  walk_row<T, TRIP>(weight + b.expert, rows_group, cols * group, [&](const typename T::Part& p, int element) {
    float w[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) w[j] = T::value(p, j);
    const int column = element % cols;
#pragma unroll
    for (int v = 0; v < GROUP_TOKENS; ++v) {
      if (v < b.count) {
        const float* xe = x + (long long)(b.slots[v] / divisor) * cols + column;
        float partial = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) partial += w[j] * xe[j];
        sums[v] += partial;
      }
    }
  });
  const int width = WAVE / group, lane = threadIdx.x % WAVE;
#pragma unroll
  for (int v = 0; v < GROUP_TOKENS; ++v) {
    if (v < b.count) {
      float sum = sums[v];
      for (int offset = width / 2; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset, WAVE);
      if (lane % width == 0) y[(long long)b.slots[v] * rows + rows_group * group + lane / width] = sum;
    }
  }
}

// matvec_experts_gated over batches: one wave per (batch, row).
template <typename T, int TRIP>
__device__ void matvec_experts_gated_grouped(const uint8_t* gate, const uint8_t* up, long long expert_bytes,
                                             const int* ids, const int* order, const int* batches,
                                             const float* x, int divisor, float* y, int rows, int cols,
                                             int max_batches) {
  const long long index = (long long)blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (index >= (long long)max_batches * rows) return;
  const int row = index % rows;
  const Batch b = batch_at(ids, order, batches, index / rows, expert_bytes);
  if (b.count == 0) return;
  float sums[2][GROUP_TOKENS] = {};
  for (int which = 0; which < 2; ++which)
    walk_row<T, TRIP>((which == 0 ? gate : up) + b.expert, row, cols, [&](const typename T::Part& p, int element) {
      float w[32];
#pragma unroll
      for (int j = 0; j < 32; ++j) w[j] = T::value(p, j);
#pragma unroll
      for (int v = 0; v < GROUP_TOKENS; ++v) {
        if (v < b.count) {
          const float* xe = x + (long long)(b.slots[v] / divisor) * cols + element;
          float partial = 0.0f;
#pragma unroll
          for (int j = 0; j < 32; ++j) partial += w[j] * xe[j];
          sums[which][v] += partial;
        }
      }
    });
#pragma unroll
  for (int v = 0; v < GROUP_TOKENS; ++v) {
    if (v < b.count) {
      float g = sums[0][v], u = sums[1][v];
#pragma unroll
      for (int offset = WAVE / 2; offset > 0; offset /= 2) {
        g += __shfl_xor(g, offset, WAVE);
        u += __shfl_xor(u, offset, WAVE);
      }
      if (threadIdx.x % WAVE == 0) y[(long long)b.slots[v] * rows + row] = g / (1.0f + expf(-g)) * u;
    }
  }
}

// W as f16, row-major, for the GEMM path.
template <typename T, int TRIP>
__device__ void dequantize(const uint8_t* weight, __half* out, int rows, int cols) {
  const int row = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (row >= rows) return;
  __half* target = out + (long long)row * cols;
  walk_row<T, TRIP>(weight, row, cols, [&](const typename T::Part& p, int element) {
#pragma unroll
    for (int j = 0; j < 32; ++j) target[element + j] = __float2half(T::value(p, j));
  });
}

// As dequantize, into BF16 (rounded to nearest even): the range of F32 for
// activations past F16's (MiniMax H3's MLP).
template <typename T, int TRIP>
__device__ void dequantize_bf16(const uint8_t* weight, uint16_t* out, int rows, int cols) {
  const int row = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (row >= rows) return;
  uint16_t* target = out + (long long)row * cols;
  walk_row<T, TRIP>(weight, row, cols, [&](const typename T::Part& p, int element) {
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      unsigned int bits = __float_as_uint(T::value(p, j));
      bits += 0x7FFFu + ((bits >> 16) & 1u);
      target[element + j] = (uint16_t)(bits >> 16);
    }
  });
}

// Rows ids[i] of an embedding table, as F32: one wave per id.
template <typename T, int TRIP>
__device__ void gather(const uint8_t* weight, const int* ids, float* out, int count, int cols) {
  const int index = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  if (index >= count) return;
  float* target = out + (long long)index * cols;
  walk_row<T, TRIP>(weight, ids[index], cols, [&](const typename T::Part& p, int element) {
#pragma unroll
    for (int j = 0; j < 32; ++j) target[element + j] = T::value(p, j);
  });
}

#define EXPERTS(name, type)                                                                          \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_experts_##name(const uint8_t* weight, long long expert_bytes, const int* ids,          \
                            const float* x, int divisor, float* y, int rows, int cols, int slots, \
                            int group) {                                                        \
    matvec_experts<type, 4096>(weight, expert_bytes, ids, x, divisor, y, rows, cols, slots, group); \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_experts_narrow_##name(const uint8_t* weight, long long expert_bytes, const int* ids,   \
                                   const float* x, int divisor, float* y, int rows, int cols,       \
                                   int slots, int group) {                                          \
    matvec_experts<type, 1024>(weight, expert_bytes, ids, x, divisor, y, rows, cols, slots, group); \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_experts_grouped_##name(const uint8_t* weight, long long expert_bytes, const int* ids,  \
                                    const int* order, const int* batches, const float* x,           \
                                    int divisor, float* y, int rows, int cols, int max_batches,     \
                                    int group) {                                                    \
    matvec_experts_grouped<type, 1024>(weight, expert_bytes, ids, order, batches, x, divisor, y,    \
                                       rows, cols, max_batches, group);                             \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_experts_gated_grouped_##name(const uint8_t* gate, const uint8_t* up,                   \
                                          long long expert_bytes, const int* ids, const int* order, \
                                          const int* batches, const float* x, int divisor,          \
                                          float* y, int rows, int cols, int max_batches) {          \
    matvec_experts_gated_grouped<type, 1024>(gate, up, expert_bytes, ids, order, batches, x,        \
                                             divisor, y, rows, cols, max_batches);                  \
  }

#define SPLIT_AND_GATED(name, type)                                                                  \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_split_##name(const uint8_t* weight, const float* x, float* y, int rows, int cols) {    \
    matvec_split<type, 1024>(weight, x, y, rows, cols);                                             \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_experts_gated_##name(const uint8_t* gate, const uint8_t* up, long long expert_bytes,   \
                                  const int* ids, const float* x, int divisor, float* y, int rows,  \
                                  int cols, int slots) {                                            \
    matvec_experts_gated<type, 1024>(gate, up, expert_bytes, ids, x, divisor, y, rows, cols, slots);\
  }

#define DEQUANTIZE(name, type)                                                                       \
  EXPERTS(name, type)                                                                               \
  SPLIT_AND_GATED(name, type)                                                                       \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      dequantize_##name(const uint8_t* weight, __half* out, int rows, int cols) {                   \
    dequantize<type, 4096>(weight, out, rows, cols);                                                \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      dequantize_bf16_##name(const uint8_t* weight, uint16_t* out, int rows, int cols) {            \
    dequantize_bf16<type, 4096>(weight, out, rows, cols);                                           \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      gather_##name(const uint8_t* weight, const int* ids, float* out, int count, int cols) {       \
    gather<type, 4096>(weight, ids, out, count, cols);                                              \
  }

#define MULTI(name, type, size, trip)                                                                \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_float_x_multi_##size##name(                                                            \
          const uint8_t* w0, float* y0, int r0, const uint8_t* w1, float* y1, int r1,               \
          const uint8_t* w2, float* y2, int r2, const uint8_t* w3, float* y3, int r3,               \
          const float* x, int cols, int vectors) {                                                  \
    matvec_values_multi<type, trip>(w0, y0, r0, w1, y1, r1, w2, y2, r2, w3, y3, r3, x, cols,        \
                                    vectors);                                                       \
  }

#define KERNELS_INT(name, type)                                                                      \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_##name(const uint8_t* weight, const int8_t* xq, const float* xd, const float* xs,      \
                    float* y, int rows, int cols, int vectors) {                                    \
    matvec_int<type, 4096>(weight, xq, xd, xs, y, rows, cols, vectors);                             \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_float_x_##name(const uint8_t* weight, const float* x, float* y, int rows, int cols,    \
                            int vectors) {                                                          \
    matvec_values<type, 4096>(weight, x, y, rows, cols, vectors);                                   \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_float_x_narrow_##name(const uint8_t* weight, const float* x, float* y, int rows,       \
                                   int cols, int vectors) {                                         \
    matvec_values<type, 1024>(weight, x, y, rows, cols, vectors);                                   \
  }                                                                                                 \
  MULTI(name, type, , 4096)                                                                         \
  MULTI(name, type, narrow_, 1024)                                                                  \
  DEQUANTIZE(name, type)

#define KERNELS_FLOAT(name, type)                                                                    \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_##name(const uint8_t* weight, const float* x, float* y, int rows, int cols,            \
                    int vectors) {                                                                  \
    matvec_float<type, 4096>(weight, x, y, rows, cols, vectors);                                    \
  }                                                                                                 \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                              \
      matvec_narrow_##name(const uint8_t* weight, const float* x, float* y, int rows, int cols,     \
                           int vectors) {                                                           \
    matvec_float<type, 1024>(weight, x, y, rows, cols, vectors);                                    \
  }                                                                                                 \
  DEQUANTIZE(name, type)

KERNELS_INT(q8_0, Q8_0)
KERNELS_INT(q4_0, Q4_0)
KERNELS_INT(q4_k, Q4_K)
KERNELS_INT(q6_k, Q6_K)
KERNELS_INT(q5_1, Q5_1)
KERNELS_INT(q5_k, Q5_K)
KERNELS_INT(iq4_nl, IQ4_NL)
KERNELS_INT(iq4_xs, IQ4_XS)
KERNELS_INT(q4_0_rocmfp4, RocmFp4<2>)
KERNELS_INT(q4_0_rocmfp4_fast, RocmFp4<1>)
KERNELS_FLOAT(f32, Dense<DenseKind::F32>)
KERNELS_FLOAT(f16, Dense<DenseKind::F16>)
KERNELS_FLOAT(bf16, Dense<DenseKind::BF16>)

// ---- rows read straight from memory ---------------------------------------------------
// Blocks of 256 whose layout keeps each lane's bytes aligned (Q4_K and Q5_K to
// 16 bytes, IQ4_XS to 8, eight ROCmFP4 blocks to 2) need no LDS staging: four
// lanes per block, lane j loading what elements 64j .. 64j+63 need and meeting
// them with x. A lane's x
// values are the same for every row, so a wave takes ROWS rows at once: all
// their loads in flight together, x loaded once. One vector of float x.
// A block type B gives BYTES, a Lane of loaded words, load(block, j),
// decode(lane, j) (the lane itself, or what every vector's dot shares) and
// dot(decoded, xs, sum of xs[0..31], sum of xs[32..63], j).

typedef uint32_t words4 __attribute__((ext_vector_type(4)));
typedef uint32_t words2 __attribute__((ext_vector_type(2)));

__device__ inline uint4 load16(const uint8_t* at) {
  const words4 v = *reinterpret_cast<const words4*>(at);
  return make_uint4(v.x, v.y, v.z, v.w);
}

__device__ inline uint2 load8(const uint8_t* at) {
  const words2 v = *reinterpret_cast<const words2*>(at);
  return make_uint2(v.x, v.y);
}

__device__ inline int header_byte(const uint4& h, int k) {  // byte 4 + k, k < 12
  const uint32_t word = k < 4 ? h.y : (k < 8 ? h.z : h.w);
  return int((word >> (8 * (k % 4))) & 0xFF);
}

// Q4_K's and Q5_K's d·scale and dmin·min of sub-block `sub`.
__device__ inline void k_scale_min(const uint4& h, int sub, float& scale, float& minimum) {
  int sc, m;
  if (sub < 4) {
    sc = header_byte(h, sub) & 63;
    m = header_byte(h, sub + 4) & 63;
  } else {
    sc = (header_byte(h, sub + 4) & 0xF) | ((header_byte(h, sub - 4) >> 2) & 0x30);
    m = (header_byte(h, sub + 4) >> 4) | ((header_byte(h, sub) >> 2) & 0x30);
  }
  const float d = __half2float(__ushort_as_half(uint16_t(h.x & 0xFFFF)));
  const float dmin = __half2float(__ushort_as_half(uint16_t(h.x >> 16)));
  scale = d * float(sc);
  minimum = dmin * float(m);
}

// Q4_K: d, dmin, scales[12], then qs; lane j's 32 bytes of qs hold its
// elements' low nibbles (sub-block 2j) and high nibbles (2j + 1).
struct Q4KDirect {
  static constexpr int BYTES = 144;
  struct Lane { uint4 header, low, high; };
  __device__ static Lane load(const uint8_t* b, int j) {
    return {load16(b), load16(b + 16 + 32 * j), load16(b + 32 + 32 * j)};
  }
  using Decoded = Lane;
  __device__ static const Lane& decode(const Lane& l, int) { return l; }
  __device__ static float dot(const Lane& l, const float* xs, float x_low, float x_high, int j) {
#pragma clang fp contract(off)  // see direct_rows
    const uint32_t q[8] = {l.low.x, l.low.y, l.low.z, l.low.w, l.high.x, l.high.y, l.high.z, l.high.w};
    float dot_low = 0.0f, dot_high = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (q[i] >> (8 * b)) & 0xFF;
        dot_low = fmaf(float(byte & 0xF), xs[4 * i + b], dot_low);
        dot_high = fmaf(float(byte >> 4), xs[32 + 4 * i + b], dot_high);
      }
    float scale_low, min_low, scale_high, min_high;
    k_scale_min(l.header, 2 * j, scale_low, min_low);
    k_scale_min(l.header, 2 * j + 1, scale_high, min_high);
    return scale_low * dot_low - min_low * x_low + scale_high * dot_high - min_high * x_high;
  }
};

// Q5_K: Q4_K with qh[32] before the nibbles; bit 2j of qh byte i is element
// 64j + i's fifth bit, bit 2j + 1 element 64j + 32 + i's.
struct Q5KDirect {
  static constexpr int BYTES = 176;
  struct Lane { uint4 header, qh0, qh1, low, high; };
  __device__ static Lane load(const uint8_t* b, int j) {
    return {load16(b), load16(b + 16), load16(b + 32), load16(b + 48 + 32 * j), load16(b + 64 + 32 * j)};
  }
  using Decoded = Lane;
  __device__ static const Lane& decode(const Lane& l, int) { return l; }
  __device__ static float dot(const Lane& l, const float* xs, float x_low, float x_high, int j) {
#pragma clang fp contract(off)  // see direct_rows
    const uint32_t q[8] = {l.low.x, l.low.y, l.low.z, l.low.w, l.high.x, l.high.y, l.high.z, l.high.w};
    const uint32_t h[8] = {l.qh0.x, l.qh0.y, l.qh0.z, l.qh0.w, l.qh1.x, l.qh1.y, l.qh1.z, l.qh1.w};
    float dot_low = 0.0f, dot_high = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t low_bits = ((h[i] >> (2 * j)) & 0x01010101) << 4;
      const uint32_t high_bits = ((h[i] >> (2 * j + 1)) & 0x01010101) << 4;
      const uint32_t lows = (q[i] & 0x0F0F0F0F) | low_bits, highs = ((q[i] >> 4) & 0x0F0F0F0F) | high_bits;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        dot_low = fmaf(float((lows >> (8 * b)) & 0xFF), xs[4 * i + b], dot_low);
        dot_high = fmaf(float((highs >> (8 * b)) & 0xFF), xs[32 + 4 * i + b], dot_high);
      }
    }
    float scale_low, min_low, scale_high, min_high;
    k_scale_min(l.header, 2 * j, scale_low, min_low);
    k_scale_min(l.header, 2 * j + 1, scale_high, min_high);
    return scale_low * dot_low - min_low * x_low + scale_high * dot_high - min_high * x_high;
  }
};

// IQ4_XS: d, scales_h (u16), scales_l[4], then eight runs of 16 bytes; lane j
// takes runs 2j and 2j + 1 (elements 64j .. 64j+63), each scaled by d (ls - 32).
struct IQ4XSDirect {
  static constexpr int BYTES = 136;
  struct Lane { uint2 header, q0, q1, q2, q3; };
  __device__ static Lane load(const uint8_t* b, int j) {
    const uint8_t* runs = b + 8 + 32 * j;
    return {load8(b), load8(runs), load8(runs + 8), load8(runs + 16), load8(runs + 24)};
  }
  using Decoded = Lane;
  __device__ static const Lane& decode(const Lane& l, int) { return l; }
  __device__ static float dot(const Lane& l, const float* xs, float, float, int j) {
#pragma clang fp contract(off)  // see direct_rows
    const uint32_t q[8] = {l.q0.x, l.q0.y, l.q1.x, l.q1.y, l.q2.x, l.q2.y, l.q3.x, l.q3.y};
    const float d = __half2float(__ushort_as_half(uint16_t(l.header.x & 0xFFFF)));
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int k = 2 * j + r;
      const int ls = int((l.header.y >> (4 * k)) & 0xF) | int(((l.header.x >> (16 + 2 * k)) & 3) << 4);
      float run = 0.0f;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const uint32_t lows = iq4_values(q[4 * r + w] & 0x0F0F0F0F);
        const uint32_t highs = iq4_values((q[4 * r + w] >> 4) & 0x0F0F0F0F);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          run = fmaf(float(signed_byte(lows, b)), xs[32 * r + 4 * w + b], run);
          run = fmaf(float(signed_byte(highs, b)), xs[32 * r + 16 + 4 * w + b], run);
        }
      }
      sum += d * float(ls - 32) * run;
    }
    return sum;
  }
};

// ROCmFP4: eight blocks of 32 taken as one of 256 (136 or 144 bytes, 8-byte
// aligned). Lane j takes blocks 2j and 2j + 1, 2 (16 + SCALES) bytes starting
// `shift` = 0 or 2 bytes into a word: it loads nine words from that word (a
// dwordx4 load needs only word alignment) and joins the codes with
// v_alignbyte. Block h then starts shift + h × SCALES < 4 bytes into word 4h.
typedef uint32_t words4_at_word __attribute__((ext_vector_type(4), aligned(4)));

template <int SCALES>
struct RocmFp4Direct {
  static constexpr int BLOCK = 16 + SCALES, BYTES = 8 * BLOCK;
  struct Lane { uint32_t w[9]; int shift; };
  __device__ static Lane load(const uint8_t* b, int j) {
    const uint8_t* start = b + 2 * BLOCK * j;
    const int shift = int(reinterpret_cast<uintptr_t>(start) & 3);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(start - shift);
    const words4_at_word first = *reinterpret_cast<const words4_at_word*>(words);
    const words4_at_word second = *reinterpret_cast<const words4_at_word*>(words + 4);
    return {{first.x, first.y, first.z, first.w, second.x, second.y, second.z, second.w, words[8]}, shift};
  }
  // Each code plus 10 as an unsigned byte (0 .. 20), so four convert to float
  // straight from a word (v_cvt_f32_ubyte): block h's low nibbles in words
  // 8h .. 8h+3 and its high nibbles in 8h+4 .. 8h+7; and the four half-block
  // scales. Decoded once per row, it meets every vector of a verification.
  struct Decoded { uint32_t codes[16]; float scales[4]; };
  __device__ static Decoded decode(const Lane& l, int) {
    Decoded d;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int at = l.shift + h * SCALES;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t codes = __builtin_amdgcn_alignbyte(l.w[4 * h + i + 1], l.w[4 * h + i], at);
        d.codes[8 * h + i] = biased_codes(codes & 0x0F0F0F0F);
        d.codes[8 * h + 4 + i] = biased_codes((codes >> 4) & 0x0F0F0F0F);
      }
      const uint32_t scales = l.w[4 * h + 4] >> (8 * at);
      d.scales[2 * h] = rocmfp4_half_scale(scales & 0xFF);
      d.scales[2 * h + 1] = rocmfp4_half_scale((scales >> (8 * (SCALES - 1))) & 0xFF);
    }
    return d;
  }
  // sum (c + 10) x - 10 sum x per half block
  __device__ static float dot(const Decoded& d, const float* xs, float x_low, float x_high, int) {
#pragma clang fp contract(off)  // see direct_rows
    float sum = 0.0f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float low = 0.0f, high = 0.0f, x_first = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          low = fmaf(float(unsigned_byte(d.codes[8 * h + i], b)), xs[32 * h + 4 * i + b], low);
          high = fmaf(float(unsigned_byte(d.codes[8 * h + 4 + i], b)), xs[32 * h + 16 + 4 * i + b], high);
          if (SCALES == 2) x_first += xs[32 * h + 4 * i + b];
        }
      const float x_block = h == 0 ? x_low : x_high;
      if (SCALES == 1)
        sum += d.scales[2 * h] * (low + high - 10.0f * x_block);
      else
        sum += d.scales[2 * h] * (low - 10.0f * x_first) + d.scales[2 * h + 1] * (high - 10.0f * (x_block - x_first));
    }
    return sum;
  }
  // Four nibbles to four codes + 10: rocmfp4_codes' tables shifted by 10.
  __device__ static uint32_t biased_codes(uint32_t nibbles) {
    const uint32_t index = nibbles & 0x07070707;
    const uint32_t positive = __builtin_amdgcn_perm(0x1412100Eu, 0x0D0C0B0Au, index);  // 10 11 12 13 | 14 16 18 20
    const uint32_t negative = __builtin_amdgcn_perm(0x00020406u, 0x0708090Au, index);  // 10 9 8 7 | 6 4 2 0
    const uint32_t select = 0x03020100u | ((nibbles & 0x08080808) >> 1);
    return __builtin_amdgcn_perm(negative, positive, select);
  }
};

// sums[r] += the lane's share of virtual row r: rows[r] is its first byte and it
// has `blocks` blocks; x repeats every `cols` values. A virtual row is one real
// row, or several consecutive short ones (fewer than eight blocks, dividing
// eight): lanes 4b .. 4b + 3 take its block b, so each real row's sum then
// gathers over `width` = 4 × its blocks lanes.
template <typename B, int ROWS>
__device__ __forceinline__ void direct(const uint8_t* const (&rows)[ROWS], int blocks, const float* x, int cols,
                                       float (&sums)[ROWS]) {
  const int lane = threadIdx.x % WAVE, j = lane % 4;
  for (int pass = 0; pass < blocks; pass += WAVE / 4) {
    const int block = pass + lane / 4;
    if (block >= blocks) break;
    typename B::Lane loaded[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) loaded[r] = B::load(rows[r] + block * B::BYTES, j);
    float xs[64];
    const float4* xv = reinterpret_cast<const float4*>(x + (block * 256 + 64 * j) % cols);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float4 v = xv[i];
      xs[4 * i] = v.x;
      xs[4 * i + 1] = v.y;
      xs[4 * i + 2] = v.z;
      xs[4 * i + 3] = v.w;
    }
    float x_low = 0.0f, x_high = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      x_low += xs[i];
      x_high += xs[32 + i];
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) sums[r] += B::dot(B::decode(loaded[r], j), xs, x_low, x_high, j);
  }
}

// The wave's index in the grid.
__device__ inline long long wave_index() { return (long long)blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE; }

// Sums over each `width` lanes: lane l's sum then belongs to real row l / width
// of its virtual row.
__device__ inline float gather_sum(float sum, int width) {
  for (int offset = width / 2; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset, WAVE);
  return sum;
}

// y = W x: each wave takes four consecutive rows (the last ones repeated at the
// end, not written). The blocks' dot functions fuse only the products they
// spell out (fmaf): the compiler would otherwise contract their sums
// differently where they are inlined here and in direct_vector_rows, and a
// verification's tokens would not get exactly a decode step's results.
constexpr int DIRECT_ROWS = 4;

template <typename B>
__device__ __forceinline__ void direct_rows(const uint8_t* weight, const float* x, float* y, int first_row, int rows,
                                            int cols) {
  const long long row_bytes = (long long)(cols / 256) * B::BYTES;
  const uint8_t* starts[DIRECT_ROWS];
#pragma unroll
  for (int r = 0; r < DIRECT_ROWS; ++r) starts[r] = weight + min(first_row + r, rows - 1) * row_bytes;
  float sums[DIRECT_ROWS] = {};
  direct<B, DIRECT_ROWS>(starts, cols / 256, x, cols, sums);
#pragma unroll
  for (int r = 0; r < DIRECT_ROWS; ++r) {
    const float sum = gather_sum(sums[r], WAVE);
    if (threadIdx.x % WAVE == 0 && first_row + r < rows) y[first_row + r] = sum;
  }
}

template <typename B>
__device__ void matvec_direct(const uint8_t* weight, const float* x, float* y, int rows, int cols) {
  const int first_row = (blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE) * DIRECT_ROWS;
  if (first_row >= rows) return;
  direct_rows<B>(weight, x, y, first_row, rows, cols);
}

// Several vectors of x (a verification's tokens, up to MAX_VECTORS), y
// [vectors][rows]: each lane's loaded words meet every vector's x, so the
// weights are read once. Two rows per wave: registers hold every vector's sums.
constexpr int DIRECT_VECTOR_ROWS = 2;

template <typename B>
__device__ __forceinline__ void direct_vector_rows(const uint8_t* weight, const float* x, float* y, int first_row,
                                                   int rows, int cols, int vectors) {
  const long long row_bytes = (long long)(cols / 256) * B::BYTES;
  const int blocks = cols / 256, lane = threadIdx.x % WAVE, j = lane % 4;
  const uint8_t* starts[DIRECT_VECTOR_ROWS];
#pragma unroll
  for (int r = 0; r < DIRECT_VECTOR_ROWS; ++r) starts[r] = weight + min(first_row + r, rows - 1) * row_bytes;
  float sums[DIRECT_VECTOR_ROWS][MAX_VECTORS] = {};
  for (int pass = 0; pass < blocks; pass += WAVE / 4) {
    const int block = pass + lane / 4;
    if (block >= blocks) break;
    typename B::Lane loaded[DIRECT_VECTOR_ROWS];
#pragma unroll
    for (int r = 0; r < DIRECT_VECTOR_ROWS; ++r) loaded[r] = B::load(starts[r] + block * B::BYTES, j);
    typename B::Decoded decoded[DIRECT_VECTOR_ROWS];
#pragma unroll
    for (int r = 0; r < DIRECT_VECTOR_ROWS; ++r) decoded[r] = B::decode(loaded[r], j);
#pragma unroll
    for (int v = 0; v < MAX_VECTORS; ++v) {
      if (v < vectors) {
        float xs[64];
        const float4* xv = reinterpret_cast<const float4*>(x + (long long)v * cols + block * 256 + 64 * j);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float4 value = xv[i];
          xs[4 * i] = value.x;
          xs[4 * i + 1] = value.y;
          xs[4 * i + 2] = value.z;
          xs[4 * i + 3] = value.w;
        }
        float x_low = 0.0f, x_high = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          x_low += xs[i];
          x_high += xs[32 + i];
        }
#pragma unroll
        for (int r = 0; r < DIRECT_VECTOR_ROWS; ++r) sums[r][v] += B::dot(decoded[r], xs, x_low, x_high, j);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < DIRECT_VECTOR_ROWS; ++r)
#pragma unroll
    for (int v = 0; v < MAX_VECTORS; ++v)
      if (v < vectors) {
        const float sum = gather_sum(sums[r][v], WAVE);
        if (threadIdx.x % WAVE == 0 && first_row + r < rows) y[(long long)v * rows + first_row + r] = sum;
      }
}

template <typename B>
__device__ void matvec_direct_vectors(const uint8_t* weight, const float* x, float* y, int rows, int cols,
                                      int vectors) {
  const int first_row = (blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE) * DIRECT_VECTOR_ROWS;
  if (first_row >= rows) return;
  direct_vector_rows<B>(weight, x, y, first_row, rows, cols, vectors);
}

// Which of up to four matrices a wave works on when their waves are numbered
// one matrix after the other, ceil(rows / rows_per_wave) each; false past the
// last.
struct Matrix {
  const uint8_t* weight;
  float* y;
  int rows, wave;
};
__device__ inline bool pick_matrix(int rows_per_wave, const uint8_t* w0, float* y0, int r0, const uint8_t* w1,
                                   float* y1, int r1, const uint8_t* w2, float* y2, int r2, const uint8_t* w3,
                                   float* y3, int r3, Matrix& m) {
  auto waves = [&](int rows) { return (rows + rows_per_wave - 1) / rows_per_wave; };
  int wave = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
  m = {w0, y0, r0, wave};
  if (wave < waves(r0)) return true;
  wave -= waves(r0);
  m = {w1, y1, r1, wave};
  if (wave < waves(r1)) return true;
  wave -= waves(r1);
  m = {w2, y2, r2, wave};
  if (wave < waves(r2)) return true;
  wave -= waves(r2);
  m = {w3, y3, r3, wave};
  return wave < waves(r3);
}

// Up to four matrices over the same x, as matvec_values_multi.
template <typename B>
__device__ void matvec_direct_multi(const uint8_t* w0, float* y0, int r0, const uint8_t* w1, float* y1, int r1,
                                    const uint8_t* w2, float* y2, int r2, const uint8_t* w3, float* y3, int r3,
                                    const float* x, int cols) {
  Matrix m;
  if (!pick_matrix(DIRECT_ROWS, w0, y0, r0, w1, y1, r1, w2, y2, r2, w3, y3, r3, m)) return;
  direct_rows<B>(m.weight, x, m.y, m.wave * DIRECT_ROWS, m.rows, cols);
}

template <typename B>
__device__ void matvec_direct_multi_vectors(const uint8_t* w0, float* y0, int r0, const uint8_t* w1, float* y1,
                                            int r1, const uint8_t* w2, float* y2, int r2, const uint8_t* w3,
                                            float* y3, int r3, const float* x, int cols, int vectors) {
  Matrix m;
  if (!pick_matrix(DIRECT_VECTOR_ROWS, w0, y0, r0, w1, y1, r1, w2, y2, r2, w3, y3, r3, m)) return;
  direct_vector_rows<B>(m.weight, x, m.y, m.wave * DIRECT_VECTOR_ROWS, m.rows, cols, vectors);
}

// Experts' products as matvec_experts, `group` consecutive rows (1, or 8 /
// the row's blocks) walked as one virtual row, two virtual rows per wave.
template <typename B>
__device__ __forceinline__ void matvec_experts_direct(long long index, const uint8_t* weight, long long expert_bytes, const int* ids,
                                      const float* x, int divisor, float* y, int rows, int cols, int slots,
                                      int group) {
  constexpr int VIRTUAL = 2;
  const int pairs = rows / (group * VIRTUAL);
  if (index >= (long long)slots * pairs) return;
  const int slot = index / pairs, first_row = (index % pairs) * group * VIRTUAL;
  const long long row_bytes = (long long)(cols / 256) * B::BYTES;
  const uint8_t* expert = weight + ids[slot] * expert_bytes;
  const uint8_t* starts[VIRTUAL] = {expert + first_row * row_bytes, expert + (first_row + group) * row_bytes};
  float sums[VIRTUAL] = {};
  direct<B, VIRTUAL>(starts, cols / 256 * group, x + (long long)(slot / divisor) * cols, cols, sums);
  const int width = WAVE / group, lane = threadIdx.x % WAVE;
#pragma unroll
  for (int v = 0; v < VIRTUAL; ++v) {
    const float sum = gather_sum(sums[v], width);
    if (lane % width == 0) y[(long long)slot * rows + first_row + v * group + lane / width] = sum;
  }
}

// silu(gate · x) × (up · x) as matvec_experts_gated: two rows per wave, their
// gate and up rows walked together. After the slots' waves, `shared_slots`
// more (one per token, or none) take the shared expert, of the same shapes:
// shared_gate and shared_up into shared_y [shared_slots][rows], so it pays no
// launch of its own. Gives which slot the wave served (the shared ones after
// the others), -1 for none.
template <typename B>
__device__ __forceinline__ int matvec_experts_gated_direct(long long index, const uint8_t* gate, const uint8_t* up,
                                            long long expert_bytes, const int* ids, const float* x, int divisor,
                                            float* y, int rows, int cols, int slots, const uint8_t* shared_gate,
                                            const uint8_t* shared_up, float* shared_y, int shared_slots) {
  const int pairs = rows / 2;
  const long long row_bytes = (long long)(cols / 256) * B::BYTES;
  const float* xs = x;
  const uint8_t *gates = gate, *ups = up;
  float* out = y;
  int slot, done;  // done: which of the slots, then the shared ones, the wave served
  if (index < (long long)slots * pairs) {
    slot = index / pairs;
    done = slot;
    const long long expert = ids[slot] * expert_bytes;
    gates += expert;
    ups += expert;
    xs += (long long)(slot / divisor) * cols;
  } else {
    index -= (long long)slots * pairs;
    if (index >= (long long)shared_slots * pairs) return -1;
    slot = index / pairs;
    done = slots + slot;
    gates = shared_gate;
    ups = shared_up;
    out = shared_y;
    xs += (long long)slot * cols;
  }
  const int row = (index % pairs) * 2;
  const long long at = row * row_bytes;
  const uint8_t* starts[4] = {gates + at, ups + at, gates + at + row_bytes, ups + at + row_bytes};
  float sums[4] = {};
  direct<B, 4>(starts, cols / 256, xs, cols, sums);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float g = gather_sum(sums[2 * r], WAVE), u = gather_sum(sums[2 * r + 1], WAVE);
    if (threadIdx.x % WAVE == 0) out[(long long)slot * rows + row + r] = g / (1.0f + expf(-g)) * u;
  }
  return done;
}

// ---- ROCmFP4 over int8 x --------------------------------------------------------------

// ROCmFP4 meets x quantized per 32 values (quantize_x), as its fork's MMVQ
// does: its codes are small integers, so four products take one v_dot4. Each
// lane loads its blocks as RocmFp4Direct does. Per weight and vector that is a
// quarter of an instruction where float x takes about three, which is what a
// verification's several vectors cost. One routine serves any number of
// vectors, so a verification's tokens get exactly a decode step's results.
// Each wave takes four consecutive rows (the last ones repeated at the end,
// not written); y [vectors][rows]. Rows of fewer than eight blocks of 256
// would leave lanes idle: GROUP (8 / the row's blocks) consecutive rows are
// then walked as one, lane l's sums belonging to row l / (WAVE / GROUP), four
// such groups per wave. A row's sum is the same either way: the lanes a
// group leaves out added zeros.
constexpr int INT8_ROWS = 2;

template <int SCALES, int GROUP>
__device__ __forceinline__ void rocmfp4_int8_rows(const uint8_t* weight, const int8_t* xq, const float* xd, float* y,
                                                  int first_row, int rows, int cols, int vectors) {
#pragma clang fp contract(off)  // see direct_rows
  using B = RocmFp4Direct<SCALES>;
  const long long row_bytes = (long long)(cols / 256) * B::BYTES;
  const int row_blocks = cols / 256, blocks = row_blocks * GROUP, lane = threadIdx.x % WAVE, j = lane % 4;
  const uint8_t* starts[INT8_ROWS];
#pragma unroll
  for (int r = 0; r < INT8_ROWS; ++r) starts[r] = weight + min(first_row + r * GROUP, rows - GROUP) * row_bytes;
  float sums[INT8_ROWS][MAX_VECTORS] = {};
  for (int pass = 0; pass < blocks; pass += WAVE / 4) {
    const int block = pass + lane / 4, x_block = GROUP == 1 ? block : block % row_blocks;
    if (block >= blocks) break;
    typename B::Lane loaded[INT8_ROWS];
#pragma unroll
    for (int r = 0; r < INT8_ROWS; ++r) loaded[r] = B::load(starts[r] + block * B::BYTES, j);
    // signed codes: block h's low nibbles (elements 0-15) in words 8h .. 8h+3,
    // its high nibbles (16-31) in 8h+4 .. 8h+7; the half-block scales
    uint32_t codes[INT8_ROWS][16];
    float scales[INT8_ROWS][4];
#pragma unroll
    for (int r = 0; r < INT8_ROWS; ++r)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int at = loaded[r].shift + h * SCALES;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t nibbles = __builtin_amdgcn_alignbyte(loaded[r].w[4 * h + i + 1], loaded[r].w[4 * h + i], at);
          codes[r][8 * h + i] = rocmfp4_codes(nibbles & 0x0F0F0F0F);
          codes[r][8 * h + 4 + i] = rocmfp4_codes((nibbles >> 4) & 0x0F0F0F0F);
        }
        const uint32_t scale_bytes = loaded[r].w[4 * h + 4] >> (8 * at);
        scales[r][2 * h] = rocmfp4_half_scale(scale_bytes & 0xFF);
        scales[r][2 * h + 1] = rocmfp4_half_scale((scale_bytes >> (8 * (SCALES - 1))) & 0xFF);
      }
#pragma unroll
    for (int v = 0; v < MAX_VECTORS; ++v)
      if (v < vectors) {
        const uint4* xw = reinterpret_cast<const uint4*>(xq + (long long)v * cols + x_block * 256 + 64 * j);
        uint32_t words[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 value = xw[i];
          words[4 * i] = value.x;
          words[4 * i + 1] = value.y;
          words[4 * i + 2] = value.z;
          words[4 * i + 3] = value.w;
        }
        const float2 d = *reinterpret_cast<const float2*>(xd + (long long)v * (cols / 32) + x_block * 8 + 2 * j);
#pragma unroll
        for (int r = 0; r < INT8_ROWS; ++r) {
          float sum = 0.0f;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            int low = 0, high = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              low = dot_ss(codes[r][8 * h + i], int(words[8 * h + i]), low);
              high = dot_ss(codes[r][8 * h + 4 + i], int(words[8 * h + 4 + i]), high);
            }
            sum += (h == 0 ? d.x : d.y) * (float(low) * scales[r][2 * h] + float(high) * scales[r][2 * h + 1]);
          }
          sums[r][v] += sum;
        }
      }
  }
  constexpr int WIDTH = WAVE / GROUP;
#pragma unroll
  for (int r = 0; r < INT8_ROWS; ++r)
#pragma unroll
    for (int v = 0; v < MAX_VECTORS; ++v)
      if (v < vectors) {
        const float sum = gather_sum(sums[r][v], WIDTH);
        const int row = first_row + r * GROUP + lane / WIDTH;
        if (lane % WIDTH == 0 && first_row + r * GROUP < rows) y[(long long)v * rows + row] = sum;
      }
}

// matvec_int8, and matvec_int8_group<G> for rows of 8 / G blocks of 256 (a
// multiple of G of them).
#define ROCMFP4_INT8_GROUP(name, scales, group)                                                            \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                                    \
      matvec_int8_group##group##_##name(const uint8_t* weight, const int8_t* xq, const float* xd, float* y, \
                                        int rows, int cols, int vectors) {                                \
    const int first_row = (blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE) * INT8_ROWS * group;        \
    if (first_row >= rows) return;                                                                        \
    rocmfp4_int8_rows<scales, group>(weight, xq, xd, y, first_row, rows, cols, vectors);                  \
  }
#define ROCMFP4_INT8(name, scales)                                                                         \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                                    \
      matvec_int8_##name(const uint8_t* weight, const int8_t* xq, const float* xd, float* y, int rows,     \
                         int cols, int vectors) {                                                         \
    const int first_row = (blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE) * INT8_ROWS;                \
    if (first_row >= rows) return;                                                                        \
    rocmfp4_int8_rows<scales, 1>(weight, xq, xd, y, first_row, rows, cols, vectors);                      \
  }                                                                                                       \
  ROCMFP4_INT8_GROUP(name, scales, 2)                                                                     \
  ROCMFP4_INT8_GROUP(name, scales, 4)                                                                     \
  ROCMFP4_INT8_GROUP(name, scales, 8)                                                                     \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK) matvec_int8_multi_##name(          \
      const uint8_t* w0, float* y0, int r0, const uint8_t* w1, float* y1, int r1, const uint8_t* w2,      \
      float* y2, int r2, const uint8_t* w3, float* y3, int r3, const int8_t* xq, const float* xd,         \
      int cols, int vectors) {                                                                            \
    Matrix m;                                                                                             \
    if (!pick_matrix(INT8_ROWS, w0, y0, r0, w1, y1, r1, w2, y2, r2, w3, y3, r3, m)) return;               \
    rocmfp4_int8_rows<scales, 1>(m.weight, xq, xd, m.y, m.wave * INT8_ROWS, m.rows, cols, vectors);       \
  }

ROCMFP4_INT8(q4_0_rocmfp4, 2)
ROCMFP4_INT8(q4_0_rocmfp4_fast, 1)

// ---- a prompt's experts on the matrix units (ROCmFP4) ----------------------------------
// An expert's batch of up to 16 slots (group_slots) against 16 of its weight
// rows per wave, as tiles of v_wmma_f32_16x16x16_f16: A = the slots' x rounded
// to F16 (as the dense GEMMs round x), B = the rows decoded to F16, which is
// exact (a code of at most 10 times a scale of 4 significant bits), the sums in
// F32. Lane l's A is slot l % 16's x, its B row l % 16's weights (lanes 16-31
// repeat 0-15); its accumulator holds, in element v, slot 2v + l / 16 and row
// l % 16 (as attention.hip's). With `up`, silu(gate) × up.

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// Values 16 step .. 16 step + 15 of a lane's decoded 64 (RocmFp4Direct): block
// step / 2, its low nibbles (elements 0-15) or high ones (16-31).
template <int SCALES>
__device__ inline half16 rocmfp4_half16(const typename RocmFp4Direct<SCALES>::Decoded& d, int step) {
  const int h = step / 2, part = step % 2;
  const float scale = d.scales[2 * h + (SCALES == 2 ? part : 0)];
  half16 out;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int b = 0; b < 4; ++b) out[4 * i + b] = (_Float16)(float(unsigned_byte(d.codes[8 * h + 4 * part + i], b) - 10) * scale);
  return out;
}

template <int SCALES>
__device__ void experts_matrix(const uint8_t* gate, const uint8_t* up, long long expert_bytes, const int* ids,
                               const int* order, const int* batches, const float* x, int divisor, float* y, int rows,
                               int cols, int max_batches) {
  using B = RocmFp4Direct<SCALES>;
  const long long index = wave_index();
  const int tiles = rows / 16;
  if (index >= (long long)max_batches * tiles) return;
  const int batch = index / tiles, first_row = (index % tiles) * 16;
  const int start = batches[2 * batch], count = batches[2 * batch + 1];
  if (count == 0) return;
  const int lane = threadIdx.x % WAVE, l = lane % 16;
  const float* xs = l < count ? x + (long long)(order[start + l] / divisor) * cols : nullptr;
  const long long row_bytes = (long long)(cols / 256) * B::BYTES;
  const long long at = ids[order[start]] * expert_bytes + (first_row + l) * row_bytes;
  float8 gates = {}, ups = {};
  for (int unit = 0; unit < cols / 64; ++unit) {
    const int j = unit % 4;
    const long long block = (long long)(unit / 4) * B::BYTES;
    const typename B::Decoded g = B::decode(B::load(gate + at + block, j), j);
    typename B::Decoded u;
    if (up) u = B::decode(B::load(up + at + block, j), j);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      half16 a = {};
      if (xs) {
        const float4* xv = reinterpret_cast<const float4*>(xs + unit * 64 + step * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float4 v = xv[i];
          a[4 * i] = (_Float16)v.x;
          a[4 * i + 1] = (_Float16)v.y;
          a[4 * i + 2] = (_Float16)v.z;
          a[4 * i + 3] = (_Float16)v.w;
        }
      }
      gates = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, rocmfp4_half16<SCALES>(g, step), gates);
      if (up) ups = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, rocmfp4_half16<SCALES>(u, step), ups);
    }
  }
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    const int r = 2 * v + lane / 16;
    if (r < count) {
      const float value = up ? gates[v] / (1.0f + expf(-gates[v])) * ups[v] : gates[v];
      y[(long long)order[start + r] * rows + first_row + l] = value;
    }
  }
}

#define EXPERTS_MATRIX(name, scales)                                                                    \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK) matvec_experts_matrix_##name(   \
      const uint8_t* gate, const uint8_t* up, long long expert_bytes, const int* ids, const int* order,  \
      const int* batches, const float* x, int divisor, float* y, int rows, int cols, int max_batches) { \
    experts_matrix<scales>(gate, up, expert_bytes, ids, order, batches, x, divisor, y, rows, cols,       \
                           max_batches);                                                                \
  }
EXPERTS_MATRIX(q4_0_rocmfp4, 2)
EXPERTS_MATRIX(q4_0_rocmfp4_fast, 1)

#define DIRECT(name, type)                                                                            \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                               \
      matvec_direct_##name(const uint8_t* weight, const float* x, float* y, int rows, int cols) {    \
    matvec_direct<type>(weight, x, y, rows, cols);                                                   \
  }                                                                                                  \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK) matvec_direct_multi_##name(   \
      const uint8_t* w0, float* y0, int r0, const uint8_t* w1, float* y1, int r1, const uint8_t* w2, \
      float* y2, int r2, const uint8_t* w3, float* y3, int r3, const float* x, int cols) {           \
    matvec_direct_multi<type>(w0, y0, r0, w1, y1, r1, w2, y2, r2, w3, y3, r3, x, cols);              \
  }                                                                                                  \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK) matvec_direct_vectors_##name( \
      const uint8_t* weight, const float* x, float* y, int rows, int cols, int vectors) {            \
    matvec_direct_vectors<type>(weight, x, y, rows, cols, vectors);                                  \
  }                                                                                                  \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                               \
      matvec_direct_multi_vectors_##name(const uint8_t* w0, float* y0, int r0, const uint8_t* w1,    \
                                         float* y1, int r1, const uint8_t* w2, float* y2, int r2,    \
                                         const uint8_t* w3, float* y3, int r3, const float* x,       \
                                         int cols, int vectors) {                                    \
    matvec_direct_multi_vectors<type>(w0, y0, r0, w1, y1, r1, w2, y2, r2, w3, y3, r3, x, cols,       \
                                      vectors);                                                      \
  }                                                                                                  \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK) matvec_experts_direct_##name( \
      const uint8_t* weight, long long expert_bytes, const int* ids, const float* x, int divisor,    \
      float* y, int rows, int cols, int slots, int group) {                                          \
    matvec_experts_direct<type>(wave_index(), weight, expert_bytes, ids, x, divisor, y, rows, cols, slots, \
                                group);                                                              \
  }                                                                                                  \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)                               \
      matvec_experts_gated_direct_##name(const uint8_t* gate, const uint8_t* up,                     \
                                         long long expert_bytes, const int* ids, const float* x,     \
                                         int divisor, float* y, int rows, int cols, int slots,       \
                                         const uint8_t* shared_gate, const uint8_t* shared_up,       \
                                         float* shared_y, int shared_slots) {                        \
    matvec_experts_gated_direct<type>(wave_index(), gate, up, expert_bytes, ids, x, divisor, y, rows, \
                                      cols, slots,                                                   \
                                      shared_gate, shared_up, shared_y, shared_slots);               \
  }

DIRECT(q4_k, Q4KDirect)
DIRECT(q5_k, Q5KDirect)
DIRECT(iq4_xs, IQ4XSDirect)
DIRECT(q4_0_rocmfp4, RocmFp4Direct<2>)
DIRECT(q4_0_rocmfp4_fast, RocmFp4Direct<1>)

// F32 matrices of few rows (routers): a workgroup per F32_ROWS rows, each
// thread loading its float4s of every row straight from memory (all in flight
// at once, x loaded once), each row's sum gathered through LDS; one vector
// after another, each summed as a lone vector is (the rows' later reads come
// from the cache). A row sums the same whatever the rows beside it.
constexpr int F32_ROWS = 4;

__device__ __forceinline__ void rows_f32(int first_row, const float* weight, const float* x, float* y, int rows,
                                         int cols, int vectors) {
#pragma clang fp contract(off)  // only the fmaf below: the same sums for any vector count
  __shared__ float partial[F32_ROWS][WAVES_PER_BLOCK];
  typedef float floats4 __attribute__((ext_vector_type(4)));
  const floats4* row[F32_ROWS];
#pragma unroll
  for (int r = 0; r < F32_ROWS; ++r)
    row[r] = reinterpret_cast<const floats4*>(weight + (long long)min(first_row + r, rows - 1) * cols);
  for (int vector = 0; vector < vectors; ++vector) {
    const floats4* xv = reinterpret_cast<const floats4*>(x + (long long)vector * cols);
    float sum[F32_ROWS] = {};
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) {
      const floats4 v = xv[i];
#pragma unroll
      for (int r = 0; r < F32_ROWS; ++r) {
        const floats4 w = vectors == 1 ? __builtin_nontemporal_load(row[r] + i) : row[r][i];
        sum[r] = fmaf(w.w, v.w, fmaf(w.z, v.z, fmaf(w.y, v.y, fmaf(w.x, v.x, sum[r]))));
      }
    }
#pragma unroll
    for (int r = 0; r < F32_ROWS; ++r) {
      sum[r] = gather_sum(sum[r], WAVE);
      if (threadIdx.x % WAVE == 0) partial[r][threadIdx.x / WAVE] = sum[r];
    }
    __syncthreads();
    if (threadIdx.x < F32_ROWS && first_row + threadIdx.x < rows) {
      float total = 0.0f;
      for (int w = 0; w < WAVES_PER_BLOCK; ++w) total += partial[threadIdx.x][w];
      y[(long long)vector * rows + first_row + threadIdx.x] = total;
    }
    __syncthreads();  // partial is written again for the next vector
  }
}

extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)
    matvec_rows_f32(const float* weight, const float* x, float* y, int rows, int cols, int vectors) {
  rows_f32(blockIdx.x * F32_ROWS, weight, x, y, rows, cols, vectors);
}

// A router and its top-k routing (route.h) in one launch: the router's
// logits as matvec_rows_f32 makes them, then the last workgroup to finish
// (`finished` counts them, and is left at 0) routes each vector with one of
// its waves (vectors <= WAVES_PER_BLOCK), reading the other workgroups'
// logits past the caches.
extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK)
    matvec_rows_f32_route(const float* weight, const float* x, float* y, int rows, int cols, int vectors, int* ids,
                          float* route_weights, int k, unsigned* finished) {
  __shared__ bool last;
  rows_f32(blockIdx.x * F32_ROWS, weight, x, y, rows, cols, vectors);
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(finished, 1u) == gridDim.x - 1;
    if (last) *finished = 0;
    __threadfence();
  }
  __syncthreads();
  const int vector = threadIdx.x / WAVE;
  if (!last || vector >= vectors) return;
  const float* logits = y + (long long)vector * rows;
  route_token([&](int i) { return __hip_atomic_load(logits + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); },
              ids + (long long)vector * k, route_weights + (long long)vector * k, rows, k);
}

// ---- a mixture of experts in one launch (ROCmFP4) ------------------------------------
// A decode step's experts once routed: the shared expert's and the routed
// experts' gate and up (matvec_experts_gated_direct), the shared expert's down
// over int8 x (rocmfp4_int8_rows), the routed experts' down
// (matvec_experts_direct), then their sum into the residual stream and its
// RMS norm (moe_combine_norm), each computed as its own launch computes it, so
// the results are those launches' to the bit. The phases follow one another in
// the grid, and a wave that needs an earlier phase's results waits for them: a
// slot's down waits for that slot's gate and up only, so the products' ramps
// and tails overlap. The workgroups take their places in the order they start
// (a ticket), so the ones a workgroup waits on have started and never wait on
// it.
//
// The sum: 32 waves per token, wave w taking the values thread w * 32 + lane of
// moe_combine_norm's 1024 would (every 1024th from there), so that its sums of
// squares reduce as that thread's wave's did; each wave's total goes through
// `totals`, and every wave adds the 32 in order.
//
// `sync` holds counters, zero between launches (the last workgroup to finish
// leaves them so): the ticket, the workgroups finished, the down waves done,
// each token's sum waves done, then the gate and up waves done per routed slot
// and per token's shared expert.
enum MoeCounter { TICKET, FINISHED, DOWN_WAVES, SUMMED, GATE_WAVES = SUMMED + MAX_VECTORS };
constexpr int SUM_WAVES = 32;

__device__ inline void wait_for(const unsigned* counter, unsigned count) {
  while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < count) __builtin_amdgcn_s_sleep(2);
}

__device__ inline unsigned add_release(unsigned* counter) {
  return __hip_atomic_fetch_add(counter, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}

// The shared expert's down input quantized per 32 values in LDS: up to this
// many.
constexpr int MOE_SHARED_VALUES = 8192;

// What the sum takes besides the experts' outputs: the route's weights, the
// shared expert's gate vector, `gates` (each token's sigmoid(x · router),
// computed early by a workgroup of its own) followed by the waves' `totals`
// (SUM_WAVES per token), the residual stream (in and out) and the norm into
// `normed`, quantized too when `xq` is given. No residual: no sum, the experts'
// outputs only.
struct MoeSum {
  const float* route_weights;
  const float* shared_router;
  float* gates;
  float* residual;
  const float* norm_weight;
  float* normed;
  float eps, offset;
  Quantized quantized;
};

// GROUP: rows of the down projections walked at once (8 / the intermediate
// size's blocks of 256), as matvec_int8_group and matvec_experts_direct do.
template <int SCALES, int GROUP>
__device__ void moe_rocmfp4(const float* x, const int* ids, int hidden, int tokens, int k, const uint8_t* gate,
                            const uint8_t* up, long long gate_bytes, float* gate_out, int intermediate,
                            const uint8_t* shared_gate, const uint8_t* shared_up, float* shared_gate_out,
                            const uint8_t* down, long long down_bytes, float* expert_out, const uint8_t* shared_down,
                            float* shared_out, const MoeSum& sum, unsigned* sync) {
  using B = RocmFp4Direct<SCALES>;
  __shared__ unsigned ticket;
  __shared__ float partial[256];
  __shared__ int8_t shared_xq[MOE_SHARED_VALUES];
  __shared__ float shared_xd[MOE_SHARED_VALUES / 32], shared_xs[MOE_SHARED_VALUES / 32];
  if (threadIdx.x == 0) ticket = add_release(sync + TICKET);
  __syncthreads();
  const int wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
  const int slots = tokens * k, pairs = intermediate / 2;
  // the phases' workgroups: the shared expert's gates, its gate and up, its
  // down, the routed gate and up, the routed down
  auto workgroups = [](long long waves) { return int((waves + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK); };
  const long long shared_gate_waves = (long long)tokens * pairs, gate_waves = (long long)slots * pairs;
  const long long down_waves = (long long)slots * (hidden / (GROUP * 2));
  const int shared_downs = workgroups(hidden / (INT8_ROWS * GROUP)), gate_workgroups = sum.residual ? tokens : 0;
  int at = ticket;
  if (at < gate_workgroups) {
    const float gate_value = shared_expert_gate(x + (long long)at * hidden, sum.shared_router, hidden, partial);
    if (threadIdx.x == 0) {
      sum.gates[at] = gate_value;
      add_release(sync + DOWN_WAVES);  // the sum waits for the gates with the downs
    }
  } else if ((at -= gate_workgroups) < workgroups(shared_gate_waves)) {
    const long long index = (long long)at * WAVES_PER_BLOCK + wave;
    if (index < shared_gate_waves) {
      const int done = matvec_experts_gated_direct<B>(gate_waves + index, gate, up, gate_bytes, ids, x, k, gate_out,
                                                      intermediate, hidden, slots, shared_gate, shared_up,
                                                      shared_gate_out, tokens);
      if (lane == 0) add_release(sync + GATE_WAVES + done);
    }
  } else if ((at -= workgroups(shared_gate_waves)) < shared_downs) {
    // every token's shared gate and up, quantized here as quantize_x would
    for (int t = 0; t < tokens; ++t) wait_for(sync + GATE_WAVES + slots + t, pairs);
    for (int i = threadIdx.x; i < tokens * intermediate; i += blockDim.x)
      quantize_block(shared_gate_out[i], i / 32, shared_xq, shared_xd, shared_xs);
    __syncthreads();
    const int first_row = (at * WAVES_PER_BLOCK + wave) * INT8_ROWS * GROUP;
    if (first_row < hidden)
      rocmfp4_int8_rows<SCALES, GROUP>(shared_down, shared_xq, shared_xd, shared_out, first_row, hidden,
                                       intermediate, tokens);
    if (lane == 0) add_release(sync + DOWN_WAVES);
  } else if ((at -= shared_downs) < workgroups(gate_waves)) {
    const long long index = (long long)at * WAVES_PER_BLOCK + wave;
    if (index < gate_waves) {
      const int done = matvec_experts_gated_direct<B>(index, gate, up, gate_bytes, ids, x, k, gate_out, intermediate,
                                                      hidden, slots, shared_gate, shared_up, shared_gate_out, 0);
      if (lane == 0) add_release(sync + GATE_WAVES + done);
    }
  } else if ((at -= workgroups(gate_waves)) < workgroups(down_waves)) {
    const long long index = (long long)at * WAVES_PER_BLOCK + wave;
    if (index < down_waves) {
      wait_for(sync + GATE_WAVES + index / (hidden / (GROUP * 2)), pairs);
      matvec_experts_direct<B>(index, down, down_bytes, ids, gate_out, 1, expert_out, hidden, intermediate, slots,
                               GROUP);
      if (lane == 0) add_release(sync + DOWN_WAVES);
    }
  } else if (sum.residual && (at -= workgroups(down_waves)) < tokens * SUM_WAVES / WAVES_PER_BLOCK) {
    // every down wave, routed and shared, and every gate
    const int t = at / (SUM_WAVES / WAVES_PER_BLOCK), w = at % (SUM_WAVES / WAVES_PER_BLOCK) * WAVES_PER_BLOCK + wave;
    wait_for(sync + DOWN_WAVES, unsigned(down_waves + shared_downs * WAVES_PER_BLOCK + tokens));
    const float gate_value = __hip_atomic_load(sum.gates + t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    float* row = sum.residual + (long long)t * hidden;
    float squares = 0.0f;
    for (int c = w * WAVE + lane; c < hidden; c += SUM_WAVES * WAVE) {
      const float value = combined(expert_out, sum.route_weights, shared_out, gate_value, sum.residual, t, c, k, hidden);
      row[c] = value;
      squares += value * value;
    }
    for (int step = WAVE / 2; step > 0; step /= 2) squares += __shfl_xor(squares, step);
    float* totals = sum.gates + tokens + t * SUM_WAVES;
    if (lane == 0) {
      totals[w] = squares;
      add_release(sync + SUMMED + t);
    }
    wait_for(sync + SUMMED + t, SUM_WAVES);
    float total = 0.0f;
    for (int i = 0; i < SUM_WAVES; ++i) total += __hip_atomic_load(totals + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const float scale = rsqrtf(total / hidden + sum.eps);
    for (int c = w * WAVE + lane; c < hidden; c += SUM_WAVES * WAVE) {
      const float value = row[c] * scale * (sum.norm_weight[c] + sum.offset);
      sum.normed[(long long)t * hidden + c] = value;
      if (sum.quantized.xq) quantize_block(value, ((long long)t * hidden + c) / 32, sum.quantized.xq, sum.quantized.xd, sum.quantized.xs);
    }
  }
  // the last workgroup to finish leaves the counters at zero
  __syncthreads();
  if (threadIdx.x == 0 && add_release(sync + FINISHED) == gridDim.x - 1)
    for (int i = 0; i < GATE_WAVES + slots + tokens; ++i) sync[i] = 0;
}

#define MOE_ROCMFP4(name, scales, group)                                                                    \
  extern "C" __global__ void __launch_bounds__(WAVE * WAVES_PER_BLOCK) moe_group##group##_##name(         \
      const float* x, const int* ids, int hidden, int tokens, int k, const uint8_t* gate, const uint8_t* up,  \
      long long gate_bytes, float* gate_out, int intermediate, const uint8_t* shared_gate,                 \
      const uint8_t* shared_up, float* shared_gate_out, const uint8_t* down, long long down_bytes,         \
      float* expert_out, const uint8_t* shared_down, float* shared_out, const float* route_weights,        \
      const float* shared_router, float* gates, float* residual, const float* norm_weight, float* normed,  \
      float eps, float offset, int8_t* xq, float* xd, float* xs, unsigned* sync) {                          \
    const MoeSum sum{route_weights, shared_router, gates, residual, norm_weight, normed, eps, offset,        \
                     Quantized{xq, xd, xs}};                                                                \
    moe_rocmfp4<scales, group>(x, ids, hidden, tokens, k, gate, up, gate_bytes, gate_out, intermediate,     \
                               shared_gate, shared_up, shared_gate_out, down, down_bytes, expert_out,       \
                               shared_down, shared_out, sum, sync);                                         \
  }
#define MOE_ROCMFP4_GROUPS(name, scales) \
  MOE_ROCMFP4(name, scales, 1)           \
  MOE_ROCMFP4(name, scales, 2)           \
  MOE_ROCMFP4(name, scales, 4)           \
  MOE_ROCMFP4(name, scales, 8)

MOE_ROCMFP4_GROUPS(q4_0_rocmfp4, 2)
MOE_ROCMFP4_GROUPS(q4_0_rocmfp4_fast, 1)
