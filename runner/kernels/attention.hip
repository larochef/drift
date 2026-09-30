#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "block.h"
#include "rope.h"

// Flash attention over a paged F16 KV cache, with the WMMA instructions of
// RDNA 3 (v_wmma_f32_16x16x16_f16, wave32).
//
// The cache: pages of P tokens (P a multiple of 16), keys as [page][P][kv head][D]
// and values transposed within a page, [page][kv head][D][P], so that both
// matrix operands below load as 32 contiguous bytes per lane. A sequence's page
// table maps its position / P to a page.
//
// One wave per (tile of 16 query rows, kv head, key split). A query row is a
// (token, query head of the kv head's group) pair, so a GQA group shares every
// key and value load. Per block of 16 keys:
//   Sᵀ = K · Qᵀ   (A = 16 keys × 16 dims, B = 16 dims × 16 rows, over D/16 steps)
//   online softmax per row
//   Oᵀ += Vᵀ · Pᵀ (A = 16 dims × 16 keys, B = 16 keys × 16 rows, per 16 dims)
// The accumulator of lane l holds, in element v, row 2v + l/16 and column
// l % 16 (checked on gfx1151); with the products transposed, the column is the
// query row, so each lane owns one row: its softmax needs one shuffle with lane
// l ^ 16, and rescaling O is one factor per lane.
//
// With several key splits, each wave writes its row's (max, sum, O) and
// attention_combine merges them (flash decoding).
//
// With `token_tiles` (a decode step, a verification's few tokens), a tile holds
// one token's rows only, keys past its token are zeroed, and its splits cut the
// token's own keys (up to TOKEN_SPLITS runs of at least MIN_SPLIT_BLOCKS
// blocks; the splits past them stay empty), so a token's sums are the same
// however many tokens come with it: the matrix units' sums move by an ulp when
// a masked key's value is there rather than zero.

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int WAVE = 32;

__device__ inline float8 wmma(half16 a, half16 b, float8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

struct AttentionArguments {
  const float* q;          // [tokens, q heads, D]
  const _Float16* keys;    // [pages, P, kv heads, D]
  const _Float16* values;  // [pages, kv heads, D, P]
  const int* page_table;
  const float* sinks;      // [q heads] or null
  float* out;              // [tokens, q heads, D]
  float* partial_o;        // [kv heads, rows, splits, D]
  float* partial_ml;       // [kv heads, rows, splits, 2]
  int tokens, q_heads, kv_heads, page_size;
  int query_start;         // position of the first query token
  int key_count;           // keys in the cache, the queries' own included
  float scale;
  int causal;
  int window;              // 0: none; else keys with position > query - window
  float softcap;           // 0: none
  int splits;
  int token_tiles;         // 1: one token per tile, SPLIT_BLOCKS blocks per split
};

constexpr int TOKEN_SPLITS = 32, MIN_SPLIT_BLOCKS = 2;

template <int D>
__device__ void attention(const AttentionArguments& a) {
  constexpr int CHUNKS = D / 16;
  __shared__ __attribute__((aligned(32))) _Float16 q_tile[16][D];

  const int lane = threadIdx.x, half = lane / 16, column = lane % 16;
  const int kv_head = blockIdx.y, split = blockIdx.z;
  const int group = a.q_heads / a.kv_heads;
  const int rows = a.tokens * group;
  // the tile's rows: 16 consecutive ones, or 16 of one token's
  const int token_tiles = (group + 15) / 16;
  const int tile_token = a.token_tiles ? blockIdx.x / token_tiles : 0;
  const int first_row = a.token_tiles ? tile_token * group + (blockIdx.x % token_tiles) * 16 : blockIdx.x * 16;
  const int end_row = a.token_tiles ? (tile_token + 1) * group : rows;

  // the tile's queries, as F16, in LDS: four values per load
  for (int i = lane; i < 16 * D / 4; i += WAVE) {
    const int r = i / (D / 4), d = (i % (D / 4)) * 4, row = first_row + r;
    float4 value = {0.0f, 0.0f, 0.0f, 0.0f};
    if (row < end_row) {
      const int token = row / group, head = kv_head * group + row % group;
      value = *reinterpret_cast<const float4*>(a.q + ((long long)token * a.q_heads + head) * D + d);
    }
    q_tile[r][d] = (_Float16)value.x;
    q_tile[r][d + 1] = (_Float16)value.y;
    q_tile[r][d + 2] = (_Float16)value.z;
    q_tile[r][d + 3] = (_Float16)value.w;
  }
  __syncthreads();

  // this lane's row
  const int row = first_row + column;
  const bool row_valid = row < end_row;
  const int token = row_valid ? row / group : 0;
  const int head = kv_head * group + (row_valid ? row % group : 0);
  const int position = a.query_start + token;

  // the keys this tile can see, in blocks of 16, and this split's share
  int last_key = a.key_count - 1;
  int first_key = 0;
  if (a.causal) last_key = min(last_key, a.query_start + min(first_row + 15, end_row - 1) / group);
  if (a.window > 0) first_key = max(0, a.query_start + first_row / group - a.window + 1);
  const int first_block = first_key / 16, blocks = last_key / 16 + 1 - first_block;
  const int split_blocks = max(MIN_SPLIT_BLOCKS, (blocks + TOKEN_SPLITS - 1) / TOKEN_SPLITS);
  const int begin = a.token_tiles ? first_block + split * split_blocks
                                  : first_block + (int)((long long)blocks * split / a.splits);
  const int end = a.token_tiles ? min(first_block + blocks, begin + split_blocks)
                                : first_block + (int)((long long)blocks * (split + 1) / a.splits);

  float8 o[CHUNKS];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) o[c] = float8{};
  float m = -INFINITY, l = 0.0f;

  for (int block = begin; block < end; ++block) {
    const int key0 = block * 16;
    const int page = a.page_table[key0 / a.page_size];
    const int offset = key0 % a.page_size;

    // Sᵀ = K · Qᵀ
    float8 s = {};
    const int my_key = key0 + column;
    const _Float16* key_row =
        a.keys + (((long long)page * a.page_size + offset + column) * a.kv_heads + kv_head) * D;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      half16 k = my_key <= last_key ? *reinterpret_cast<const half16*>(key_row + c * 16) : half16{};
      const half16 q = *reinterpret_cast<const half16*>(&q_tile[column][c * 16]);
      s = wmma(k, q, s);
    }

    // scale, cap and mask this lane's eight keys: 2v + half
    float p[8];
    float block_max = -INFINITY;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const int key = key0 + 2 * v + half;
      float value = s[v] * a.scale;
      if (a.softcap > 0.0f) value = a.softcap * tanhf(value / a.softcap);
      const bool visible = row_valid && key < a.key_count && (!a.causal || key <= position) &&
                           (a.window <= 0 || key > position - a.window);
      p[v] = visible ? value : -INFINITY;
      block_max = fmaxf(block_max, p[v]);
    }
    block_max = fmaxf(block_max, __shfl_xor(block_max, 16, WAVE));
    // No lane may skip the WMMAs below (they need the whole wave), so a row
    // with nothing visible yet gets zero weights rather than a `continue`.
    const float m_new = fmaxf(m, block_max);
    const bool any = m_new != -INFINITY;
    const float alpha = any ? __expf(m - m_new) : 1.0f;
    float sum = 0.0f;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      p[v] = any ? __expf(p[v] - m_new) : 0.0f;
      sum += p[v];
    }
    sum += __shfl_xor(sum, 16, WAVE);
    l = l * alpha + sum;
    m = m_new;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) o[c] *= alpha;

    // Pᵀ as B: this lane's row over the 16 keys, the odd or even half from its partner
    half16 pt;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const float other = __shfl_xor(p[v], 16, WAVE);
      pt[2 * v] = (_Float16)(half == 0 ? p[v] : other);
      pt[2 * v + 1] = (_Float16)(half == 0 ? other : p[v]);
    }

    // Oᵀ += Vᵀ · Pᵀ, keys past the tile's last zeroed (past the cache, their
    // pages may hold anything)
    const _Float16* value_rows =
        a.values + (((long long)page * a.kv_heads + kv_head) * D + column) * a.page_size + offset;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      half16 vt = *reinterpret_cast<const half16*>(value_rows + (long long)c * 16 * a.page_size);
      if (key0 + 15 > last_key) {
#pragma unroll
        for (int k = 0; k < 16; ++k)
          if (key0 + k > last_key) vt[k] = (_Float16)0.0f;
      }
      o[c] = wmma(vt, pt, o[c]);
    }
  }

  if (!row_valid) return;
  // element v of chunk c is dimension 16c + 2v + half of this lane's row
  if (a.splits == 1) {
    if (a.sinks) {
      const float sink = a.sinks[head];
      const float m_new = fmaxf(m, sink);
      l = l * __expf(m - m_new) + __expf(sink - m_new);
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c) o[c] *= __expf(m - m_new);
    }
    // divided as attention_combine divides, which a lone split's sums pass
    // through unchanged
    float* target = a.out + ((long long)token * a.q_heads + head) * D;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
      for (int v = 0; v < 8; ++v) target[16 * c + 2 * v + half] = l > 0.0f ? o[c][v] / l : 0.0f;
  } else {
    const long long slot = ((long long)kv_head * rows + row) * a.splits + split;
    float* target = a.partial_o + slot * D;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
      for (int v = 0; v < 8; ++v) target[16 * c + 2 * v + half] = o[c][v];
    if (half == 0) {
      a.partial_ml[slot * 2] = m;
      a.partial_ml[slot * 2 + 1] = l;
    }
  }
}

#define ATTENTION(d)                                                                            \
  extern "C" __global__ void __launch_bounds__(WAVE) attention_d##d(AttentionArguments a) {    \
    attention<d>(a);                                                                           \
  }
ATTENTION(64)
ATTENTION(80)
ATTENTION(128)
ATTENTION(256)

constexpr float LOG2E = 1.4426950408889634f;

// The same attention for many query rows at once (prefill, diffusion): a
// workgroup of WAVES waves, each its own 16 rows as above, shares every block
// of KEYS keys, loaded once into LDS (keys [KEYS][D], values transposed
// [D][KEYS], rows padded by 16 halves against bank conflicts), so the cache
// is read once per workgroup instead of once per wave. The queries stay in
// registers as the products' B operands. No key splits.
template <int D, int WAVES, int KEYS>
__device__ void attention_tiled(const AttentionArguments& a) {
  constexpr int CHUNKS = D / 16, TILES = KEYS / 16, PAD = 16, THREADS = WAVES * WAVE;
  __shared__ __attribute__((aligned(32))) _Float16 k_tile[KEYS][D + PAD];
  __shared__ __attribute__((aligned(32))) _Float16 v_tile[D][KEYS + PAD];

  const int wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE, half = lane / 16, column = lane % 16;
  const int kv_head = blockIdx.y;
  const int group = a.q_heads / a.kv_heads;
  const int rows = a.tokens * group;
  const int group_first_row = blockIdx.x * WAVES * 16;
  const int first_row = group_first_row + wave * 16;

  // this lane's row and its queries, as F16
  const int row = first_row + column;
  const bool row_valid = row < rows;
  const int token = row_valid ? row / group : 0;
  const int head = kv_head * group + (row_valid ? row % group : 0);
  const int position = a.query_start + token;
  half16 q[CHUNKS];
  {
    const float* source = a.q + ((long long)token * a.q_heads + head) * D;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
      for (int i = 0; i < 16; i += 4) {
        const float4 value = row_valid ? *reinterpret_cast<const float4*>(source + c * 16 + i)
                                       : float4{0.0f, 0.0f, 0.0f, 0.0f};
        q[c][i] = (_Float16)value.x;
        q[c][i + 1] = (_Float16)value.y;
        q[c][i + 2] = (_Float16)value.z;
        q[c][i + 3] = (_Float16)value.w;
      }
  }

  // the keys the workgroup can see, in blocks of KEYS
  int last_key = a.key_count - 1;
  int first_key = 0;
  const int last_token = min(a.tokens - 1, (group_first_row + WAVES * 16 - 1) / group);
  if (a.causal) last_key = min(last_key, a.query_start + last_token);
  if (a.window > 0) first_key = max(0, a.query_start + group_first_row / group - a.window + 1);

  float8 o[CHUNKS];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) o[c] = float8{};
  float m = -INFINITY, l = 0.0f;

  // a block's keys and values from the cache, 8 halves per load
  // (the last round may be partial: heads of 80 are not a multiple of it)
  constexpr int LOADS = (KEYS * D / 8 + THREADS - 1) / THREADS;
  uint4 key_loads[LOADS], value_loads[LOADS];
  auto fetch = [&](int key0) {
#pragma unroll
    for (int n = 0; n < LOADS; ++n) {
      const int i = threadIdx.x + n * THREADS;
      if (i >= KEYS * D / 8) break;
      {
        const int key = i / (D / 8), part = i % (D / 8), k = key0 + key;
        key_loads[n] = uint4{0, 0, 0, 0};
        if (k < a.key_count) {
          const int page = a.page_table[k / a.page_size];
          key_loads[n] = *reinterpret_cast<const uint4*>(
              a.keys + (((long long)page * a.page_size + k % a.page_size) * a.kv_heads + kv_head) * D + part * 8);
        }
      }
      {
        // values are transposed in the cache: 8 keys of one dimension
        const int dim = i / (KEYS / 8), part = i % (KEYS / 8), k = key0 + part * 8;
        value_loads[n] = uint4{0, 0, 0, 0};
        if (k < a.key_count) {
          const int page = a.page_table[k / a.page_size];
          value_loads[n] = *reinterpret_cast<const uint4*>(
              a.values + (((long long)page * a.kv_heads + kv_head) * D + dim) * a.page_size + k % a.page_size);
        }
      }
    }
  };
  // into LDS, keys past the cache zeroed (their pages may hold anything)
  auto store = [&](int key0) {
#pragma unroll
    for (int n = 0; n < LOADS; ++n) {
      const int i = threadIdx.x + n * THREADS;
      if (i >= KEYS * D / 8) break;
      *reinterpret_cast<uint4*>(&k_tile[i / (D / 8)][i % (D / 8) * 8]) = key_loads[n];
      const int dim = i / (KEYS / 8), part = i % (KEYS / 8), k = key0 + part * 8;
      uint4 value = value_loads[n];
      if (k + 8 > a.key_count) {
        const int kept = max(0, a.key_count - k);
        unsigned* words = reinterpret_cast<unsigned*>(&value);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          if (2 * w >= kept) words[w] = 0;
          else if (2 * w + 1 >= kept) words[w] &= 0xffffu;
        }
      }
      *reinterpret_cast<uint4*>(&v_tile[dim][part * 8]) = value;
    }
  };

  // (fetching the next block during the products instead costs occupancy
  // and was slower)
  for (int key0 = first_key / KEYS * KEYS; key0 <= last_key; key0 += KEYS) {
    fetch(key0);
    store(key0);
    __syncthreads();

    // Sᵀ = K · Qᵀ, per 16 keys
    // (the tiles' products interleaved: independent accumulators)
    float8 s[TILES];
#pragma unroll
    for (int t = 0; t < TILES; ++t) s[t] = float8{};
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
      for (int t = 0; t < TILES; ++t)
        s[t] = wmma(*reinterpret_cast<const half16*>(&k_tile[16 * t + column][c * 16]), q[c], s[t]);

    // scale (in log2 units: exp2 below), cap and mask this lane's keys,
    // 16t + 2v + half; a block every row sees whole skips the mask
    const bool whole = row_valid && key0 >= first_key && key0 + KEYS <= a.key_count &&
                       (!a.causal || key0 + KEYS - 1 <= position) &&
                       (a.window <= 0 || key0 > position - a.window);
    float block_max = -INFINITY;
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        float value = s[t][v] * a.scale;
        if (a.softcap > 0.0f) value = a.softcap * tanhf(value / a.softcap);
        value *= LOG2E;
        if (!whole) {
          const int key = key0 + 16 * t + 2 * v + half;
          const bool visible = row_valid && key < a.key_count && key >= first_key &&
                               (!a.causal || key <= position) && (a.window <= 0 || key > position - a.window);
          value = visible ? value : -INFINITY;
        }
        s[t][v] = value;
        block_max = fmaxf(block_max, value);
      }
    block_max = fmaxf(block_max, __shfl_xor(block_max, 16, WAVE));
    const float m_new = fmaxf(m, block_max);
    const bool any = m_new != -INFINITY;
    const float alpha = any ? exp2f(m - m_new) : 1.0f;
    float sum = 0.0f;
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        s[t][v] = any ? exp2f(s[t][v] - m_new) : 0.0f;
        sum += s[t][v];
      }
    sum += __shfl_xor(sum, 16, WAVE);
    l = l * alpha + sum;
    m = m_new;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) o[c] *= alpha;

    // Oᵀ += Vᵀ · Pᵀ, Pᵀ per 16 keys as B: this lane's row, the odd or even
    // half of the keys from its partner
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
      half16 pt;
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const float other = __shfl_xor(s[t][v], 16, WAVE);
        pt[2 * v] = (_Float16)(half == 0 ? s[t][v] : other);
        pt[2 * v + 1] = (_Float16)(half == 0 ? other : s[t][v]);
      }
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c)
        o[c] = wmma(*reinterpret_cast<const half16*>(&v_tile[c * 16 + column][16 * t]), pt, o[c]);
    }
    __syncthreads();  // the tiles are reloaded next
  }

  if (!row_valid) return;
  if (a.sinks) {
    // m is in log2 units here
    const float sink = a.sinks[head] * LOG2E;
    const float m_new = fmaxf(m, sink);
    l = l * exp2f(m - m_new) + exp2f(sink - m_new);
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) o[c] *= exp2f(m - m_new);
  }
  const float inverse = l > 0.0f ? 1.0f / l : 0.0f;
  float* target = a.out + ((long long)token * a.q_heads + head) * D;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
    for (int v = 0; v < 8; ++v) target[16 * c + 2 * v + half] = o[c][v] * inverse;
}

// 8 waves over blocks of 64 keys measured fastest at 4608 and 8704 tokens
// (FLUX.2 at 1024², alone and with a reference), against 4 or 16 waves and
// blocks of 32.
constexpr int TILED_WAVES = 8, TILED_KEYS = 64;
// Each head width with the waves per SIMD it is compiled for, which caps its
// registers: left free, the compiler kept up to 256 live at heads of 64 and
// spilled inside the loop (half the speed), and fit fewer workgroups per WGP
// (two waves each per SIMD) at heads of 80 and 128. 8 leaves 192 registers
// (heads of 64 and 80), 6 leaves 240 (128); what spills then is a few
// scalars.
#define ATTENTION_TILED(d, waves_per_simd)                                                   \
  extern "C" __global__ void __launch_bounds__(WAVE * TILED_WAVES)                           \
      __attribute__((amdgpu_waves_per_eu(waves_per_simd))) attention_tiled_d##d(             \
          AttentionArguments a) {                                                            \
    attention_tiled<d, TILED_WAVES, TILED_KEYS>(a);                                          \
  }
ATTENTION_TILED(64, 8)
ATTENTION_TILED(80, 8)
ATTENTION_TILED(128, 6)

// Merges the splits of one (row, kv head): one thread per dimension.
extern "C" __global__ void attention_combine(AttentionArguments a, int d) {
  const int row = blockIdx.x, kv_head = blockIdx.y, dim = threadIdx.x;
  const int group = a.q_heads / a.kv_heads, rows = a.tokens * group;
  const long long first = ((long long)kv_head * rows + row) * a.splits;
  float m = -INFINITY;
  for (int s = 0; s < a.splits; ++s) m = fmaxf(m, a.partial_ml[(first + s) * 2]);
  const int token = row / group, head = kv_head * group + row % group;
  if (a.sinks) m = fmaxf(m, a.sinks[head]);
  float l = 0.0f, o = 0.0f;
  if (m != -INFINITY) {
    for (int s = 0; s < a.splits; ++s) {
      const float ms = a.partial_ml[(first + s) * 2];
      if (ms == -INFINITY) continue;
      const float factor = __expf(ms - m);
      l += a.partial_ml[(first + s) * 2 + 1] * factor;
      o += a.partial_o[(first + s) * d + dim] * factor;
    }
    if (a.sinks) l += __expf(a.sinks[head] - m);
  }
  a.out[((long long)token * a.q_heads + head) * d + dim] = l > 0.0f ? o / l : 0.0f;
}

// Writes tokens' keys and values, F32 [tokens, kv heads, D], into the cache at
// positions start .. start + tokens - 1.
extern "C" __global__ void cache_write(const float* k, const float* v, _Float16* keys, _Float16* values,
                                       const int* page_table, int tokens, int kv_heads, int d, int page_size,
                                       int start) {
  const long long count = (long long)tokens * kv_heads * d;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (long long)gridDim.x * blockDim.x) {
    const int dim = i % d, head = (i / d) % kv_heads, token = i / ((long long)d * kv_heads);
    const int position = start + token;
    const long long page = page_table[position / page_size];
    const int offset = position % page_size;
    keys[((page * page_size + offset) * kv_heads + head) * d + dim] = (_Float16)k[i];
    values[((page * kv_heads + head) * d + dim) * page_size + offset] = (_Float16)v[i];
  }
}

// A gated attention's inputs from its projections, one block of 256 threads
// per (token, head): query head h takes the first half of query_and_gate's
// head h (the second is the output gate, into `gate`), RMS-normed with q_norm
// and turned (rope.h) into `queries`; key head j is normed with k_norm, turned
// and written to the cache with value head j, as cache_write writes them. The
// arithmetic of split_halves, rms_norm_f32, rope_f32 and cache_write, in one
// launch. Head dimension d <= ATTENTION_INPUT_DIMENSION.
constexpr int ATTENTION_INPUT_DIMENSION = 512;

extern "C" __global__ void __launch_bounds__(256)
    attention_inputs(const float* query_and_gate, const float* k, const float* v, const float* q_norm,
                     const float* k_norm, float eps, float offset, const int* positions, int tokens, int heads,
                     int kv_heads, int d, int rotary, float theta, int layout, int sections, int s0, int s1, int s2,
                     int s3, float* queries, float* gate, _Float16* keys, _Float16* values, const int* page_table,
                     int page_size, int start) {
  __shared__ float partial[256], normed[ATTENTION_INPUT_DIMENSION];
  const int t = blockIdx.x / (heads + kv_heads), head = blockIdx.x % (heads + kv_heads);
  const bool query = head < heads;
  const int kv = head - heads;
  const float* row = query ? query_and_gate + ((long long)t * heads + head) * 2 * d : k + ((long long)t * kv_heads + kv) * d;
  const float* weight = query ? q_norm : k_norm;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    sum += row[c] * row[c];
    if (query) gate[((long long)t * heads + head) * d + c] = row[d + c];
  }
  const float scale = rsqrtf(block_sum(sum, partial) / d + eps);
  for (int c = threadIdx.x; c < d; c += blockDim.x) normed[c] = row[c] * scale * (weight[c] + offset);
  __syncthreads();
  const int position = start + t;
  const long long page = page_table[position / page_size];
  const int at = position % page_size;
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    const float value = rotated(normed, c, positions, t, tokens, rotary, theta, layout, sections, s0, s1, s2, s3);
    if (query)
      queries[((long long)t * heads + head) * d + c] = value;
    else {
      keys[((page * page_size + at) * kv_heads + kv) * d + c] = (_Float16)value;
      values[((page * kv_heads + kv) * d + c) * page_size + at] = (_Float16)v[((long long)t * kv_heads + kv) * d + c];
    }
  }
}

// Attention over short sequences, many at once, with no mask: q and out
// [S][B][heads][D], k and v [S][B][kv_heads][D] (sequence-major: position s of
// sequence b is row s * B + b), query head h reading key head h / (heads /
// kv_heads). One wave per (position, sequence, head), lanes across D, an
// online softmax over the S keys. For text-fusion blocks (S of 12 across a
// token's layers, or one sequence of the prompt's tokens). bias, when not
// null, is added to the scaled scores: [heads][S][S] (T5's relative positions).
extern "C" __global__ void short_attention_f32(const float* q, const float* k, const float* v, float* out, int S,
                                               int B, int heads, int kv_heads, int D, float scale,
                                               const float* bias) {
  const long long wave = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (wave >= (long long)S * B * heads) return;
  const int h = wave % heads;
  const long long row = wave / heads;  // s * B + b
  const int b = row % B;
  const int kv = h / (heads / kv_heads);
  const float* query = q + (row * heads + h) * D;
  float acc[16] = {};  // D <= 512
  float running_max = -INFINITY, total = 0.0f;
  for (int s = 0; s < S; ++s) {
    const long long key_row = (long long)s * B + b;
    const float* key = k + (key_row * kv_heads + kv) * D;
    float dot = 0.0f;
    for (int d = lane; d < D; d += 32) dot += query[d] * key[d];
    for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_xor(dot, offset, 32);
    const float score =
        dot * scale + (bias ? bias[((long long)h * S + row / B) * S + s] : 0.0f);
    const float new_max = fmaxf(running_max, score);
    const float correction = expf(running_max - new_max), weight = expf(score - new_max);
    total = total * correction + weight;
    const float* value = v + (key_row * kv_heads + kv) * D;
    for (int i = 0, d = lane; d < D; ++i, d += 32) acc[i] = acc[i] * correction + weight * value[d];
    running_max = new_max;
  }
  for (int i = 0, d = lane; d < D; ++i, d += 32) out[(row * heads + h) * D + d] = acc[i] / total;
}
