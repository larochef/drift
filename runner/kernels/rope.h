#pragma once
#include <hip/hip_runtime.h>

// Rotary position embedding of one head's values `head` of token t: value d
// as it turns. The first `rotary` values turn in pairs, the rest are copied.
//
// Pair i turns by position * theta^(-2i / rotary), in float as transformers
// computes it (inv_freq in float32, then position * inv_freq), so long
// contexts round the way the model was trained.
// - layout 0 (NeoX, "rotate half"): pair (i, i + rotary/2);
//   layout 1 (GPT-J, interleaved): pair (2i, 2i + 1).
// - sections 0: one position per token, positions[t];
//   1 (mRoPE): pairs [0, s0) take positions[0][t], [s0, s0+s1) positions[1][t],
//   the rest positions[2][t];
//   2 (interleaved mRoPE, Qwen3-VL and on): pair i takes section 1 when
//   i % 3 == 1 and i < 3 s1, section 2 when i % 3 == 2 and i < 3 s2, else 0;
//   3 (per-axis, Krea 2 and FLUX.2): up to four sections s0..s3 in turn,
//   section a taking positions[a][t], and pair j of a section of n turns by
//   theta^(-j / n).
__device__ inline float rotated(const float* head, int d, const int* positions, int t, int tokens, int rotary,
                               float theta, int layout, int sections, int s0, int s1, int s2, int s3) {
  if (d >= rotary) return head[d];
  const int half = rotary / 2;
  int pair, partner;
  bool first;
  if (layout == 0) {
    pair = d % half;
    first = d < half;
    partner = first ? d + half : d - half;
  } else {
    pair = d / 2;
    first = d % 2 == 0;
    partner = first ? d + 1 : d - 1;
  }
  int section = 0;
  if (sections == 1) section = pair < s0 ? 0 : (pair < s0 + s1 ? 1 : 2);
  if (sections == 3) section = pair < s0 ? 0 : (pair < s0 + s1 ? 1 : (pair < s0 + s1 + s2 ? 2 : 3));
  if (sections == 2) section = (pair % 3 == 1 && pair < 3 * s1) ? 1 : ((pair % 3 == 2 && pair < 3 * s2) ? 2 : 0);
  float inverse_frequency;
  if (sections == 3) {
    const int start = section == 0 ? 0 : (section == 1 ? s0 : (section == 2 ? s0 + s1 : s0 + s1 + s2));
    const int count = section == 0 ? s0 : (section == 1 ? s1 : (section == 2 ? s2 : s3));
    inverse_frequency = 1.0f / powf(theta, float(pair - start) / float(count));
  } else {
    inverse_frequency = 1.0f / powf(theta, float(2 * pair) / float(rotary));
  }
  const float angle = float(positions[(long long)section * tokens + t]) * inverse_frequency;
  const float c = cosf(angle), s = sinf(angle);
  const float a = head[first ? d : partner], b = head[first ? partner : d];
  return first ? a * c - b * s : a * s + b * c;
}
