#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Sound tensors channels-last: [time][channels], as the vocoders run them.

// A 1-D convolution's input patches, rows of output samples [first, first +
// count): row p holds, for each input channel c, the taps values the sample
// reads at column c * taps + k (a weight [out][in][taps] flattened the same
// way), as BF16; zeros outside x.
extern "C" __global__ void im2col_1d_bf16(const float* x, __hip_bfloat16* patches, int length, int channels,
                                          int taps, int dilation, int stride, int pad_left, long long first,
                                          long long count) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long columns = (long long)channels * taps;
  if (index >= count * columns) return;
  const long long p = first + index / columns;
  const int column = index % columns, c = column / taps, k = column % taps;
  const long long s = p * stride + (long long)k * dilation - pad_left;
  const float value = (s >= 0 && s < length) ? x[s * channels + c] : 0.0f;
  patches[index] = __float2bfloat16(value);
}

// im2col_1d_bf16's patches kept in F32.
extern "C" __global__ void im2col_1d_f32(const float* x, float* patches, int length, int channels, int taps,
                                         int dilation, int stride, int pad_left, long long first, long long count) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long columns = (long long)channels * taps;
  if (index >= count * columns) return;
  const long long p = first + index / columns;
  const int column = index % columns, c = column / taps, k = column % taps;
  const long long s = p * stride + (long long)k * dilation - pad_left;
  patches[index] = (s >= 0 && s < length) ? x[s * channels + c] : 0.0f;
}

// DAC's Snake: out = x + sin²(alpha x) / (alpha + 1e-9), alpha per channel.
extern "C" __global__ void snake_f32(const float* x, const float* alpha, float* out, long long count,
                                     int channels) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float a = alpha[index % channels], value = x[index];
  const float s = sinf(a * value);
  out[index] = value + s * s / (a + 1e-9f);
}

// A transposed 1-D convolution's overlap-add: out[p][o] = bias[o] + the sum of
// columns[t][o * taps + k] over t * stride + k - pad = p.
extern "C" __global__ void overlap_add_f32(const float* columns, const float* bias, float* out, int length,
                                           int out_length, int channels, int taps, int stride, int pad) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)out_length * channels) return;
  const int o = index % channels;
  const long long p = index / channels;
  float sum = bias[o];
  for (int k = 0; k < taps; ++k) {
    const long long shifted = p + pad - k;
    if (shifted >= 0 && shifted % stride == 0 && shifted / stride < length)
      sum += columns[(shifted / stride) * channels * taps + (long long)o * taps + k];
  }
  out[index] = sum;
}

// BigVGAN's anti-aliased SnakeBeta, one output value per thread: the ×2
// upsampled signal (a transposed convolution by up, times 2, of x
// replicate-padded by 5, cropped by 15), activated as
// u + inverse_magnitude * sin²(frequency * u), then filtered by down at
// stride 2 (replicate-padded by 5 before, 6 after). Both filters 12 taps.
extern "C" __global__ void anti_aliased_snake_f32(const float* x, const float* frequency,
                                                  const float* inverse_magnitude, const float* up,
                                                  const float* down, float* out, int length, int channels) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)length * channels) return;
  const int c = index % channels;
  const long long t = index / channels;
  const float f = frequency[c], m = inverse_magnitude[c];
  float sum = 0.0f;
  for (int j = 0; j < 12; ++j) {
    const long long n = min(max(2 * t + j - 5, 0LL), 2LL * length - 1);
    float u = 0.0f;
    for (int k = (n + 15) % 2; k < 12; k += 2) {
      const long long i = min(max((n + 15 - k) / 2 - 5, 0LL), (long long)length - 1);
      u += up[k] * x[i * channels + c];
    }
    u *= 2.0f;
    const float s = sinf(f * u);
    sum += down[j] * (u + m * s * s);
  }
  out[index] = sum;
}
