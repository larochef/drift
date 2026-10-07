#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// Image tensors channels-last: [height][width][channels] (a pixel's channels
// contiguous), as the VAE runs them.

// A 3×3 convolution's input patches, rows of output pixels [first, first +
// count) of an output out_width wide: row p holds, for each input channel c,
// the 9 values the pixel reads at column c * 9 + ky * 3 + kx (a weight
// [out][in][3][3] flattened the same way), as BF16. Stride 1 pads by 1 on
// every side; stride 2 by 1 at the bottom and right only (diffusers'
// downsampler: pad (0, 1, 0, 1), then no padding). `padding` 1 pads with the
// nearest edge pixel instead of zeros (stride 1), 2 with the reflection
// (−1 reads 1, height reads height − 2). F16 patches (im2col_3x3_f16) for F16
// weights.
__device__ inline __hip_bfloat16 to_patch(float value, __hip_bfloat16*) { return __float2bfloat16(value); }
__device__ inline __half to_patch(float value, __half*) { return __float2half(value); }

template <typename T>
__device__ void im2col_3x3(const float* x, T* patches, int height, int width, int channels, int out_width,
                           int stride, int padding, long long first, long long count) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long columns = (long long)channels * 9;
  if (index >= count * columns) return;
  const long long p = first + index / columns;
  const int column = index % columns, c = column / 9, ky = column % 9 / 3, kx = column % 3;
  const int pad = stride == 1 ? 1 : 0;
  int y = p / out_width * stride + ky - pad, xx = p % out_width * stride + kx - pad;
  if (padding == 1) {
    y = min(max(y, 0), height - 1);
    xx = min(max(xx, 0), width - 1);
  } else if (padding == 2) {
    y = y < 0 ? -y : (y >= height ? 2 * height - 2 - y : y);
    xx = xx < 0 ? -xx : (xx >= width ? 2 * width - 2 - xx : xx);
  }
  const float value =
      (y >= 0 && y < height && xx >= 0 && xx < width) ? x[((long long)y * width + xx) * channels + c] : 0.0f;
  patches[index] = to_patch(value, patches);
}

extern "C" __global__ void im2col_3x3_bf16(const float* x, __hip_bfloat16* patches, int height, int width,
                                           int channels, int out_width, int stride, int padding,
                                           long long first, long long count) {
  im2col_3x3(x, patches, height, width, channels, out_width, stride, padding, first, count);
}

extern "C" __global__ void im2col_3x3_f16(const float* x, __half* patches, int height, int width,
                                          int channels, int out_width, int stride, int padding,
                                          long long first, long long count) {
  im2col_3x3(x, patches, height, width, channels, out_width, stride, padding, first, count);
}

// Nearest-neighbour ×2: out[y][x] = in[y / 2][x / 2].
extern "C" __global__ void upsample_2x_f32(const float* x, float* out, int height, int width, int channels) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 4LL * height * width * channels) return;
  const int c = index % channels;
  const long long p = index / channels;
  const int y = p / (2 * width), xx = p % (2 * width);
  out[index] = x[((long long)(y / 2) * width + xx / 2) * channels + c];
}

// Wan 2.2's up-shortcut on one frame (DupUp3D, the last of `frames` kept):
// out[y][x][o] = in[y / 2][x / 2][j / repeats], where
// j = ((o * frames + frames - 1) * 2 + y % 2) * 2 + x % 2 and
// repeats = out_channels * frames * 4 / channels.
extern "C" __global__ void duplicate_up_f32(const float* x, float* out, int height, int width, int channels,
                                            int out_channels, int frames) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 4LL * height * width * out_channels) return;
  const int o = index % out_channels;
  const long long p = index / out_channels;
  const int y = p / (2 * width), xx = p % (2 * width);
  const int j = ((o * frames + frames - 1) * 2 + y % 2) * 2 + xx % 2;
  const int repeats = out_channels * frames * 4 / channels;
  out[index] = x[((long long)(y / 2) * width + xx / 2) * channels + j / repeats];
}

// Wan 2.2's down-shortcut on one frame (AvgDown3D, zero frames before it):
// out[y][x][o] = the mean over g < group of value k = o * group + g, where
// k = ((c * frames + t) * 2 + dy) * 2 + dx is in[2y + dy][2x + dx][c] for the
// last frame t and zero for the others; group = channels * frames * 4 / out_channels.
extern "C" __global__ void average_down_f32(const float* x, float* out, int height, int width, int channels,
                                            int out_channels, int frames) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)(height / 2) * (width / 2) * out_channels) return;
  const int o = index % out_channels;
  const long long p = index / out_channels;
  const int y = p / (width / 2), xx = p % (width / 2);
  const int group = channels * frames * 4 / out_channels;
  float sum = 0.0f;
  for (int g = 0; g < group; ++g) {
    const int k = o * group + g;
    if ((k / 4) % frames != frames - 1) continue;
    sum += x[((long long)(2 * y + k / 2 % 2) * width + 2 * xx + k % 2) * channels + k / (4 * frames)];
  }
  out[index] = sum / group;
}

// out[c][r] = x[r][c].
extern "C" __global__ void transpose_f32(const float* x, float* out, int rows, int cols) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)rows * cols) return;
  const int r = index / cols, c = index % cols;
  out[(long long)c * rows + r] = x[index];
}

// A depthwise 3×3 convolution, zeros around the image: out[y][x][c] =
// bias[c] + Σ weight[c][ky * 3 + kx] × x[y + ky − 1][x + kx − 1][c].
extern "C" __global__ void depthwise_conv_3x3_f32(const float* x, const float* weight, const float* bias,
                                                  float* out, int height, int width, int channels) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)height * width * channels) return;
  const int c = index % channels;
  const long long p = index / channels;
  const int y = p / width, xx = p % width;
  const float* taps = weight + (long long)c * 9;
  float sum = bias[c];
  for (int ky = 0; ky < 3; ++ky) {
    const int sy = y + ky - 1;
    if (sy < 0 || sy >= height) continue;
    for (int kx = 0; kx < 3; ++kx) {
      const int sx = xx + kx - 1;
      if (sx < 0 || sx >= width) continue;
      sum += taps[ky * 3 + kx] * x[((long long)sy * width + sx) * channels + c];
    }
  }
  out[index] = sum;
}

// Each column's mean over the rows, one block per column: its threads sum
// every blockDim-th row, then their sums.
extern "C" __global__ void column_mean_f32(const float* x, float* out, long long rows, int columns) {
  __shared__ float partial[256];
  const int c = blockIdx.x;
  float sum = 0.0f;
  for (long long r = threadIdx.x; r < rows; r += blockDim.x) sum += x[r * columns + c];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[c] = partial[0] / rows;
}

// transpose_f32 of 16-bit values as they are stored (F16, BF16).
extern "C" __global__ void transpose_16(const unsigned short* x, unsigned short* out, int rows, int cols) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)rows * cols) return;
  const int r = index / cols, c = index % cols;
  out[(long long)c * rows + r] = x[index];
}

// out[ids[s]][c] += weights[s] * rows[s][c]: several slots may add to one
// token, so the sums are atomic.
extern "C" __global__ void scatter_add_rows_f32(const float* rows, const int* ids, const float* weights,
                                                float* out, long long slots, int columns) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= slots * columns) return;
  const long long slot = index / columns;
  atomicAdd(out + (long long)ids[slot] * columns + index % columns, weights[slot] * rows[index]);
}

// Packed patches [grid_h * grid_w][channels * patch * patch] (feature
// c * patch² + py * patch + px) back to an image [grid_h * patch][grid_w *
// patch][channels].
extern "C" __global__ void unpack_patches_f32(const float* packed, float* image, int grid_h, int grid_w,
                                              int channels, int patch) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int width = grid_w * patch;
  if (index >= (long long)grid_h * patch * width * channels) return;
  const int c = index % channels;
  const long long p = index / channels;
  const int y = p / width, x = p % width;
  const long long token = (long long)(y / patch) * grid_w + x / patch;
  image[index] = packed[token * channels * patch * patch + (long long)c * patch * patch + (y % patch) * patch + x % patch];
}

// An image [grid_h * patch][grid_w * patch][channels] into packed patches
// [grid_h * grid_w][channels * patch * patch], unpack_patches' inverse.
extern "C" __global__ void pack_patches_f32(const float* image, float* packed, int grid_h, int grid_w,
                                            int channels, int patch) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int features = channels * patch * patch;
  if (index >= (long long)grid_h * grid_w * features) return;
  const long long token = index / features;
  const int feature = index % features, c = feature / (patch * patch), py = feature % (patch * patch) / patch,
            px = feature % patch;
  const long long y = token / grid_w * patch + py, x = token % grid_w * patch + px;
  packed[index] = image[(y * grid_w * patch + x) * channels + c];
}

// out[r][offset + c] = x[r][c] for c < cols: one part of a concatenation of
// columns into rows of out_cols.
extern "C" __global__ void copy_columns_f32(const float* x, float* out, long long rows, int cols, int out_cols,
                                            int offset) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * cols) return;
  out[index / cols * out_cols + offset + index % cols] = x[index];
}

// An image [H][W][C] as rows of patch-ordered pixels [L * P²][C]: row
// l * P² + py * P + px holds pixel (hs * P + py, ws * P + px) of patch
// l = hs * (W / P) + ws. `back` runs it the other way.
extern "C" __global__ void patch_pixels_f32(const float* from, float* to, int height, int width, int channels,
                                            int patch, int back) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (long long)height * width * channels) return;
  const int c = index % channels;
  const long long row = index / channels;
  const int area = patch * patch, grid_w = width / patch;
  const long long l = row / area;
  const int p = row % area;
  const long long y = l / grid_w * patch + p / patch, x = l % grid_w * patch + p % patch;
  const long long pixel = (y * width + x) * channels + c;
  if (back) to[pixel] = from[index];
  else to[index] = from[pixel];
}
