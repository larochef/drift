// Dequantizes ROCmFP4 blocks with the fork's own reference code, so the
// runner's decoder is checked against it rather than against a second reading
// of the same README. Built by generate.py against a clone of the fork.
//
//   rocmfp4_dequantize dual|fast <block count>  < blocks  > float32 values
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocmfp4.h"

// The quantizers and dot products in rocmfp4.c need these from the rest of
// ggml; dequantization never calls them.
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void) type; (void) ne; abort(); }
float ggml_fp16_to_fp32(ggml_fp16_t value) { (void) value; abort(); }

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const int fast = strcmp(argv[1], "fast") == 0;
    const long count = atol(argv[2]);
    const size_t block_bytes = fast ? sizeof(block_rocmfp4_fast) : sizeof(block_rocmfp4);
    void * blocks = malloc(block_bytes * count);
    float * values = malloc(sizeof(float) * QK_ROCMFP4 * count);
    if (fread(blocks, block_bytes, count, stdin) != (size_t) count) return 3;
    if (fast) rocmfp4_dequantize_row_q4_0_fast(blocks, values, QK_ROCMFP4 * count);
    else rocmfp4_dequantize_row_q4_0(blocks, values, QK_ROCMFP4 * count);
    fwrite(values, sizeof(float), QK_ROCMFP4 * count, stdout);
    return 0;
}
