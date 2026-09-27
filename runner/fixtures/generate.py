# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "safetensors", "ml_dtypes", "pyyaml", "tqdm", "requests"]
# ///
"""Golden fixtures for the runner's format tests (specs/42, Testing).

Every expected value comes from an implementation other than the runner's:
gguf-py's numpy dequantizers for the GGML types, the ROCmFP4 fork's own C
reference for ROCmFP4, ml_dtypes for fp8. Run by hand; the outputs are
committed, so the tests need no Python.

    uv run runner/fixtures/generate.py <clone of PlunderStruck/rocmfp4-turboquant>

The clone gives gguf-py (a superset of upstream's that knows the ROCmFP4 type
ids) and rocmfp4.c. Fixtures were generated from commit
9c37bc7727a4bbc7e5b392e726cb0ef03a3bb6af of branch turboquant-port.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file

fork = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(fork / "gguf-py"))
from gguf import GGMLQuantizationType as T, GGUFWriter  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

here = Path(__file__).resolve().parent
out = here.parent / "test" / "resources" / "fixtures"
(out / "quants").mkdir(parents=True, exist_ok=True)
(out / "sharded").mkdir(parents=True, exist_ok=True)
random = np.random.default_rng(42)

# type: (elements per block, bytes per block, byte offsets of its f16 fields)
layouts = {
    T.Q4_0: (32, 18, [0]),
    T.Q4_1: (32, 20, [0, 2]),
    T.Q5_0: (32, 22, [0]),
    T.Q5_1: (32, 24, [0, 2]),
    T.Q8_0: (32, 34, [0]),
    T.Q2_K: (256, 84, [80, 82]),
    T.Q3_K: (256, 110, [108]),
    T.Q4_K: (256, 144, [0, 2]),
    T.Q5_K: (256, 176, [0, 2]),
    T.Q6_K: (256, 210, [208]),
}


def random_blocks(count, block_bytes, half_offsets):
    """Random bytes, so every bit pattern of the quants is exercised, with
    finite f16 scales; the first block is all zeros, the second all ones but
    its scales."""
    blocks = random.integers(0, 256, size=(count, block_bytes), dtype=np.uint8)
    blocks[0] = 0
    blocks[1] = 0xFF
    for offset in half_offsets:
        halves = (random.standard_normal(count) * 0.05).astype(np.float16)
        blocks[:, offset:offset + 2] = halves.view(np.uint8).reshape(count, 2)
    return blocks


def reference_rocmfp4(kind, blocks):
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "rocmfp4_dequantize"
        subprocess.run(
            ["clang", "-O1", "-ffp-contract=off", "-std=c11",
             f"-I{fork}/ggml/include", f"-I{fork}/ggml/src", f"-I{fork}/ggml/rocmfp4",
             str(here / "rocmfp4_dequantize.c"), str(fork / "ggml/rocmfp4/rocmfp4.c"),
             "-lm", "-o", str(binary)],
            check=True,
        )
        result = subprocess.run(
            [str(binary), kind, str(len(blocks))], input=blocks.tobytes(),
            capture_output=True, check=True,
        )
    return np.frombuffer(result.stdout, dtype=np.float32).reshape(len(blocks), 32)


# --- quant types: blocks and what they decode to --------------------------------
count = 16
for quant, (elements, block_bytes, halves) in layouts.items():
    blocks = random_blocks(count, block_bytes, halves)
    expected = dequantize(blocks.reshape(-1), quant).reshape(count, elements)
    save_file({"blocks": blocks, "expected": expected.astype(np.float32)},
              out / "quants" / f"{quant.name.lower()}.safetensors")

for kind, quant, block_bytes, scales in [
    ("dual", T.Q4_0_ROCMFP4, 18, [16, 17]),
    ("fast", T.Q4_0_ROCMFP4_FAST, 17, [16]),
]:
    blocks = random.integers(0, 256, size=(count, block_bytes), dtype=np.uint8)
    for offset in scales:  # finite UE4M3 scales only: 0x00–0x7e
        blocks[:, offset] = random.integers(0, 0x7F, size=count, dtype=np.uint8)
    blocks[0, :16] = 0
    blocks[1, :16] = 0xFF
    save_file({"blocks": blocks, "expected": reference_rocmfp4(kind, blocks)},
              out / "quants" / f"{quant.name.lower()}.safetensors")

# --- fp8: every one of the 256 codes -------------------------------------------
codes = np.arange(256, dtype=np.uint8)
save_file({
    "e4m3": codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32),
    "e5m2": codes.view(ml_dtypes.float8_e5m2).astype(np.float32),
}, out / "fp8.safetensors")

# --- a safetensors file with one tensor per dtype, and each one's values -------
values = (random.standard_normal((2, 3)) * 4).astype(np.float32)
mixed = {
    "f32": values,
    "f16": values.astype(np.float16),
    "bf16": values.astype(ml_dtypes.bfloat16),
    "f8_e4m3": values.astype(ml_dtypes.float8_e4m3fn),
    "f8_e5m2": values.astype(ml_dtypes.float8_e5m2),
    "i8": np.array([[-128, -1, 0], [1, 64, 127]], dtype=np.int8),
    "u8": np.array([[0, 1, 2], [128, 254, 255]], dtype=np.uint8),
    "layer.weight": np.ones((2, 2), dtype=np.float32),
    "layer.weight_scale": np.array([0.5], dtype=np.float32),
}
save_file(mixed, out / "mixed.safetensors", metadata={"format": "pt", "note": "fixture"})
save_file({name: np.asarray(tensor).astype(np.float32) for name, tensor in mixed.items()},
          out / "mixed.expected.safetensors")

# --- a model sharded in two files and its index --------------------------------
shards = {
    "model-00001-of-00002.safetensors": {"a.weight": np.arange(6, dtype=np.float32).reshape(2, 3)},
    "model-00002-of-00002.safetensors": {"b.weight": np.arange(4, dtype=np.float32) * -1,
                                         "c.bias": np.array([7], dtype=np.float32)},
}
weight_map = {}
for file, tensors in shards.items():
    save_file(tensors, out / "sharded" / file)
    weight_map.update({name: file for name in tensors})
(out / "sharded" / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": 40}, "weight_map": weight_map}, indent=2))

# --- a GGUF with every metadata type and a few tensor types --------------------
writer = GGUFWriter(str(out / "tiny.gguf"), "llama")
writer.add_uint8("test.u8", 200)
writer.add_int8("test.i8", -100)
writer.add_uint16("test.u16", 60000)
writer.add_int16("test.i16", -30000)
writer.add_uint32("test.u32", 4000000000)
writer.add_int32("test.i32", -2000000000)
writer.add_float32("test.f32", 1.5)
writer.add_uint64("test.u64", 2**63 + 5)
writer.add_int64("test.i64", -(2**62))
writer.add_float64("test.f64", 2.25)
writer.add_bool("test.bool", True)
writer.add_string("test.string", "héllo")
writer.add_array("test.ints", [1, 2, 3])
writer.add_array("test.strings", ["a", "bc"])
expected = {}
f32 = np.arange(15, dtype=np.float32).reshape(3, 5)
writer.add_tensor("f32", f32)
expected["f32"] = f32
writer.add_tensor("f16", f32.astype(np.float16))
expected["f16"] = f32
for quant, name in [(T.Q8_0, "q8_0"), (T.Q4_K, "q4_k"), (T.Q4_0_ROCMFP4, "q4_0_rocmfp4")]:
    fixture = out / "quants" / f"{name}.safetensors"
    from safetensors.numpy import load_file
    loaded = load_file(fixture)
    blocks, values = loaded["blocks"][2:4], loaded["expected"][2:4]
    # two blocks as one row of 2 × block elements
    writer.add_tensor(name, blocks.reshape(1, -1), raw_dtype=quant)  # a byte shape: the writer derives the element shape
    expected[name] = values.reshape(1, -1)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
save_file(expected, out / "tiny.expected.safetensors")

# --- the non-linear 4-bit quants (added after the rest: earlier fixtures keep
# their random draws) -------------------------------------------------------------
for quant, (elements, block_bytes, halves) in {
    T.IQ4_NL: (32, 18, [0]),
    T.IQ4_XS: (256, 136, [0]),
}.items():
    blocks = random_blocks(count, block_bytes, halves)
    expected = dequantize(blocks.reshape(-1), quant).reshape(count, elements)
    save_file({"blocks": blocks, "expected": expected.astype(np.float32)},
              out / "quants" / f"{quant.name.lower()}.safetensors")
print(f"fixtures written to {out}")
