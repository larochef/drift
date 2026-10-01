# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Golden fixtures for ComfyUI's quantized linears (specs/42, LTX 2.5's mixed
checkpoints): `comfy_quant.safetensors` holds three modules as a checkpoint
stores them, `comfy_quant.expected.safetensors` their float weights.

The expected values come from the formats' own arithmetic, not the runner's:
the 4-bit decode is `dequantize_rotated` of 0xDELUXA/comfy-int8-convrot-to-w4a8
(`convert_int8_convrot_to_w4a8.py`), the int8 one Comfy-Org/comfy-quants'
`docs/formats/int8_tensorwise.md`, and the rotation is the explicit 256 × 256
matrix. Which regular Hadamard matrix that is was settled against real weights:
with this one, a REDGraft LTX 2.5 layer decodes to the official BF16 layer
(cosine 1.00), with the other candidates to noise.

    python3 runner/fixtures/comfy_quant.py
"""

import json
import struct
from pathlib import Path

import numpy as np

out = Path(__file__).resolve().parent.parent / "test" / "resources" / "fixtures"
random = np.random.default_rng(7)

GROUP = 256


def e4m3(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exponent, mantissa = (byte >> 3) & 0xF, byte & 7
    if exponent == 0:
        return sign * mantissa / 8 * 2.0**-6
    return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


E4M3 = np.array([e4m3(b) for b in range(256)], np.float32)

# the normalized regular Hadamard matrix: symmetric, its own inverse
h4 = np.ones((4, 4)) - 2 * np.eye(4)[::-1]
hadamard = np.array([[1.0]])
while hadamard.shape[0] < GROUP:
    hadamard = np.kron(hadamard, h4)
hadamard = (hadamard / np.sqrt(GROUP)).astype(np.float32)
assert np.allclose(hadamard @ hadamard, np.eye(GROUP), atol=1e-5)


def unrotated(weight):
    rows, columns = weight.shape
    return (weight.reshape(rows, columns // GROUP, GROUP) @ hadamard).reshape(
        rows, columns
    )


def save(path, tensors):
    header, data = {}, b""
    for name, (dtype, array) in tensors.items():
        raw = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def marker(**fields):
    return ("U8", np.frombuffer(json.dumps(fields).encode(), np.uint8))


stored, expected = {}, {}

# int8, rotated: one scale a row
weight = random.integers(-128, 128, (3, 512), dtype=np.int8)
scale = random.uniform(1e-4, 1e-2, (3, 1)).astype(np.float32)
stored["rotated.weight"] = ("I8", weight)
stored["rotated.weight_scale"] = ("F32", scale)
stored["rotated.comfy_quant"] = marker(
    format="int8_tensorwise", convrot=True, convrot_groupsize=GROUP
)
expected["rotated.weight"] = unrotated(weight.astype(np.float32) * scale)

# int8, not rotated: its columns are no multiple of the group
weight = random.integers(-128, 128, (2, 24), dtype=np.int8)
scale = random.uniform(1e-4, 1e-2, (2, 1)).astype(np.float32)
stored["plain.weight"] = ("I8", weight)
stored["plain.weight_scale"] = ("F32", scale)
stored["plain.comfy_quant"] = marker(format="int8_tensorwise")
expected["plain.weight"] = weight.astype(np.float32) * scale

# 4-bit: two codes a byte, a 16-level codebook, fp8 group scales, a row scale
rows, columns, group_size = 3, 512, 16
packed = random.integers(0, 256, (rows, columns // 2), dtype=np.uint8)
codebook = np.sort(random.uniform(-1, 1, 16)).astype(np.float32)
# positive scales from 16 to 240: levels on both sides of the int8 clamp
relative = random.integers(0x58, 0x77, (rows, columns // group_size), dtype=np.uint8)
channel = random.uniform(1e-4, 1e-3, rows).astype(np.float32)
stored["four.weight"] = ("I8", packed.view(np.int8))
stored["four.weight_codebook"] = ("F32", codebook)
stored["four.weight_s_rel"] = ("F8_E4M3", relative)
stored["four.weight_s_channel"] = ("F32", channel)
stored["four.comfy_quant"] = marker(
    format="asym_w4a8_int8", group_size=group_size, convrot_groupsize=GROUP
)
raw = packed.astype(np.int32)
codes = np.empty((rows, columns), np.int32)
codes[:, 0::2] = raw & 0xF
codes[:, 1::2] = (raw >> 4) & 0xF
values = codebook[codes].reshape(rows, -1, group_size) * E4M3[relative][:, :, None]
int8 = np.clip(np.round(values), -127, 127).astype(np.float32)
assert (np.abs(np.round(values)) > 127).any(), "the clamp is exercised"
expected["four.weight"] = unrotated(
    (int8 * channel.reshape(rows, 1, 1)).reshape(rows, columns)
)

save(out / "comfy_quant.safetensors", stored)
save(
    out / "comfy_quant.expected.safetensors",
    {name: ("F32", value.astype(np.float32)) for name, value in expected.items()},
)
print("wrote", ", ".join(f"{name} {value.shape}" for name, value in expected.items()))
