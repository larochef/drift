"""Tiny HiDream O1 golden fixture generator.

Builds the OFFICIAL `Qwen3VLForConditionalGeneration` with its pixel head
(HiDream-ai's HiDream-O1-Image, `models/qwen3_vl_transformers.py`) from a
config shrunk down, with seeded random weights, and runs ONE generation forward
in float32 the way `models/pipeline.py` drives it — on the CPU, through the
masked attention path (`use_flash_attn=False`: the prompt causal, the generated
tokens seeing everything), which the flash path computes the same in two
passes. Two cases:

  text        a prompt, the timestep token and a 2 × 3 grid of 32 px patches
  references  the same with two reference pictures: each read by the vision
              tower in the prompt (with deepstack) and as pixel patches after
              the target's, placed by the official `get_rope_index_fix_point`
  LoRAs       the first case with a LoRA merged into the weights at 0.8, one
              per naming the published files use (`lora_*.safetensors`):
              ai-toolkit's (Civitai's trainer), DiffSynth-Studio's, a ComfyUI
              extraction with its whole-tensor differences, kohya's

    HIDREAM_O1_OFFICIAL=~/dev/hidream-o1-reference \\
    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        --with torch --with transformers==4.57.1 --with safetensors --with pillow \\
        python runner/fixtures/tiny_hidream_o1.py

Outputs are committed under runner/test/resources/fixtures/tiny/hidream_o1/.
"""

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

official = Path(
    os.environ.get("HIDREAM_O1_OFFICIAL", "~/dev/hidream-o1-reference")
).expanduser()
sys.path.insert(0, str(official))

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig  # noqa: E402

# the model and the position helper alone: `models.pipeline` wants torchvision
# and diffusers, which nothing here uses
from models import qwen3_vl_transformers as transformer  # noqa: E402
from models.utils import get_rope_index_fix_point  # noqa: E402

out = (
    Path(__file__).resolve().parent.parent
    / "test"
    / "resources"
    / "fixtures"
    / "tiny"
    / "hidream_o1"
)
torch.manual_seed(0)

PATCH = 32
IMAGE, VISION_START, VISION_END, VIDEO, BOI, TIMESTEP = 80, 81, 82, 83, 89, 90


def build():
    text = dict(
        vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128, rms_norm_eps=1e-6,
        max_position_embeddings=8192, rope_theta=5000000,
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True},
    )
    vision = dict(
        depth=2, hidden_size=64, num_heads=4, intermediate_size=128, patch_size=16,
        spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
        num_position_embeddings=16, deepstack_visual_indexes=[0, 1],
    )
    config = Qwen3VLConfig(
        text_config=text, vision_config=vision, image_token_id=IMAGE, video_token_id=VIDEO,
        vision_start_token_id=VISION_START, vision_end_token_id=VISION_END, tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = transformer.Qwen3VLForConditionalGeneration(config).eval().float()
    model.model.tms_token_id = TIMESTEP
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            elif name.endswith("bias") or "pos_embed" in name:
                parameter.copy_(0.2 * torch.randn_like(parameter))
            elif "final_layer2" in name:  # the pixel head starts at zero
                parameter.copy_(0.05 * torch.randn_like(parameter))
    return config, model


def patches(pixels):
    """[H, W, 3] in [−1, 1] → [(H/32)(W/32), 3072], each patch channel-major."""
    h, w, _ = pixels.shape
    x = pixels.permute(2, 0, 1).reshape(3, h // PATCH, PATCH, w // PATCH, PATCH)
    return x.permute(1, 3, 0, 2, 4).reshape((h // PATCH) * (w // PATCH), 3 * PATCH * PATCH)


def vision_patches(pixels, patch=16, merge=2):
    """A still as the tower's processor gives it: [patches, 3 × 2 × 16²], in
    merge-window order, the frame doubled."""
    h, w, _ = pixels.shape
    gh, gw = h // patch, w // patch
    x = pixels.permute(2, 0, 1)[None].repeat(2, 1, 1, 1)  # [2, 3, H, W]
    x = x.reshape(1, 2, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gh * gw, 3 * 2 * patch * patch), torch.tensor([[1, gh, gw]])


def run(model, ids, grids_seen, grid_target, grids_read, vinputs, timestep, pixel_values=None, image_grid=None):
    """One forward as `generate_image` builds it: `ids` the text (the tower's
    placeholders expanded), then the target's and the references' patches."""
    target_len = grid_target[0] * grid_target[1]
    read_lens = [h * w for h, w in grids_read]
    tokens = []
    for length in [target_len] + read_lens:
        part = torch.full((1, length), IMAGE, dtype=ids.dtype)
        part[0, 0] = VISION_START
        tokens.append(part)
    padded = torch.cat([ids] + tokens, dim=-1)
    grid_all = torch.tensor(
        [[1, h, w] for h, w in grids_seen] + [[1, *grid_target]] + [[1, h, w] for h, w in grids_read]
    )
    position_ids, _ = get_rope_index_fix_point(
        1, IMAGE, VIDEO, VISION_START, input_ids=padded, image_grid_thw=grid_all,
        video_grid_thw=None, attention_mask=None,
        skip_vision_start_token=[0] * len(grids_seen) + [1] + [1] * len(grids_read),
    )
    text_len = ids.shape[-1]
    types = torch.zeros((1, position_ids.shape[-1]), dtype=ids.dtype)
    types[0, text_len - 1:] = 1  # the timestep token and every patch
    kwargs = dict(
        input_ids=ids, position_ids=position_ids, vinputs=vinputs,
        timestep=torch.tensor([timestep]), token_types=types, use_flash_attn=False,
    )
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values
        kwargs["image_grid_thw"] = image_grid
    with torch.no_grad():
        x_pred = model(**kwargs).x_pred
    return x_pred[0, text_len:text_len + target_len].float(), position_ids[:, 0].to(torch.int32)


LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
HEADS = ["t_embedder1.mlp.0", "t_embedder1.mlp.2", "x_embedder.proj1", "x_embedder.proj2",
         "final_layer2.linear"]
NORMS = ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"]
RANK = 4
MULTIPLIER = 0.8


def pair(weight, generator):
    """A rank-4 update of `weight` ([out, in]): down [4, in], up [out, 4]."""
    return (0.1 * torch.randn((RANK, weight.shape[1]), generator=generator),
            0.1 * torch.randn((weight.shape[0], RANK), generator=generator))


def lora_files(model, generator):
    """Each naming's file and what it adds to the state dict at multiplier 1:
    `{name: (tensors, {state key: delta})}`."""
    state = model.state_dict()
    layers = [f"language_model.layers.{i}.{part}" for i in range(3) for part in LINEARS]
    files = {}

    # ai-toolkit: the checkpoint's names less `model.`, under ComfyUI's prefix,
    # lora_A / lora_B, no alpha
    tensors, deltas = {}, {}
    for site in layers + HEADS:
        down, up = pair(state[f"model.{site}.weight"], generator)
        tensors[f"diffusion_model.{site}.lora_A.weight"] = down
        tensors[f"diffusion_model.{site}.lora_B.weight"] = up
        deltas[f"model.{site}.weight"] = up @ down
    files["ai_toolkit"] = (tensors, deltas)

    # DiffSynth-Studio: the checkpoint's names whole, PEFT's `default` adapter,
    # the vision tower's linears too (which no prompt-only forward reads)
    tensors, deltas = {}, {}
    for site in layers:
        down, up = pair(state[f"model.{site}.weight"], generator)
        tensors[f"model.{site}.lora_A.default.weight"] = down
        tensors[f"model.{site}.lora_B.default.weight"] = up
        deltas[f"model.{site}.weight"] = up @ down
    down, up = pair(state["model.visual.blocks.0.attn.qkv.weight"], generator)
    tensors["model.visual.blocks.0.attn.qkv.lora_A.default.weight"] = down
    tensors["model.visual.blocks.0.attn.qkv.lora_B.default.weight"] = up
    files["diffsynth"] = (tensors, deltas)

    # a ComfyUI extraction (one checkpoint less another): lora_down / lora_up,
    # the norms' weights and the heads' biases as whole differences, the token
    # embeddings and the vision tower among the rest
    tensors, deltas = {}, {}
    for site in layers + HEADS:
        down, up = pair(state[f"model.{site}.weight"], generator)
        tensors[f"diffusion_model.{site}.lora_down.weight"] = down
        tensors[f"diffusion_model.{site}.lora_up.weight"] = up
        deltas[f"model.{site}.weight"] = up @ down
    norms = [f"language_model.layers.{i}.{part}" for i in range(3) for part in NORMS]
    for site in norms + ["language_model.norm"]:
        difference = 0.1 * torch.randn(state[f"model.{site}.weight"].shape, generator=generator)
        tensors[f"diffusion_model.{site}.diff"] = difference
        deltas[f"model.{site}.weight"] = difference
    for site in [h for h in HEADS if f"model.{h}.bias" in state]:
        difference = 0.1 * torch.randn(state[f"model.{site}.bias"].shape, generator=generator)
        tensors[f"diffusion_model.{site}.diff_b"] = difference
        deltas[f"model.{site}.bias"] = difference
    down, up = pair(state["model.language_model.embed_tokens.weight"], generator)
    tensors["diffusion_model.language_model.embed_tokens.lora_down.weight"] = down
    tensors["diffusion_model.language_model.embed_tokens.lora_up.weight"] = up
    tensors["diffusion_model.visual.blocks.0.norm1.diff"] = 0.1 * torch.randn(
        state["model.visual.blocks.0.norm1.weight"].shape, generator=generator)
    files["comfy_extracted"] = (tensors, deltas)

    # kohya: `lora_unet_`, `_` for every `.`, an alpha (2 over a rank of 4)
    tensors, deltas = {}, {}
    for site in layers + HEADS:
        down, up = pair(state[f"model.{site}.weight"], generator)
        stem = "lora_unet_" + site.replace(".", "_")
        tensors[f"{stem}.lora_down.weight"] = down
        tensors[f"{stem}.lora_up.weight"] = up
        tensors[f"{stem}.alpha"] = torch.tensor(2.0)
        deltas[f"model.{site}.weight"] = (2.0 / RANK) * (up @ down)
    files["kohya"] = (tensors, deltas)
    return files


def main():
    config, model = build()
    out.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(out)
    save_file(
        {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()},
        out / "model.safetensors",
        metadata={"format": "pt"},
    )
    generator = torch.Generator().manual_seed(3)
    target = (2, 3)  # 64 × 96 pixels
    z = 8.0 * torch.randn((1, target[0] * target[1], 3 * PATCH * PATCH), generator=generator)

    # a prompt alone
    ids = torch.randint(0, 80, (1, 12), generator=generator)
    ids[0, -2], ids[0, -1] = BOI, TIMESTEP
    x_text, positions_text = run(model, ids, [], target, [], z, 0.37)

    # two references: a 64 × 96 picture seen as 2 × 3 tokens and read as 2 × 3
    # patches, a 64 × 64 one seen as 2 × 2 and read as 2 × 2
    pictures = [
        torch.randint(0, 256, (64, 96, 3), generator=generator).float(),
        torch.randint(0, 256, (64, 64, 3), generator=generator).float(),
    ]
    seen = [vision_patches(p / 127.5 - 1) for p in pictures]
    pixel_values = torch.cat([s[0] for s in seen])
    image_grid = torch.cat([s[1] for s in seen])
    grids_seen = [(int(g[1]) // 2, int(g[2]) // 2) for g in image_grid]
    parts = [torch.randint(0, 80, (1, 3), generator=generator)]
    for h, w in grids_seen:
        parts += [torch.tensor([[VISION_START]]), torch.full((1, h * w), IMAGE), torch.tensor([[VISION_END]])]
    parts += [torch.randint(0, 80, (1, 5), generator=generator), torch.tensor([[BOI, TIMESTEP]])]
    ids_references = torch.cat(parts, dim=-1)
    read = [patches(p / 127.5 - 1) for p in pictures]
    grids_read = [(2, 3), (2, 2)]
    x_references, positions_references = run(
        model, ids_references, grids_seen, target, grids_read,
        torch.cat([z, torch.cat(read)[None]], dim=1), 0.62,
        pixel_values=pixel_values, image_grid=image_grid,
    )
    loras = {}
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    for name, (tensors, deltas) in lora_files(model, generator).items():
        save_file({k: v.contiguous() for k, v in tensors.items()}, out / f"lora_{name}.safetensors")
        merged = dict(original)
        for key, delta in deltas.items():
            merged[key] = original[key] + MULTIPLIER * delta
        model.load_state_dict(merged)
        loras[f"lora_{name}_x"] = run(model, ids, [], target, [], z, 0.37)[0].contiguous()
    model.load_state_dict(original)
    save_file(
        {
            **loras,
            "z": z[0].contiguous(),
            "text_ids": ids[0].to(torch.int32),
            "text_positions": positions_text.contiguous(),
            "text_x": x_text.contiguous(),
            "references_ids": ids_references[0].to(torch.int32),
            "references_positions": positions_references.contiguous(),
            "references_x": x_references.contiguous(),
            "picture_0": pictures[0].contiguous(),
            "picture_1": pictures[1].contiguous(),
        },
        out / "expected.safetensors",
    )
    print(f"hidream_o1: text {tuple(x_text.shape)}, references {tuple(x_references.shape)}, "
          f"positions {tuple(positions_references.shape)}")


main()
