"""Tiny FLUX 3 Image Klein golden fixtures, from the OFFICIAL code
(black-forest-labs/flux-3-image-klein, `src/flux3`): its transformer
(`flux3.dit.DiT`), packed by its pipeline's own `pack` and `scaled_ids`,
built small with seeded random weights and run once in float32 on the CPU.

  flux3       the step-distilled model (a flow map): 1 block per mode and 2
              shared blocks of 2 heads of 128 (RoPE axes 32 × 4, as released),
              32 latent channels, text features of 64: one velocity at σ 0.7
              jumping to 0.55 of a 3 × 4 target with two references (2 × 3,
              3 × 2), on 9 text tokens — every mode, and the references' own
              RoPE axis. Then the same with a LoRA at 0.8 under BFL's names
              (`diffusion_model.`, the MLP's fused input whole) and kohya's.
              -> tiny/flux3/
  flux3_base  the base model (no flow map) at σ 0.4 with one reference.
              -> tiny/flux3_base/

The timesteps are BF16, as the official pack holds them; the timestep's
sinusoid is BF16 too (the official `timestep_embedding` casts it back to the
timestep's type), turned to float32 for the float32 weights.

The official code is not packaged on PyPI: clone it first and pass its `src`
folder as FLUX3_OFFICIAL.

Outputs (runner/test/resources/fixtures/tiny/<folder>/):
  model.safetensors     the weights, rounded to BF16, under their names
  lora.safetensors      (flux3) the LoRA
  expected.safetensors  inputs + outputs

Run from the repository root:
  git clone https://github.com/black-forest-labs/flux-3-image-klein /tmp/flux3-official
  FLUX3_OFFICIAL=/tmp/flux3-official/src uv run --index https://download.pytorch.org/whl/cpu \\
      --index-strategy unsafe-best-match --with torch --with safetensors --with numpy --with pillow \\
      python runner/fixtures/tiny_flux3.py [flux3] [flux3_base]
"""

import os
import sys
import types

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TINY_DIR = os.path.join(HERE, "..", "test", "resources", "fixtures", "tiny")
sys.path.insert(0, os.environ["FLUX3_OFFICIAL"])

# the pipeline module imports the text encoder (transformers), which the
# transformer's fixtures never run
sys.modules["flux3.text_encoder"] = types.SimpleNamespace(TextEncoder=None)

from flux3 import dit  # noqa: E402

dit.HIDDEN = 256
dit.HEADS = 2
dit.DEPTH = 1
dit.DEPTH_SINGLE = 2
dit.IN_CHANNELS = {"image": 32, "image_cond": 32, "text": 64}

official_embedding = dit.timestep_embedding
dit.timestep_embedding = lambda t, *a, **k: official_embedding(t, *a, **k).float()

from flux3.autoencoder import latent_ids  # noqa: E402
from flux3.pipeline import pack, scaled_ids  # noqa: E402

TEXT_TOKENS = 9


def bf16(tensor):
    return tensor.to(torch.bfloat16).float()


@torch.no_grad()
def randomize(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.ndim == 1:
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        else:
            p.copy_(torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5)
        p.copy_(bf16(p))


def inputs(seed, references):
    """text, target and reference tokens with their official ids."""
    g = torch.Generator().manual_seed(seed)
    text = bf16(torch.randn(TEXT_TOKENS, dit.IN_CHANNELS["text"], generator=g))
    positions = torch.arange(TEXT_TOKENS)
    text_ids = torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1).float()
    h, w = 3, 4
    latents = torch.randn(h * w, dit.IN_CHANNELS["image"], generator=g)
    image = (latents, scaled_ids(latent_ids(h, w)))
    tokens, ids = [], []
    for index, (rh, rw) in enumerate(references):
        tokens.append(bf16(torch.randn(rh * rw, dit.IN_CHANNELS["image_cond"], generator=g)))
        i = scaled_ids(latent_ids(rh, rw))
        i[:, 3] = -(1 + index)
        ids.append(i)
    cond = (torch.cat(tokens), torch.cat(ids)) if tokens else None
    grid = [h, w] + [side for reference in references for side in reference]
    return (text, text_ids), image, cond, grid


def velocity(model, text, cond, image, timestep, target):
    packed = pack([[text, cond, image]])
    packed.timesteps["image"].fill_(timestep)
    reference = torch.tensor([target]) if target is not None else None
    return model(packed, timesteps_ref=reference)


def lora_for(model, run, targets):
    """A LoRA for the tiny model (`lora_A` down [rank, in], `lora_B` up
    [out, rank]; kohya's `lora_unet_` names `lora_down`/`lora_up`), some with
    an `.alpha`. Returns the file's tensors and the velocity with every delta
    merged into the weights at multiplier 0.8."""
    multiplier = 0.8
    state = model.state_dict()
    g = torch.Generator().manual_seed(7)
    lora, merged = {}, {}
    for name, key, rank, alpha, (down_name, up_name) in targets:
        weight = state[name + ".weight"]
        down = bf16(torch.randn(rank, weight.shape[1], generator=g) / weight.shape[1] ** 0.5)
        up = bf16(torch.randn(weight.shape[0], rank, generator=g) * 0.2)
        scale = multiplier * (alpha / rank if alpha is not None else 1.0)
        merged[name + ".weight"] = weight + scale * (up @ down)
        lora[f"{key}.{down_name}"] = down.contiguous()
        lora[f"{key}.{up_name}"] = up.contiguous()
        if alpha is not None:
            lora[key + ".alpha"] = torch.tensor(float(alpha))
    base = {k: v.clone() for k, v in state.items()}
    model.load_state_dict({**state, **merged})
    result = run()
    model.load_state_dict(base)
    return lora, result


def write(folder, model, expected, lora=None):
    out = os.path.join(TINY_DIR, folder)
    os.makedirs(out, exist_ok=True)
    weights = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    save_file(weights, os.path.join(out, "model.safetensors"))
    save_file({k: v.float().contiguous() for k, v in expected.items()}, os.path.join(out, "expected.safetensors"))
    if lora is not None:
        save_file({k: v.to(torch.bfloat16).contiguous() if v.ndim else v for k, v in lora.items()},
                  os.path.join(out, "lora.safetensors"))
    for k, v in expected.items():
        print(f"  {folder}/{k}: {tuple(v.shape)}  |max| {v.abs().max().item():.4f}")


@torch.no_grad()
def flux3():
    model = dit.DiT(flow_map=True)
    randomize(model, 1)
    text, image, cond, grid = inputs(2, [(2, 3), (3, 2)])
    timestep, target = 0.7, 0.55
    run = lambda: velocity(model, text, cond, image, timestep, target)
    result = run()
    lora, lora_result = lora_for(model, run, [
        ("single_blocks.0.mlp_in", "diffusion_model.single_blocks.0.mlp_in", 4, None, ("lora_A.weight", "lora_B.weight")),
        ("content_mode_blocks.image_cond.0.q_proj", "diffusion_model.content_mode_blocks.image_cond.0.q_proj", 4, 2.0,
         ("lora_A.weight", "lora_B.weight")),
        ("txt_mode_blocks.0.attn_out", "lora_unet_txt_mode_blocks_0_attn_out", 2, 1.0, ("lora_down.weight", "lora_up.weight")),
        ("single_blocks.1.mlp_out", "lora_unet_single_blocks_1_mlp_out", 4, None, ("lora_down.weight", "lora_up.weight")),
        ("single_stream_modulations.txt.lin", "diffusion_model.single_stream_modulations.txt.lin", 4, None,
         ("lora_A.weight", "lora_B.weight")),
        ("ref_time_in.out_layer", "diffusion_model.ref_time_in.out_layer", 4, None, ("lora_A.weight", "lora_B.weight")),
    ])
    write("flux3", model, {
        "grid": torch.tensor(grid, dtype=torch.float32),
        "text": text[0],
        "latents": image[0],
        "references": cond[0],
        "timestep": torch.tensor([timestep]),
        "next": torch.tensor([target]),
        "velocity": result,
        "lora_velocity": lora_result,
    }, lora)


@torch.no_grad()
def flux3_base():
    model = dit.DiT(flow_map=False)
    randomize(model, 3)
    text, image, cond, grid = inputs(4, [(3, 3)])
    timestep = 0.4
    result = velocity(model, text, cond, image, timestep, None)
    write("flux3_base", model, {
        "grid": torch.tensor(grid, dtype=torch.float32),
        "text": text[0],
        "latents": image[0],
        "references": cond[0],
        "timestep": torch.tensor([timestep]),
        "velocity": result,
    })


if __name__ == "__main__":
    chosen = sys.argv[1:] or ["flux3", "flux3_base"]
    for name in chosen:
        globals()[name]()
