"""Tiny Nucleus-Image golden fixture: diffusers' `NucleusMoEImageTransformer2DModel`
built small with seeded random weights and run once in float32.

  3 blocks of 2 query heads over 1 key-value head of 128 (RoPE axes 16/56/56,
  as released), the first dense and two with 4 routed experts of 64 plus the
  shared one (capacity factors 4 and 2: every token to every expert, then half
  of them), text features of 48, 16 packed latent features in and out: one
  output at σ 0.75 of a 5 × 6 grid on 7 text tokens. The timestep is rounded to
  BF16 as the released pipeline (which runs the transformer in BF16) rounds
  it: the scheduler's `1000 σ`, its thousandth, and the sinusoid's sines and
  cosines.
  -> tiny/nucleus/ (model.safetensors, config.json, expected.safetensors)

Run from the repository root (the model is in diffusers' main branch):
  uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
      --with torch --with git+https://github.com/huggingface/diffusers --with transformers==5.17.0 \\
      --with safetensors --with numpy python runner/fixtures/tiny_nucleus.py
"""

import json
import os

import torch
from safetensors.torch import save_file

import diffusers.models.embeddings as embeddings
from diffusers import NucleusMoEImageTransformer2DModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "test", "resources", "fixtures", "tiny", "nucleus")

sinusoid = embeddings.get_timestep_embedding
embeddings.get_timestep_embedding = lambda *a, **k: sinusoid(*a, **k).to(torch.bfloat16).float()

CONFIG = dict(
    patch_size=2, in_channels=16, out_channels=4, num_layers=3, attention_head_dim=128, num_attention_heads=2,
    num_key_value_heads=1, joint_attention_dim=48, axes_dims_rope=(16, 56, 56), mlp_ratio=4.0, moe_enabled=True,
    dense_moe_strategy="leave_first_block_dense", num_experts=4, moe_intermediate_dim=64,
    capacity_factors=[0.0, 4.0, 2.0], use_sigmoid=False, route_scale=2.5, use_grouped_mm=False,
)


@torch.no_grad()
def main() -> None:
    model = NucleusMoEImageTransformer2DModel(**CONFIG).eval()
    g = torch.Generator().manual_seed(0)
    for name, p in model.named_parameters():
        if p.ndim == 1 and name.endswith(".weight"):
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        elif p.ndim == 1:
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
        elif "experts." in name:  # [experts, in, out]
            p.copy_(torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5)
        else:
            p.copy_(torch.randn(p.shape, generator=g) / p[0].numel() ** 0.5)
        p.copy_(p.to(torch.bfloat16).float())
    grid = (5, 6)
    latents = torch.randn(1, grid[0] * grid[1], 16, generator=g)
    text = torch.randn(1, 7, 48, generator=g)
    sigma = 0.75
    # the pipeline's: the scheduler's timestep in BF16, over 1000 in BF16
    timestep = (torch.tensor([sigma * 1000]).to(torch.bfloat16) / 1000).float()
    output = model(
        hidden_states=latents, img_shapes=[(1, *grid)], encoder_hidden_states=text,
        timestep=timestep, return_dict=False,
    )[0]
    os.makedirs(OUT, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
              os.path.join(OUT, "model.safetensors"))
    json.dump(dict(model.config), open(os.path.join(OUT, "config.json"), "w"), indent=2)
    expected = {
        "latents": latents[0], "text": text[0], "sigma": torch.tensor([sigma]),
        "grid": torch.tensor(grid, dtype=torch.float32), "output": output[0],
    }
    save_file({k: v.float().contiguous() for k, v in expected.items()}, os.path.join(OUT, "expected.safetensors"))
    for k, v in expected.items():
        print(f"  {k}: {tuple(v.shape)}  |max| {v.abs().max().item():.4f}")


if __name__ == "__main__":
    main()
