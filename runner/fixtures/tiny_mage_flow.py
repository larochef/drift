"""Tiny Mage-Flow golden fixtures, from the OFFICIAL code (microsoft/Mage,
`mage_flow/models`): its transformer (`MageFlow`) and its VAE's two halves
(`_DConvEncoder`, `_DConvDenoiser`), each built small with seeded random
weights and run once in float32.

  transformer  2 blocks of 2 heads of 128 (RoPE axes 16/56/56, as released), 16
               latent channels, text features of 48: one velocity at σ 0.7 of a
               5 × 6 target with a 4 × 3 reference after it, on 7 text tokens.
               The timestep's sinusoid is rounded to BF16 the way the released
               pipeline rounds it (its transformer runs in BF16): the level and
               the frequencies before the product, the sines and cosines after.
               -> tiny/mage_flow/
  vae          hidden 64 (head 96, 2 + 2 blocks), 16 latent channels, a
               per-pixel decoder of 16 with 3 blocks, windows of 4 in place of
               32 (the released decoder's are fixed at 32): a 96 × 80 image
               encoded to its mean and log-variance, and a 6 × 5 latent decoded
               (the windows padded to 8 × 8).
               -> tiny/mage_vae/

The official code is not packaged: clone it first (the fixtures were made at
76bec2b) and pass its `mage_flow` folder's parent as MAGE_OFFICIAL.

Outputs (runner/test/resources/fixtures/tiny/<folder>/):
  model.safetensors     the weights, rounded to BF16 as the released
                        checkpoints store them, under their names
  expected.safetensors  inputs + outputs

Run from the repository root:
  git clone https://github.com/microsoft/Mage /tmp/mage-official
  MAGE_OFFICIAL=/tmp/mage-official uv run --index https://download.pytorch.org/whl/cpu \\
      --index-strategy unsafe-best-match --with torch --with diffusers==0.40.0 \\
      --with transformers==5.17.0 --with safetensors --with numpy --with einops \\
      --with loguru --with pydantic python runner/fixtures/tiny_mage_flow.py [transformer] [vae]
"""

import importlib
import os
import sys
import types

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.join(os.environ["MAGE_OFFICIAL"], "mage_flow")
TINY_DIR = os.path.join(HERE, "..", "test", "resources", "fixtures", "tiny")

# The packages' __init__ files import the whole pipeline (the text encoder,
# transformers' Qwen3-VL): the two modules wanted are loaded under bare
# packages instead.
for name, path in [
    ("mage_flow", OFFICIAL),
    ("mage_flow.models", os.path.join(OFFICIAL, "models")),
    ("mage_flow.models.modules", os.path.join(OFFICIAL, "models", "modules")),
]:
    package = types.ModuleType(name)
    package.__path__ = [path]
    sys.modules[name] = package

attn_backend = importlib.import_module("mage_flow.models.modules._attn_backend")
attn_backend.set_attn_backend("sdpa")
layers = importlib.import_module("mage_flow.models.modules.mage_layers")
vae_modules = importlib.import_module("mage_flow.models.modules.mage_vae")


@torch.no_grad()
def randomize(model: torch.nn.Module, seed: int) -> None:
    """Seeded weights that hide no path (norm scales near 1, biases not zero),
    rounded to BF16."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.ndim == 1 and name.endswith(".weight"):
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        elif p.ndim == 1:
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
        else:
            fan_in = p[0].numel()
            p.copy_(torch.randn(p.shape, generator=g) / fan_in**0.5)
        p.copy_(p.to(torch.bfloat16).float())


def write(folder: str, weights: dict, expected: dict) -> None:
    out = os.path.join(TINY_DIR, folder)
    os.makedirs(out, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in weights.items()}, os.path.join(out, "model.safetensors"))
    save_file({k: v.float().contiguous() for k, v in expected.items()}, os.path.join(out, "expected.safetensors"))
    for k, v in expected.items():
        print(f"  {folder}/{k}: {tuple(v.shape)}  |max| {v.abs().max().item():.4f}")


@torch.no_grad()
def transformer() -> None:
    # MageFlow itself lives in mage_flow.py beside the model wrapper, which
    # imports the text encoder: the class is rebuilt from its own source here.
    source = open(os.path.join(OFFICIAL, "models", "mage_flow.py")).read()
    start, end = source.index("@dataclass\nclass MageFlowParams"), source.index("class MageFlowModel")
    scope = {
        "torch": torch, "nn": torch.nn, "Tensor": torch.Tensor, "Any": object,
        "dataclass": __import__("dataclasses").dataclass,
        "AdaLayerNormContinuous": layers.AdaLayerNormContinuous,
        "MageFlowEmbedRope": layers.MageFlowEmbedRope,
        "MageFlowTimestepProjEmbeddings": layers.MageFlowTimestepProjEmbeddings,
        "MageFlowTransformerBlock": layers.MageFlowTransformerBlock,
        "RMSNorm": layers.RMSNorm,
    }
    exec(source[start:end], scope)

    # the released pipeline's BF16 sinusoid, in a float32 model
    sinusoid = layers.get_timestep_embedding
    layers.get_timestep_embedding = lambda t, *a, **k: sinusoid(t.to(torch.bfloat16), *a, **k).to(torch.bfloat16).float()

    model = scope["MageFlow"](scope["MageFlowParams"](
        in_channels=16, out_channels=16, context_in_dim=48, hidden_size=256, num_heads=2, depth=2,
        axes_dim=[16, 56, 56], checkpoint=False,
    )).eval()
    randomize(model, 0)
    g = torch.Generator().manual_seed(1)
    target, reference = (5, 6), (4, 3)
    tokens = target[0] * target[1] + reference[0] * reference[1]
    img = torch.randn(1, tokens, 16, generator=g)
    txt = torch.randn(1, 7, 48, generator=g)
    sigma = 0.7
    out = model(
        img=img, txt=txt, timesteps=torch.tensor([sigma]),
        img_shapes=[[(1, *target), (1, *reference)]],
        img_cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32),
        txt_cu_seqlens=torch.tensor([0, 7], dtype=torch.int32),
    )
    alone = model(
        img=img[:, : target[0] * target[1]], txt=txt, timesteps=torch.tensor([sigma]),
        img_shapes=[[(1, *target)]],
        img_cu_seqlens=torch.tensor([0, target[0] * target[1]], dtype=torch.int32),
        txt_cu_seqlens=torch.tensor([0, 7], dtype=torch.int32),
    )
    write("mage_flow", model.state_dict(), {
        "img": img[0], "txt": txt[0], "sigma": torch.tensor([sigma]),
        "target": torch.tensor(target, dtype=torch.float32), "reference": torch.tensor(reference, dtype=torch.float32),
        "output": out[0], "output_alone": alone[0],
    })


@torch.no_grad()
def vae() -> None:
    latent, hidden, window = 16, 64, 4
    encoder = vae_modules._DConvEncoder(
        z_ch=latent, hidden_size=hidden, num_blocks=2, patch_size=16, head_size=96, num_head_blocks=2,
    ).eval()
    decoder = vae_modules._DConvDenoiser(
        patch_size=16, hidden_size=hidden, hidden_size_x=16, num_blocks=5, num_cond_blocks=2, bottleneck_dim=latent,
    ).eval()
    for block in decoder.y_embedder.decoder.block:
        if isinstance(block, vae_modules.AttnBlock):
            block.patch_size = window
    randomize(encoder, 2)
    randomize(decoder, 3)
    g = torch.Generator().manual_seed(4)

    # MageVAE._moments
    image = torch.rand(1, 3, 96, 80, generator=g) * 2 - 1
    moments = encoder.forward_pred(torch.zeros(1, latent, 6, 5), torch.zeros(1), image)
    # MageVAE.decode
    z = torch.randn(1, latent, 6, 5, generator=g)
    cond = decoder.y_embedder.decoder(z)
    decoded = decoder(torch.zeros(1, 3, 96, 80), torch.zeros(1), cond)

    weights = {f"student.dconv_encoder.{k}": v for k, v in encoder.state_dict().items()}
    weights.update({f"pipeline.{k}": v for k, v in decoder.state_dict().items()})
    # channels-last, as the runner holds images
    write("mage_vae", weights, {
        "image": image[0].permute(1, 2, 0), "moments": moments[0].permute(1, 2, 0),
        "latent": z[0].permute(1, 2, 0), "condition": cond[0].permute(1, 2, 0),
        "decoded": decoded[0].permute(1, 2, 0), "window": torch.tensor([float(window)]),
    })


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["transformer", "vae"]
    for name in wanted:
        {"transformer": transformer, "vae": vae}[name]()
