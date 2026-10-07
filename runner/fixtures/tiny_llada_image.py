"""Tiny LLaDA-Image golden fixtures, from the OFFICIAL code (inclusionAI's
`LLaDA-Image`, `src/models/transformer_llada_image.py`, and the LLaDA2 text
encoder's `modeling_llada2uni_moe.py` from the model repository), each
component built small with seeded random weights and run once in float32.

  transformer  1 refiner block of each kind and 2 main
               blocks of 2 heads of 128 (RoPE axes 32/48/48, as released), 16
               latent features, caption features of 40, semantic features of
               24: a 5 × 6 latent at σ 0.5 on 7 caption tokens (text to
               image), and the same with a clean 5 × 6 source latent and 5
               semantic tokens (editing). The timestep's sinusoid is rounded
               to BF16, as the released pipeline (in BF16) hands it to its MLP.
               -> tiny/llada_image/
  connectors   the QueryFormer (8 queries, 2 heads of 128, one block) on 7
               token embeddings, and the text projection (2 blocks of 4 heads
               of 64, to 40 features) on 15 hidden states.
               -> tiny/llada_connectors/ (names prefixed `queryformer.`,
               `text_projection.` and `sigvq.` as sd-cpp's connector files)
  sigvq        in the same file: 2 blocks of 2 heads of 96, a 4 × 4 position
               table, a codebook of 32: a 48 × 32 image to its tokens and
               semantic features.
  backbone     the LLaDA2 text model: 2 layers (one dense, one with 16
               experts in 4 groups, 4 chosen among 2 groups), 2 query heads
               over 1 key-value head of 128, half of each head rotated: 7 token
               embeddings then 8 queries, the text blind to the queries.
               -> tiny/llada2/

Pass the official code's folder as LLADA_OFFICIAL and the folder
holding the text encoder's three .py files as LLADA_TEXT_ENCODER.

Run from the repository root:
  git clone https://github.com/inclusionAI/LLaDA-Image /tmp/llada-official
  (text encoder code: modeling_llada2uni_moe.py, configuration_llada2uni_moe.py and fused_moe_ops.py
   from https://huggingface.co/inclusionAI/LLaDA-Image/tree/main/text_encoder into /tmp/llada-te)
  LLADA_OFFICIAL=/tmp/llada-official LLADA_TEXT_ENCODER=/tmp/llada-te \\
      uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
      --with torch --with diffusers==0.39.0 --with transformers==4.57.6 --with safetensors --with numpy \\
      python runner/fixtures/tiny_llada_image.py [transformer] [connectors] [backbone]
"""

import importlib
import importlib.util
import os
import sys
import types

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TINY_DIR = os.path.join(HERE, "..", "test", "resources", "fixtures", "tiny")


def official():
    """The official models module alone (the package's __init__ pulls the pipeline)."""
    path = os.path.join(os.environ["LLADA_OFFICIAL"], "src", "models", "transformer_llada_image.py")
    spec = importlib.util.spec_from_file_location("transformer_llada_image", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformer_llada_image"] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def randomize(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.ndim == 1 and name.endswith(".weight"):
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        elif p.ndim == 1:
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
        elif "pad_token" in name or "meta_queries" in name or "embedding" in name:
            p.copy_(0.5 * torch.randn(p.shape, generator=g))
        else:
            p.copy_(torch.randn(p.shape, generator=g) / p[0].numel() ** 0.5)
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
    m = official()

    def embed(self, timestep, hidden_dtype):
        import math
        half = self.frequency_embedding_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        return self.mlp(embedding.to(torch.bfloat16).float())

    m.LLaDAImageTimestepEmbedder.forward = embed
    model = m.LLaDAImageTransformer2DModel(
        in_channels=16, dim=256, n_layers=2, n_refiner_layers=1, n_heads=2, cap_feat_dim=40, semantic_feat_dim=24,
    ).eval()
    randomize(model, 0)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(16, 1, 5, 6, generator=g)
    source = torch.randn(16, 1, 5, 6, generator=g)
    caption = torch.randn(7, 40, generator=g)
    semantic = torch.randn(5, 24, generator=g)
    sigma = torch.tensor([0.5])
    plain = model(x=[x], t=sigma, cap_feats=[caption]).sample[0]
    edited = model(x=[x], t=sigma, cap_feats=[caption], glm_cap_feats=[semantic], source_latents=[source]).sample[0]
    # channels-last rows, as the runner holds latents
    rows = lambda t: t[:, 0].permute(1, 2, 0).reshape(-1, t.shape[0])
    write("llada_image", model.state_dict(), {
        "latents": rows(x), "source": rows(source), "caption": caption, "semantic": semantic, "sigma": sigma,
        "grid": torch.tensor([5.0, 6.0]), "output": rows(plain), "output_editing": rows(edited),
    })


@torch.no_grad()
def connectors() -> None:
    m = official()
    queryformer = m.LLaDAImageQueryFormerModel(
        num_queries=8, hidden_size=256, num_hidden_layers=1, num_attention_heads=2, intermediate_size=512,
    ).eval()
    projection = m.LLaDAImageTextProjectionModel(
        hidden_size=256, intermediate_size=384, num_hidden_layers=2, num_attention_heads=4, projection_dim=40,
    ).eval()
    sigvq = m.LLaDAImageSigVQModel(
        image_size=64, patch_size=16, hidden_size=192, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, codebook_size=32, codebook_embed_dim=24, semantic_embed_dim=24,
    ).eval()
    randomize(queryformer, 2)
    randomize(projection, 3)
    randomize(sigvq, 4)
    g = torch.Generator().manual_seed(5)
    embeds = torch.randn(1, 7, 256, generator=g)
    hidden = torch.randn(1, 15, 256, generator=g)
    image = torch.rand(1, 3, 48, 32, generator=g) * 2 - 1
    queries = queryformer(embeds, torch.ones(1, 7)).query_embeds[0]
    projected = projection(hidden).hidden_states[0]
    seen = sigvq(pixel_values=image)
    weights = {}
    for prefix, model in [("queryformer", queryformer), ("text_projection", projection), ("sigvq", sigvq)]:
        weights.update({f"{prefix}.{k}": v for k, v in model.state_dict().items()})
    write("llada_connectors", weights, {
        "embeds": embeds[0], "queries": queries, "hidden": hidden[0], "projected": projected,
        "image": image[0].permute(1, 2, 0), "semantic": seen.semantic_features[0],
        "tokens": seen.token_ids[0].float(),
    })


@torch.no_grad()
def backbone() -> None:
    folder = os.environ["LLADA_TEXT_ENCODER"]
    package = types.ModuleType("llada_te")
    package.__path__ = [folder]
    sys.modules["llada_te"] = package
    configuration = importlib.import_module("llada_te.configuration_llada2uni_moe")
    modeling = importlib.import_module("llada_te.modeling_llada2uni_moe")
    os.environ["LLADA_MOE_BACKEND"] = "eager"
    config = configuration.LLaDA2MoeConfig(
        vocab_size=64, hidden_size=256, intermediate_size=320, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=128, moe_intermediate_size=64, num_experts=16, num_experts_per_tok=4,
        n_group=4, topk_group=2, num_shared_experts=1, first_k_dense_replace=1, partial_rotary_factor=0.5,
        rope_theta=600000, routed_scaling_factor=2.5, norm_topk_prob=True, score_function="sigmoid",
        moe_router_enable_expert_bias=True, use_qk_norm=True, use_qkv_bias=False, use_bias=False,
        max_position_embeddings=128, pad_token_id=0, rms_norm_eps=1e-6, use_cache=False,
        rope_scaling={"mrope_section": [16, 24, 24], "rope_type": "default", "type": "default"},
    )
    model = modeling.LLaDA2MoeModel(config).eval()
    randomize(model, 6)
    g = torch.Generator().manual_seed(7)
    for name, buffer in model.named_buffers():
        if name.endswith("expert_bias"):
            buffer.copy_((0.05 * torch.randn(buffer.shape, generator=g)).to(torch.bfloat16).float())
    text, queries = 7, 8
    embeds = torch.randn(1, text + queries, 256, generator=g)
    # the pipeline's mask: every token valid, the text blind to the queries
    mask = torch.zeros(1, 1, text + queries, text + queries)
    mask[:, :, :text, text:] = torch.finfo(torch.float32).min
    hidden = model(
        inputs_embeds=embeds, attention_mask=mask, position_ids=torch.arange(text + queries)[None], return_dict=True,
    ).last_hidden_state
    weights = dict(model.state_dict())
    write("llada2", weights, {"embeds": embeds[0], "text": torch.tensor([float(text)]), "hidden": hidden[0]})
    import json
    json.dump(config.to_dict(), open(os.path.join(TINY_DIR, "llada2", "config.json"), "w"), indent=2)
    for k, v in weights.items():
        print("   ", k, tuple(v.shape))


if __name__ == "__main__":
    for name in sys.argv[1:] or ["transformer", "connectors", "backbone"]:
        {"transformer": transformer, "connectors": connectors, "backbone": backbone}[name]()
