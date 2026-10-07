"""Tiny GRN golden fixtures, from the OFFICIAL code (bytedance's GRN,
`grn/models/grn.py` and `grn/models/hbq_tokenizer.py`), each built small with
seeded random weights and run once in float32 on the CPU.

  transformer  2 blocks of 2 heads of 128 (the released 3-D RoPE, 21/21/22
               pairs), bits of 8 latent channels × 4 rounds: one refinement
               pass of a 5 × 6 grid of bits on 7 text tokens (features of 48)
               at progress 0.3 — the logits of every bit. The official loop
               wants a GPU; the pass is its body's (bits one-hot through
               `word_embed`, [bits ; text ; progress token], the blocks, the
               head).
               -> tiny/grn/
  tokenizer    the HBQ tokenizer's decoder (Wan 2.2's VAE in RGB, pixels
               patchified 2 × 2), 8 latent channels, widths of 32: a 3 × 2
               latent decoded to a 48 × 32 image.
               -> tiny/grn_tokenizer/

Pass the official repository's folder as GRN_OFFICIAL.

Run from the repository root:
  git clone https://github.com/MGenAI/GRN /tmp/grn-official
  GRN_OFFICIAL=/tmp/grn-official uv run --index https://download.pytorch.org/whl/cpu \\
      --index-strategy unsafe-best-match --with torch --with safetensors --with numpy --with einops \\
      --with timm --with opencv-python-headless --with tqdm --with pillow \\
      python runner/fixtures/tiny_grn.py [transformer] [tokenizer]
"""

import os
import sys
import types

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TINY_DIR = os.path.join(HERE, "..", "test", "resources", "fixtures", "tiny")
sys.path.insert(0, os.environ["GRN_OFFICIAL"])


@torch.no_grad()
def randomize(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if name.endswith("gamma") or (p.ndim == 1 and name.endswith(".weight")):
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        elif p.ndim == 1:
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
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
    # the sequence-parallel helpers import distributed code the pass never uses
    for name in ("grn.utils_t2iv.dist", "grn.utils_t2iv.sequence_parallel"):
        stub = types.ModuleType(name)
        stub.get_device = lambda: "cpu"
        stub.for_visualize = lambda f: f
        stub.SequenceParallelManager = types.SimpleNamespace(sp_on=lambda: False, get_sp_size=lambda: 1)
        stub.sp_gather_sequence_by_dim = stub.sp_split_sequence_by_dim = stub.sp_all_to_all = None
        sys.modules[name] = stub
        setattr(__import__("grn.utils_t2iv", fromlist=["x"]), name.rsplit(".", 1)[1], stub)
    from grn.models.grn import GRN, build_attn_mask
    from grn.schedules.global_refine import get_visual_rope_embeds
    from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input

    latent, rounds = 8, 4
    args = types.SimpleNamespace(
        detail_scale_dim=latent, detail_num_lvl=2, hbq_round=rounds, refine_mode="ar_discrete_GRN_bit",
        dynamic_scale_schedule="GRN_vae_stride16", train_h_div_w_list="[]", video_frames=81,
        use_ada_layer_norm=0, temporal_compress_rate=4, add_scale_token=1,
    )
    model = GRN(
        vae_local=types.SimpleNamespace(codebook_dim=latent), arch="qwen", qwen_qkvo_bias=False,
        text_channels=48, text_maxlen=512, embed_dim=256, depth=2, num_heads=2, num_key_value_heads=2,
        mlp_ratio=3.55, block_chunks=1, rope2d_normalized_by_hw=2, other_args=args,
    ).eval()
    randomize(model, 0)
    g = torch.Generator().manual_seed(1)
    pt, ph, pw = 1, 5, 6
    bits = torch.randint(0, 2, (1, latent * rounds, pt, ph, pw), generator=g)
    text = torch.randn(7, 48, generator=g)
    progress = 0.3

    # the pipeline's: the trained shape nearest the picture's
    import numpy as np
    template = model.h_div_w_templates[np.argmin(np.abs(model.h_div_w_templates - ph / pw))]

    visual = model.embeds_codes2input(multiclass_labels2onehot_input(bits, 2))
    prefix = model.text_proj(text)
    token = model.pt_embedder(torch.tensor([progress])).unsqueeze(0)
    sequence = torch.cat((visual, prefix.unsqueeze(0), token), dim=1)
    rope = torch.cat([
        get_visual_rope_embeds(model.rope2d_freqs_grid, (pt, ph, pw), "cpu", template, t_offset=0),
        model.rope2d_freqs_grid["freqs_text"][:, :, :, :, :7],
        model.rope2d_freqs_grid["freqs_text"][:, :, :, :, 512:513],
    ], dim=4)
    rope = rope[:, 0].permute(0, 1, 3, 2, 4)
    length = sequence.shape[1]
    mask = build_attn_mask([length], "cpu")
    hidden = sequence
    for chunk in model.block_chunks:
        hidden = chunk(x=hidden, cu_seqlens=torch.tensor([0, length], dtype=torch.int32), max_seqlen=length, e0=None,
                       attn_bias_or_two_vector=mask, attn_fn=None, rope2d_freqs_grid=rope, last_diffusion_step=False)
    logits = model.get_logits_during_infer(hidden)[0, : pt * ph * pw]
    # bits as the runner holds them: a row per latent, a column per bit
    write("grn", model.state_dict(), {
        "bits": bits[0, :, 0].permute(1, 2, 0).reshape(ph * pw, -1), "text": text,
        "progress": torch.tensor([progress]), "grid": torch.tensor([float(ph), float(pw)]), "logits": logits,
    })
    for k, v in model.state_dict().items():
        print("   ", k, tuple(v.shape))


@torch.no_grad()
def tokenizer() -> None:
    from grn.models.hbq_tokenizer import HBQ_Tokenizer
    model = HBQ_Tokenizer(args=None, dim=32, dec_dim=32, latent_channels=8, encoder_out_type="feature_tanh").eval()
    randomize(model, 2)
    g = torch.Generator().manual_seed(3)
    z = torch.rand(1, 8, 1, 3, 2, generator=g) * 2 - 1
    image = model.decode(z)
    write("grn_tokenizer", model.state_dict(), {
        "latent": z[0, :, 0].permute(1, 2, 0), "decoded": image[0, :, 0].permute(1, 2, 0),
    })


if __name__ == "__main__":
    for name in sys.argv[1:] or ["transformer", "tokenizer"]:
        {"transformer": transformer, "tokenizer": tokenizer}[name]()
