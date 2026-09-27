"""Tiny PiD 1.5 golden fixture generator.

Builds the OFFICIAL PidNet (pid/_src/networks/pid_net.py) with the same flags as
the released configs but tiny sizes, random weights, and runs ONE network
forward in float32; and the official tokenizers' latent encoders
(pid/_src/tokenizers) the other variants condition on. Variants:

  flux2           `pid_sr4x_v1pt5_for_flux2` (pid_1.5_flux2_1024_to_4096_4step):
                  the FLUX.2 latent, 128 packed features at 1/16
                  -> tiny/pid/
  sixteen         `pid_sr4x_v1pt5` (pid_1.5_flux1_… and pid_1.5_qwenimage_…,
                  the same network): a 16-channel latent at 1/8
                  -> tiny/pid_sixteen/
  flux1_vae       flux_vae.py's AutoEncoder: an image encoded to the FLUX.1
                  latent (mean, 0.3611 (z - 0.1159)), and a latent decoded
                  -> tiny/pid_flux1_vae/
  qwen_image_vae  qwenimage_vae.py's WanVAE2d_: an image encoded to the Qwen
                  Image latent ((mu - mean) / std, Wan 2.1's statistics), saved
                  under the 3-D checkpoint's shapes -> tiny/pid_qwen_image_vae/

The official code is not packaged: clone it first (the fixtures were made at
2c8814c) and pass its folder as PID_OFFICIAL.

Outputs (runner/test/resources/fixtures/tiny/<folder>/):
  model.safetensors     the weights (the networks' rounded to BF16, as the real
                        checkpoints store them, and computed in float32), under
                        the real checkpoints' names
  expected.safetensors  inputs + outputs (+ debug taps), see the printout

Run from the repository root:
  git clone https://github.com/nv-tlabs/PiD /tmp/pid-official
  PID_OFFICIAL=/tmp/pid-official uv run --index https://download.pytorch.org/whl/cpu \\
      --index-strategy unsafe-best-match --with torch --with safetensors --with numpy \\
      --with einops python runner/fixtures/tiny_pid.py [variant ...]
"""

import json
import math
import os
import re
import struct
import sys
import types

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.environ["PID_OFFICIAL"]
TINY_DIR = os.path.join(HERE, "..", "test", "resources", "fixtures", "tiny")
REAL_DIR = (
    "/home/francois/.cache/huggingface/hub/models--Comfy-Org--PixelDiT/snapshots/"
    "8df2107641178235a534f1dc014ad3056922a89c/diffusion_models/"
)

# ---------------------------------------------------------------------------
# Minimal stubs for heavy / distributed imports of the official repo.
# pid_net.py imports `pid._ext.imaginaire.utils.log`; pixeldit_official.py and
# pid_net.py import the two context-parallel helpers (only used when CP is on);
# the tokenizers import the lazy config, the distributed helpers, a state-dict
# loader and their interface, none of which the bare modules use.
# ---------------------------------------------------------------------------
sys.path.insert(0, OFFICIAL)


def stub(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module


_log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
stub("pid._ext.imaginaire.utils", log=_log)


def _cp_unused(*args, **kwargs):
    raise RuntimeError("context parallel is not used in the fixture")


stub("pid._src.utils.context_parallel", cat_outputs_cp_with_grad=_cp_unused, split_inputs_cp=_cp_unused)
stub("pid._ext.imaginaire.utils.distributed", get_rank=lambda: 0, sync_model_states=lambda *a, **k: None)
stub("pid._ext.imaginaire.lazy_config", LazyCall=lambda f: (lambda *a, **k: None), LazyDict=dict)
stub("pid._src.models.utils", load_state_dict=None)
stub("pid._src.tokenizers.interface", VideoTokenizerInterface=object)

from pid._src.networks.pid_net import PidNet  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny net: same structure/flags as PID_SR4X_V1PT5_FOR_FLUX2
# (pid/_src/configs/common/defaults/net.py), small sizes.
# ---------------------------------------------------------------------------
TINY = dict(
    in_channels=3,
    num_groups=3,  # real 24   -> patch head_dim 64 (same as real)
    hidden_size=192,  # real 1536; SwiGLU hidden int(2*768/3) = 512, whole blocks of 32
    pixel_hidden_size=16,  # real 16 (kept)
    pixel_attn_hidden_size=144,  # real 1152
    pixel_num_groups=2,  # real 16   -> pixel head_dim 72 (same as real)
    patch_depth=4,  # real 14
    pixel_depth=2,  # real 2 (kept)
    patch_size=8,  # real 16; 8 keeps the per-pixel tables 4x smaller (latent upsampled x4)
    txt_embed_dim=64,  # real 2304 (Gemma 2 2B hidden)
    txt_max_length=12,  # real 300
    use_text_rope=True,
    text_rope_theta=10000.0,
    rope_mode="ntk_aware",
    # real: 2048 (ref grid 128). Tiny: 32 px -> ref grid 2 so the NTK factor is
    # non-trivial and different per axis (Hs/2 = 4, Ws/2 = 2), like 4096 output (factor 2).
    rope_ref_h=32,
    rope_ref_w=32,
    repa_encoder_index=6,  # only records tokens; no effect on output
    enable_ed=False,
    lq_inject_mode="controlnet",
    lq_in_channels=0,  # image branch off
    lq_latent_channels=128,  # Flux2 packed BN-normalized latent
    lq_hidden_dim=64,  # real 1024 (GroupNorm(4) needs /4)
    lq_num_res_blocks=4,  # kept -> latent_proj.3..6
    lq_latent_unpatchify_factor=2,
    lq_aux_rgb_head=False,  # real config True, but training-only and absent from the checkpoint
    lq_aux_rgb_head_latent_block_idx=-1,
    lq_conv_padding_mode="replicate",
    lq_gate_type="sigma_aware_per_token",
    lq_interval=2,  # injections before blocks 0 and 2 (real: 0,2,...,12)
    zero_init_lq=True,  # overwritten below with random values
    train_lq_proj_only=False,
    sr_scale=4,
    latent_spatial_down_factor=16,
    pit_lq_inject=True,
)

# PID_SR4X_V1PT5 (FLUX.1's and Qwen Image's): a 16-channel latent at 1/8, as it is.
SIXTEEN = dict(TINY, lq_latent_channels=16, lq_latent_unpatchify_factor=1, latent_spatial_down_factor=8)


# ---------------------------------------------------------------------------
# Random, non-degenerate weights (the official init zero-inits final_layer and
# lq output heads, which would hide paths).
# ---------------------------------------------------------------------------
@torch.no_grad()
def randomize(model: torch.nn.Module) -> None:
    g = torch.Generator().manual_seed(0)
    for name, p in model.named_parameters():
        if name.endswith("log_alpha"):
            p.copy_(torch.tensor(math.log(5.0)) + 0.3 * torch.randn((), generator=g))
        elif name == "y_pos_embedding":
            p.copy_(0.5 * torch.randn(p.shape, generator=g))
        elif p.ndim == 1 and name.endswith(".weight"):
            # every 1-D weight is an RMSNorm / GroupNorm scale
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        elif p.ndim == 1:
            # biases (incl. GroupNorm bias)
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
        else:
            fan_in = p[0].numel()
            p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(fan_in))
    # Keep adaLN gates/scales moderate so the residual stream stays O(1).
    for name, p in model.named_parameters():
        if "adaLN_modulation" in name:
            p.mul_(0.5)


def network(config: dict, folder: str, real_checkpoint: str) -> None:
    torch.manual_seed(0)
    net = PidNet(**config).float().eval()
    randomize(net)
    # the real checkpoint is BF16: the same weights, rounded, computed in float32
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(p.bfloat16().float())

    # -----------------------------------------------------------------------
    # Inputs, in the exact form the real inference feeds them.
    #   target H x W = 128 x 64  -> patch grid Hs x Ws = 8 x 4 (L = 32)
    #   source image = target / sr_scale = 32 x 16
    #   Flux2 VAE: 8x encoder + 2x2 patchify = 16x, 128 channels, BN-normalized
    #   -> LQ latent [1, 128, 32/16, 16/16] = [1, 128, 2, 1]
    #   16-channel VAEs: 8x -> LQ latent [1, 16, 4, 2]
    # (H, W must be multiples of 64 for sr_scale 4 x Flux2 16x; the requested 64x48
    #  would give a fractional latent, so 128x64 is used, non-square on purpose.)
    # -----------------------------------------------------------------------
    H, W = 128, 64
    down = config["sr_scale"] * config["latent_spatial_down_factor"]
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 3, H, W, generator=g)  # noisy RGB x_t (data range [-1, 1], noise N(0,1))
    t = torch.tensor([866.0])  # sigma * 1000 (fm_timescale), here student_t_list[1] = 0.866
    y = torch.randn(1, config["txt_max_length"], config["txt_embed_dim"], generator=g)
    # The real path never masks text (padded Gemma positions are attended), so the last
    # 4 "padding" rows are just ordinary rows here; they share one vector like real pads do.
    y[:, -4:, :] = y[:, -1:, :].clone()
    latent = torch.randn(1, config["lq_latent_channels"], H // down, W // down, generator=g)
    degrade_sigma = torch.tensor([0.0])  # value used at inference for a clean latent
    degrade_sigma_alt = torch.tensor([0.3])  # exercises the -exp(log_alpha)*sigma gate term

    # -----------------------------------------------------------------------
    # Debug taps
    # -----------------------------------------------------------------------
    taps = {}

    def tap(name):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        taps[f"{name}.{i}"] = o.detach().clone()
            else:
                taps[name] = output.detach().clone()

        return hook

    hooks = [
        net.t_embedder.register_forward_hook(tap("debug.t_emb")),
        net.y_embedder.register_forward_hook(tap("debug.y_embedder")),
        net.s_embedder.register_forward_hook(tap("debug.s_embedder")),
        net.lq_proj.register_forward_hook(tap("debug.lq_proj")),  # .0..N-1 = output_heads, last = pit_head
        net.pit_lq_gate.register_forward_hook(tap("debug.pit_lq_gate")),  # s_cond tokens [B, L, D]
        net.pixel_embedder.register_forward_hook(tap("debug.pixel_embedder")),
        net.final_layer.register_forward_hook(tap("debug.final_layer")),
    ]
    for i, blk in enumerate(net.patch_blocks):
        hooks.append(blk.register_forward_hook(tap(f"debug.patch_blocks.{i}")))  # .0 = image, .1 = text
    for i, blk in enumerate(net.pixel_blocks):
        hooks.append(blk.register_forward_hook(tap(f"debug.pixel_blocks.{i}")))
    for i, gm in enumerate(net.lq_proj.gate_modules):
        hooks.append(gm.register_forward_hook(tap(f"debug.lq_gate.{i}")))

    with torch.no_grad():
        output = net(x, t, y, lq_video_or_image=None, lq_latent=latent, degrade_sigma=degrade_sigma)
        debug = dict(taps)
        taps.clear()
        output_alt = net(x, t, y, lq_video_or_image=None, lq_latent=latent, degrade_sigma=degrade_sigma_alt)
        taps.clear()
    for h in hooks:
        h.remove()

    # Sanity: every path matters.
    assert output.shape == (1, 3, H, W)
    assert not torch.allclose(output, output_alt), "degrade_sigma had no effect"

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_dir = os.path.join(TINY_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)
    state = {f"net.{k}": v.detach().contiguous().bfloat16() for k, v in net.state_dict().items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"config": json.dumps(config)})

    expected = {
        "x": x,
        "t": t,
        "y": y,
        "latent": latent,
        "degrade_sigma": degrade_sigma,
        "output": output,
        "degrade_sigma_alt": degrade_sigma_alt,
        "output_alt": output_alt,
    }
    expected.update(debug)
    save_file({k: v.contiguous().float() for k, v in expected.items()}, os.path.join(out_dir, "expected.safetensors"))

    # -----------------------------------------------------------------------
    # Naming report vs the real checkpoint
    # -----------------------------------------------------------------------
    def collapse(name: str) -> str:
        return re.sub(r"\.\d+(?=\.|$)", ".N", name)

    def pattern_table(shapes: dict) -> dict:
        table = {}
        for k, s in shapes.items():
            table.setdefault(collapse(k), set()).add(tuple(s))
        return table

    tiny_patterns = pattern_table({k: list(v.shape) for k, v in state.items()})
    print(f"{folder}: {len(state)} tensors, {sum(v.numel() for v in state.values()):,} params")
    for k in sorted(tiny_patterns):
        print(f"  {k:60s} {sorted(tiny_patterns[k])}")

    real_path = os.path.join(REAL_DIR, real_checkpoint)
    if os.path.exists(real_path):
        with open(real_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        real = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
        real_patterns = pattern_table(real)
        only_tiny = sorted(set(tiny_patterns) - set(real_patterns))
        only_real = sorted(set(real_patterns) - set(tiny_patterns))
        exact_tiny = {re.sub(r"(patch_blocks|output_heads|gate_modules)\.\d+", r"\1.N", k) for k in state}
        exact_real = {re.sub(r"(patch_blocks|output_heads|gate_modules)\.\d+", r"\1.N", k) for k in real}
        print(f"\nreal checkpoint: {len(real)} tensors")
        print("name patterns only in tiny:", only_tiny or "none")
        print("name patterns only in real:", only_real or "none")
        print(
            "exact names (block indices collapsed only for patch_blocks/output_heads/gate_modules) "
            f"only in tiny: {sorted(exact_tiny - exact_real) or 'none'}; only in real: {sorted(exact_real - exact_tiny) or 'none'}"
        )
        latent_in = "net.lq_proj.latent_proj.0.weight"
        print(f"latent stack input: tiny {list(state[latent_in].shape)}, real {real[latent_in]}")

    print("\nexpected.safetensors:")
    for k, v in expected.items():
        print(f"  {k:36s} {list(v.shape)}")
    print(
        f"\noutput stats: mean={output.mean():.6f} std={output.std():.6f}; "
        f"|output - output_alt| max={((output - output_alt).abs().max()):.6f}"
    )


# ---------------------------------------------------------------------------
# Latent encoders
# ---------------------------------------------------------------------------
@torch.no_grad()
def randomize_vae(model: torch.nn.Module) -> None:
    for name, p in model.named_parameters():
        if "norm" in name and (name.endswith("weight") or name.endswith("gamma")):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        else:
            fan_in = p[0].numel() if p.dim() > 1 else 16
            p.copy_(torch.randn_like(p) / fan_in**0.5)


def channels_last(z):  # [1, C, h, w] -> [h, w, C]
    return z[0].permute(1, 2, 0).contiguous()


def flux1_vae() -> None:
    """flux_vae.py's AutoEncoder, three levels of 32/64/64 channels, one residual
    block per level, 16 latent channels: a 32 x 24 image encoded to the
    normalized mean (not sampled), and a random latent on its 4 x 3 grid
    decoded. LDM names, no quant convolutions, as Z-Image's ae.safetensors."""
    from pid._src.tokenizers.flux_vae import AutoEncoder, AutoEncoderParams

    torch.manual_seed(0)
    params = AutoEncoderParams(ch=32, ch_mult=[1, 2, 2], num_res_blocks=1, z_channels=16)
    model = AutoEncoder(params).eval().float()
    randomize_vae(model)
    with torch.no_grad():
        model.decoder.conv_out.weight.mul_(0.1)  # keep the image inside [-1, 1], unclamped
    image = torch.rand(1, 3, 32, 24) * 2 - 1
    latent = torch.randn(1, 16, 4, 3)
    with torch.no_grad():
        encoded = model.encode(image)
        decoded = model.decode(latent)[0].float()
    out_dir = os.path.join(TINY_DIR, "pid_flux1_vae")
    os.makedirs(out_dir, exist_ok=True)
    tensors = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    save_file(
        {
            "image": channels_last(image),
            "latents": channels_last(encoded),
            "latent": channels_last(latent),
            "decoded": decoded.permute(1, 2, 0).contiguous(),
        },
        os.path.join(out_dir, "expected.safetensors"),
    )
    print(f"pid_flux1_vae: {len(tensors)} tensors, latents {tuple(encoded.shape)}, |max| {decoded.abs().max():.3f}")
    print("  names:", sorted({re.sub(r"\.\d+\.", ".N.", n) for n in tensors}))


def qwen_image_vae() -> None:
    """qwenimage_vae.py's WanVAE2d_ (the 2-D Wan 2.1 VAE the PiD Qwen Image
    student was trained with), 8 channels at its base (32 in the middle), 16
    latent channels: a 32 x 24 image encoded to (mu - mean) / std. Saved as the
    3-D checkpoint stores it (the ComfyUI repackage's qwen_image_vae): causal
    3x3x3 kernels whose last temporal slice is the 2-D one (the earlier ones
    random, unused on one frame), 1x1x1 shortcuts and quant convolution, the
    downsamplers' `time_conv` (unused on one frame), gammas [C, 1, 1, 1]."""
    from pid._src.tokenizers.qwenimage_vae import _LATENTS_MEAN, _LATENTS_STD, WanVAE2d_

    torch.manual_seed(0)
    model = WanVAE2d_(dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                      temperal_downsample=[False, True, True]).eval().float()
    randomize_vae(model)
    image = torch.rand(1, 3, 32, 24) * 2 - 1
    mean = torch.tensor(_LATENTS_MEAN)
    std = torch.tensor(_LATENTS_STD)
    with torch.no_grad():
        encoded = model.encode(image, [mean, 1.0 / std])

    def three_d(name, value):
        if name.endswith("gamma") and not name.startswith("encoder.middle.1") and not name.startswith("decoder.middle.1"):
            return value.reshape(value.shape[0], 1, 1, 1)
        if value.dim() == 4 and ".resample." not in name and ".middle.1." not in name:
            depth = value.shape[-1] if value.shape[-1] == 3 else 1
            earlier = torch.randn(value.shape[0], value.shape[1], depth - 1, *value.shape[2:]) / value[0].numel() ** 0.5
            return torch.cat([earlier, value.unsqueeze(2)], dim=2)
        return value

    tensors = {k: three_d(k, v.detach().float()).contiguous() for k, v in model.state_dict().items()}
    temporal = [i for i, m in enumerate(model.encoder.downsamples) if hasattr(m, "mode") and m.mode == "downsample2d"]
    for i in temporal[1:]:  # the real encoder's last two downsamplers are temporal too
        dim = model.encoder.downsamples[i].dim
        tensors[f"encoder.downsamples.{i}.time_conv.weight"] = torch.randn(dim, dim, 3, 1, 1)
        tensors[f"encoder.downsamples.{i}.time_conv.bias"] = torch.randn(dim)
    out_dir = os.path.join(TINY_DIR, "pid_qwen_image_vae")
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    save_file(
        {"image": channels_last(image), "latents": channels_last(encoded)},
        os.path.join(out_dir, "expected.safetensors"),
    )
    print(f"pid_qwen_image_vae: {len(tensors)} tensors, latents {tuple(encoded.shape)}, |max| {encoded.abs().max():.3f}")
    print("  encoder names:", sorted({re.sub(r"\.\d+\.", ".N.", n) + f" {list(v.shape)}" for n, v in tensors.items()
                                      if n.startswith("encoder") or n.startswith("conv1")}))


variants = {
    "flux2": lambda: network(TINY, "pid", "pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors"),
    "sixteen": lambda: network(SIXTEEN, "pid_sixteen", "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors"),
    "flux1_vae": flux1_vae,
    "qwen_image_vae": qwen_image_vae,
}
for variant in sys.argv[1:] or variants:
    variants[variant]()
