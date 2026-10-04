"""SeedVR2 golden fixture generator.

Builds the reference networks (numz/ComfyUI-SeedVR2_VideoUpscaler: `NaDiT`,
`VideoAutoencoderKLWrapper`) from their released configurations and runs them
in float32 **on the CPU only**. Variants:

  transformer  the 3B `NaDiT` (configs_3b/main.yaml) with tiny sizes and random
               weights: one forward over 5 latent frames of 40 x 48 (a token
               grid of 5 x 20 x 24, so every axis has several windows that
               the shifted layers move, time included), at two timesteps, with taps after every block and
               around the first two attentions -> tiny/seedvr2/
  transformer7b  the 7B `NaDiT` (configs_7b/main.yaml) alike, the velocities
               alone -> tiny/seedvr2_7b/
  vae          the video VAE (s8_c16_t4_inflation_sd3.yaml) with tiny widths
               and random weights: one frame and 5 frames of 48 x 64 encoded
               (the posterior's mode) and decoded -> tiny/seedvr2_vae/
  rules        no network: the windows of both methods for a list of token
               grids -> tiny/seedvr2_rules/
  picture      the REAL 3B weights and VAE on a synthetic 64 x 64 picture
               upscaled to 128 x 128, every stage dumped, with the exact RoPE
               frequencies in place of the file's rounded ones
               -> tiny/seedvr2_picture/
  clip         the same on 5 frames -> tiny/seedvr2_clip/
  picture7b    the picture through the REAL 7B -> tiny/seedvr2_picture_7b/

The reference is not packaged: clone it and pass its folder as SEEDVR2_OFFICIAL,
and for `picture` and `clip` the folder holding the weights as SEEDVR2_MODELS
(`seedvr2_ema_3b_fp8_e4m3fn.safetensors`, `ema_vae_fp16.safetensors`).

Outputs (runner/test/resources/fixtures/tiny/<folder>/):
  model.safetensors     the tiny weights (the transformer's rounded to BF16 and
                        the VAE's to F16, as the real checkpoints store them,
                        and computed in float32), under the real names
  expected.safetensors  inputs + outputs (+ debug taps), see the printout
`picture` also writes runner/resources/seedvr2/text.safetensors: the fixed
positive text embedding (`pos_emb.pt` of the reference), as the runner ships it.

PyTorch must never reach the GPU on this machine: run it with the devices
hidden, from the repository root, with the reference's own environment:
  HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" \\
      SEEDVR2_OFFICIAL=~/dev/redraw-experiments/SeedVR2 \\
      SEEDVR2_MODELS=~/dev/redraw-experiments/models/seedvr2 \\
      ~/dev/redraw-experiments/SeedVR2/.venv/bin/python \\
      runner/fixtures/tiny_seedvr2.py [variant ...]
"""

import math
import os
import sys

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

assert not torch.cuda.is_available() and torch.cuda.device_count() == 0, "the GPU must stay hidden"

from omegaconf import OmegaConf  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.expanduser(os.environ["SEEDVR2_OFFICIAL"])
TINY_DIR = os.path.join(HERE, "..", "test", "resources", "fixtures", "tiny")
CPU = torch.device("cpu")

sys.path.insert(0, OFFICIAL)

from src.common.config import create_object, load_config  # noqa: E402
from src.core.infer import VideoDiffusionInfer  # noqa: E402
from src.models.dit_3b import na  # noqa: E402
from src.models.dit_3b.window import get_window_op  # noqa: E402
from src.utils.color_fix import lab_color_transfer  # noqa: E402


class Quiet:
    """The reference's `Debug`, silent."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def configuration(model: str = "3b"):
    config = load_config(os.path.join(OFFICIAL, f"configs_{model}", "main.yaml"))
    OmegaConf.set_readonly(config, False)
    vae = load_config(os.path.join(OFFICIAL, "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
    config.vae.model = OmegaConf.merge(config.vae.model, vae)
    return config


def write(folder: str, name: str, tensors: dict) -> None:
    directory = os.path.join(TINY_DIR, folder)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    print(f"{folder}/{name}  {os.path.getsize(path) / 1e6:.1f} MB")
    for k, v in tensors.items() if name != "model.safetensors" else ():
        print(f"  {k:32s} {str(v.dtype).replace('torch.', ''):8s} {list(v.shape)}")


@torch.no_grad()
def randomize(model: torch.nn.Module, stored: torch.dtype) -> None:
    """Random, non-degenerate weights, rounded to the dtype the real checkpoint
    stores (the initial ones zero some outputs, which would hide paths)."""
    g = torch.Generator().manual_seed(0)
    for name, p in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf.endswith(("_shift", "_gate")) or (p.ndim == 1 and leaf == "bias"):
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
        elif leaf.endswith("_scale"):
            p.copy_(0.2 * torch.randn(p.shape, generator=g))
        elif p.ndim == 1:
            # the scales of the RMS, layer and group norms
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        else:
            p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p[0].numel()))
        p.copy_(p.to(stored).float())


def stored_weights(model: torch.nn.Module, stored: torch.dtype) -> dict:
    return {k: v.to(stored) for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# The transformer
# ---------------------------------------------------------------------------
def tiny_transformer_config(model_name: str):
    model = configuration(model_name).dit.model
    model.vid_dim = 256  # real 2560 (7B 3072); heads of 128 and their RoPE are kept
    model.heads = 2  # real 20 (7B 24)
    model.txt_in_dim = 96  # real 5120
    model.num_layers = 4  # real 32 (7B 36): plain, shifted, plain, shifted windows
    if model_name == "3b":
        model.mm_layers = 2  # real 10: two blocks apart, two shared (the last one video only)
    return model


@torch.no_grad()
def transformer(model_name: str = "3b") -> None:
    torch.manual_seed(0)
    net = create_object(tiny_transformer_config(model_name)).float().eval()
    randomize(net, torch.bfloat16)
    small = model_name != "3b"
    folder = "seedvr2" if model_name == "3b" else f"seedvr2_{model_name}"

    g = torch.Generator().manual_seed(1)
    # 16 of noise, 16 of the low-quality latent, the mask at 1
    latent = torch.randn(5, 40, 48, 33, generator=g)
    latent[..., -1] = 1.0
    text = torch.randn(12, 96, generator=g)
    vid, vid_shape = na.flatten([latent])
    txt, txt_shape = na.flatten([text])

    taps = {}

    def keep(name, value):
        taps[name] = value.detach().float().clone()

    def block_hook(index):
        def hook(_module, _args, output):
            keep(f"block{index}.vid", output[0])
            keep(f"block{index}.txt", output[1])

        return hook

    def attention_hook(index):
        def hook(_module, args, output):
            keep(f"attention{index}.vid_in", args[0])
            keep(f"attention{index}.txt_in", args[1])
            keep(f"attention{index}.vid_out", output[0])
            keep(f"attention{index}.txt_out", output[1])

        return hook

    handles = [net.vid_in.register_forward_hook(lambda _m, _a, out: keep("vid_in", out[0]))]
    handles.append(net.emb_in.register_forward_hook(lambda _m, _a, out: keep("emb", out)))
    handles.append(net.txt_in.register_forward_hook(lambda _m, _a, out: keep("txt_in", out)))
    for i, block in enumerate(net.blocks if not small else ()):
        handles.append(block.register_forward_hook(block_hook(i)))
        if i < 2:
            handles.append(block.attn.register_forward_hook(attention_hook(i)))

    def forward(timestep: float) -> torch.Tensor:
        return net(
            vid=vid, txt=txt, vid_shape=vid_shape, txt_shape=txt_shape, timestep=torch.tensor([timestep])
        ).vid_sample

    # the one step of the released sampler, then a mid one for the embedding
    velocity = forward(1000.0)
    for handle in handles:
        handle.remove()
    velocity_mid = forward(637.5)

    write(folder, "model.safetensors", stored_weights(net, torch.bfloat16))
    write(
        folder,
        "expected.safetensors",
        {
            "latent": latent,
            "text": text,
            "velocity": na.unflatten(velocity, vid_shape)[0],
            "velocity_mid": na.unflatten(velocity_mid, vid_shape)[0],
            **taps,
        },
    )


# ---------------------------------------------------------------------------
# The VAE
# ---------------------------------------------------------------------------
def build_vae(config, model_config) -> torch.nn.Module:
    vae = create_object(model_config).float().eval()
    vae.set_causal_slicing(**config.vae.slicing)
    vae.set_memory_limit(**config.vae.memory_limit)
    return vae


@torch.no_grad()
def vae() -> None:
    torch.manual_seed(0)
    config = configuration()
    model = config.vae.model
    model.block_out_channels = [32, 64, 128, 128]  # real 128, 256, 512, 512; 32 groups kept
    net = build_vae(config, model)
    randomize(net, torch.float16)

    g = torch.Generator().manual_seed(1)
    expected = {}
    for name, frames in (("picture", 1), ("clip", 5)):
        pixels = torch.rand(1, 3, frames, 48, 64, generator=g) * 2 - 1
        posterior = net.encode(pixels).posterior
        latent = posterior.mode()
        # a latent of its own too, so the decoder is checked apart from the encoder
        noise = torch.randn(latent.shape, generator=g)
        expected[f"{name}.pixels"] = pixels[0]
        expected[f"{name}.mean"] = latent[0]
        expected[f"{name}.decoded"] = net.decode(latent).sample[0].reshape(3, frames, 48, 64)
        expected[f"{name}.latent"] = noise[0]
        expected[f"{name}.latent_decoded"] = net.decode(noise).sample[0].reshape(3, frames, 48, 64)

    write("seedvr2_vae", "model.safetensors", stored_weights(net, torch.float16))
    write("seedvr2_vae", "expected.safetensors", expected)


# ---------------------------------------------------------------------------
# The rules that are not a network
# ---------------------------------------------------------------------------
# latent token grids (frames, height, width): the tiny one, a 128 px picture,
# 1024 and 1664 px pictures, a 4k one, 720p and 1080p clips of 5, 21 and 49
# frames, and odd ones for the rounding
GRIDS = [
    (5, 20, 24),
    (1, 8, 8),
    (1, 64, 64),
    (1, 104, 104),
    (1, 135, 240),
    (2, 45, 80),
    (6, 45, 80),
    (13, 68, 120),
    (33, 45, 80),
    (1, 37, 91),
    (5, 51, 29),
]


def rules() -> None:
    expected = {"grids": torch.tensor(GRIDS, dtype=torch.int32)}
    for index, grid in enumerate(GRIDS):
        for name, method in (("plain", "720pwin_by_size_bysize"), ("shifted", "720pswin_by_size_bysize")):
            windows = get_window_op(method)(grid, (4, 3, 3))
            expected[f"windows.{index}.{name}"] = torch.tensor(
                [[t.start, t.stop, h.start, h.stop, w.start, w.stop] for t, h, w in windows], dtype=torch.int32
            )
    write("seedvr2_rules", "expected.safetensors", expected)


# ---------------------------------------------------------------------------
# The real model, end to end
# ---------------------------------------------------------------------------
def subject(frames: int, size: int = 64) -> torch.Tensor:
    """A synthetic low-quality input, [frames, size, size, 3] of bytes: a
    gradient, a disc, stripes and grain, drifting one pixel a frame."""
    g = torch.Generator().manual_seed(2)
    grain = torch.randn(size, size + frames, 3, generator=g)
    images = []
    for frame in range(frames):
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size) + frame, indexing="ij")
        image = torch.stack([x / (size + frames), y / size, 1 - x / (size + frames)], dim=-1) * 0.6 + 0.2
        disc = ((x - 24) ** 2 + (y - 28) ** 2) < 14**2
        image[disc] = torch.tensor([0.85, 0.62, 0.5])
        stripes = (x > 44) & ((y // 3) % 2 == 0)
        image[stripes] *= 0.4
        image += 0.03 * grain[:, frame : frame + size]
        images.append(image)
    return (torch.stack(images).clamp(0, 1) * 255).round().to(torch.uint8)


@torch.no_grad()
def real(
    folder: str, frames: int, source=None, side: int = 128, exact_frequencies: bool = True, model: str = "3b"
) -> None:
    from src.core.generation_utils import prepare_video_transforms

    models = os.path.expanduser(os.environ["SEEDVR2_MODELS"])
    config = configuration(model)
    config.diffusion.cfg.scale = 1.0
    config.diffusion.cfg.rescale = 0.0
    config.diffusion.timesteps.sampling.steps = 1

    runner = VideoDiffusionInfer(config, Quiet())
    runner.dit = create_object(config.dit.model).float().eval()
    weights = load_file(os.path.join(models, f"seedvr2_ema_{model}_fp8_e4m3fn.safetensors"))
    runner.dit.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
    # the fp8 file rounds the RoPE frequencies it stores too (the 3B's 0.645 to
    # 0.625 and its lowest to 0, the 7B's 91.8 to 88), and the reference takes
    # them as they are; the runner computes them, so the golden model gets the
    # exact ones back
    for block in runner.dit.blocks if exact_frequencies else ():
        freqs = block.attn.rope.rope.freqs
        if model == "3b":
            freqs.copy_(1.0 / (10000 ** (torch.arange(0, 2 * len(freqs), 2).float() / (2 * len(freqs)))))
        else:
            freqs.copy_(torch.linspace(1.0, 128.0, len(freqs)) * math.pi)
    runner.vae = build_vae(config, config.vae.model)
    weights = load_file(os.path.join(models, "ema_vae_fp16.safetensors"))
    runner.vae.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
    config.vae.dtype = "float32"
    runner.configure_diffusion(device=CPU, dtype=torch.float32)

    positive = torch.load(os.path.join(OFFICIAL, "pos_emb.pt"), map_location="cpu")
    negative = torch.load(os.path.join(OFFICIAL, "neg_emb.pt"), map_location="cpu")

    source = subject(frames) if source is None else source
    # the reference's own preparation: bicubic to a short side of 128, clamp,
    # pad to 16, [-1, 1], channels first
    video = prepare_video_transforms(side)(source.permute(0, 3, 1, 2).float() / 255.0)

    latent = runner.vae_encode([video])[0]
    torch.manual_seed(42)
    noise = torch.randn_like(latent)
    condition = runner.get_condition(noise, task="sr", latent_blur=latent)

    taps = {}
    handle = runner.dit.register_forward_hook(
        lambda _m, _a, kwargs, out: taps.update(
            velocity=na.unflatten(out.vid_sample, torch.tensor([list(latent.shape[:3])]))[0].clone(),
            timestep=kwargs["timestep"].float().clone(),
        ),
        with_kwargs=True,
    )
    restored = runner.inference(
        noises=[noise], conditions=[condition], texts_pos=[positive.float()], texts_neg=[negative.float()]
    )[0]
    handle.remove()

    decoded = runner.vae_decode([restored])[0]
    decoded = decoded.unsqueeze(1) if decoded.ndim == 3 else decoded
    # channels, frames -> frames, channels, as the colour match takes them
    result = decoded.permute(1, 0, 2, 3)
    # on copies: the colour match works in place
    matched = lab_color_transfer(result.clone(), video.permute(1, 0, 2, 3).clone(), Quiet(), luminance_weight=0.8)

    write(
        folder,
        "expected.safetensors",
        {
            "source": source,
            "video": video,
            "latent": latent,
            "noise": noise,
            "restored": restored,
            "decoded": result,
            "matched": matched,
            **taps,
        },
    )
    if frames == 1 and model == "3b":
        # the runner's own copy: the positive embedding is the only one used
        resources = os.path.join(HERE, "..", "resources", "seedvr2")
        os.makedirs(resources, exist_ok=True)
        save_file({"positive": positive}, os.path.join(resources, "text.safetensors"))


VARIANTS = {
    "transformer": transformer,
    "transformer7b": lambda: transformer("7b"),
    "vae": vae,
    "rules": rules,
    "picture": lambda: real("seedvr2_picture", 1),
    "clip": lambda: real("seedvr2_clip", 5),
    "picture7b": lambda: real("seedvr2_picture_7b", 1, model="7b"),
}

if __name__ == "__main__":
    for variant in sys.argv[1:] or list(VARIANTS):
        VARIANTS[variant]()
