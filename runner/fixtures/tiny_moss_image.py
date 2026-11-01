"""Tiny MOSS-Image golden fixtures, from the OFFICIAL code (OpenMOSS's
`pipeline_moss_image.py`, shipped beside the weights on Hugging Face): its
transformer (`MOSSImageTransformer`) and scheduler (`MOSSImageScheduler`),
built small with seeded random weights and run once in float32 on the CPU.
diffusers and transformers are stood in for by the few names the file
imports; neither runs.

  moss_image  3 blocks of 2 heads of 128 (RoPE 16 frequencies per axis, as
              released), an MLP of 384, text features of 64, 32 latent
              features: the text refiner's output for 7 tokens, one velocity
              at σ 0.62 of a 3 × 4 image, the same without block 1 (the
              negative branch's skip), and with a LoRA at 0.8 under
              OpenMOSS's names (`diffusion_model.`, split q and fused
              ffn_up_gate and text qkv) and kohya's. The weights as released
              (q, k, v apart) and fused as ComfyUI and leejet store them
              (`proj.qkv`). The scheduler's levels for a few step counts and
              shifts.
              -> tiny/moss_image/

Outputs (runner/test/resources/fixtures/tiny/moss_image/):
  model.safetensors        the weights, rounded to BF16, under their names
  model_fused.safetensors  the same with each block's q, k and v as one
  lora.safetensors         the LoRA
  expected.safetensors     inputs + outputs

Run from the repository root:
  curl -L -o /tmp/pipeline_moss_image.py \\
      https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/resolve/main/pipeline_moss_image.py
  MOSS_OFFICIAL=/tmp/pipeline_moss_image.py uv run --index https://download.pytorch.org/whl/cpu \\
      --index-strategy unsafe-best-match --with torch --with safetensors --with numpy \\
      python runner/fixtures/tiny_moss_image.py
"""

import importlib.util
import inspect
import os
import sys
import types

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TINY_DIR = os.path.join(HERE, "..", "test", "resources", "fixtures", "tiny")


class ConfigMixin:
    def register_to_config(self, **values):
        if "_values" not in self.__dict__:
            self.__dict__["_values"] = {}
        self.__dict__["_values"].update(values)

    @property
    def config(self):
        return types.SimpleNamespace(**self.__dict__["_values"])


def register_to_config(init):
    signature = inspect.signature(init)

    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        init(self, *args, **kwargs)
        ConfigMixin.register_to_config(self, **{k: v for k, v in bound.arguments.items() if k != "self"})

    return wrapped


class FlowMatchEulerDiscreteScheduler(ConfigMixin):
    """What the official scheduler hands its levels to."""

    def __init__(self, num_train_timesteps=1000, shift=1.0):
        self.register_to_config(num_train_timesteps=num_train_timesteps, shift=shift)

    def set_timesteps(self, device=None, sigmas=None):
        self.levels = sigmas


class Placeholder:
    pass


def stub(name, **names):
    module = types.ModuleType(name)
    module.__dict__.update(names)
    sys.modules[name] = module


stub("diffusers", AutoencoderKLFlux2=Placeholder, DiffusionPipeline=Placeholder,
     FlowMatchEulerDiscreteScheduler=FlowMatchEulerDiscreteScheduler)
stub("diffusers.schedulers")
stub("diffusers.schedulers.scheduling_utils", SchedulerMixin=Placeholder)
stub("diffusers.configuration_utils", ConfigMixin=ConfigMixin, register_to_config=register_to_config)
stub("diffusers.image_processor", VaeImageProcessor=Placeholder)
stub("diffusers.models")
stub("diffusers.models.modeling_utils", ModelMixin=torch.nn.Module)
stub("diffusers.models.modeling_outputs", Transformer2DModelOutput=lambda sample: types.SimpleNamespace(sample=sample))
stub("diffusers.pipelines")
stub("diffusers.pipelines.pipeline_utils", ImagePipelineOutput=Placeholder)
stub("diffusers.utils")
stub("diffusers.utils.torch_utils", randn_tensor=None)
stub("transformers", AutoModel=Placeholder, AutoTokenizer=Placeholder)

spec = importlib.util.spec_from_file_location("pipeline_moss_image", os.environ["MOSS_OFFICIAL"])
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)

TEXT_TOKENS = 7
HEIGHT, WIDTH = 3, 4
SCHEDULES = [(1, 5.0), (4, 5.0), (8, 3.0), (50, 5.0), (200, 5.0)]


def bf16(tensor):
    return tensor.to(torch.bfloat16).float()


@torch.no_grad()
def randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.ndim == 1:
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        else:
            p.copy_(torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5)
        p.copy_(bf16(p))


def lora_for(model, run, targets):
    """A LoRA for the tiny model (`lora_A` down [rank, in], `lora_B` up
    [out, rank]; kohya's `lora_unet_` names `lora_down`/`lora_up`), some with
    an `.alpha`. Returns the file's tensors and the result with every delta
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


def fused(state):
    """Each block's q, k and v as one `proj.qkv`, as ComfyUI stores them."""
    out = {}
    for name, value in state.items():
        if name.endswith(".proj.q.weight"):
            prefix = name[: -len("q.weight")]
            out[prefix + "qkv.weight"] = torch.cat(
                [state[prefix + part + ".weight"] for part in ("q", "k", "v")])
        elif not (name.endswith(".proj.k.weight") or name.endswith(".proj.v.weight")):
            out[name] = value
    return out


@torch.no_grad()
def moss_image():
    model = official.MOSSImageTransformer(
        dim=256, ffn_dim=384, num_heads=2, num_layers=3, text_dim=64, freq_dim=256, latent_channels=32)
    randomize(model, 1)
    g = torch.Generator().manual_seed(2)
    text = bf16(torch.randn(1, TEXT_TOKENS, 64, generator=g))
    latents = torch.randn(1, 32, HEIGHT, WIDTH, generator=g)
    sigma = 0.62
    # the scheduler's timestep: σ × 1000 in float32
    timestep = torch.tensor([sigma], dtype=torch.float32) * 1000

    def velocity(skip=None):
        out = model(latents, timestep, text, skip_layers=skip).sample
        return out[0].flatten(1).transpose(0, 1)

    rope = model.rope(HEIGHT, WIDTH, TEXT_TOKENS, "cpu")
    embedded = model.text_embedder(text[0:1], rope[:, HEIGHT * WIDTH:])[0]
    result = velocity()
    skipped = velocity([1])
    lora, lora_result = lora_for(model, velocity, [
        ("layers.1.proj.q", "diffusion_model.layers.1.proj.q", 4, 2.0, ("lora_A.weight", "lora_B.weight")),
        ("layers.0.proj.ffn_up_gate", "diffusion_model.layers.0.proj.ffn_up_gate", 4, None,
         ("lora_A.weight", "lora_B.weight")),
        ("text_embedder.blocks.0.qkv", "diffusion_model.text_embedder.blocks.0.qkv", 4, None,
         ("lora_A.weight", "lora_B.weight")),
        ("layers.2.proj.o", "lora_unet_layers_2_proj_o", 2, 1.0, ("lora_down.weight", "lora_up.weight")),
        ("shared_block_modulation", "diffusion_model.shared_block_modulation", 4, None,
         ("lora_A.weight", "lora_B.weight")),
    ])
    expected = {
        "grid": torch.tensor([HEIGHT, WIDTH], dtype=torch.float32),
        "text": text[0],
        "embedded": embedded,
        "latents": latents[0].flatten(1).transpose(0, 1),
        "timestep": torch.tensor([sigma]),
        "velocity": result,
        "skipped_velocity": skipped,
        "lora_velocity": lora_result,
    }
    for steps, shift in SCHEDULES:
        scheduler = official.MOSSImageScheduler(shift=shift)
        scheduler.set_timesteps(steps)
        expected[f"schedule_{steps}_{int(shift)}"] = torch.tensor(scheduler.levels + [0.0])

    out = os.path.join(TINY_DIR, "moss_image")
    os.makedirs(out, exist_ok=True)
    weights = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    weights["rope.bands"] = model.rope.bands.float().contiguous()
    save_file(weights, os.path.join(out, "model.safetensors"))
    save_file({k: v.contiguous() for k, v in fused(weights).items()},
              os.path.join(out, "model_fused.safetensors"))
    save_file({k: v.float().contiguous() for k, v in expected.items()},
              os.path.join(out, "expected.safetensors"))
    save_file({k: v.to(torch.bfloat16).contiguous() if v.ndim else v for k, v in lora.items()},
              os.path.join(out, "lora.safetensors"))
    for k, v in expected.items():
        print(f"  moss_image/{k}: {tuple(v.shape)}  |max| {v.abs().max().item():.4f}")


if __name__ == "__main__":
    moss_image()
