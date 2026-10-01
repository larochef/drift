"""Golden tiny Wan 2.2 A14B inputs beside `tiny_diffusion.py`'s (the tiny I2V
transformer and the tiny Wan 2.1 VAE it writes, read back into diffusers):
LoRAs for the two experts in every naming the published files use, and the
I2V conditioning with an end image.

    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        --with torch --with diffusers==0.40.0 --with transformers==5.17.0 --with safetensors \\
        python runner/fixtures/tiny_wan.py [family ...]

Outputs are committed under runner/test/resources/fixtures/tiny/.
"""

import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from tiny_diffusion import out, wan_original_name, wan_video_vae_original_name

torch.manual_seed(0)


def tiny_wan():
    """`tiny_diffusion.wan`'s transformer, its weights read back."""
    from diffusers import WanTransformer3DModel
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=128, in_channels=36, out_channels=16,
        text_dim=32, freq_dim=32, ffn_dim=128, num_layers=2, cross_attn_norm=True, eps=1e-6,
    ).eval().float()
    stored = load_file(out / "wan" / "model.safetensors")
    model.load_state_dict({name: stored[wan_original_name(name)] for name in model.state_dict()})
    return model


def wan_lora():
    """Two LoRAs for the tiny Wan, as the published ones hold them, one per
    expert: the high-noise one in the original names (ComfyUI's
    `diffusion_model.`, `lora_A`/`lora_B`, one float alpha) and diffusers'
    (`transformer.`: `attn1.to_out.0`, `proj_out`, the condition embedder);
    the low-noise one in kohya's (`lora_unet_`, `lora_down`/`lora_up`, integer
    alphas as I64 scalars) of both namings, in BF16 as the Civitai files. The
    high-noise file is F32, as the turbo pair: the runner converts it to BF16,
    so the merged deltas are made of the BF16-rounded matrices. Each file's
    velocity with its deltas merged into diffusers' weights at its
    multiplier."""
    model = tiny_wan()
    golden = load_file(out / "wan" / "expected.safetensors")
    latents = golden["latents"].permute(3, 0, 1, 2).unsqueeze(0)
    text = golden["text"].unsqueeze(0)
    timestep = golden["timestep"]

    def run():
        with torch.no_grad():
            return model(latents, timestep, text, return_dict=False)[0][0].float()

    base = run()
    assert (base.permute(1, 2, 3, 0) - golden["velocity"]).abs().max() < 1e-5
    experts = {
        "high": (0.8, [  # (the diffusers weight, the file's name, rank, alpha)
            ("blocks.0.attn1.to_q", "diffusion_model.blocks.0.self_attn.q", 4, None),
            ("blocks.1.attn2.to_k", "diffusion_model.blocks.1.cross_attn.k", 4, 2.0),
            ("blocks.0.ffn.net.0.proj", "diffusion_model.blocks.0.ffn.0", 4, None),
            ("blocks.1.attn1.to_out.0", "transformer.blocks.1.attn1.to_out.0", 4, None),
            ("proj_out", "transformer.proj_out", 4, None),
            ("condition_embedder.time_proj", "transformer.condition_embedder.time_proj", 2, None),
            ("condition_embedder.text_embedder.linear_1", "transformer.condition_embedder.text_embedder.linear_1",
             2, None),
        ]),
        "low": (1.5, [
            ("blocks.0.attn2.to_v", "lora_unet_blocks_0_cross_attn_v", 4, 8),
            ("blocks.1.ffn.net.2", "lora_unet_blocks_1_ffn_2", 4, None),
            ("blocks.0.attn1.to_k", "lora_unet_blocks_0_attn1_to_k", 4, 2),
            ("blocks.1.attn2.to_out.0", "lora_unet_blocks_1_attn2_to_out_0", 4, None),
            ("condition_embedder.time_embedder.linear_1", "lora_unet_time_embedding_0", 2, None),
            ("proj_out", "lora_unet_head_head", 2, None),
        ]),
    }
    state = {name: value.clone() for name, value in model.state_dict().items()}
    expected = {}
    folder = out / "wan_lora"
    folder.mkdir(parents=True, exist_ok=True)
    for expert, (multiplier, targets) in experts.items():
        lora, merged = {}, {}
        for name, prefix, rank, alpha in targets:
            weight = state[name + ".weight"]
            in_features = weight.shape[1]
            down = (torch.randn(rank, in_features) / in_features ** 0.5).bfloat16().float()
            up = (torch.randn(weight.shape[0], rank) * 0.2).bfloat16().float()
            if expert == "low":
                down, up = down.bfloat16(), up.bfloat16()
            scale = multiplier * (alpha / rank if alpha is not None else 1.0)
            merged[name + ".weight"] = weight + scale * (up.float() @ down.float())
            kohya = prefix.startswith("lora_unet_")
            lora[prefix + (".lora_down.weight" if kohya else ".lora_A.weight")] = down.contiguous()
            lora[prefix + (".lora_up.weight" if kohya else ".lora_B.weight")] = up.contiguous()
            if alpha is not None:
                lora[prefix + ".alpha"] = torch.tensor(alpha)
        save_file(lora, folder / f"{expert}.safetensors", metadata={"format": "pt"})
        model.load_state_dict({**state, **merged})
        expected[f"{expert}_velocity"] = run().permute(1, 2, 3, 0).contiguous()
        expected[f"{expert}_multiplier"] = torch.tensor([multiplier])
        model.load_state_dict(state)
    save_file(expected, folder / "expected.safetensors")
    print(f"wan_lora: {', '.join(f'{k} {tuple(v.shape)}' for k, v in expected.items())}")


def wan_conditioning():
    """The I2V conditioning (mask of 4, then the latents of 16) of a 9-frame
    32 × 32 video through the tiny Wan 2.1 VAE: with a first image, with a
    first and a last (diffusers' `WanImageToVideoPipeline.prepare_latents`
    with `last_image`), and with a last alone (sd-cpp's `vid_gen`: the video's
    other frames zeros, the mask on the last frame alone; diffusers requires a
    first image)."""
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    vae = AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                           temperal_downsample=[False, True, True]).eval().float()
    stored = load_file(out / "wan_video_vae" / "model.safetensors")
    vae.load_state_dict({name: stored[wan_video_vae_original_name(name)] for name in vae.state_dict()})
    pipeline = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=None)
    frames, size = 9, 32
    generator = np.random.default_rng(0)
    first, last = (torch.from_numpy(generator.integers(0, 256, (size, size, 3)).astype(np.float32)) for _ in range(2))

    def pixels(image):  # [H, W, 3] of 0‥255 → [1, 3, H, W] in [−1, 1], as the runner's `Images.pixels`
        return (image / 127.5 - 1).permute(2, 0, 1).unsqueeze(0)

    def prepared(image, last_image):
        with torch.no_grad():
            _, condition = pipeline.prepare_latents(pixels(image), 1, 16, size, size, frames, torch.float32, "cpu",
                                                    None, last_image=last_image)
        return condition

    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    video = torch.zeros(1, 3, frames, size, size)
    video[:, :, -1] = pixels(last)[0]
    with torch.no_grad():
        latents = (vae.encode(video).latent_dist.mode() - mean) / std
    mask = torch.zeros(1, 4, latents.shape[2], size // 8, size // 8)
    mask[:, 3, -1] = 1
    conditions = {"first": prepared(first, None), "first_last": prepared(first, pixels(last)),
                  "last": torch.cat([mask, latents], dim=1)}
    folder = out / "wan_conditioning"
    folder.mkdir(parents=True, exist_ok=True)
    # the runner's layout: [T, h, w, 20]
    save_file({"first_image": first.contiguous(), "last_image": last.contiguous(),
               **{name: c[0].permute(1, 2, 3, 0).contiguous() for name, c in conditions.items()}},
              folder / "expected.safetensors")
    print(f"wan_conditioning: {', '.join(f'{k} {tuple(v.shape)}' for k, v in conditions.items())}, "
          f"last mask {conditions['first_last'][0, :4, -1, 0, 0].tolist()}")


families = {"wan_lora": wan_lora, "wan_conditioning": wan_conditioning}
if __name__ == "__main__":
    for family in sys.argv[1:] or families:
        families[family]()
