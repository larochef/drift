"""Golden tiny MiniMax H3 cases beyond text to video (specs/42, step 14): LoRAs
on the transformer (full and pruned), torch's CPU `randn`, the video and audio
VAEs' encoders, the fl2va keyframe layout and the guides' layout. Each built by
diffusers 0.40 (ComfyUI where diffusers has nothing) from a shrunk config with
seeded random weights, written under the names the released files use.

    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        --with torch --with diffusers==0.40.0 --with transformers==5.17.0 --with safetensors \\
        python runner/fixtures/tiny_minimax_h3.py [case ...]

Outputs are committed under runner/test/resources/fixtures/tiny/minimax_h3_*.
"""

import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_diffusion import minimax_h3_original, out, swap_halves  # noqa: E402


def tiny_transformer(seed):
    """The tiny transformer of `tiny_diffusion.minimax_h3`, weights from `seed`."""
    from diffusers import MiniMaxH3Transformer3DModel
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=64, hidden_size=64, num_layers=2, num_refiner_layers=1,
        ffn_dim=128, in_channels=8, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=32, freq_dim=32,
        time_embed_hidden_dim=64, time_embed_dim=32, rope_freq_dim=8, rope_theta=10000.0,
    ).eval().float()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight") and parameter.dim() == 1:
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            else:
                fan_in = parameter.shape[-1] if parameter.dim() > 1 else 16
                parameter.copy_(torch.randn_like(parameter) / fan_in ** 0.5)
    return model


def run(model, text_tags, frames, latent_h, latent_w, audio_latents, video_rows, audio_rows, text,
        row_timesteps=None, video_t=None, audio_t=None, anchors=()):
    """The transformer over diffusers' own `[text | keyframes | audio | video]`
    layout; `row_timesteps` (else the video's for all but the audio's)."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    positions, token_tags, video_indices, audio_indices, text_indices, conditions, _ = \
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(text_tags, frames, latent_h, latent_w, audio_latents,
                                                          (1, 2, 2), 2, 2, 0, anchors)
    if row_timesteps is None:
        row_timesteps = torch.full((positions.shape[0],), video_t)
        row_timesteps[audio_indices] = audio_t
    timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
    with torch.no_grad():
        video, audio = model(video_rows[None], audio_rows[None], text[None], timesteps, timestep_indices, token_tags,
                             positions.float(), video_indices, audio_indices, text_indices, return_dict=False)
    return positions, token_tags, video[0], audio[0]


# ---- LoRAs ---------------------------------------------------------------------------------------------------

def lora():
    """A LoRA over the tiny transformer in the three namings the published H3
    files use (ai-toolkit's `diffusion_model.` + lora_A/B, musubi-tuner's
    flattened `lora_unet_` + lora_down/up + alpha, diffusers' `transformer.`),
    its targets the fused qkv and fc1, the MLP's output, a refiner block, the
    AdaLN projections, the text projection and the time MLP; the reference
    merges `W += multiplier × scale × up · down` into the weights. Run on the
    full file and on a pruned one (its time MLP replaced by the table of its
    post-SiLU outputs on 65 points), where the time MLP's update has no weight
    and is left out."""
    model = tiny_transformer(1)
    torch.manual_seed(11)
    multiplier = 0.8
    hidden, inner, intermediate = 64, 128, 128

    def pair(rank, outputs, inputs):
        return torch.randn(rank, inputs) / inputs ** 0.5, 0.5 * torch.randn(outputs, rank) / rank ** 0.5

    # (file names, down, up, scale, diffusers' parameter and how the update lands on it)
    entries = []

    def add(names, rank, outputs, inputs, target, alpha=None, rows=None):
        down, up = pair(rank, outputs, inputs)
        entries.append((names, down, up, alpha, target, rows))

    add(("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"),
        4, 3 * inner, hidden, "qkv:transformer_blocks.0")
    add(("diffusion_model.blocks.1.mlp.fc1.lora_A.weight", "diffusion_model.blocks.1.mlp.fc1.lora_B.weight"),
        4, 2 * intermediate, hidden, "fc1:transformer_blocks.1")
    add(("diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_A.weight",
         "diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_B.weight"),
        3, hidden, inner, "token_refiner.refiner_blocks.0.attn.to_out.0.weight")
    add(("diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight",
         "diffusion_model.blocks.0.adaln_proj.linear.lora_B.weight"),
        2, 6 * 3 * hidden, 32, "transformer_blocks.0.adaln_proj.linear.weight")
    add(("lora_unet_blocks_1_mlp_fc2.lora_down.weight", "lora_unet_blocks_1_mlp_fc2.lora_up.weight"),
        4, hidden, intermediate, "transformer_blocks.1.ff.net.2.weight", alpha=2.0)
    add(("lora_unet_condition_proj.lora_down.weight", "lora_unet_condition_proj.lora_up.weight"),
        2, hidden, 32, "context_embedder.weight", alpha=1.0)
    add(("lora_unet_final_layer_adaln_proj_linear.lora_down.weight",
         "lora_unet_final_layer_adaln_proj_linear.lora_up.weight"),
        2, 2 * hidden, 32, "norm_out.linear.weight", alpha=4.0)
    add(("transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight",
         "transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"),
        4, 2 * intermediate, hidden, "transformer_blocks.0.ff.net.0.proj.weight")
    add(("transformer.transformer_blocks.1.attn.to_k.lora_A.weight",
         "transformer.transformer_blocks.1.attn.to_k.lora_B.weight"),
        4, inner, hidden, "transformer_blocks.1.attn.to_k.weight")
    add(("diffusion_model.time_embedder.proj_in.lora_A.weight", "diffusion_model.time_embedder.proj_in.lora_B.weight"),
        2, 64, 32, "time_embedder.linear_1.weight")

    tensors = {}
    for (down_name, up_name), down, up, alpha, _, _ in entries:
        tensors[down_name], tensors[up_name] = down, up
        if alpha is not None:
            tensors[down_name.replace(".lora_down.weight", ".alpha")] = torch.tensor(alpha)

    inputs = dict(text_tags=torch.ones(5, dtype=torch.long), frames=3, latent_h=4, latent_w=8, audio_latents=3)
    torch.manual_seed(12)
    video_rows, audio_rows, text = torch.randn(3 * 2 * 4, 32), torch.randn(6, 32), torch.randn(5, 32)
    grid = 65
    video_t, audio_t = 20 / (grid - 1), 35 / (grid - 1)

    # the pruned file: the time MLP's post-SiLU outputs on the grid
    with torch.no_grad():
        points = torch.linspace(0, 1, grid)
        table = torch.nn.functional.silu(model.time_embedder(model.time_proj(points)))
    original = minimax_h3_original(model)
    pruned = {n: v for n, v in original.items() if not n.startswith("time_embedder.")}
    pruned["adaln_t_table"] = table.contiguous()

    def merged(include_time):
        state = {n: v.clone() for n, v in model.state_dict().items()}
        for _, down, up, alpha, target, _ in entries:
            if target.startswith("time_embedder") and not include_time:
                continue
            scale = multiplier * (alpha / down.shape[0] if alpha is not None else 1.0)
            delta = scale * up @ down
            if target.startswith("qkv:"):
                block = target[4:]
                for i, p in enumerate("qkv"):
                    state[f"{block}.attn.to_{p}.weight"] += delta[i * inner:(i + 1) * inner]
            elif target.startswith("fc1:"):
                # the file's [gate; value] is diffusers' [value; gate]
                state[f"{target[4:]}.ff.net.0.proj.weight"] += swap_halves(delta)
            else:
                state[target] += delta
        copy = tiny_transformer(1)
        copy.load_state_dict(state)
        return copy

    expected = {}
    for name, include_time in (("full", True), ("pruned", False)):
        _, _, video, audio = run(merged(include_time), **inputs, video_rows=video_rows, audio_rows=audio_rows,
                                 text=text, video_t=video_t, audio_t=audio_t)
        expected[f"{name}_video"], expected[f"{name}_audio"] = video.contiguous(), audio.contiguous()
    folder = out / "minimax_h3_lora"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(original, folder / "model.safetensors")
    save_file(pruned, folder / "pruned.safetensors")
    save_file({n: v.contiguous() for n, v in tensors.items()}, folder / "lora.safetensors")
    save_file({**expected, "video_rows": video_rows, "audio_rows": audio_rows, "text": text,
               "timesteps": torch.tensor([video_t, audio_t]), "multiplier": torch.tensor([multiplier])},
              folder / "expected.safetensors")
    print(f"minimax_h3_lora: {len(tensors)} tensors, video |max| {expected['full_video'].abs().max():.3f}")


# ---- torch's CPU randn -----------------------------------------------------------------------------------------

def randn():
    """`torch.randn` on the CPU generator: seeds 0, 42 and 12345 at sizes 5,
    16, 100 and 1000 (the scalar path under 16, the vectorized one and its
    tail), each draw after the last from the same generator."""
    draws = {}
    for seed in (0, 42, 12345):
        generator = torch.Generator().manual_seed(seed)
        draws[f"seed_{seed}"] = torch.cat([torch.randn(size, generator=generator) for size in (5, 16, 100, 1000)])
    folder = out / "minimax_h3_randn"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(draws, folder / "expected.safetensors")
    print(f"minimax_h3_randn: {sum(v.numel() for v in draws.values())} values")


# ---- the video VAE's encoder ------------------------------------------------------------------------------------

def video_encoder_original_name(name):
    """diffusers' encoder names → the released file's (ComfyUI's
    `EncoderFCN3D`)."""
    name = re.sub(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.down.\1.block.\2.", name)
    name = re.sub(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", name)
    return name.replace(".conv_shortcut.", ".nin_shortcut.")


def video_encoder():
    """MiniMax H3's video VAE encoder shrunk to 32 then 64 channels a level
    (whole blocks of 32, as the kernels' linears take; the released six
    levels, time halved at the second and third), groups of 8:
    one 96 × 112 frame in 64-pixel tiles overlapping by 16 (four tiles,
    blended), as its posterior's mean and as diffusers' keyframe recipe (a draw
    under seed 42, rounded to F16); and a 22-frame clip of 32 × 48 (two
    chunks of 17, the last 3 latent frames dropped: 7), as its mean. Pixels in
    [−1, 1]; latents normalized."""
    from diffusers import AutoencoderKLMiniMaxH3
    torch.manual_seed(21)
    model = AutoencoderKLMiniMaxH3(
        latent_channels=8, block_out_channels=(32, 32, 64, 64, 64, 64), norm_num_groups=8, decoder_num_layers=1,
        decoder_num_attention_heads=2, decoder_attention_head_dim=64, decoder_num_register_tokens=4,
        latents_mean=tuple(0.1 * i for i in range(8)), latents_std=tuple(0.5 + 0.1 * i for i in range(8)),
    ).eval().float()
    model.enable_tiling(64, 64, 16, 16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if re.search(r"norm\d?\.weight$|norm_out\.weight$", name):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            else:
                fan_in = parameter[0].numel() if parameter.dim() > 1 else 16
                parameter.copy_(torch.randn_like(parameter) / fan_in ** 0.5)
    mean = torch.tensor(model.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(model.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

    def encoded(pixels, sample):
        # `encode_vae_condition` on pixels in [−1, 1]
        with torch.no_grad():
            posterior = model.encode(((pixels + 1) / 2 - pixel_mean) / pixel_std, return_dict=False)[0]
        if sample:
            latents = posterior.sample(generator=torch.Generator().manual_seed(42)).to(torch.float16).float()
        else:
            latents = posterior.mode()
        return (latents - mean) / std

    frame = torch.rand(1, 3, 1, 96, 112) * 2 - 1
    clip = torch.rand(1, 3, 22, 32, 48) * 2 - 1

    def rows(z):  # [1, C, T, h, w] → [T, h, w, C]
        return z[0].permute(1, 2, 3, 0).contiguous()

    def pixels(x):  # [1, 3, T, H, W] → [T, H, W, 3]
        return x[0].permute(1, 2, 3, 0).contiguous()

    tensors = {video_encoder_original_name(n): v.detach().float().contiguous() for n, v in model.state_dict().items()
               if n.startswith("encoder.") or n.startswith("quant_conv.")}
    tensors["latents_mean"] = torch.tensor(model.config.latents_mean)
    tensors["latents_std"] = torch.tensor(model.config.latents_std)
    folder = out / "minimax_h3_video_encoder"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    expected = {"frame": pixels(frame), "clip": pixels(clip), "frame_mean": rows(encoded(frame, False)),
                "frame_sample": rows(encoded(frame, True)), "clip_mean": rows(encoded(clip, False))}
    save_file(expected, folder / "expected.safetensors")
    print(f"minimax_h3_video_encoder: {len(tensors)} tensors, frame {tuple(expected['frame_mean'].shape)}, "
          f"clip {tuple(expected['clip_mean'].shape)}")
    print("  names:", sorted({re.sub(r"\.\d+\.", ".N.", n) for n in tensors}))


# ---- fl2va: keyframes ---------------------------------------------------------------------------------------------

def fl2va():
    """A first and a last keyframe (diffusers' `MiniMaxH3FL2VA*` blocks): the
    presentation's tags (each keyframe's label as text, its vision block as
    video), the `[text | keyframes | audio | video]` layout of 7 latent frames
    (22 pixel frames) of 4 × 8 latents and 4 audio latents a channel, the draws
    of seed 42 in the request's order (each keyframe's conditioning noise,
    mixed at t = 0.999; the video noise as a latent tensor; the audio rows),
    then one step of the tiny transformer with the keyframe rows held at
    max(t, 0.999)."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents)
    model = tiny_transformer(2)
    torch.manual_seed(31)
    channels, frames, latent_h, latent_w, audio_latents = 8, 7, 4, 8, 4
    tags = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    keyframes = [torch.randn(1, channels, 1, latent_h, latent_w) for _ in range(2)]
    text = torch.randn(tags.shape[0], 32)
    aug = 0.999
    generator = torch.Generator().manual_seed(42)
    condition_rows = []
    for condition in keyframes:
        noise = torch.randn(condition.shape, generator=generator, dtype=torch.float32)
        condition_rows.append(patchify_video_latents(aug * condition + (1 - aug) * noise, (1, 2, 2)))
    video_noise = torch.randn((1, channels, frames, latent_h, latent_w), generator=generator, dtype=torch.float32)
    video_rows = patchify_video_latents(video_noise, (1, 2, 2))
    audio_rows = torch.randn((2 * audio_latents, 32), generator=generator, dtype=torch.float32)
    positions, token_tags, video_indices, audio_indices, text_indices, conditions, _ = \
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(tags, frames, latent_h, latent_w, audio_latents,
                                                          (1, 2, 2), 2, 2, 0, ("first", "last"))
    video_t, audio_t = 0.3, 0.55
    timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, conditions, 0, tags.shape[0], video_t, audio_t, max(video_t, aug), 1.0)
    all_video = torch.cat(condition_rows + [video_rows])
    with torch.no_grad():
        video, audio = model(all_video[None], audio_rows[None], text[None], timesteps, timestep_indices, token_tags,
                             positions.float(), video_indices, audio_indices, text_indices, return_dict=False)
    folder = out / "minimax_h3_fl2va"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(minimax_h3_original(model), folder / "model.safetensors")
    save_file({"tags": tags.int(), "text": text, "positions": positions.float().contiguous(),
               "keyframes": torch.cat([k[0, :, 0].permute(1, 2, 0).reshape(1, -1) for k in keyframes]),
               "condition_rows": torch.cat(condition_rows).contiguous(), "video_rows": video_rows.contiguous(),
               "audio_rows": audio_rows.contiguous(), "timesteps": torch.tensor([video_t, audio_t]),
               "video": video[0, conditions:].contiguous(), "audio": audio[0].contiguous()},
              folder / "expected.safetensors")
    print(f"minimax_h3_fl2va: {positions.shape[0]} rows ({conditions} keyframe rows), "
          f"video |max| {video.abs().max():.3f}")


# ---- the audio VAE's encoder ------------------------------------------------------------------------------------

def audio_encoder():
    """MiniMax H3's audio encoder (diffusers' `AutoencoderKLMiniMaxH3Audio.encode`,
    its posterior's mean normalized as ComfyUI's `encode` gives it): DAC's
    encoder shrunk to 4 → 16 channels over strides 2 and 5 (10 samples a
    latent), `pre_block` over 32 channels in 2 causal heads of 16, pooled to 8
    latent channels (each 2 values averaged); Snake's α off its ones, weight
    norms folded as the released file has them. Two channels of 95 samples
    (padded to 100: 10 latents), encoded one after the other."""
    from diffusers import AutoencoderKLMiniMaxH3Audio
    torch.manual_seed(41)
    channels = 8
    mean, std = (0.1 * torch.randn(channels)).tolist(), (0.5 + torch.rand(channels)).tolist()
    model = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=4, encoder_rates=(2, 5), latent_dim=32, latent_channels=channels, num_attention_heads=2,
        decoder_dim=32, decoder_rates=(5, 2), decoder_kernel_sizes=(9, 4),
        latents_mean=mean, latents_std=std).eval().float()
    for module in model.modules():
        if hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("encoder.") and name.endswith(".alpha"):
                parameter.copy_(0.5 + torch.rand_like(parameter))
            elif name.startswith("pre_block.") and name.endswith("_bias"):
                parameter.copy_(0.2 * torch.randn_like(parameter))
    waveform = 0.5 * torch.randn(2, 1, 95)
    with torch.no_grad():
        latents = model.encode(waveform, return_dict=False)[0].mode()  # [2, C, 10]
    normalized = (latents - torch.tensor(mean).view(1, -1, 1)) / torch.tensor(std).view(1, -1, 1)
    tensors = {n: v for n, v in model.state_dict().items()
               if n.startswith(("encoder.", "pre_block.", "mean_proj.", "logs_proj."))}
    tensors["latents_mean"], tensors["latents_std"] = torch.tensor(mean), torch.tensor(std)
    folder = out / "minimax_h3_audio_encoder"
    folder.mkdir(parents=True, exist_ok=True)
    save_file({n: v.detach().float().contiguous() for n, v in tensors.items()}, folder / "model.safetensors")
    save_file({"waveform": waveform[:, 0].contiguous(),
               "latents": normalized.permute(0, 2, 1).reshape(-1, channels).contiguous()},  # channel-major rows
              folder / "expected.safetensors")
    print(f"minimax_h3_audio_encoder: {len(tensors)} tensors, latents {tuple(latents.shape)} "
          f"|max| {normalized.abs().max():.3f}")
    print("  names:", sorted({re.sub(r"\.\d+\.", ".N.", n) for n in tensors}))


# ---- the presentation: Qwen3-VL reading two keyframes ------------------------------------------------------------

def presentation():
    """MiniMax H3's text encoder as its GGUF holds it, shrunk: Qwen3-VL's
    language model under `model.` (no final norm read: the residual stream
    after the last layer) and its tower under `visual.`, no configuration. The
    released shapes where the runner reads them off the file: heads of 128
    turned by the interleaved mRoPE of 24, 20 and 20 pairs, a tower of 16
    heads. Two keyframes (64 × 96 and 32 × 64) each after a label, then the
    prompt, as the fl2va presentation lays them out."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from PIL import Image
    torch.manual_seed(51)
    text = dict(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                num_attention_heads=2, num_key_value_heads=1, head_dim=128, rms_norm_eps=1e-6,
                max_position_embeddings=512,
                rope_parameters={"rope_type": "default", "rope_theta": 5e6,
                                 "mrope_section": [24, 20, 20], "mrope_interleaved": True})
    vision = dict(depth=2, hidden_size=64, num_heads=16, intermediate_size=128, patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                  num_position_embeddings=16, deepstack_visual_indexes=[0, 1])
    config = Qwen3VLConfig(text_config=text, vision_config=vision, image_token_id=300, video_token_id=303,
                           vision_start_token_id=301, vision_end_token_id=302, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    model = Qwen3VLForConditionalGeneration(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            elif name.endswith("bias") or "pos_embed" in name:
                parameter.copy_(0.2 * torch.randn_like(parameter))
    tensors = {}
    for n, t in model.state_dict().items():
        if n.startswith("model.language_model.") and not n.startswith("model.language_model.norm."):
            tensors[n.replace("model.language_model.", "model.")] = t
        elif n.startswith("model.visual."):
            tensors[n.replace("model.visual.", "visual.")] = t
    folder = out / "minimax_h3_presentation"
    folder.mkdir(parents=True, exist_ok=True)
    save_file({n: t.detach().contiguous() for n, t in tensors.items()}, folder / "model.safetensors")

    generator = torch.Generator().manual_seed(3)
    images = [torch.randint(0, 256, (64, 96, 3), generator=generator, dtype=torch.uint8),
              torch.randint(0, 256, (32, 64, 3), generator=generator, dtype=torch.uint8)]
    processor = Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                         image_mean=[0.5] * 3, image_std=[0.5] * 3, do_resize=False)
    processed = processor(images=[Image.fromarray(i.numpy()) for i in images], return_tensors="pt")
    pixel_values, grid = processed["pixel_values"], processed["image_grid_thw"]
    labels = [torch.randint(0, 300, (3,), generator=generator).tolist() for _ in images]
    prompt = torch.randint(0, 300, (6,), generator=generator).tolist()
    ids = []
    for label, g in zip(labels, grid):
        ids += label + [301] + [300] * (int(g.prod()) // 4) + [302]
    ids += prompt
    ids = torch.tensor([ids])
    handle = model.model.language_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.no_grad():
            hidden = model(input_ids=ids, pixel_values=pixel_values, image_grid_thw=grid,
                           mm_token_type_ids=(ids == 300).int(), output_hidden_states=True).hidden_states[-1][0]
    finally:
        handle.remove()
    save_file({"ids": ids[0].to(torch.int32), "image_0": images[0].float().contiguous(),
               "image_1": images[1].float().contiguous(), "hidden": hidden.float().contiguous()},
              folder / "expected.safetensors")
    print(f"minimax_h3_presentation: {len(tensors)} tensors, {ids.shape[1]} tokens, hidden |max| "
          f"{hidden.abs().max():.3f}")


# ---- guides: ComfyUI's MiniMaxH3AddGuide ------------------------------------------------------------------------

COMFYUI_MODEL = Path("/opt/comfyui/comfy/ldm/minimax/model.py")


def comfyui_layout():
    """ComfyUI's `PackedLayout` and `_cond_video_rows` with the helpers they
    call, lifted from `comfy/ldm/minimax/model.py` as they stand (the module's
    imports pull ComfyUI's whole runtime)."""
    import ast
    import math
    source = COMFYUI_MODEL.read_text()
    tree = ast.parse(source)
    wanted = {"FRAME_PER_TOKEN", "FRAME_RESCALE", "VISUAL_COND_TIMESTEP", "AUDIO_COND_TIMESTEP", "patchify_video",
              "pack_audio", "_axis_from_sqrt_area", "_frame_grid", "_video_t_spans", "_video_t_grid", "_ref_t_span",
              "_audio_grid", "_video_grid", "PackedLayout"}
    kept = []
    for node in tree.body:
        names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else \
            [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        if set(names) & wanted:
            kept.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Model":
            method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "_cond_video_rows")
            kept.append(method)
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(COMFYUI_MODEL), "exec"), namespace)
    return namespace


def guides():
    """Guides as ComfyUI's `MiniMaxH3AddGuide` conditions H3 (its
    `PackedLayout`, `_cond_video_rows` and the row timesteps and tags of
    `MiniMaxH3Model._forward`, the tiny diffusers transformer for the maths):
    a 5-frame clip with its sound at frame 0 (2 latent frames, 5 audio
    latents), an image at frame 10, a sound alone at the last frame (3
    latents), over 7 latent frames of 4 × 8 and 37 audio latents. The guides'
    video rows mixed at 0.999 with the noise of a generator restarted at the
    seed (7) for each; their audio held clean at t = 1."""
    comfy = comfyui_layout()
    model = tiny_transformer(4)
    torch.manual_seed(61)
    channels, frames, latent_h, latent_w, audio_latents, text_tokens, seed = 8, 7, 4, 8, 37, 5, 7
    keyframes = [
        {"resolved_frame_index": 0, "latent": torch.randn(1, channels, 2, latent_h, latent_w),
         "audio_latent": torch.randn(1, 32, 2, 5)},
        {"resolved_frame_index": 10, "latent": torch.randn(1, channels, 1, latent_h, latent_w)},
        {"resolved_frame_index": 21, "audio_latent": torch.randn(1, 32, 2, 3)},
    ]
    layout = comfy["PackedLayout"](text_tokens, frames, latent_h, latent_w, audio_latents, keyframes=keyframes)
    payload = {"seed": seed, "cond_video_latents": [k["latent"] for k in keyframes if "latent" in k]}
    video_conditions = comfy["_cond_video_rows"](type("Model", (), {"patch_size": (1, 2, 2)})(), payload, "cpu")
    audio_conditions = torch.cat([comfy["pack_audio"](k["audio_latent"]) for k in keyframes if "audio_latent" in k])
    video_rows = torch.randn(frames * (latent_h // 2) * (latent_w // 2), 4 * channels)
    audio_rows = torch.randn(2 * audio_latents, 32)
    text = torch.randn(text_tokens, 32)
    video_t, audio_t = 0.3, 0.55
    seg_t = {"text": video_t, "video": video_t, "audio": audio_t, "cond": max(video_t, 0.999),
             "cond_audio": max(audio_t, 1.0)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "cond_audio": 2}
    row_timesteps = torch.empty(layout.seq_len)
    tags = torch.empty(layout.seq_len, dtype=torch.long)
    for a, b, kind in layout.segments:
        row_timesteps[a:b], tags[a:b] = seg_t[kind], seg_tag[kind]
    timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
    all_video = torch.cat([video_conditions, video_rows])
    all_audio = torch.cat([audio_conditions, audio_rows])
    with torch.no_grad():
        video, audio = model(all_video[None], all_audio[None], text[None], timesteps, timestep_indices, tags,
                             layout.position_ids.float(), layout.img_pos, layout.audio_pos,
                             torch.arange(text_tokens), return_dict=False)
    clean = [comfy["patchify_video"](k["latent"]) for k in keyframes if "latent" in k]
    folder = out / "minimax_h3_guides"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(minimax_h3_original(model), folder / "model.safetensors")
    save_file({"positions": layout.position_ids.float().contiguous(), "text": text,
               "clean_video_conditions": torch.cat(clean).contiguous(),
               "video_conditions": video_conditions.contiguous(), "audio_conditions": audio_conditions.contiguous(),
               "video_rows": video_rows, "audio_rows": audio_rows, "timesteps": torch.tensor([video_t, audio_t]),
               "video": video[0, video_conditions.shape[0]:].contiguous(),
               "audio": audio[0, audio_conditions.shape[0]:].contiguous()},
              folder / "expected.safetensors")
    print(f"minimax_h3_guides: {layout.seq_len} rows, segments {[(k, b - a) for a, b, k in layout.segments]}, "
          f"video |max| {video.abs().max():.3f}")


def resampling():
    """torchaudio's `functional.resample`, as ComfyUI brings a guide's sound to
    the audio VAE's 32 kHz: 1000 samples at 48 kHz and at 44.1 kHz. Needs
    `--with torchaudio`."""
    import torchaudio
    torch.manual_seed(71)
    waveform = torch.randn(1000)
    folder = out / "minimax_h3_resampling"
    folder.mkdir(parents=True, exist_ok=True)
    save_file({"waveform": waveform,
               "from_48000": torchaudio.functional.resample(waveform, 48000, 32000).contiguous(),
               "from_44100": torchaudio.functional.resample(waveform, 44100, 32000).contiguous()},
              folder / "expected.safetensors")
    print("minimax_h3_resampling: done")


cases = {"lora": lora, "randn": randn, "video_encoder": video_encoder, "fl2va": fl2va,
         "audio_encoder": audio_encoder, "presentation": presentation, "guides": guides, "resampling": resampling}

if __name__ == "__main__":
    for case in sys.argv[1:] or cases:
        cases[case]()
