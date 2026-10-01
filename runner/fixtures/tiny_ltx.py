"""Golden tiny LTX 2.5 pieces beyond text to video (specs/42, step 14): LoRAs on
the transformer and its text connectors, a conditioned transformer step
(first-frame tokens held, keyframe tokens appended) and the conv VAE's
encoder, each built by diffusers from its config shrunk down, with seeded
random weights, under the official single files' names. Run as
`tiny_diffusion.py` says:

    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        --with torch --with diffusers==0.40.0 --with transformers==5.17.0 --with safetensors \\
        python runner/fixtures/tiny_ltx.py [family ...]

Outputs are committed under runner/test/resources/fixtures/tiny/ltx_*.
"""

import re
import sys

import torch
from safetensors.torch import load_file, save_file

from tiny_diffusion import ltx_transformer_original_name, out

torch.manual_seed(1)

FRAMES, HEIGHT, WIDTH, AUDIO_FRAMES, FPS = 2, 3, 4, 9, 16.0


def randomize(model):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if re.search(r"norm_[qk]\.weight$", name):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            elif "scale_shift_table" in name:
                parameter.copy_(0.2 * torch.randn_like(parameter))
            elif "registers" in name:
                parameter.copy_(torch.randn_like(parameter))
            else:
                fan_in = parameter.shape[-1] if parameter.dim() > 1 else 16
                parameter.copy_(torch.randn_like(parameter) / fan_in ** 0.5)
    return model


def tiny_transformer():
    """`tiny_diffusion.ltx_transformer`'s configuration: 2 blocks, video 2 heads
    of 128 over 32 latent channels, audio 2 heads of 64."""
    from diffusers import LTX2VideoTransformer3DModel
    return randomize(LTX2VideoTransformer3DModel(
        in_channels=32, out_channels=32, num_attention_heads=2, attention_head_dim=128, cross_attention_dim=256,
        gated_attn=True, cross_attn_mod=True, audio_in_channels=32, audio_out_channels=32,
        audio_num_attention_heads=2, audio_attention_head_dim=64, audio_cross_attention_dim=128,
        audio_gated_attn=True, audio_cross_attn_mod=True, num_layers=2, caption_channels=32,
        rope_type="split", use_prompt_embeddings=False, ff_bias=False, audio_ff_bias=True,
        use_prompt_adaln_single=True, use_keyframes_abs_pos_embedding=True,
    ).eval().float())


def tiny_connectors():
    """`tiny_diffusion.ltx_connectors`' configuration, its widths the tiny
    transformer's (video 256, audio 128)."""
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    model = LTX2TextConnectors(
        caption_channels=32, text_proj_in_factor=3, video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=128, video_connector_num_layers=2,
        video_connector_num_learnable_registers=8, video_gated_attn=True, audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=64, audio_connector_num_layers=2,
        audio_connector_num_learnable_registers=8, audio_gated_attn=True, rope_type="split",
        per_modality_projections=True, video_hidden_dim=256, audio_hidden_dim=128, proj_bias=True,
    ).eval().float()
    randomize(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
    return model


def connectors_original_name(name):
    for pattern, replacement in [
        (r"^video_text_proj_in\.", "text_embedding_projection.video_aggregate_embed."),
        (r"^audio_text_proj_in\.", "text_embedding_projection.audio_aggregate_embed."),
        (r"^(video|audio)_connector\.transformer_blocks\.", r"\1_embeddings_connector.transformer_1d_blocks."),
        (r"^(video|audio)_connector\.", r"\1_embeddings_connector."),
        (r"\.norm_q\.", ".q_norm."), (r"\.norm_k\.", ".k_norm."),
    ]:
        name = re.sub(pattern, replacement, name)
    return name


def transformer_inputs():
    return (torch.randn(1, FRAMES * HEIGHT * WIDTH, 32), torch.randn(1, AUDIO_FRAMES, 32),
            torch.randn(1, 16, 256), torch.randn(1, 16, 128), torch.tensor([640.0]))


def run_transformer(model, video, audio, text, audio_text, t):
    with torch.no_grad():
        return model(video, audio, text, audio_text, timestep=t, audio_timestep=t, sigma=t,
                     num_frames=FRAMES, height=HEIGHT, width=WIDTH, fps=FPS, audio_num_frames=AUDIO_FRAMES,
                     use_cross_timestep=True, return_dict=False)


def run_connectors(model, hidden, mask):
    with torch.no_grad():
        video, audio, _ = model(hidden, mask, padding_side="left")
    return video, audio


# (the module the update covers, in diffusers' names; the LoRA file's target; rank; alpha; its naming)
LORA_TARGETS = [
    ("transformer", "transformer_blocks.0.attn1.to_q", "diffusion_model.transformer_blocks.0.attn1.to_q", 4, 2.0, "AB"),
    ("transformer", "transformer_blocks.1.ff.net.0.proj", "diffusion_model.transformer_blocks.1.ff.net.0.proj", 4,
     None, "down_up"),
    ("transformer", "transformer_blocks.0.audio_to_video_attn.to_k",
     "diffusion_model.transformer_blocks.0.audio_to_video_attn.to_k", 2, None, "AB"),
    ("transformer", "transformer_blocks.1.video_to_audio_attn.to_out.0",
     "diffusion_model.transformer_blocks.1.video_to_audio_attn.to_out.0", 2, 1.0, "AB"),
    ("transformer", "transformer_blocks.0.attn2.to_gate_logits",
     "diffusion_model.transformer_blocks.0.attn2.to_gate_logits", 2, None, "AB"),
    ("transformer", "transformer_blocks.1.audio_ff.net.2", "diffusion_model.transformer_blocks.1.audio_ff.net.2", 2,
     None, "AB"),
    ("transformer", "time_embed.linear", "diffusion_model.adaln_single.linear", 4, None, "AB"),
    ("transformer", "audio_proj_out", "diffusion_model.audio_proj_out", 2, None, "AB"),
    ("transformer", "proj_in", "transformer.proj_in", 4, None, "AB"),
    ("transformer", "audio_time_embed.emb.timestep_embedder.linear_1",
     "transformer.audio_time_embed.emb.timestep_embedder.linear_1", 2, None, "AB"),
    ("transformer", "av_cross_attn_audio_v2a_gate.linear", "transformer.av_cross_attn_audio_v2a_gate.linear", 2,
     None, "AB"),
    ("connectors", "video_connector.transformer_blocks.0.attn1.to_v",
     "diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_v", 4, None, "AB"),
    ("connectors", "audio_connector.transformer_blocks.1.ff.net.2",
     "connectors.audio_connector.transformer_blocks.1.ff.net.2", 2, None, "AB"),
]
LORA_MULTIPLIER = 0.8


def ltx_lora():
    """LoRAs on the tiny transformer and text connectors, in one file as the
    official transformer holds both: the published files' namings (the
    original names under `diffusion_model.`, `lora_A`/`lora_B` or
    `lora_down`/`lora_up`, an `.alpha` on some) and diffusers' (`transformer.`,
    `connectors.`), on attention, MLP, AdaLN, gate and projection linears. The
    outputs without the LoRA, and with it merged into the weights at 0.8."""
    transformer, connectors = tiny_transformer(), tiny_connectors()
    video, audio, text, audio_text, t = transformer_inputs()
    hidden = torch.randn(1, 16, 32, 3)
    mask = torch.tensor([[0] * 6 + [1] * 10])
    video_out, audio_out = run_transformer(transformer, video, audio, text, audio_text, t)
    text_video, text_audio = run_connectors(connectors, hidden, mask)
    # copies: the merge below changes the modules' weights in place
    tensors = {ltx_transformer_original_name(n): v.detach().float().clone()
               for n, v in transformer.state_dict().items()}
    tensors.update({connectors_original_name(n): v.detach().float().clone()
                    for n, v in connectors.state_dict().items()})
    lora = {}
    modules = {"transformer": transformer, "connectors": connectors}
    with torch.no_grad():
        for owner, module_name, target, rank, alpha, naming in LORA_TARGETS:
            weight = modules[owner].get_submodule(module_name).weight
            down = torch.randn(rank, weight.shape[1]) / weight.shape[1] ** 0.5
            up = 0.3 * torch.randn(weight.shape[0], rank) / rank ** 0.5
            # BF16 as the runner keeps the updates
            down, up = down.bfloat16().float(), up.bfloat16().float()
            scale = (alpha / rank if alpha is not None else 1.0) * LORA_MULTIPLIER
            weight.add_(scale * up @ down)
            lora[target + (".lora_A.weight" if naming == "AB" else ".lora_down.weight")] = down.contiguous()
            lora[target + (".lora_B.weight" if naming == "AB" else ".lora_up.weight")] = up.contiguous()
            if alpha is not None:
                lora[target + ".alpha"] = torch.tensor(alpha)
    lora_video, lora_audio = run_transformer(transformer, video, audio, text, audio_text, t)
    lora_text_video, lora_text_audio = run_connectors(connectors, hidden, mask)
    folder = out / "ltx_lora"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    save_file(lora, folder / "lora.safetensors", metadata={"format": "pt"})
    states = hidden[0, 6:].permute(2, 0, 1).reshape(3 * 10, 32)
    save_file({"video": video[0].contiguous(), "audio": audio[0].contiguous(), "text": text[0].contiguous(),
               "audio_text": audio_text[0].contiguous(), "timestep": t, "states": states.contiguous(),
               "video_out": video_out[0].contiguous(), "audio_out": audio_out[0].contiguous(),
               "lora_video": lora_video[0].contiguous(), "lora_audio": lora_audio[0].contiguous(),
               "text_video": text_video[0].contiguous(), "text_audio": text_audio[0].contiguous(),
               "lora_text_video": lora_text_video[0].contiguous(),
               "lora_text_audio": lora_text_audio[0].contiguous()},
              folder / "expected.safetensors")
    print(f"ltx_lora: {len(tensors)} tensors, {len(lora)} LoRA tensors, "
          f"video change {(lora_video - video_out).abs().max():.3f} of {video_out.abs().max():.3f}, "
          f"text change {(lora_text_video - text_video).abs().max():.3f}")


def ltx_condition():
    """One conditioned step of `ltx_transformer`'s tiny transformer (diffusers' condition
    blocks): 3 latent frames of 3 × 4, the first frame's tokens held (timestep
    0, `conditioning_mask` 1), then two keyframes appended with their own
    positions (`_prepare_keyframe_coords`), held too: a still at pixel frame 13
    and a clip of 2 latent frames from pixel frame 9."""
    from diffusers.modular_pipelines.ltx2.before_denoise import _prepare_keyframe_coords
    model = tiny_transformer()
    # `ltx_transformer`'s weights: the runner reads that fixture's file
    stored = load_file(out / "ltx_transformer" / "model.safetensors")
    model.load_state_dict({n: stored[ltx_transformer_original_name(n)] for n in model.state_dict()})
    frames, height, width = 3, HEIGHT, WIDTH
    per_frame = height * width
    keyframes = [(13, 1, True), (9, 2, False)]  # (pixel frame, latent frames, one pixel frame)
    appended = sum(count for _, count, _ in keyframes) * per_frame
    tokens = frames * per_frame + appended
    video = torch.randn(1, tokens, 32)
    audio = torch.randn(1, AUDIO_FRAMES, 32)
    text, audio_text = torch.randn(1, 16, 256), torch.randn(1, 16, 128)
    t = torch.tensor([640.0])
    mask = torch.zeros(1, tokens)
    mask[:, :per_frame] = 1
    mask[:, frames * per_frame:] = 1
    coords = model.rope.prepare_video_coords(1, frames, height, width, "cpu", fps=FPS)
    extra = [_prepare_keyframe_coords(count, height, width, index, 1 if single else 8 * count + 1, FPS, 1, 1,
                                      (8, 32, 32), "cpu")
             for index, count, single in keyframes]
    video_coords = torch.cat([coords] + extra, dim=2)
    with torch.no_grad():
        video_out, audio_out = model(video, audio, text, audio_text, timestep=t * (1 - mask), audio_timestep=t,
                                     sigma=t, num_frames=frames, height=height, width=width, fps=FPS,
                                     audio_num_frames=AUDIO_FRAMES, video_coords=video_coords,
                                     use_cross_timestep=True, return_dict=False)
    folder = out / "ltx_condition"
    folder.mkdir(parents=True, exist_ok=True)
    save_file({"video": video[0].contiguous(), "audio": audio[0].contiguous(), "text": text[0].contiguous(),
               "audio_text": audio_text[0].contiguous(), "timestep": t,
               "video_out": video_out[0].contiguous(), "audio_out": audio_out[0].contiguous()},
              folder / "expected.safetensors")
    print(f"ltx_condition: {tokens} video tokens ({appended} appended), video |max| {video_out.abs().max():.3f}")


def ltx_video_encoder():
    """LTX 2.5's conv VAE encoder, shrunk: diffusers' own LTX 2 blocks
    (`LTX2VideoCausalConv3d`, `LTX2VideoResnetBlock3d`,
    `LTX2VideoDownsampler3d`, `PerChannelRMSNorm`) in the official file's order
    (the config's `encoder_blocks`: residuals, ×2 in space, residuals, ×2 in
    time, residuals, ×2 everywhere doubling the channels, residuals, ×2
    everywhere, residuals), causal, the encoder's own patchify, the mode
    normalized by the statistics; beside `ltx_video_vae`'s decoder, whose
    statistics it shares. 9 frames of 64 × 64 into 2 latent frames of 2 × 2,
    and 1 frame into 1."""
    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import (
        LTX2VideoCausalConv3d, LTX2VideoDownsampler3d, LTX2VideoResnetBlock3d, PerChannelRMSNorm)
    decoder = load_file(out / "ltx_video_vae" / "model.safetensors")
    mean = decoder["per_channel_statistics.mean-of-means"]
    std = decoder["per_channel_statistics.std-of-means"]
    latent = mean.numel()
    stages = [("res", 16, 2), ("down", 16, 32, (1, 2, 2)), ("res", 32, 1), ("down", 32, 64, (2, 1, 1)),
              ("res", 64, 1), ("down", 64, 128, (2, 2, 2)), ("res", 128, 1), ("down", 128, 128, (2, 2, 2)),
              ("res", 128, 1)]
    conv_in = LTX2VideoCausalConv3d(48, 16, 3)
    blocks = []
    for stage in stages:
        if stage[0] == "res":
            blocks.append(torch.nn.ModuleList([LTX2VideoResnetBlock3d(stage[1], stage[1])
                                               for _ in range(stage[2])]))
        else:
            _, inputs, outputs, stride = stage
            blocks.append(LTX2VideoDownsampler3d(inputs, outputs, stride=stride))
    conv_out = LTX2VideoCausalConv3d(128, latent + 1, 3)
    modules = torch.nn.ModuleList([conv_in, *blocks, conv_out]).eval().float()
    with torch.no_grad():
        for parameter in modules.parameters():
            fan_in = parameter[0].numel() if parameter.dim() > 1 else 16
            parameter.copy_(torch.randn_like(parameter) / fan_in ** 0.5)
    norm = PerChannelRMSNorm()

    def encode(pixels):  # [1, 3, F, H, W]
        b, c, f, h, w = pixels.shape
        x = pixels.reshape(b, c, f, 1, h // 4, 4, w // 4, 4).permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        with torch.no_grad():
            x = conv_in(x, causal=True)
            for block in blocks:
                if isinstance(block, torch.nn.ModuleList):
                    for resnet in block:
                        x = resnet(x, causal=True)
                else:
                    x = block(x, causal=True)
            x = conv_out(torch.nn.functional.silu(norm(x)), causal=True)
        means = x[:, :latent]
        return (means - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)

    clip = torch.rand(1, 3, 9, 64, 64) * 2 - 1
    still = torch.rand(1, 3, 1, 64, 64) * 2 - 1
    clip_latents, still_latents = encode(clip), encode(still)
    tensors = dict(decoder)
    tensors.update({"encoder.conv_in.conv.weight": conv_in.conv.weight, "encoder.conv_in.conv.bias": conv_in.conv.bias,
                    "encoder.conv_out.conv.weight": conv_out.conv.weight,
                    "encoder.conv_out.conv.bias": conv_out.conv.bias})
    for i, block in enumerate(blocks):
        if isinstance(block, torch.nn.ModuleList):
            for j, resnet in enumerate(block):
                for part in ("conv1", "conv2"):
                    conv = getattr(resnet, part).conv
                    tensors[f"encoder.down_blocks.{i}.res_blocks.{j}.{part}.conv.weight"] = conv.weight
                    tensors[f"encoder.down_blocks.{i}.res_blocks.{j}.{part}.conv.bias"] = conv.bias
        else:
            tensors[f"encoder.down_blocks.{i}.conv.conv.weight"] = block.conv.conv.weight
            tensors[f"encoder.down_blocks.{i}.conv.conv.bias"] = block.conv.conv.bias
    folder = out / "ltx_video_encoder"
    folder.mkdir(parents=True, exist_ok=True)
    save_file({n: v.detach().float().contiguous() for n, v in tensors.items()}, folder / "model.safetensors")
    # channels-last: frames [F, H, W, 3], latents [F, h, w, C]
    save_file({"clip": clip[0].permute(1, 2, 3, 0).contiguous(), "still": still[0].permute(1, 2, 3, 0).contiguous(),
               "clip_latents": clip_latents[0].permute(1, 2, 3, 0).contiguous(),
               "still_latents": still_latents[0].permute(1, 2, 3, 0).contiguous()},
              folder / "expected.safetensors")
    print(f"ltx_video_encoder: clip latents {tuple(clip_latents.shape)} |max| {clip_latents.abs().max():.3f}, "
          f"still {tuple(still_latents.shape)}")


families = {"ltx_lora": ltx_lora, "ltx_condition": ltx_condition, "ltx_video_encoder": ltx_video_encoder}
if __name__ == "__main__":
    for family in sys.argv[1:] or families:
        families[family]()
