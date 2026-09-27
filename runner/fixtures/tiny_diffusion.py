"""Golden tiny diffusion models (specs/42, Testing, layer 4): each built by
diffusers from its config shrunk down, with seeded random weights, written
under the names the published single-file checkpoints use, with its output on
fixed inputs. The runner loads the same weights through its own loaders and
must give the same outputs.

    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        --with torch --with diffusers==0.40.0 --with transformers==5.17.0 --with safetensors \\
        python runner/fixtures/tiny_diffusion.py [family ...]

Outputs are committed under runner/test/resources/fixtures/tiny/.
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

out = Path(__file__).resolve().parent.parent / "test" / "resources" / "fixtures" / "tiny"
torch.manual_seed(0)


def krea2_original_name(name):
    """diffusers' Krea 2 names → the original ones (ComfyUI's, sd-cpp's, the
    Civitai single files')."""
    block = [
        (r"norm1\.weight$", "prenorm.scale"), (r"norm2\.weight$", "postnorm.scale"),
        (r"attn\.to_q\.", "attn.wq."), (r"attn\.to_k\.", "attn.wk."), (r"attn\.to_v\.", "attn.wv."),
        (r"attn\.to_gate\.", "attn.gate."), (r"attn\.to_out\.0\.", "attn.wo."),
        (r"attn\.norm_q\.weight$", "attn.qknorm.qnorm.scale"), (r"attn\.norm_k\.weight$", "attn.qknorm.knorm.scale"),
        (r"ff\.(gate|up|down)\.", r"mlp.\1."), (r"scale_shift_table$", "mod.lin"),
    ]
    top = [
        (r"^img_in\.", "first."), (r"^time_embed\.linear_1\.", "tmlp.0."), (r"^time_embed\.linear_2\.", "tmlp.2."),
        (r"^time_mod_proj\.", "tproj.1."), (r"^txt_in\.norm\.weight$", "txtmlp.0.scale"),
        (r"^txt_in\.linear_1\.", "txtmlp.1."), (r"^txt_in\.linear_2\.", "txtmlp.3."),
        (r"^text_fusion\.", "txtfusion."), (r"^transformer_blocks\.", "blocks."),
        (r"^final_layer\.scale_shift_table$", "last.modulation.lin"), (r"^final_layer\.norm\.weight$", "last.norm.scale"),
        (r"^final_layer\.linear\.", "last.linear."),
    ]
    for pattern, replacement in top:
        name = re.sub(pattern, replacement, name)
    if name.startswith("blocks.") or name.startswith("txtfusion.layerwise") or name.startswith("txtfusion.refiner"):
        for pattern, replacement in block:
            name = re.sub(pattern, replacement, name)
    return name


def krea2():
    """Krea 2's transformer: two blocks of 2 query heads over 1 key-value head
    of 64 (RoPE axes 16/24/24), a text fusion of 2 layerwise and 2 refiner
    blocks over 3 text-encoder layers of 64, on a 4 × 6 latent grid (24 image
    tokens) after 7 text tokens."""
    from diffusers import Krea2Transformer2DModel
    model = Krea2Transformer2DModel(
        in_channels=64, num_layers=2, attention_head_dim=64, num_attention_heads=2, num_key_value_heads=1,
        intermediate_size=256, timestep_embed_dim=256, text_hidden_dim=64, num_text_layers=3,
        text_num_attention_heads=2, text_num_key_value_heads=2, text_intermediate_size=128,
        num_layerwise_text_blocks=2, num_refiner_text_blocks=2, axes_dims_rope=(16, 24, 24), rope_theta=1000.0,
        norm_eps=1e-5,
    ).eval().float()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(0.2 * torch.randn_like(parameter))  # stored minus one: (1 + w)
            elif "scale_shift_table" in name:
                parameter.copy_(0.2 * torch.randn_like(parameter))
            else:
                fan_in = parameter.shape[-1] if parameter.dim() > 1 else 16
                parameter.copy_(torch.randn_like(parameter) / fan_in ** 0.5)
    text_tokens, grid_h, grid_w = 7, 4, 6
    latents = torch.randn(1, grid_h * grid_w, 64)
    text = torch.randn(1, text_tokens, 3, 64)
    timestep = torch.tensor([0.75])
    positions = torch.zeros(text_tokens + grid_h * grid_w, 3, dtype=torch.long)
    gy, gx = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    positions[text_tokens:, 1] = gy.flatten()
    positions[text_tokens:, 2] = gx.flatten()
    with torch.no_grad():
        velocity = model(latents, text, timestep, positions, return_dict=False)[0][0].float()
    lora, lora_velocity = krea2_lora(model, latents, text, timestep, positions)
    tensors = {}
    for name, value in model.state_dict().items():
        original = krea2_original_name(name)
        if original.endswith("mod.lin"):
            value = value.reshape(-1)
        tensors[original] = value.detach().float().contiguous()
    folder = out / "krea2"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    save_file(lora, folder / "lora.safetensors", metadata={"format": "pt"})
    save_file({"latents": latents[0].contiguous(), "text": text[0].contiguous(), "timestep": timestep,
               "velocity": velocity.contiguous(), "lora_velocity": lora_velocity.contiguous(),
               "grid": torch.tensor([grid_h, grid_w], dtype=torch.int32)}, folder / "expected.safetensors")
    print(f"krea2: {len(tensors)} tensors, velocity {tuple(velocity.shape)}, |max| {velocity.abs().max():.3f}")
    print("  names:", sorted({re.sub(r"\.\d+\.", ".N.", n) for n in tensors})[:60])


def krea2_lora(model, latents, text, timestep, positions):
    """A LoRA for the tiny Krea 2, as the published ones hold them: pairs
    `lora_A` (down, [rank, in]) and `lora_B` (up, [out, rank]), some under the
    original names (`diffusion_model.`), some under diffusers' (`transformer.`),
    one with an `.alpha` (scale alpha / rank, else 1), and two on tables rather
    than linears (the final modulation, the text projector). Returns the file's
    tensors and the velocity with every delta merged into the weights at
    multiplier 0.8, as diffusers (and sd-cpp) would apply it."""
    multiplier = 0.8
    targets = [  # (diffusers name, original prefix or None for diffusers naming, rank, alpha)
        ("transformer_blocks.0.attn.to_q", "diffusion_model.blocks.0.attn.wq", 4, None),
        ("transformer_blocks.1.ff.down", "diffusion_model.blocks.1.mlp.down", 8, 4.0),
        ("transformer_blocks.1.attn.to_gate", None, 4, None),
        ("text_fusion.refiner_blocks.0.attn.to_v", None, 2, None),
        ("img_in", "diffusion_model.first", 4, None),
        ("time_mod_proj", "diffusion_model.tproj.1", 4, None),
        ("final_layer.scale_shift_table", "diffusion_model.last.modulation.lin", 2, None),
        ("text_fusion.projector", "diffusion_model.txtfusion.projector", 1, None),
    ]
    state = model.state_dict()
    lora = {}
    merged = {}
    for name, original, rank, alpha in targets:
        weight_name = name if name.endswith("scale_shift_table") else name + ".weight"
        weight = state[weight_name]
        out_features, in_features = weight.shape
        down = torch.randn(rank, in_features) / in_features ** 0.5
        up = torch.randn(out_features, rank) * 0.2
        scale = multiplier * (alpha / rank if alpha is not None else 1.0)
        merged[weight_name] = weight + scale * (up @ down)
        prefix = original if original is not None else "transformer." + name
        kohya = prefix.startswith("lora_unet_")
        lora[prefix + (".lora_down.weight" if kohya else ".lora_A.weight")] = down.contiguous()
        lora[prefix + (".lora_up.weight" if kohya else ".lora_B.weight")] = up.contiguous()
        if alpha is not None:
            lora[prefix + ".alpha"] = torch.tensor(alpha)
    base = {k: v.clone() for k, v in state.items()}
    model.load_state_dict({**state, **merged})
    with torch.no_grad():
        velocity = model(latents, text, timestep, positions, return_dict=False)[0][0].float()
    model.load_state_dict(base)
    return lora, velocity


def wan_vae_original_name(name):
    """diffusers' Wan 2.1 VAE decoder names → the original Wan ones (the
    ComfyUI repackage's, sd-cpp's)."""
    name = re.sub(r"^post_quant_conv\.", "conv2.", name)
    name = re.sub(r"^decoder\.conv_in\.", "decoder.conv1.", name)
    name = re.sub(r"^decoder\.norm_out\.", "decoder.head.0.", name)
    name = re.sub(r"^decoder\.conv_out\.", "decoder.head.2.", name)
    name = re.sub(r"^decoder\.mid_block\.attentions\.0\.", "decoder.middle.1.", name)
    name = re.sub(r"^decoder\.mid_block\.resnets\.(\d+)\.", lambda m: f"decoder.middle.{2 * int(m[1])}.", name)
    name = re.sub(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.",
                  lambda m: f"decoder.upsamples.{4 * int(m[1]) + int(m[2])}.", name)
    name = re.sub(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.",
                  lambda m: f"decoder.upsamples.{4 * int(m[1]) + 3}.", name)
    for diffusers, original in [("norm1.gamma", "residual.0.gamma"), ("conv1.", "residual.2."),
                                ("norm2.gamma", "residual.3.gamma"), ("conv2.", "residual.6."),
                                ("conv_shortcut.", "shortcut.")]:
        if name.startswith("decoder.middle.") or name.startswith("decoder.upsamples."):
            name = re.sub(r"\." + re.escape(diffusers), "." + original, name)
    return name


def wan_vae():
    """The Wan 2.1 VAE's decoder, 8 channels at its base (32 in the middle),
    decoding a 4 × 4 latent into a 32 × 32 image; the latents are the
    diffusion model's, un-normalized with Wan's statistics first, as the Krea 2
    pipeline does."""
    from diffusers import AutoencoderKLQwenImage
    model = AutoencoderKLQwenImage(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                                   temperal_downsample=[False, True, True]).eval().float()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("gamma"):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            else:
                fan_in = parameter[0].numel() if parameter.dim() > 1 else 16
                parameter.copy_(torch.randn_like(parameter) / fan_in ** 0.5)
        model.decoder.conv_out.weight.mul_(0.1)  # keep the image inside [−1, 1], unclamped
    latents = torch.randn(1, 16, 1, 4, 4)
    mean = torch.tensor(model.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(model.config.latents_std).view(1, 16, 1, 1, 1)
    with torch.no_grad():
        image = model.decode(latents * std + mean, return_dict=False)[0][0, :, 0].float()
    tensors = {wan_vae_original_name(name): value.detach().float().contiguous()
               for name, value in model.state_dict().items()
               if name.startswith("decoder.") or name.startswith("post_quant_conv.")}
    folder = out / "wan_vae"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    save_file({"latents": latents[0, :, 0].permute(1, 2, 0).contiguous(),
               "image": image.permute(1, 2, 0).contiguous()}, folder / "expected.safetensors")
    print(f"wan_vae: {len(tensors)} tensors, image {tuple(image.shape)}, |max| {image.abs().max():.3f}")
    print("  names:", sorted({re.sub(r"\.\d+\.", ".N.", n) for n in tensors}))


def flux2_original_name(name):
    """diffusers' FLUX.2 transformer names → ComfyUI's single-file ones (BFL's
    under `model.diffusion_model.`, norms as `.weight`); the fused q/k/v are
    joined by the caller."""
    rules = [
        (r"^x_embedder\.", "img_in."), (r"^context_embedder\.", "txt_in."),
        (r"^time_guidance_embed\.timestep_embedder\.linear_1\.", "time_in.in_layer."),
        (r"^time_guidance_embed\.timestep_embedder\.linear_2\.", "time_in.out_layer."),
        (r"^time_guidance_embed\.guidance_embedder\.linear_1\.", "guidance_in.in_layer."),
        (r"^time_guidance_embed\.guidance_embedder\.linear_2\.", "guidance_in.out_layer."),
        (r"^(double_stream_modulation_img|double_stream_modulation_txt|single_stream_modulation)\.linear\.",
         r"\1.lin."),
        (r"^norm_out\.linear\.", "final_layer.adaLN_modulation.1."), (r"^proj_out\.", "final_layer.linear."),
        (r"^transformer_blocks\.(\d+)\.attn\.norm_q\.", r"double_blocks.\1.img_attn.norm.query_norm."),
        (r"^transformer_blocks\.(\d+)\.attn\.norm_k\.", r"double_blocks.\1.img_attn.norm.key_norm."),
        (r"^transformer_blocks\.(\d+)\.attn\.norm_added_q\.", r"double_blocks.\1.txt_attn.norm.query_norm."),
        (r"^transformer_blocks\.(\d+)\.attn\.norm_added_k\.", r"double_blocks.\1.txt_attn.norm.key_norm."),
        (r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.", r"double_blocks.\1.img_attn.proj."),
        (r"^transformer_blocks\.(\d+)\.attn\.to_add_out\.", r"double_blocks.\1.txt_attn.proj."),
        (r"^transformer_blocks\.(\d+)\.ff\.linear_in\.", r"double_blocks.\1.img_mlp.0."),
        (r"^transformer_blocks\.(\d+)\.ff\.linear_out\.", r"double_blocks.\1.img_mlp.2."),
        (r"^transformer_blocks\.(\d+)\.ff_context\.linear_in\.", r"double_blocks.\1.txt_mlp.0."),
        (r"^transformer_blocks\.(\d+)\.ff_context\.linear_out\.", r"double_blocks.\1.txt_mlp.2."),
        (r"^single_transformer_blocks\.(\d+)\.attn\.to_qkv_mlp_proj\.", r"single_blocks.\1.linear1."),
        (r"^single_transformer_blocks\.(\d+)\.attn\.to_out\.", r"single_blocks.\1.linear2."),
        (r"^single_transformer_blocks\.(\d+)\.attn\.norm_q\.", r"single_blocks.\1.norm.query_norm."),
        (r"^single_transformer_blocks\.(\d+)\.attn\.norm_k\.", r"single_blocks.\1.norm.key_norm."),
    ]
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return "model.diffusion_model." + name


def flux2():
    """FLUX.2's transformer: two double-stream and two single-stream blocks of
    2 heads of 64 (RoPE axes 16 × 4, theta 2000), text features 96 wide,
    32 latent features; a 4 × 6 target grid after 7 text tokens, then one
    2 × 4 reference. No guidance embedding (Klein)."""
    flux2_transformer("flux2", guidance=None, lora_targets=[  # (the diffusers weights the update covers in turn, the file's name, rank, alpha)
        ([f"transformer_blocks.0.attn.to_{x}" for x in "qkv"], "diffusion_model.double_blocks.0.img_attn.qkv", 4, None),
        (["transformer_blocks.1.attn.add_k_proj"], "transformer.transformer_blocks.1.attn.add_k_proj", 2, 4.0),
        (["transformer_blocks.1.ff_context.linear_in"], "diffusion_model.double_blocks.1.txt_mlp.0", 4, None),
        (["single_transformer_blocks.0.attn.to_qkv_mlp_proj"], "diffusion_model.single_blocks.0.linear1", 8, 4.0),
        (["single_transformer_blocks.1.attn.to_out"], "transformer.single_transformer_blocks.1.attn.to_out", 4, None),
        (["norm_out.linear"], "transformer.norm_out.linear", 2, None),
        (["x_embedder"], "diffusion_model.img_in", 4, None),
        (["double_stream_modulation_txt.linear"], "transformer.double_stream_modulation_txt.linear", 4, None),
    ])


def flux2_dev():
    """FLUX.2 [dev]'s transformer: Klein's tiny shape with the guidance
    embedding, at guidance 3.5; its LoRA partly in kohya's names
    (`lora_unet_…`, `lora_down`/`lora_up`), partly on the guidance MLP."""
    flux2_transformer("flux2_dev", guidance=3.5, lora_targets=[
        ([f"transformer_blocks.0.attn.to_{x}" for x in "qkv"], "lora_unet_double_blocks_0_img_attn_qkv", 4, 2.0),
        (["transformer_blocks.1.ff.linear_in"], "lora_unet_double_blocks_1_img_mlp_0", 4, None),
        (["single_transformer_blocks.1.attn.to_out"], "lora_unet_single_blocks_1_linear2", 4, None),
        (["time_guidance_embed.guidance_embedder.linear_1"], "diffusion_model.guidance_in.in_layer", 4, None),
        (["time_guidance_embed.guidance_embedder.linear_2"],
         "transformer.time_guidance_embed.guidance_embedder.linear_2", 2, None),
    ])


def flux2_transformer(folder_name, guidance, lora_targets):
    """The tiny FLUX.2 transformer written under `folder_name`: with a guidance
    embedding run at `guidance` when it is not None."""
    from diffusers import Flux2Transformer2DModel
    model = Flux2Transformer2DModel(
        in_channels=32, num_layers=2, num_single_layers=2, attention_head_dim=64, num_attention_heads=2,
        joint_attention_dim=96, timestep_guidance_channels=256, mlp_ratio=3.0, axes_dims_rope=(16, 16, 16, 16),
        rope_theta=2000, eps=1e-6, guidance_embeds=guidance is not None,
    ).eval().float()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and parameter.dim() == 1:
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            else:
                parameter.copy_(torch.randn_like(parameter) / parameter.shape[-1] ** 0.5)
    text_tokens, (grid_h, grid_w), (ref_h, ref_w) = 7, (4, 6), (2, 4)
    latents = torch.randn(1, grid_h * grid_w, 32)
    reference = torch.randn(1, ref_h * ref_w, 32)
    text = torch.randn(1, text_tokens, 96)
    timestep = torch.tensor([0.75])
    txt_ids = torch.zeros(text_tokens, 4, dtype=torch.long)
    txt_ids[:, 3] = torch.arange(text_tokens)
    def ids(t, h, w):
        return torch.cartesian_prod(torch.tensor([t]), torch.arange(h), torch.arange(w), torch.arange(1))
    img_ids = torch.cat([ids(0, grid_h, grid_w), ids(10, ref_h, ref_w)])
    def run():
        with torch.no_grad():
            return model(hidden_states=torch.cat([latents, reference], 1), encoder_hidden_states=text,
                         timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                         guidance=None if guidance is None else torch.tensor([guidance]),
                         return_dict=False)[0][0, :grid_h * grid_w].float()
    velocity = run()
    lora, lora_velocity = flux2_lora(model, run, lora_targets)
    state = model.state_dict()
    tensors = {}
    for i in range(len(model.transformer_blocks)):
        b = f"transformer_blocks.{i}.attn."
        tensors[f"model.diffusion_model.double_blocks.{i}.img_attn.qkv.weight"] = torch.cat(
            [state.pop(b + f"to_{x}.weight") for x in "qkv"])
        tensors[f"model.diffusion_model.double_blocks.{i}.txt_attn.qkv.weight"] = torch.cat(
            [state.pop(b + f"add_{x}_proj.weight") for x in "qkv"])
    for name, value in state.items():
        if name == "norm_out.linear.weight":  # diffusers (scale ; shift), BFL (shift ; scale)
            scale, shift = value.chunk(2)
            value = torch.cat([shift, scale])
        tensors[flux2_original_name(name)] = value
    tensors = {name: value.detach().float().contiguous() for name, value in tensors.items()}
    folder = out / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    save_file(lora, folder / "lora.safetensors", metadata={"format": "pt"})
    save_file({"latents": latents[0].contiguous(), "reference": reference[0].contiguous(),
               "text": text[0].contiguous(), "timestep": timestep, "velocity": velocity.contiguous(),
               "lora_velocity": lora_velocity.contiguous(),
               "guidance": torch.tensor([guidance if guidance is not None else 0.0]),
               "grid": torch.tensor([grid_h, grid_w, ref_h, ref_w], dtype=torch.int32)},
              folder / "expected.safetensors")
    print(f"{folder_name}: {len(tensors)} tensors, velocity {tuple(velocity.shape)}, |max| {velocity.abs().max():.3f}")
    print("  names:", sorted({re.sub(r"\.\d+\.", ".N.", n) for n in tensors}))


def flux2_lora(model, run, targets):
    """A LoRA for the tiny FLUX.2 as the published ones hold them (`lora_A`
    down [rank, in], `lora_B` up [out, rank]; kohya's `lora_unet_` names
    `lora_down`/`lora_up`): some under BFL's names (`diffusion_model.`, fused
    weights whole: qkv, the MLP's input, linear1), some under diffusers'
    (`transformer.`, including the final modulation, whose halves diffusers
    stores the other way round), some with an `.alpha`. Returns the file's
    tensors and the velocity with every delta merged into diffusers' weights
    at multiplier 0.8."""
    multiplier = 0.8
    state = model.state_dict()
    lora, merged = {}, {}
    for names, prefix, rank, alpha in targets:
        weights = [state[name + ".weight"] for name in names]
        in_features = weights[0].shape[1]
        down = torch.randn(rank, in_features) / in_features ** 0.5
        up = torch.randn(sum(w.shape[0] for w in weights), rank) * 0.2
        scale = multiplier * (alpha / rank if alpha is not None else 1.0)
        first = 0
        for name, weight in zip(names, weights):
            rows = weight.shape[0]
            merged[name + ".weight"] = weight + scale * (up[first:first + rows] @ down)
            first += rows
        lora[prefix + ".lora_A.weight"] = down.contiguous()
        lora[prefix + ".lora_B.weight"] = up.contiguous()
        if alpha is not None:
            lora[prefix + ".alpha"] = torch.tensor(alpha)
    base = {k: v.clone() for k, v in state.items()}
    model.load_state_dict({**state, **merged})
    velocity = run()
    model.load_state_dict(base)
    return lora, velocity


def flux2_vae_original_name(name, levels):
    """diffusers' FLUX.2 VAE names → the original LDM ones (the ComfyUI
    repackage's)."""
    name = re.sub(r"^quant_conv\.", "encoder.quant_conv.", name)
    name = re.sub(r"^post_quant_conv\.", "decoder.post_quant_conv.", name)
    name = re.sub(r"^(encoder|decoder)\.conv_norm_out\.", r"\1.norm_out.", name)
    name = re.sub(r"^encoder\.down_blocks\.(\d+)\.resnets\.", r"encoder.down.\1.block.", name)
    name = re.sub(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", name)
    name = re.sub(r"^decoder\.up_blocks\.(\d+)\.resnets\.",
                  lambda m: f"decoder.up.{levels - 1 - int(m[1])}.block.", name)
    name = re.sub(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.",
                  lambda m: f"decoder.up.{levels - 1 - int(m[1])}.upsample.", name)
    name = re.sub(r"\.mid_block\.resnets\.(\d+)\.", lambda m: f".mid.block_{int(m[1]) + 1}.", name)
    name = re.sub(r"\.mid_block\.attentions\.0\.", ".mid.attn_1.", name)
    for diffusers, original in [("group_norm.", "norm."), ("to_q.", "q."), ("to_k.", "k."), ("to_v.", "v."),
                                ("to_out.0.", "proj_out."), ("conv_shortcut.", "nin_shortcut.")]:
        name = name.replace("." + diffusers, "." + original)
    return name


def flux2_vae():
    """The FLUX.2 VAE both ways, three levels of 32/64/64 channels (one
    residual block per level in the encoder, two in the decoder), 8 latent
    channels: a 32 × 24 image encoded to packed, normalized latents on a 4 × 3
    grid, and random packed latents on that grid decoded, as the Klein
    pipeline does."""
    from diffusers import AutoencoderKLFlux2
    model = AutoencoderKLFlux2(block_out_channels=(32, 64, 64), down_block_types=("DownEncoderBlock2D",) * 3,
                               up_block_types=("UpDecoderBlock2D",) * 3, layers_per_block=1,
                               latent_channels=8).eval().float()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            else:
                fan_in = parameter[0].numel() if parameter.dim() > 1 else 16
                parameter.copy_(torch.randn_like(parameter) / fan_in ** 0.5)
        model.decoder.conv_out.weight.mul_(0.1)  # keep the image inside [−1, 1], unclamped
        model.bn.running_mean.copy_(0.1 * torch.randn(32))
        model.bn.running_var.copy_(0.5 + torch.rand(32))
    bn_mean = model.bn.running_mean.view(1, -1, 1, 1)
    bn_std = torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps)
    def patchify(z):
        b, c, h, w = z.shape
        return z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    def unpatchify(z):
        b, c, h, w = z.shape
        return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
    image = torch.rand(1, 3, 32, 24) * 2 - 1
    grid_h, grid_w = 4, 3
    packed = torch.randn(1, 32, grid_h, grid_w)
    with torch.no_grad():
        encoded = (patchify(model.encode(image).latent_dist.mode()) - bn_mean) / bn_std
        decoded = model.decode(unpatchify(packed * bn_std + bn_mean), return_dict=False)[0][0].float()
    def rows(z):  # [1, C, h, w] → [h × w, C]
        return z[0].permute(1, 2, 0).reshape(-1, z.shape[1]).contiguous()
    tensors = {flux2_vae_original_name(name, 3): value.detach().float().contiguous()
               for name, value in model.state_dict().items() if value.is_floating_point()}
    folder = out / "flux2_vae"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    # the same weights under diffusers' own names (ERNIE-Image's flux2-vae): the runner maps them at load
    save_file({name: value.detach().float().contiguous() for name, value in model.state_dict().items()
               if value.is_floating_point()}, folder / "diffusers.safetensors")
    save_file({"image": image[0].permute(1, 2, 0).contiguous(), "latents": rows(encoded),
               "packed": rows(packed), "decoded": decoded.permute(1, 2, 0).contiguous(),
               "grid": torch.tensor([grid_h, grid_w], dtype=torch.int32)}, folder / "expected.safetensors")
    print(f"flux2_vae: {len(tensors)} tensors, latents {tuple(rows(encoded).shape)}, image {tuple(decoded.shape)}, "
          f"|max| {decoded.abs().max():.3f}")
    print("  names:", sorted({re.sub(r"block\.\d+", "block.N", n) for n in tensors}))


def qwen_image21_lora(model, run):
    """A LoRA for the tiny Qwen Image 2.1 as the published ones hold them:
    diffusers' names (`transformer.`, the MLP's `proj` and `gate_layer` apart)
    and ComfyUI's (`diffusion_model.`, `lora_down`/`lora_up`, the MLP's fused
    `gate_up` whole), one with an `.alpha`. Returns the file's tensors and the
    velocity with every delta merged into diffusers' weights at multiplier
    0.8."""
    multiplier = 0.8
    targets = [  # (the diffusers weights the update covers in turn, the file's name, rank, alpha)
        (["transformer_blocks.0.attn.to_q"], "transformer.transformer_blocks.0.attn.to_q", 4, None),
        (["transformer_blocks.0.img_mlp.proj"], "transformer.transformer_blocks.0.img_mlp.proj", 2, 4.0),
        (["transformer_blocks.1.img_mlp.gate_layer", "transformer_blocks.1.img_mlp.proj"],
         "diffusion_model.transformer_blocks.1.img_mlp.gate_up", 4, None),
        (["transformer_blocks.1.attn.to_out.0"], "diffusion_model.transformer_blocks.1.attn.to_out.0", 4, None),
        (["txt_in.in_layer"], "transformer.txt_in.in_layer", 2, None),
        (["modulation.1"], "diffusion_model.modulation.1", 4, None),
        (["norm_out.linear"], "transformer.norm_out.linear", 2, None),
        (["img_in"], "diffusion_model.img_in", 4, None),
    ]
    state = model.state_dict()
    lora, merged = {}, {}
    for names, prefix, rank, alpha in targets:
        weights = [state[name + ".weight"] for name in names]
        in_features = weights[0].shape[1]
        down = torch.randn(rank, in_features) / in_features ** 0.5
        up = torch.randn(sum(w.shape[0] for w in weights), rank) * 0.2
        scale = multiplier * (alpha / rank if alpha is not None else 1.0)
        first = 0
        for name, weight in zip(names, weights):
            rows = weight.shape[0]
            merged[name + ".weight"] = weight + scale * (up[first:first + rows] @ down)
            first += rows
        comfy = prefix.startswith("diffusion_model.")
        lora[prefix + (".lora_down.weight" if comfy else ".lora_A.weight")] = down.contiguous()
        lora[prefix + (".lora_up.weight" if comfy else ".lora_B.weight")] = up.contiguous()
        if alpha is not None:
            lora[prefix + ".alpha"] = torch.tensor(alpha)
    base = {k: v.clone() for k, v in state.items()}
    model.load_state_dict({**state, **merged})
    velocity = run()
    model.load_state_dict(base)
    return lora, velocity


def qwen_image21():
    """Qwen Image 2.1's transformer: two single-stream blocks of 2 heads of
    128 (RoPE axes 16/56/56), 64 latent channels, text of 64, on a 4 × 6 latent
    grid after 7 text tokens (block-causal: the text causal, the image seeing
    all). Saved as ComfyUI's file holds it: diffusers' names, the MLP's gate and
    proj fused into `gate_up`. Also the scheduler's sigmas for two sizes.
    Needs diffusers newer than 0.40 (`QwenImage21Transformer2DModel`):
    `--with git+https://github.com/huggingface/diffusers@bdc2bea37a36038c44452811610489ea30ede229`."""
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    model = QwenImage21Transformer2DModel(in_channels=64, out_channels=64, num_layers=2, attention_head_dim=128,
                                          num_attention_heads=2, context_in_dim=64, mlp_ratio=3).eval().float()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("text_norm.weight"):
                parameter.copy_(0.2 * torch.randn_like(parameter))  # stored minus one: (1 + w)
            elif "norm" in name and parameter.dim() == 1:
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            else:
                parameter.copy_(torch.randn_like(parameter) / parameter.shape[-1] ** 0.5)
    text_tokens, grid_h, grid_w = 7, 4, 6
    latents = torch.randn(1, grid_h * grid_w, 64)
    text = torch.randn(1, text_tokens, 64)
    timestep = torch.tensor([0.75])
    img_mask = torch.cat([torch.zeros(1, text_tokens, dtype=torch.bool),
                          torch.ones(1, grid_h * grid_w // 4, dtype=torch.bool)], dim=1)
    def run():
        with torch.no_grad():
            out = model(hidden_states=latents, encoder_hidden_states=text, timestep=timestep,
                        img_shapes=[[(1, grid_h, grid_w)]], img_mask=img_mask, return_dict=False)[0]
        return out[0, -grid_h * grid_w:].float()
    velocity = run()
    edit = qwen_image21_edit(model)
    lora, lora_velocity = qwen_image21_lora(model, run)
    tensors = {}
    state = model.state_dict()
    for name, value in state.items():
        if name.endswith("img_mlp.gate_layer.weight"):
            block = name[:-len("gate_layer.weight")]
            tensors[block + "gate_up.weight"] = torch.cat([value, state[block + "proj.weight"]]).contiguous()
        elif not name.endswith("img_mlp.proj.weight"):
            tensors[name] = value.detach().float().contiguous()
    scheduler = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 8192, "max_shift": 0.9,
        "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": 0.02, "time_shift_type": "exponential",
        "use_dynamic_shifting": True})
    schedules = {}
    for steps, tokens in [(8, 4096), (25, 1024)]:
        mu = calculate_shift(tokens, 256, 8192, 0.5, 0.9)
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
        schedules[f"sigmas_{steps}_{tokens}"] = scheduler.sigmas.float().contiguous()
    folder = out / "qwen_image21"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    save_file(lora, folder / "lora.safetensors", metadata={"format": "pt"})
    save_file({"latents": latents[0].contiguous(), "text": text[0].contiguous(), "timestep": timestep,
               "velocity": velocity.contiguous(), "lora_velocity": lora_velocity.contiguous(),
               "grid": torch.tensor([grid_h, grid_w], dtype=torch.int32), **schedules, **edit},
              folder / "expected.safetensors")
    print(f"qwen_image21: {len(tensors)} tensors, velocity {tuple(velocity.shape)}, |max| {velocity.abs().max():.3f}")
    print("  names:", sorted({re.sub(r"\.\d+\.", ".N.", n) for n in tensors}))


def qwen_image21_edit(model):
    """Editing: two references among 12 text rows, a 4 × 4 grid in slots 2–5
    and a 2 × 6 in 7–9 (each slot 2 × 2 latents), then the 4 × 6 image. Its own
    generator, so the other goldens keep theirs."""
    generator = torch.Generator().manual_seed(7)
    def randn(*shape):
        return torch.randn(*shape, generator=generator)
    text = randn(1, 12, 64)
    first, second, target = randn(1, 16, 64), randn(1, 12, 64), randn(1, 24, 64)
    slots = torch.zeros(1, 12 + 6, dtype=torch.bool)
    slots[0, 2:6] = True
    slots[0, 7:10] = True
    slots[0, 12:] = True
    with torch.no_grad():
        out = model(hidden_states=torch.cat([first, second, target], dim=1), encoder_hidden_states=text,
                    timestep=torch.tensor([0.75]), img_shapes=[[(1, 4, 4), (1, 2, 6), (1, 4, 6)]],
                    img_mask=slots, return_dict=False)[0]
    return {"edit_text": text[0].contiguous(), "edit_first": first[0].contiguous(),
            "edit_second": second[0].contiguous(), "edit_latents": target[0].contiguous(),
            "edit_velocity": out[0, -24:].float().contiguous()}


def qwen_image21_vae_original_name(name):
    """diffusers' Qwen Image 2.1 VAE names → the original Wan 2.2 ones (the
    ComfyUI repackage's): levels nested as `upsamples.N.upsamples.M`, the
    resampler last in its level."""
    name = re.sub(r"^quant_conv\.", "conv1.", name)
    name = re.sub(r"^post_quant_conv\.", "conv2.", name)
    name = re.sub(r"^(encoder|decoder)\.conv_in\.", r"\1.conv1.", name)
    name = re.sub(r"^(encoder|decoder)\.norm_out\.", r"\1.head.0.", name)
    name = re.sub(r"^(encoder|decoder)\.conv_out\.", r"\1.head.2.", name)
    name = re.sub(r"^(encoder|decoder)\.mid_block\.attentions\.0\.", r"\1.middle.1.", name)
    name = re.sub(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.", lambda m: f"{m[1]}.middle.{2 * int(m[2])}.", name)
    name = re.sub(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.downsamples.\1.downsamples.\2.", name)
    name = re.sub(r"^encoder\.down_blocks\.(\d+)\.downsampler\.", r"encoder.downsamples.\1.downsamples.2.", name)
    name = re.sub(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.", r"decoder.upsamples.\1.upsamples.\2.", name)
    name = re.sub(r"^decoder\.up_blocks\.(\d+)\.upsampler\.", r"decoder.upsamples.\1.upsamples.3.", name)
    for diffusers, original in [("norm1.gamma", "residual.0.gamma"), ("conv1.", "residual.2."),
                                ("norm2.gamma", "residual.3.gamma"), ("conv2.", "residual.6."),
                                ("conv_shortcut.", "shortcut.")]:
        if re.match(r"^(encoder|decoder)\.(middle|downsamples|upsamples)\.", name):
            name = re.sub(r"\." + re.escape(diffusers), "." + original, name)
    return name


def qwen_image21_vae():
    """The Qwen Image 2.1 VAE both ways (Wan 2.2's residual design: channel
    shuffles around each resampling level), 8 channels at the encoder's base,
    12 at the decoder's, the real 64 latent channels and their statistics, RGBA:
    a 64 × 48 image encoded to normalized latents on a 4 × 3 grid, and random
    latents on that grid decoded, as the pipeline does. Needs diffusers newer
    than 0.40, as `qwen_image21`."""
    from diffusers import AutoencoderKLQwenImage21
    model = AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=12).eval().float()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("gamma"):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            else:
                fan_in = parameter[0].numel() if parameter.dim() > 1 else 16
                parameter.copy_(torch.randn_like(parameter) / fan_in ** 0.5)
        model.decoder.conv_out.weight.mul_(0.1)  # keep the image inside [−1, 1], unclamped
    mean = torch.tensor(model.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(model.config.latents_std).view(1, 64, 1, 1, 1)
    image = torch.rand(1, 4, 1, 64, 48) * 2 - 1
    grid_h, grid_w = 4, 3
    latents = torch.randn(1, 64, 1, grid_h, grid_w)
    with torch.no_grad():
        encoded = (model.encode(image).latent_dist.mode() - mean) / std
        decoded = model.decode(latents * std + mean, return_dict=False)[0][0, :, 0].float()
    def rows(z):  # [1, C, 1, h, w] → [h, w, C]
        return z[0, :, 0].permute(1, 2, 0).contiguous()
    tensors = {qwen_image21_vae_original_name(name): value.detach().float().contiguous()
               for name, value in model.state_dict().items()}
    folder = out / "qwen_image21_vae"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    save_file({"image": rows(image), "latents": rows(encoded), "packed": rows(latents),
               "decoded": decoded.permute(1, 2, 0).contiguous(),
               "grid": torch.tensor([grid_h, grid_w], dtype=torch.int32)}, folder / "expected.safetensors")
    print(f"qwen_image21_vae: {len(tensors)} tensors, latents {tuple(rows(encoded).shape)}, "
          f"image {tuple(decoded.shape)}, |max| {decoded.abs().max():.3f}")
    print("  names:", sorted({re.sub(r"\.\d+\.", ".N.", n) for n in tensors}))


families = {"krea2": krea2, "wan_vae": wan_vae, "flux2": flux2, "flux2_dev": flux2_dev, "flux2_vae": flux2_vae,
            "qwen_image21": qwen_image21, "qwen_image21_vae": qwen_image21_vae}
for family in sys.argv[1:] or families:
    families[family]()
