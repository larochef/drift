"""Golden tiny models (specs/42, Testing, layer 4): each family built by
transformers from its config shrunk down, with seeded random weights, its
weights saved as safetensors and its logits on a fixed sequence. The runner
loads the same weights through its own loaders and must give the same logits.

Needs torch and transformers (ComfyUI's venv has both), and names the
families to build (all when none is named):

    /opt/comfyui/venv/bin/python runner/fixtures/tiny_models.py qwen3 qwen35 qwen35moe

mistral also needs the `gguf` package (`--with gguf` below); qwen4exp needs
transformers 5.17 (the first release with `qwen4_exp`):

    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        --with torch --with transformers==5.17.0 --with safetensors \
        python runner/fixtures/tiny_models.py qwen4exp

Outputs are committed under runner/test/resources/fixtures/tiny/.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
import transformers

out = Path(__file__).resolve().parent.parent / "test" / "resources" / "fixtures" / "tiny"
torch.manual_seed(0)


def randomize_norms(model):
    # a norm of ones would hide a weight read the wrong way
    for name, parameter in model.named_parameters():
        if "norm" in name:
            with torch.no_grad():
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))


def qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(
        vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, rope_theta=1e6,
        rms_norm_eps=1e-6, tie_word_embeddings=False, max_position_embeddings=512,
        torch_dtype="float32",
    )
    model = Qwen3ForCausalLM(config).eval()
    randomize_norms(model)
    folder = out / "qwen3"
    model.save_pretrained(folder, safe_serialization=True)
    ids = torch.randint(0, config.vocab_size, (1, 20))
    with torch.no_grad():
        output = model(ids, output_hidden_states=True)
    logits = output.logits[0].float()
    # the residual stream after 0 and 1 layers (the last entry is normed)
    hidden = torch.stack(output.hidden_states[:2])[:, 0].float()
    # the same with the last 8 tokens padding (id 7), masked as keys (FLUX.2's encoder)
    mask = torch.ones_like(ids)
    mask[:, 12:] = 0
    with torch.no_grad():
        padded = model(torch.where(mask.bool(), ids, 7), attention_mask=mask, output_hidden_states=True)
    padded_hidden = torch.stack(padded.hidden_states[:2])[:, 0].float()
    save_file({"ids": ids[0].to(torch.int32), "logits": logits.contiguous(), "hidden": hidden.contiguous(),
               "padded_hidden": padded_hidden.contiguous()},
              folder / "expected.safetensors")
    print(f"qwen3: logits {tuple(logits.shape)}, |max| {logits.abs().max():.3f}")


def gemma2():
    """Gemma 2 as a text encoder (PiD's): 3 layers of 64, 4 query heads over 2
    of 64, the query scalar (48) apart from the head size, a softcap low
    enough to bind (2) and a sliding window (8) shorter than the 20 tokens.
    The residual stream after every layer and the final normed state."""
    from transformers import Gemma2Config, Gemma2Model
    config = Gemma2Config(
        vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, query_pre_attn_scalar=48,
        attn_logit_softcapping=2.0, final_logit_softcapping=30.0, sliding_window=8,
        rope_theta=10000.0, rms_norm_eps=1e-6, max_position_embeddings=512, torch_dtype="float32",
        attn_implementation="eager",
    )
    model = Gemma2Model(config).eval()
    randomize_norms(model)
    folder = out / "gemma2"
    model.save_pretrained(folder, safe_serialization=True)
    ids = torch.randint(0, config.vocab_size, (1, 20))
    with torch.no_grad():
        output = model(ids, output_hidden_states=True)
    # hidden_states[k] is the residual stream after k layers, but the last,
    # which transformers norms
    hidden = torch.stack(output.hidden_states[:-1])[:, 0].float()
    save_file({"ids": ids[0].to(torch.int32), "hidden": hidden.contiguous(),
               "normed": output.last_hidden_state[0].float().contiguous()}, folder / "expected.safetensors")
    print(f"gemma2: hidden {tuple(hidden.shape)}, |max| {hidden.abs().max():.3f}")


def mistral():
    """Mistral Small 3.x's language model as a text encoder (FLUX.2 [dev]'s):
    3 layers of 64, 4 query heads over 2 of 64 (a head size apart from
    hidden / heads, as the 24B's 128 against 5120 / 32). The residual stream after 1
    and 2 layers of 20 tokens. Written twice: transformers' safetensors, and a
    GGUF as llama.cpp converts it (architecture `llama`, q and k permuted to
    rotate interleaved pairs). Needs the `gguf` package."""
    import gguf
    import numpy as np
    from transformers import MistralConfig, MistralForCausalLM
    config = MistralConfig(
        vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, rope_theta=10000.0,
        rms_norm_eps=1e-5, tie_word_embeddings=False, max_position_embeddings=512,
        sliding_window=None, torch_dtype="float32",
    )
    model = MistralForCausalLM(config).eval()
    randomize_norms(model)
    folder = out / "mistral"
    model.save_pretrained(folder, safe_serialization=True)
    ids = torch.randint(0, config.vocab_size, (1, 20))
    with torch.no_grad():
        output = model(ids, output_hidden_states=True)
    hidden = torch.stack(output.hidden_states[1:3])[:, 0].float()
    save_file({"ids": ids[0].to(torch.int32), "hidden": hidden.contiguous()}, folder / "expected.safetensors")

    def permute(weights, heads):  # convert_hf_to_gguf's LlamaModel.permute
        return (weights.reshape(heads, 2, weights.shape[0] // heads // 2, *weights.shape[1:])
                .swapaxes(1, 2).reshape(weights.shape))
    writer = gguf.GGUFWriter(str(folder / "model.gguf"), "llama")
    writer.add_block_count(config.num_hidden_layers)
    writer.add_context_length(512)
    writer.add_embedding_length(config.hidden_size)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_key_length(config.head_dim)
    writer.add_value_length(config.head_dim)
    writer.add_rope_freq_base(10000.0)  # rope_theta above (transformers 5 nests it)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    state = {name: value.detach().float().numpy() for name, value in model.state_dict().items()}
    parts = {"input_layernorm": "attn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
             "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
             "post_attention_layernorm": "ffn_norm", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
             "mlp.down_proj": "ffn_down"}
    writer.add_tensor("token_embd.weight", state["model.embed_tokens.weight"])
    writer.add_tensor("output_norm.weight", state["model.norm.weight"])
    writer.add_tensor("output.weight", state["lm_head.weight"])
    for i in range(config.num_hidden_layers):
        for part, name in parts.items():
            value = state[f"model.layers.{i}.{part}.weight"]
            if part == "self_attn.q_proj":
                value = permute(value, config.num_attention_heads)
            elif part == "self_attn.k_proj":
                value = permute(value, config.num_key_value_heads)
            writer.add_tensor(f"blk.{i}.{name}.weight", np.ascontiguousarray(value))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"mistral: hidden {tuple(hidden.shape)}, |max| {hidden.abs().max():.3f}")


def qwen35_moe():
    # Qwen 3.5/3.6's shape: three gated-DeltaNet layers then one full-attention
    # layer, every MLP a mixture of experts with a shared expert
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    config = Qwen3_5MoeTextConfig(
        vocab_size=320, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, shared_expert_intermediate_size=64, rms_norm_eps=1e-6,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        tie_word_embeddings=False, max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    )
    model = Qwen3_5MoeForCausalLM(config).eval()
    randomize_qwen35(model.named_parameters())
    folder = out / "qwen35moe"
    model.save_pretrained(folder, safe_serialization=True)
    ids = torch.randint(0, config.vocab_size, (1, 20))
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    mtp_logits = qwen35_mtp(model, config, ids, folder / "model.safetensors")
    save_file({"ids": ids[0].to(torch.int32), "logits": logits.contiguous(), "mtp_logits": mtp_logits.contiguous()},
              folder / "expected.safetensors")
    print(f"qwen35moe: logits {tuple(logits.shape)}, |max| {logits.abs().max():.3f}; "
          f"MTP logits {tuple(mtp_logits.shape)}, |max| {mtp_logits.abs().max():.3f}")


def qwen35():
    # the dense Qwen 3.5/3.8 (27B): the same layers with a SwiGLU for the experts
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    config = Qwen3_5TextConfig(
        vocab_size=320, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4, intermediate_size=96, rms_norm_eps=1e-6,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        tie_word_embeddings=False, max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    )
    model = Qwen3_5ForCausalLM(config).eval()
    randomize_qwen35(model.named_parameters())
    folder = out / "qwen35"
    model.save_pretrained(folder, safe_serialization=True)
    ids = torch.randint(0, config.vocab_size, (1, 20))
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    mtp_logits = qwen35_mtp(model, config, ids, folder / "model.safetensors")
    save_file({"ids": ids[0].to(torch.int32), "logits": logits.contiguous(), "mtp_logits": mtp_logits.contiguous()},
              folder / "expected.safetensors")
    print(f"qwen35: logits {tuple(logits.shape)}, |max| {logits.abs().max():.3f}; "
          f"MTP logits {tuple(mtp_logits.shape)}, |max| {mtp_logits.abs().max():.3f}")


def qwen35_moe_vision():
    # Qwen 3.5/3.6 seeing: the vision tower (two blocks, heads of 16, a 4 × 4
    # position table sampled for a 4 × 6 grid of patches) and the language
    # model with an image's six tokens among text, turning by mRoPE on its grid
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from PIL import Image
    text = dict(
        vocab_size=320, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, shared_expert_intermediate_size=64, rms_norm_eps=1e-6,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    )
    vision = dict(depth=2, hidden_size=64, num_heads=4, intermediate_size=128, patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                  num_position_embeddings=16, deepstack_visual_indexes=[])
    config = Qwen3_5MoeConfig(text_config=text, vision_config=vision, image_token_id=300,
                              video_token_id=303, vision_start_token_id=301, vision_end_token_id=302,
                              tie_word_embeddings=False)
    config._attn_implementation = "eager"
    model = Qwen3_5MoeForConditionalGeneration(config).eval()
    randomize_qwen35(model.named_parameters())
    with torch.no_grad():
        for name, parameter in model.model.visual.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            elif name.endswith("bias") or "pos_embed" in name:
                parameter.copy_(0.2 * torch.randn_like(parameter))
    folder = out / "qwen35moe_vision"
    folder.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(folder)
    # the language model under ForCausalLM's names, as the runner reads them
    tensors = {}
    for name, tensor in model.state_dict().items():
        name = name.replace("model.language_model.", "model.")
        tensors[name] = tensor.detach().contiguous()
    save_file(tensors, folder / "model.safetensors", metadata={"format": "pt"})

    generator = torch.Generator().manual_seed(1)
    image = torch.randint(0, 256, (64, 96, 3), generator=generator, dtype=torch.uint8)
    processor = Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                         image_mean=[0.5] * 3, image_std=[0.5] * 3, do_resize=False)
    processed = processor(images=[Image.fromarray(image.numpy())], return_tensors="pt")
    pixel_values, grid = processed["pixel_values"], processed["image_grid_thw"]
    # a photo-like image resized as the runner sizes it (64 to 4096 tokens' pixels)
    photo = torch.randint(0, 256, (70, 100, 3), generator=generator, dtype=torch.uint8)
    photo = torch.nn.functional.avg_pool2d(photo.permute(2, 0, 1)[None].float(), 5, 1, 2)[0].permute(1, 2, 0)
    photo = photo.round().to(torch.uint8)
    resizer = Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                       image_mean=[0.5] * 3, image_std=[0.5] * 3,
                                       min_pixels=64 * 32 * 32, max_pixels=4096 * 32 * 32)
    resized = resizer(images=[Image.fromarray(photo.numpy())], return_tensors="pt")

    ids = torch.randint(0, 300, (1, 22), generator=generator)
    ids[0, 5] = 301
    ids[0, 6:12] = 300
    ids[0, 12] = 302
    mm_types = (ids == 300).int()
    with torch.no_grad():
        features = model.model.get_image_features(pixel_values, grid).pooler_output
        features = torch.cat(features) if isinstance(features, (list, tuple)) else features
        positions, _ = model.model.get_rope_index(ids, mm_types, image_grid_thw=grid)
        logits = model(input_ids=ids, pixel_values=pixel_values, image_grid_thw=grid,
                       mm_token_type_ids=mm_types).logits[0].float()
    save_file({
        "ids": ids[0].to(torch.int32),
        "image": image.float().contiguous(),
        "pixel_values": pixel_values.float().contiguous(),
        "features": features.float().contiguous(),
        "positions": positions[:, 0].to(torch.int32).contiguous(),
        "logits": logits.contiguous(),
        "photo": photo.float().contiguous(),
        "photo_pixel_values": resized["pixel_values"].float().contiguous(),
        "photo_grid": resized["image_grid_thw"][0].to(torch.int32),
    }, folder / "expected.safetensors")
    print(f"qwen35moe_vision: features {tuple(features.shape)}, logits {tuple(logits.shape)}, "
          f"photo grid {resized['image_grid_thw'][0].tolist()}")


def qwen3vl_vision():
    # Qwen3-VL as Qwen Image 2.1's text encoder reads a reference: the tower's
    # tokens and its deepstack taps (after both blocks) added to the first two
    # layers' outputs, mRoPE on the image's grid; the last layer's residual
    # stream (before the final norm) is what the encoder gives
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    text = dict(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6,
                max_position_embeddings=512,
                rope_parameters={"rope_type": "default", "rope_theta": 5e6,
                                 "mrope_section": [12, 10, 10], "mrope_interleaved": True})
    vision = dict(depth=2, hidden_size=64, num_heads=4, intermediate_size=128, patch_size=16,
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
    folder = out / "qwen3vl_vision"
    folder.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(folder)
    tensors = {n.replace("model.language_model.", "model."): t.detach().contiguous()
               for n, t in model.state_dict().items()}
    save_file(tensors, folder / "model.safetensors", metadata={"format": "pt"})

    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from PIL import Image
    generator = torch.Generator().manual_seed(2)
    image = torch.randint(0, 256, (64, 96, 3), generator=generator, dtype=torch.uint8)
    processor = Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                         image_mean=[0.5] * 3, image_std=[0.5] * 3, do_resize=False)
    processed = processor(images=[Image.fromarray(image.numpy())], return_tensors="pt")
    pixel_values, grid = processed["pixel_values"], processed["image_grid_thw"]
    ids = torch.randint(0, 300, (1, 16), generator=generator)
    ids[0, 3] = 301
    ids[0, 4:10] = 300
    ids[0, 10] = 302
    mm_types = (ids == 300).int()
    text_model = model.model.language_model
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.no_grad():
            visual = model.model.visual(pixel_values, grid_thw=grid)
            hidden = model(input_ids=ids, pixel_values=pixel_values, image_grid_thw=grid,
                           mm_token_type_ids=mm_types, output_hidden_states=True).hidden_states[-1][0].float()
    finally:
        handle.remove()
    deepstack = visual.deepstack_features if hasattr(visual, "deepstack_features") else visual[1]
    features = visual.pooler_output if hasattr(visual, "pooler_output") else visual[0]
    save_file({
        "ids": ids[0].to(torch.int32),
        "image": image.float().contiguous(),
        "features": features.float().contiguous(),
        "deepstack": torch.stack(list(deepstack)).float().contiguous(),
        "hidden": hidden.contiguous(),
    }, folder / "expected.safetensors")
    print(f"qwen3vl_vision: features {tuple(features.shape)}, deepstack {len(deepstack)}, hidden {tuple(hidden.shape)}")


def randomize_qwen35(parameters):
    with torch.no_grad():
        for name, parameter in parameters:
            if name.endswith("layernorm.weight") or name.endswith("_norm.weight") or name.endswith("norm.weight") \
                    and not name.endswith("linear_attn.norm.weight"):
                parameter.copy_(0.2 * torch.randn_like(parameter))  # stored minus one: (1 + w)
            elif name.endswith("linear_attn.norm.weight"):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            elif name.endswith("dt_bias") or name.endswith("A_log"):
                parameter.copy_(torch.randn_like(parameter))
            elif "experts" in name or "shared_expert" in name or "mlp." in name:
                parameter.copy_(0.1 * torch.randn_like(parameter))


def qwen35_mtp(model, config, ids, weights):
    """The MTP layer transformers leaves out, as vLLM's qwen3_next_mtp runs it:
    entry i joins token i+1's embedding and the model's final hidden state at i
    (each normed) through `fc`, then one full-attention decoder layer, a norm
    and the shared head give the logits of token i+2. Its weights go into the
    model's file under `mtp.` (the experts fused, as the official checkpoints
    hold them); returns every entry's logits."""
    if hasattr(config, "num_experts"):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer as DecoderLayer, Qwen3_5MoeRMSNorm as RMSNorm)
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer as DecoderLayer, Qwen3_5RMSNorm as RMSNorm)
    config._attn_implementation = "eager"
    hidden_size = config.hidden_size
    layer = DecoderLayer(config, layer_idx=3).eval()  # a full-attention layer
    modules = {
        "fc": torch.nn.Linear(2 * hidden_size, hidden_size, bias=False),
        "pre_fc_norm_embedding": RMSNorm(hidden_size, eps=config.rms_norm_eps),
        "pre_fc_norm_hidden": RMSNorm(hidden_size, eps=config.rms_norm_eps),
        "norm": RMSNorm(hidden_size, eps=config.rms_norm_eps),
    }
    randomize_qwen35(layer.named_parameters())
    randomize_qwen35((f"{n}.weight", m.weight) for n, m in modules.items() if "norm" in n)
    with torch.no_grad():
        hidden = model.model(ids).last_hidden_state
        count = ids.shape[1] - 1
        embedded = modules["pre_fc_norm_embedding"](model.model.embed_tokens(ids[:, 1:]))
        x = modules["fc"](torch.cat([embedded, modules["pre_fc_norm_hidden"](hidden[:, :-1])], -1))
        positions = torch.arange(count)
        rotary = model.model.rotary_emb(x, positions.view(1, 1, -1).expand(3, 1, -1))
        mask = torch.full((count, count), float("-inf")).triu(1)[None, None]
        x = layer(x, position_embeddings=rotary, attention_mask=mask, position_ids=positions[None])
        x = x[0] if isinstance(x, tuple) else x
        logits = model.lm_head(modules["norm"](x))[0].float()
    tensors = load_file(weights)
    tensors.update({f"mtp.{n}.weight": m.weight.detach().contiguous() for n, m in modules.items()})
    tensors.update({f"mtp.layers.0.{n}": t.detach().contiguous() for n, t in layer.state_dict().items()})
    save_file(tensors, weights, metadata={"format": "pt"})
    return logits


def qwen4exp():
    """Qwen 3.8 Flash Next's shape: four residual streams through
    hyper-connections, the n-gram embedding (PLE) before layer 1, three
    gated-DeltaNet layers with sigmoid output gates, then one Qwen Sparse
    Attention layer whose indexer keeps 2 blocks of 4 tokens: positions 0–10
    see every token (dense), later ones a selection. The sequence holds EOS
    (7) twice, so the n-gram context is cut."""
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    config = Qwen4ExpTextConfig(
        vocab_size=320, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=3,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, rms_norm_eps=1e-6,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        tie_word_embeddings=False, max_position_embeddings=512, eos_token_id=7,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        hc_count=4, hc_lowrank=32, ple_layer_ids=[2], ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=50,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=2, ple_conv_kernel_size=4,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=8, indexer_compress_ratio=4,
        output_gate_type="sigmoid", torch_dtype="float32",
    )
    config._attn_implementation = "eager"
    model = Qwen4ExpForCausalLM(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "linear_attn.norm" in name:
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            elif "norm" in name:
                parameter.copy_(0.2 * torch.randn_like(parameter))  # stored minus one: (1 + w)
            elif name.endswith("dt_bias") or name.endswith("A_log"):
                parameter.copy_(torch.randn_like(parameter))
            elif "ngram_embedding" in name or "embed_tokens" in name:
                parameter.copy_(torch.randn_like(parameter))
            else:  # every weight random, none left at zero
                parameter.copy_(torch.randn_like(parameter) / parameter.shape[-1] ** 0.5)
    folder = out / "qwen4exp"
    model.save_pretrained(folder, safe_serialization=True)
    ids = torch.randint(8, config.vocab_size, (1, 20))
    ids[0, 5] = 7
    ids[0, 13] = 7
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    mtp_logits = qwen4exp_mtp(model, config, ids, folder / "mtp.safetensors")
    save_file({"ids": ids[0].to(torch.int32), "logits": logits.contiguous(), "mtp_logits": mtp_logits.contiguous()},
              folder / "expected.safetensors")
    print(f"qwen4exp: logits {tuple(logits.shape)}, |max| {logits.abs().max():.3f}; "
          f"MTP logits {tuple(mtp_logits.shape)}, |max| {mtp_logits.abs().max():.3f}")


def qwen4exp_mtp(model, config, ids, sidecar):
    """The MTP head transformers leaves out, as llama.cpp's qwen4exp draft graph
    runs it (PR #28243): entry i takes token i+1's embedding, normed, and the
    model's four streams after its last layer at i, each normed per stream;
    each stream joins the two halves through `eh_proj`; then one decoder layer
    (hyper-connections, attention, experts) and the MTP's own stream mixer give
    the logits of token i+2. Attention is dense everywhere here, the model's
    too, as llama.cpp's draft graph and the runner compute it. The head is
    written as a sidecar with GGUF names and norms folded to (1 + w), as the
    published MTP GGUFs hold it; returns every entry's logits."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual)
    hidden_size, streams = config.hidden_size, config.hc_count
    layer = Qwen4ExpTextDecoderLayer(config, layer_idx=3).eval()  # a full-attention layer
    mixer = Qwen4ExpTextGatedResidual(config, use_combine=False).eval()
    join = torch.nn.Linear(2 * hidden_size, hidden_size, bias=False)
    embedding_norm = 0.2 * torch.randn(hidden_size)
    hidden_norm = 0.2 * torch.randn(streams * hidden_size)
    with torch.no_grad():
        for module in (layer, mixer, join):
            for name, parameter in module.named_parameters():
                if "norm" in name:
                    parameter.copy_(0.2 * torch.randn_like(parameter))
                else:
                    parameter.copy_(torch.randn_like(parameter) / parameter.shape[-1] ** 0.5)

    def dense(*_, **__):  # every token selected
        return 0.0
    for module in [layer] + [l for l in model.model.layers if hasattr(l, "self_attn")]:
        module.self_attn.indexer.forward = dense

    def norm(x, weight, groups=1):
        x = x.view(*x.shape[:-1], groups, -1)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
        return x.flatten(-2) * (1 + weight)

    captured = {}
    hook = model.model.hyper_connection_mixer.register_forward_pre_hook(
        lambda _, inputs: captured.update(streams=inputs[0]))
    with torch.no_grad():
        model(ids)
        hook.remove()
        count = ids.shape[1] - 1
        embedded = norm(model.model.embed_tokens(ids[:, 1:]), embedding_norm)
        normed = norm(captured["streams"][:, :-1], hidden_norm, streams)
        joined = torch.cat([embedded.unsqueeze(-2).expand(-1, -1, streams, -1),
                            normed.view(1, count, streams, hidden_size)], -1)
        x = join(joined).flatten(-2)
        positions = torch.arange(count)
        rotary = model.model.rotary_emb(x, positions.view(1, 1, -1).expand(3, 1, -1))
        mask = torch.full((count, count), float("-inf")).triu(1)[None, None]
        x = layer(x, position_embeddings=rotary, attention_mask=mask)
        logits = model.lm_head(mixer(x))[0].float()

    def folded(tensor):
        return (1 + tensor).contiguous()
    g = f"blk.{config.num_hidden_layers}."
    tensors = {g + "nextn.enorm.weight": folded(embedding_norm), g + "nextn.hnorm.weight": folded(hidden_norm),
               g + "nextn.eh_proj.weight": join.weight.detach().contiguous()}
    for prefix, connection in [(g + "hc_attn", layer.attn_hyper_connection), (g + "hc_ffn", layer.mlp_hyper_connection),
                               ("output_hc", mixer)]:
        tensors[prefix + "_norm.weight"] = folded(connection.hc_norm.weight.detach())
        tensors[prefix + "_down.weight"] = connection.input_mix_weight_down.weight.detach().contiguous()
        tensors[prefix + "_up.weight"] = connection.input_mix_weight_up.weight.detach().contiguous()
        if connection.block_inject_weight is not None:
            tensors[prefix + "_inject.weight"] = connection.block_inject_weight.weight.detach().contiguous()
    attention = layer.self_attn
    for name, module in [("attn_q", attention.q_proj), ("attn_k", attention.k_proj), ("attn_v", attention.v_proj),
                         ("attn_output", attention.o_proj)]:
        tensors[g + name + ".weight"] = module.weight.detach().contiguous()
    tensors[g + "attn_q_norm.weight"] = folded(attention.q_norm.weight.detach())
    tensors[g + "attn_k_norm.weight"] = folded(attention.k_norm.weight.detach())
    mlp = layer.mlp
    gate_up = mlp.experts.gate_up_proj.detach()
    intermediate = gate_up.shape[1] // 2
    tensors[g + "ffn_gate_inp.weight"] = mlp.gate.weight.detach().contiguous()
    tensors[g + "ffn_gate_exps.weight"] = gate_up[:, :intermediate].contiguous()
    tensors[g + "ffn_up_exps.weight"] = gate_up[:, intermediate:].contiguous()
    tensors[g + "ffn_down_exps.weight"] = mlp.experts.down_proj.detach().contiguous()
    tensors[g + "ffn_gate_shexp.weight"] = mlp.shared_expert.gate_proj.weight.detach().contiguous()
    tensors[g + "ffn_up_shexp.weight"] = mlp.shared_expert.up_proj.weight.detach().contiguous()
    tensors[g + "ffn_down_shexp.weight"] = mlp.shared_expert.down_proj.weight.detach().contiguous()
    tensors[g + "ffn_gate_inp_shexp.weight"] = mlp.shared_expert_gate.weight.detach().view(-1).contiguous()
    save_file(tensors, sidecar, metadata={"format": "pt"})
    return logits


families = {"qwen3": qwen3, "qwen35": qwen35, "qwen35moe": qwen35_moe, "qwen35moe_vision": qwen35_moe_vision,
            "qwen3vl_vision": qwen3vl_vision, "qwen4exp": qwen4exp, "gemma2": gemma2,
            "mistral": mistral}
for family in sys.argv[1:] or families:
    families[family]()
