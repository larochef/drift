"""Golden tiny MiniMax H3 cases for references (ref2va) and the Fun ControlNet
union (specs/42, step 14): the ref2va presentation (token ids from Qwen3's
tokenizer, the hidden state of a tiny Qwen3-VL reading an image and two clips),
the ref2va layout with its draws and one step, and one step under the
ControlNet with and without a mask. diffusers 0.40 for ref2va, ComfyUI for the
ControlNet (its patch and control model lifted from their files: the modules'
imports pull ComfyUI's whole runtime).

    uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        --with torch --with diffusers==0.40.0 --with transformers==5.17.0 --with safetensors \\
        --with pillow --with torchvision \\
        python runner/fixtures/tiny_minimax_h3_references.py [case ...]

Outputs are committed under runner/test/resources/fixtures/tiny/minimax_h3_ref2va_* and minimax_h3_control.
"""

import ast
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_diffusion import minimax_h3_original, out  # noqa: E402
from tiny_minimax_h3 import tiny_transformer  # noqa: E402

QWEN3_TOKENIZER = next((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots").glob("*/tokenizer.json"))
COMFYUI = Path("/opt/comfyui")

# the tiny Qwen3-VL's special ids, as `tiny_minimax_h3.presentation` has them
TINY_IDS = {"<|image_pad|>": 300, "<|vision_start|>": 301, "<|vision_end|>": 302, "<|video_pad|>": 303}


def lifted(path, wanted, namespace, methods=()):
    """Functions, classes and constants of `path` by name, and methods of its
    classes (`(class, method)`) as plain functions, compiled into `namespace`."""
    tree = ast.parse(path.read_text())
    kept = []
    for node in tree.body:
        names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else \
            [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        if set(names) & set(wanted):
            kept.append(node)
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and (node.name, method.name) in methods:
                    kept.append(method)
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


# ---- ref2va: the presentation ---------------------------------------------------------------------------------

def ref2va_presentation():
    """diffusers' ref2va presentation (`MiniMaxH3Ref2VATextEncoderStep`) of a
    clip with its soundtrack (30 frames of 64 × 96: read at 2 fps, frames 0, 12
    and 24, two blocks at "0.2" and "1.0" seconds), an image (64 × 64), a sound
    and a clip without one (26 frames of 32 × 64), then the prompt: the token
    ids from Qwen3's tokenizer, the vision blocks from transformers' processors
    (the video's under Qwen3-VL's bounds: 4096 to 25165824 pixels over the
    read frames); then the hidden state of a tiny Qwen3-VL (the tower's
    two-frame patches unfolded) over the same presentation, its ids folded
    into the tiny vocabulary. And the video `smart_resize` on a few shapes."""
    from PIL import Image
    from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor, smart_resize
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(QWEN3_TOKENIZER))
    generator = torch.Generator().manual_seed(81)

    def pixels(*shape):
        return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8).numpy()

    clip, image, still_clip = pixels(30, 64, 96, 3), pixels(64, 64, 3), pixels(26, 32, 64, 3)
    sound = torch.zeros(2, 3200)
    references = [MiniMaxH3VideoReference(frames=clip, fps=24.0, audio=sound, sample_rate=32000),
                  MiniMaxH3ImageReference(image=Image.fromarray(image)),
                  MiniMaxH3AudioReference(audio=sound, sample_rate=32000),
                  MiniMaxH3VideoReference(frames=still_clip, fps=24.0)]
    prompt = "A cat from <Picture 1> walks as in <Video 1>, purring like <Audio 2>; then <Video 2>."

    image_processor = Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                               image_mean=[0.5] * 3, image_std=[0.5] * 3, do_resize=False)
    video_processor = Qwen3VLVideoProcessor(size={"shortest_edge": 4096, "longest_edge": 25165824},
                                            cap_pixels_per_frame=False)
    images = image_processor(images=[Image.fromarray(image)], return_tensors="pt")
    sampled = [MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(frames, 24.0, 2.0, 2)
               for frames in (clip, still_clip)]
    videos = video_processor(videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False,
                             return_tensors="pt")
    image_counts = [int(g.prod()) // 4 for g in images["image_grid_thw"]]
    video_counts = [int(g[1]) * int(g[2]) // 4 for g in videos["video_grid_thw"]]
    ids, tags = MiniMaxH3Ref2VATextEncoderStep._build_presentation(
        tokenizer, prompt, references, image_counts, video_counts, [times for _, times in sampled])
    print("  " + tokenizer.decode(ids)[:400].replace("<|image_pad|>", "").replace("<|video_pad|>", ""))

    # the tiny Qwen3-VL of `tiny_minimax_h3.presentation`
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
    folder = out / "minimax_h3_ref2va_presentation"
    folder.mkdir(parents=True, exist_ok=True)
    save_file({n: t.detach().contiguous() for n, t in tensors.items()}, folder / "model.safetensors")

    special = {tokenizer.convert_tokens_to_ids(token): tiny for token, tiny in TINY_IDS.items()}
    tiny_ids = torch.tensor([[special.get(i, i % 300) for i in ids]])
    types_ = torch.zeros_like(tiny_ids)
    types_[tiny_ids == 300], types_[tiny_ids == 303] = 1, 2
    handle = model.model.language_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.no_grad():
            hidden = model(input_ids=tiny_ids, pixel_values=images["pixel_values"],
                           image_grid_thw=images["image_grid_thw"],
                           pixel_values_videos=videos["pixel_values_videos"],
                           video_grid_thw=videos["video_grid_thw"], mm_token_type_ids=types_,
                           output_hidden_states=True).hidden_states[-1][0]
    finally:
        handle.remove()

    shapes = [(10, 768, 1344), (30, 768, 1344), (3, 64, 96), (2, 40, 40), (7, 720, 1280), (31, 1080, 1920)]
    resized = [smart_resize(n, h, w, 2, 32, 4096, 25165824) for n, h, w in shapes]
    save_file({"ids": torch.tensor(ids, dtype=torch.int32), "tags": torch.tensor(tags, dtype=torch.int32),
               "clip": torch.from_numpy(clip).float().contiguous(), "image": torch.from_numpy(image).float(),
               "still_clip": torch.from_numpy(still_clip).float().contiguous(),
               "hidden": hidden.float().contiguous(),
               "shapes": torch.tensor(shapes, dtype=torch.int32),
               "resized": torch.tensor(resized, dtype=torch.int32)},
              folder / "expected.safetensors", metadata={"prompt": prompt})
    print(f"minimax_h3_ref2va_presentation: {len(ids)} tokens, video grids {videos['video_grid_thw'].tolist()}, "
          f"hidden |max| {hidden.abs().max():.3f}, resized {resized}")


# ---- ref2va: the normalization ---------------------------------------------------------------------------------

def ref2va_normalization():
    """diffusers' ref2va normalization (`MiniMaxH3Ref2VASetupStep`): clips
    resampled onto 24 fps (each frame's pixels its index, on a 32 × 32 canvas
    so nothing is resized) from 30, 12, 25 and 24 fps, cut to 22 frames; the
    canvas a clip's aspect resolves to (`resolve_canvas_size`); PIL's LANCZOS,
    up and down, on a 30 × 50 image."""
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    timed = {}
    for rate, count in ((30, 37), (12, 10), (25, 50), (24, 9)):
        frames = np.repeat(np.arange(count, dtype=np.uint8)[:, None, None, None], 32, 1).repeat(32, 2).repeat(3, 3)
        normalized = MiniMaxH3Ref2VASetupStep._normalize_video_condition(frames, float(rate), 22, 32, 32, 32 * 32, 24.0)
        timed[f"timed_{rate}"] = torch.from_numpy(normalized[:, 0, 0, 0].astype(np.int32))
    aspects = [(1920, 1080), (1080, 1920), (1000, 1000), (640, 480), (2560, 1080), (480, 1900), (720, 1280)]
    canvases = [resolve_canvas_size(w, h, 32, 768, 768 * 1344)[::-1] for w, h in aspects]
    generator = torch.Generator().manual_seed(111)
    image = torch.randint(0, 256, (30, 50, 3), generator=generator, dtype=torch.uint8).numpy()
    up = np.asarray(Image.fromarray(image).resize((64, 96), Image.Resampling.LANCZOS))
    down = np.asarray(Image.fromarray(image).resize((24, 16), Image.Resampling.LANCZOS))
    folder = out / "minimax_h3_ref2va_normalization"
    folder.mkdir(parents=True, exist_ok=True)
    save_file({**timed, "aspects": torch.tensor(aspects, dtype=torch.int32),
               "canvases": torch.tensor(canvases, dtype=torch.int32),
               "image": torch.from_numpy(image).float().contiguous(), "up": torch.from_numpy(up).float().contiguous(),
               "down": torch.from_numpy(down).float().contiguous()},
              folder / "expected.safetensors")
    print(f"minimax_h3_ref2va_normalization: {[(k, v.tolist()) for k, v in timed.items()][:1]}, canvases {canvases}")


# ---- ref2va: the layout, the draws, one step -------------------------------------------------------------------

def ref2va_layout():
    """diffusers' ref2va layout (`build_ref2va_packed_sequence`) of a clip
    with its soundtrack (7 latent frames of 4 × 6, 9 audio latents), an image
    (6 × 4), a sound (5 latents) and a clip without one (2 latent frames of
    4 × 8), the draws of seed 42 in its order (each visual reference's noise
    mixed at t = 0.999, the video's noise as a latent tensor, the audio's
    rows), then one step of the tiny transformer over 2 latent frames of 4 × 8
    and 6 audio latents, the references held at max(t, 0.999) and 1."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3Ref2VAPrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents)
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference)
    model = tiny_transformer(5)
    torch.manual_seed(91)
    channels, frames, latent_h, latent_w, audio_latents = 8, 2, 4, 8, 6
    blank = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    references = [MiniMaxH3VideoReference(frames=blank, fps=24.0, audio=torch.zeros(2, 10), sample_rate=32000),
                  MiniMaxH3ImageReference(image=np.zeros((32, 32, 3), dtype=np.uint8)),
                  MiniMaxH3AudioReference(audio=torch.zeros(2, 10), sample_rate=32000),
                  MiniMaxH3VideoReference(frames=blank, fps=24.0)]
    visual = [torch.randn(1, channels, 7, 4, 6), torch.randn(1, channels, 1, 6, 4),
              torch.randn(1, channels, 2, 4, 8)]
    sounds = [torch.randn(2 * 9, 32), torch.randn(2 * 5, 32)]
    tags = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    text = torch.randn(tags.shape[0], 32)
    positions, token_tags, video_indices, audio_indices, text_indices, video_conditions, audio_conditions = \
        MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            tags, references, visual, sounds, frames, latent_h, latent_w, audio_latents, (1, 2, 2), 2, 2, 0)
    aug = 0.999
    generator = torch.Generator().manual_seed(42)
    condition_rows = []
    for condition in visual:
        noise = torch.randn(condition.shape, generator=generator, dtype=torch.float32)
        condition_rows.append(patchify_video_latents(aug * condition + (1 - aug) * noise, (1, 2, 2)))
    video_noise = torch.randn((1, channels, frames, latent_h, latent_w), generator=generator, dtype=torch.float32)
    video_rows = patchify_video_latents(video_noise, (1, 2, 2))
    audio_rows = torch.randn((2 * audio_latents, 32), generator=generator, dtype=torch.float32)
    video_t, audio_t = 0.3, 0.55
    timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, video_conditions, audio_conditions, tags.shape[0], video_t, audio_t,
        max(video_t, aug), 1.0)
    all_video = torch.cat(condition_rows + [video_rows])
    all_audio = torch.cat(sounds + [audio_rows])
    with torch.no_grad():
        video, audio = model(all_video[None], all_audio[None], text[None], timesteps, timestep_indices, token_tags,
                             positions.float(), video_indices, audio_indices, text_indices, return_dict=False)
    folder = out / "minimax_h3_ref2va_layout"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(minimax_h3_original(model), folder / "model.safetensors")
    save_file({"tags": tags.int(), "text": text, "positions": positions.float().contiguous(),
               **{f"visual_{i}": v[0].permute(1, 2, 3, 0).reshape(1, -1).contiguous() for i, v in enumerate(visual)},
               **{f"sound_{i}": s for i, s in enumerate(sounds)},
               "condition_rows": torch.cat(condition_rows).contiguous(), "video_rows": video_rows.contiguous(),
               "audio_rows": audio_rows.contiguous(), "timesteps": torch.tensor([video_t, audio_t]),
               "video": video[0, video_conditions:].contiguous(), "audio": audio[0, audio_conditions:].contiguous()},
              folder / "expected.safetensors")
    print(f"minimax_h3_ref2va_layout: {positions.shape[0]} rows ({video_conditions} reference video rows, "
          f"{audio_conditions} reference audio rows), origin {positions[-1, 0]:.4f}, video |max| {video.abs().max():.3f}")


# ---- the Fun ControlNet union -------------------------------------------------------------------------------------

def comfyui_control():
    """ComfyUI's ControlNet patch (`MiniMaxH3FunControlPatch`'s
    `prepare_control_latent`, `before_block` and `after_block`, the control
    model's `init_stream`) and `common_upscale`, lifted from their files with
    their imports stubbed: no models are loaded (`load_models_gpu` does
    nothing), `pad_to_patch_size` keeps even shapes as they are."""
    namespace = {"torch": torch, "math": math, "F": F}
    lifted(COMFYUI / "comfy/utils.py", {"common_upscale"}, namespace)
    lifted(COMFYUI / "comfy/ldm/minimax/model.py", {"patchify_video"}, namespace)
    namespace["comfy"] = types.SimpleNamespace(
        utils=types.SimpleNamespace(common_upscale=namespace["common_upscale"]),
        model_management=types.SimpleNamespace(loaded_models=lambda only_currently_used=False: [],
                                               load_models_gpu=lambda models: None),
        ldm=types.SimpleNamespace(common_dit=types.SimpleNamespace(pad_to_patch_size=lambda x, patch: x)))
    lifted(COMFYUI / "comfy_extras/nodes_minimax_h3.py", set(), namespace,
           {("MiniMaxH3FunControlPatch", m) for m in
            ("prepare_control_latent", "_fit_frames", "_encode", "before_block", "after_block")})
    lifted(COMFYUI / "comfy/ldm/minimax/controlnet.py", set(), namespace, {("MiniMaxH3FunControl", "init_stream")})
    return namespace


def control():
    """One step of the tiny transformer under a tiny Fun ControlNet union as
    ComfyUI patches it: two control blocks at blocks 0 and 1 (the
    `control_blocks_places` metadata), shaped as the transformer's (diffusers'
    block runs them: ComfyUI's `ControlDiTBlock` is a `DiTBlock`), over the
    fl2va layout of a first keyframe, 7 latent frames of 4 × 8 (22 frames on a
    128 × 64 canvas) and 4 audio latents. The control: 7 frames of 40 × 72
    fitted onto the canvas (cropped, bilinear) and encoded (the encoder stubbed
    by seeded latents, its input frames recorded); with a mask (3 frames of 48
    × 96, 1 regenerates) and 10 source frames: the visibility on the latents'
    grid and the hidden source's latents after the hint. At strength 0.7, with
    and without the mask."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents)
    comfy = comfyui_control()
    model = tiny_transformer(6)
    net = tiny_transformer(7)
    torch.manual_seed(101)
    hidden = 64
    before = torch.nn.Linear(hidden, hidden)
    afters = [torch.nn.Linear(hidden, hidden) for _ in range(2)]
    project_in = torch.nn.Linear(49 * 4, hidden)
    for layer in [before, project_in, *afters]:
        with torch.no_grad():
            layer.weight.copy_(torch.randn_like(layer.weight) / layer.weight.shape[1] ** 0.5)
            layer.bias.copy_(0.3 * torch.randn_like(layer.bias))
    channels, frames, latent_h, latent_w, audio_latents, width, height = 8, 7, 4, 8, 4, 128, 64
    tags = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    text = torch.randn(tags.shape[0], 32)
    keyframe_rows = torch.randn(latent_h // 2 * latent_w // 2, 4 * channels)
    video_rows = torch.randn(frames * (latent_h // 2) * (latent_w // 2), 4 * channels)
    audio_rows = torch.randn(2 * audio_latents, 32)
    positions, token_tags, video_indices, audio_indices, text_indices, conditions, _ = \
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(tags, frames, latent_h, latent_w, audio_latents,
                                                          (1, 2, 2), 2, 2, 0, ("first",))
    video_t, audio_t = 0.3, 0.55
    timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, conditions, 0, tags.shape[0], video_t, audio_t, max(video_t, 0.999), 1.0)
    generator = torch.Generator().manual_seed(102)

    def pixels(*shape):
        return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)

    control_frames = pixels(7, 40, 72, 3)
    mask_frames = pixels(3, 48, 96)
    source_frames = pixels(10, 48, 96, 3)
    target_shape = (1, channels, frames, latent_h, latent_w)
    encoded = [torch.randn(target_shape), torch.randn(target_shape)]

    class Encoder:
        """The VAE, stubbed: the next seeded latents, its input recorded."""
        def __init__(self):
            self.inputs = []

        def spacial_compression_encode(self):
            return 16

        def encode(self, frames):
            self.inputs.append(frames.clone())
            return encoded[len(self.inputs) - 1].clone()

    layout = types.SimpleNamespace(img_pos=video_indices, audio_pos=audio_indices,
                                   img_update=torch.arange(video_indices.shape[0]) >= conditions)

    def step(index, c, t_emb, mod_segments, rope_freqs, transformer_options):
        # ComfyUI's `MiniMaxH3FunControl.step`, the block run by diffusers' (the same maths)
        temb, adaln_indices, rotary = t_emb
        c = net.transformer_blocks[index](c[None], temb, adaln_indices, rotary)[0]
        return c, afters[index](c)

    control_model = types.SimpleNamespace(
        injection_layers=(0, 1), control_in_dim=49, patch_size=(1, 2, 2), control_proj_in=project_in,
        control_blocks=[types.SimpleNamespace(before_proj=before, adaln_proj=types.SimpleNamespace(
            linear=types.SimpleNamespace(in_features=32)))], step=step)
    control_model.init_stream = lambda h, latent, layout_, t_emb: comfy["init_stream"](
        control_model, h, latent, layout_, t_emb[0])

    def patched(mask):
        encoder = Encoder()
        patch = types.SimpleNamespace(
            model_patch=types.SimpleNamespace(model=control_model), vae=encoder,
            control_video=control_frames.float().div(255).movedim(-1, 1),
            mask=mask_frames.float().div(255) if mask else None,
            source_video=source_frames.float().div(255).movedim(-1, 1) if mask else None,
            strength=0.7, control_latent=None, control_latent_shape=None, control_stream=None,
            pristine_stream=None, active=True)
        for name in ("prepare_control_latent", "_fit_frames", "_encode", "before_block", "after_block"):
            setattr(patch, name, types.MethodType(comfy[name], patch))
        patch.prepare_control_latent(target_shape)
        blocks = model.transformer_blocks
        originals = list(blocks)

        class Patched(torch.nn.Module):
            def __init__(self, index, block):
                super().__init__()
                self.index, self.block = index, block

            def forward(self, h, temb, adaln_indices, rotary):
                # ComfyUI's stream has no batch axis
                args = {"img": h[0], "layout": layout, "t_emb": (temb, adaln_indices, rotary), "mod_segments": None,
                        "rope_freqs": None, "transformer_options": {}}
                patch.before_block(self.index, args)
                result = {"img": self.block(h, temb, adaln_indices, rotary)[0]}
                return patch.after_block(self.index, args, result)["img"][None]

        try:
            for i in range(len(originals)):
                blocks[i] = Patched(i, originals[i])
            with torch.no_grad():
                video, audio = model(torch.cat([keyframe_rows, video_rows])[None], audio_rows[None], text[None],
                                     timesteps, timestep_indices, token_tags, positions.float(), video_indices,
                                     audio_indices, text_indices, return_dict=False)
        finally:
            for i, block in enumerate(originals):
                blocks[i] = block
        return video[0, conditions:], audio[0], patch.control_latent, encoder.inputs

    plain_video, plain_audio, plain_latent, plain_inputs = patched(mask=False)
    masked_video, masked_audio, masked_latent, masked_inputs = patched(mask=True)
    with torch.no_grad():
        free_video, _ = model(torch.cat([keyframe_rows, video_rows])[None], audio_rows[None], text[None], timesteps,
                              timestep_indices, token_tags, positions.float(), video_indices, audio_indices,
                              text_indices, return_dict=False)

    folder = out / "minimax_h3_control"
    folder.mkdir(parents=True, exist_ok=True)
    save_file(minimax_h3_original(model), folder / "model.safetensors")
    controlnet = {}
    for name, value in minimax_h3_original(net).items():
        if name.startswith("blocks."):
            controlnet["control_" + name] = value
    controlnet["control_blocks.0.before_proj.weight"] = before.weight.detach()
    controlnet["control_blocks.0.before_proj.bias"] = before.bias.detach()
    for i, after in enumerate(afters):
        controlnet[f"control_blocks.{i}.after_proj.weight"] = after.weight.detach()
        controlnet[f"control_blocks.{i}.after_proj.bias"] = after.bias.detach()
    controlnet["control_proj_in.weight"] = project_in.weight.detach()
    controlnet["control_proj_in.bias"] = project_in.bias.detach()
    save_file({n: v.contiguous() for n, v in controlnet.items()}, folder / "controlnet.safetensors",
              metadata={"control_blocks_places": json.dumps([0, 1])})

    def channels_last(latent):
        return latent[0].permute(1, 2, 3, 0).contiguous()

    save_file({"text": text, "keyframe_rows": keyframe_rows, "video_rows": video_rows, "audio_rows": audio_rows,
               "timesteps": torch.tensor([video_t, audio_t]),
               "control_frames": control_frames.float().contiguous(), "mask_frames": mask_frames.float().contiguous(),
               "source_frames": source_frames.float().contiguous(),
               "encoded_0": channels_last(encoded[0]), "encoded_1": channels_last(encoded[1]),
               "plain_latent": channels_last(plain_latent), "masked_latent": channels_last(masked_latent),
               "plain_input": plain_inputs[0].contiguous(), "masked_input": masked_inputs[1].contiguous(),
               "plain_video": plain_video.contiguous(), "plain_audio": plain_audio.contiguous(),
               "masked_video": masked_video.contiguous(), "masked_audio": masked_audio.contiguous()},
              folder / "expected.safetensors")
    print(f"minimax_h3_control: control moves the video by {(plain_video - free_video[0, conditions:]).abs().max():.3f}"
          f" (mask: {(masked_video - plain_video).abs().max():.3f}), audio |max| {plain_audio.abs().max():.3f}")


cases = {"ref2va_presentation": ref2va_presentation, "ref2va_normalization": ref2va_normalization,
         "ref2va_layout": ref2va_layout, "control": control}

if __name__ == "__main__":
    for case in sys.argv[1:] or cases:
        cases[case]()
