#==================================================================================================
#
#   Copyright (C) 2026-2026 turbo-offloader authors. <https://omega.gg/turbo-offloader>
#
#   Author: Benjamin Arnaud. <https://bunjee.me> <bunjee@omega.gg>
#
#   This file is part of turbo-offloader.
#
#   - GNU General Public License Usage:
#   This file may be used under the terms of the GNU General Public License version 3 as published
#   by the Free Software Foundation and appearing in the LICENSE.md file included in the packaging
#   of this file. Please review the following information to ensure the GNU General Public License
#   requirements will be met: https://www.gnu.org/licenses/gpl.html.
#
#==================================================================================================
#
#   ComfyUI's MiniMax-H3-Image, behind the calls a pipeline makes. The transformer is ComfyUI's own
#   (comfy/ldm/minimax/model.py, vendored), and so are the text encoder and the VAE. What ComfyUI
#   keeps around them is mirrored here, citing its lines (v0.45.4):
#
#     - model_detection.py:401-434, the settings read off the file (detect), the PDD heads' grid
#       from its metadata;
#     - MiniMaxH3ImageModelSampling (ldm/minimax/sampling.py:7-32), the sigmas KSampler's
#       schedulers run on (sampling);
#     - model_base.MiniMaxH3Image and MiniMaxH3.extra_conds (model_base.py:2254-2300,
#       2429-2441), the model ComfyUI samples (MiniMaxH3Image, over adapter.BaseModel);
#     - model_lora_keys_unet's entries for it (lora.py:426-434, lora_keys);
#     - CLIPLoader of type minimax on the Qwen3-VL 32B file (sd.py:1841-1843, 2126-2128,
#       TextEncoder);
#     - MiniMaxH3ImageConditioning (nodes_minimax_h3.py:673-747) on a source the edit template
#       sizes with two ResizeImageMaskNode (nodes_post_processing.py:356-416): TextEncoder.encode
#       and MiniMaxH3Image.edit.
#
#==================================================================================================

import contextlib
import json
import math

import torch

from . import adapter

import comfy.ldm.minimax.model as h3
import comfy.text_encoders.minimax
import comfy.utils

# MiniMaxH3ImageConditioning's sizes (nodes_minimax_h3.py:29, 39-40): Qwen sees a picture at a
# 1024 short edge, the VAE within a 1024 to 2048 short edge, both on the 32 canvas grid.
_GRID = 32
_VISION = (1024, 1024)
_VAE    = (1024, 2048)

# The edit template's two ResizeImageMaskNode on the source: 4 MP, then the 32 grid, by area.
_SOURCE_MEGAPIXELS = 4


def _config(weight_file):
    """The transformer's settings in its file's metadata (model_detection.py:409-411, 432-433):
    the PDD heads' count and grid. The header only, no tensor read."""
    from safetensors import safe_open

    with safe_open(adapter.as_files(weight_file)[0], framework="pt") as f:
        metadata = f.metadata() or {}

    return json.loads(metadata["config"]).get("transformer", {}) if "config" in metadata else {}


def detect(sd, config):
    """model_detection.py:401-434 (no key prefix), the image branch: the transformer's settings
    off its file, then its metadata's `config`. Comfy-Org's file is the curve form (an
    `adaln_t_table`, no time embedder)."""
    pdd_steps = config.get("pdd_steps", 1)
    head_dim = sd["blocks.0.attn.q_norm.weight"].shape[0]

    out = {"image_model": "minimax_h3_image",
           "num_layers": adapter.count_blocks(sd, "blocks.{}."),
           "token_refiner_num_layers": adapter.count_blocks(sd, "token_refiner.blocks.{}."),
           "hidden_size": sd["video_patch_proj.weight"].shape[0],
           "latents_dim": sd["final_layer.video_out.weight"].shape[0] // (4 * pdd_steps),
           "attention_head_dim": head_dim,
           "num_attention_heads": sd["blocks.0.attn.qkv_proj.weight"].shape[0] // (3 * head_dim),
           "ffn_hidden_size": sd["blocks.0.mlp.fc1.weight"].shape[0] // 2,
           "text_dim": sd["condition_proj.weight"].shape[1],
           "rope_inv_freq_len": sd["rope.inv_freq"].shape[0],
           "gate_compress": "blocks.0.attn.to_gate_compress.weight" in sd}

    if "adaln_t_table" in sd:
        out["adaln_curve_grid"], out["time_embed_dim"] = sd["adaln_t_table"].shape
    else:
        time = sd["time_embedder.proj_in.weight"]
        out["timestep_input_dim"] = time.shape[1]
        out["time_embed_hidden_size"] = time.shape[0]
        out["time_embed_dim"] = sd["time_embedder.proj_out.weight"].shape[0]

    out.update(config)
    return out


def sampling(weight_file):
    """MiniMaxH3ImageModelSampling (sampling.py:15-25) for adapter.FlowSampler: flow sampling on
    the file's PDD grid, its sigma linear in the grid index, so KSampler's schedulers at the PDD
    step count land on the grid the heads were distilled for. The multiplier is
    ModelSamplingDiscreteFlow's default."""
    grid = torch.tensor(_config(weight_file)["pdd_sigmas"], dtype=torch.float32).flip(0)
    n = grid.shape[0] - 1
    multiplier = 1000.0

    def sigma(timestep):
        pos = (timestep / multiplier).clamp(0.0, 1.0) * n
        i = pos.floor().long().clamp(max=n - 1)
        return torch.lerp(grid[i], grid[i + 1], pos - i)

    return {"multiplier": multiplier, "sigma": sigma}


def lora_keys(model, key_map):
    """model_lora_keys_unet's MiniMaxH3 entries (lora.py:425-434) over the generic `key_map`:
    bare module paths, and with a baked PDD bank a single-head LoRA on head 0 of each output
    head (video, and audio where there is one), the base of every interval's head."""
    for k, _ in model.named_parameters():
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_map[k[len("diffusion_model."):-len(".weight")]] = k

    final = model.diffusion_model.final_layer
    if final.pdd_steps > 1:
        heads = {"diffusion_model.final_layer.%s.weight" % name:
                 getattr(final, name).weight.shape[0] // final.pdd_steps
                 for name in ("video_out", "audio_out") if getattr(final, name) is not None}
        key_map.update({key: (to, (0, 0, heads[to])) for key, to in key_map.items()
                        if to in heads})
    return key_map


def _upscale(image, width, height, method):
    """common_upscale on an IMAGE, crop disabled."""
    samples = image.movedim(-1, 1)
    return comfy.utils.common_upscale(samples, width, height, method, "disabled").movedim(1, -1)


def _source(image):
    """The edit template's source: LoadImage, then ResizeImageMaskNode's scale_total_pixels to
    4 MP and scale_to_multiple_cover on the 32 grid, both by area
    (nodes_post_processing.py:356-366, 380-416)."""
    image = adapter.load_image(image)

    scale = math.sqrt(int(_SOURCE_MEGAPIXELS * 1024 * 1024) / (image.shape[2] * image.shape[1]))
    image = _upscale(image, round(image.shape[2] * scale), round(image.shape[1] * scale), "area")

    _, height, width, _ = image.shape
    target_w, target_h = width // _GRID * _GRID, height // _GRID * _GRID
    if (target_w, target_h) == (width, height):
        return image

    if target_w / width >= target_h / height:
        scaled_w, scaled_h = target_w, max(target_h, int(math.ceil(height * target_w / width)))
    else:
        scaled_w, scaled_h = max(target_w, int(math.ceil(width * target_h / height))), target_h
    image = _upscale(image, scaled_w, scaled_h, "area")

    x0, y0 = (scaled_w - target_w) // 2, (scaled_h - target_h) // 2
    return image[:, y0:y0 + target_h, x0:x0 + target_w, :]


def _short_edge(image, low, high):
    """_size_short_edge then _resize_to (nodes_minimax_h3.py:679-689): the IMAGE with its short
    edge inside [low, high], each side on the canvas grid, lanczos; untouched at its size."""
    width, height = image.shape[2], image.shape[1]
    scale = min(max(min(width, height), low), high) / min(width, height)
    size = tuple(max(_GRID, round(side * scale / _GRID) * _GRID) for side in (width, height))

    if (width, height) == size:
        return image[..., :3]
    return _upscale(image[..., :3], size[0], size[1], "lanczos")


class MiniMaxH3Image(adapter.BaseModel):
    """model_base.MiniMaxH3Image around ComfyUI's transformer: the (1, 24, 1, h, w) latent, the
    sigma × 1000 as its timestep, and the conditions extra_conds made once for the generation.
    Its forward is _apply_model's (model_base.py:221-263): the latent and conditioning cast to the
    compute dtype, ComfyUI's transformer run on them, its output in float32 as ComfyUI samples
    in."""

    def __init__(self, diffusion_model, config, dtype):
        super().__init__(diffusion_model, config, dtype)
        self._edit = None

    @contextlib.contextmanager
    def edit(self, images, vae):
        """The edit template's sources for one generation, each image sized once: yielded for
        TextEncoder.encode, their references encoded by `vae` in extra_conds, as
        MiniMaxH3ImageConditioning's one pass gives both its copies. No image, no references
        (text to image)."""
        sources = [_source(image) for image in images]
        self._edit = (sources, vae) if sources else None
        try:
            yield sources
        finally:
            self._edit = None

    def extra_conds(self, context, tags, shape, seed, device):
        """MiniMaxH3.extra_conds and MiniMaxH3Image's (model_base.py:2254-2300, 2433-2441) for a
        latent of `shape`: the text through the condition projection and the token refiner once,
        and the payload (the token tags, the references and their latents, KSampler's seed, the
        packed layout). Before sampling, as ComfyUI makes them."""
        # MiniMaxH3ImageConditioning's reference latents (nodes_minimax_h3.py:692-708): each
        # source at a short edge within 1024 to 2048, through ComfyUI's VAE (in [-1, 1], one
        # frame). Encoded after the text encoder and before the transformer, since the VAE sizes
        # its tiles to the free VRAM: this order gives ComfyUI's images exactly.
        refs = None
        if self._edit is not None:
            sources, vae = self._edit
            refs = []
            for source in sources:
                pixels = _short_edge(source, *_VAE).movedim(-1, 1).unsqueeze(2)
                z = vae.encode(pixels * 2.0 - 1.0).latent_dist.mode()
                refs.append({"kind": "image", "latent_h": z.shape[-2], "latent_w": z.shape[-1],
                             "latent": z})

        context = self.diffusion_model.preprocess_text_embeds(
            context.to(device=device, dtype=self._dtype))

        payload = {}
        if tags is not None:
            payload["text_token_tags"] = tags.cpu()
        if refs is not None:
            payload["refs"] = refs
        # _cond_latents (model_base.py:2302-2307): the references' latents, no keyframe or audio
        payload["cond_video_latents"] = [r["latent"] for r in refs or ()]
        payload["cond_audio_latents"] = []
        payload["seed"] = seed
        payload["audio_scale"] = 1.0
        payload["layout"] = h3.PackedLayout(context.shape[1], shape[2], (shape[3] + 1) // 2 * 2,
                                            (shape[4] + 1) // 2 * 2, 0, refs=payload.get("refs"))

        return {"context": context, "minimax_payload": payload}

    def forward(self, hidden_states, timestep, context, minimax_payload, sample_sigmas,
                return_dict=False):
        options = self.transformer_options()
        options["sample_sigmas"] = sample_sigmas  # samplers.py: the PDD heads read the schedule

        t = timestep.float().reshape(-1).expand(hidden_states.shape[0])

        out = self.diffusion_model(hidden_states.to(self._dtype), t, context.to(self._dtype),
                                   transformer_options=options, minimax_payload=minimax_payload)

        return (out.float(),)

    def lora_keys(self):
        return lora_keys(self, super().lora_keys())


class TextEncoder(adapter.CLIP):
    """CLIPLoader of type minimax on ComfyUI's Qwen3-VL 32B file (sd.py:1841-1843, 2126-2128),
    as its own state dict names it; `encode` is MiniMaxH3ImageConditioning's
    (nodes_minimax_h3.py:736-739)."""

    def __init__(self, path, dtype=None):
        sd = adapter.load_clip_sd(path)

        super().__init__(comfy.text_encoders.minimax.te(**adapter.llama_detect(sd)),
                         comfy.text_encoders.minimax.MiniMaxH3Tokenizer, sd, dtype)

    def encode(self, text, sources=()):
        """The conditioning and its token tags: each of MiniMaxH3Image.edit's sources seen at a
        1024 short edge, labeled "Picture n: " before the prompt."""
        vision = [_short_edge(source, *_VISION) for source in sources]

        cond, extra = self.encode_from_tokens(
            self.tokenize(text, images=vision, picture_label="Picture %d: "))

        return cond, extra.get("minimax_token_tags")


def load(weight_file, config, load_device, dtype, **options):
    """ComfyUI's transformer for its file, behind the call the pipeline makes."""
    file_config = _config(weight_file)

    def build(sd, operations):
        return h3.MiniMaxH3ImageModel(**detect(sd, file_config), dtype=dtype,
                                      operations=operations)

    return MiniMaxH3Image.load(build, weight_file, config, load_device, dtype, **options)
