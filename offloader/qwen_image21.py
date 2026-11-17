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
#   ComfyUI's Qwen-Image 2.1, behind the call diffusers' QwenImage21Pipeline makes. The
#   transformer is ComfyUI's own (comfy/ldm/qwen_image21/model.py, vendored), so its fused kernels,
#   its per-block prefetch and its prefix K/V cache come with each re-sync. What ComfyUI keeps
#   around it is mirrored here, citing its lines (v0.45.4):
#
#     - model_detection.py:991-1010, the settings read off the file (detect);
#     - model_base.QwenImage21 (model_base.py:2825-2879), the model ComfyUI samples
#       (QwenImage21, over qwen_image.QwenImage as ComfyUI's is, with its LoRA keys);
#     - ModelPatcher.writable_models around sampling (samplers.py:1237,
#       model_patcher.py:1408-1417), opened on a run's first forward, once its conditioning is
#       known.
#
#==================================================================================================

import contextlib

import torch

from . import adapter, qwen_image

import comfy.model_management as mm
import comfy.ldm.qwen_image21.model as qwen_image21
import comfy.text_encoders.qwen_image21


def detect(sd):
    """model_detection.py:991-1010 (no key prefix): the transformer's settings off its file."""
    head_dim = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
    inner_dim = sd["img_in.weight"].shape[0]
    gate_up = sd.get("transformer_blocks.0.img_mlp.gate_up.weight", None)

    num_layers = adapter.count_blocks(sd, "transformer_blocks.{}.")

    if gate_up is not None:
        mlp_ratio = gate_up.shape[0] // 2 // inner_dim
    else:
        mlp_ratio = sd["transformer_blocks.0.img_mlp.proj.weight"].shape[0] // inner_dim

    return {"in_channels": sd["img_in.weight"].shape[1],
            "out_channels": sd["proj_out.weight"].shape[0],
            "num_layers": num_layers,
            "attention_head_dim": head_dim,
            "num_attention_heads": inner_dim // head_dim,
            "context_in_dim": sd["txt_in.text_norm.weight"].shape[0],
            "mlp_ratio": mlp_ratio,
            "fused_mlp": gate_up is not None}


class QwenImage21(qwen_image.QwenImage):
    """model_base.QwenImage21 around ComfyUI's transformer, called the way diffusers'
    QwenImage21Pipeline calls its own: packed latent tokens (the references, then the target) and
    the text encoder's output with a slot per reference image. It answers what the pipeline reads
    off its transformer (transformer_blocks, cache_context), and its forward is _apply_model's
    (model_base.py:221-263): the latents and conditioning cast to the compute dtype, ComfyUI's
    transformer run on them, its output in float32 as ComfyUI samples in. The pipeline's own K/V
    cache goes unused: ComfyUI's transformer keeps its prefix itself."""

    def __init__(self, diffusion_model, config, dtype):
        super().__init__(diffusion_model, config, dtype)
        self._current_patcher = None
        self._context = "cond"
        self._writable = {}  # what writable_models opened this run, by cache context

    @property
    def transformer_blocks(self):
        return self.diffusion_model.transformer_blocks

    def get_dynamic_vram__units(self):
        # model_base.py:2829-2830
        return list(self.diffusion_model.transformer_blocks), []

    @property
    def current_patcher(self):
        return self._current_patcher

    @current_patcher.setter
    def current_patcher(self, patcher):
        # model_base.py:2836-2843: the prefix K/V cache lives for one sampling run. A run's end
        # closes what writable_models opened first, as samplers.py:1237 exits before cleanup.
        for _cache, stack in self._writable.values():
            stack.close()
        self._writable = {}

        self._current_patcher = patcher
        self.diffusion_model.current_patcher = patcher
        self.diffusion_model.reset_prefix_cache(patcher is not None
                                                and len(patcher.hook_patches) == 0)

    @contextlib.contextmanager
    def cache_context(self, name):
        # The pipeline names the cond it runs ("cond", "uncond"), each with its own prefix.
        self._context = name
        yield

    def _get_writable(self, context, ref_latents, batch, device, transformer_options):
        """get_writable_model (model_base.py:2845-2872) for the cond of this cache context, entered
        as writable_models does (model_patcher.py:1408-1417) and kept until the run ends."""
        if self._context in self._writable:
            return self._writable[self._context][0]

        patcher = self._current_patcher
        if patcher is None or not patcher.is_dynamic():
            return None

        options = transformer_options.get("qwen_image21_cache", {})
        model = self.diffusion_model
        if (options.get("device", "auto") != "auto"
            or
            not model.can_cache_prefix(transformer_options)):
            return None

        heads = model.transformer_blocks[0].attn.heads
        length = context.shape[1] + sum(t.shape[-2] * t.shape[-1] for t in ref_latents)
        shape = (batch, 2, length, heads, model.inner_dim // heads)
        slot = qwen_image21.PrefixCacheSlot(shape, len(model.transformer_blocks), self._dtype,
                                            options.get("dtype", "default"))

        cache = qwen_image21.KVCacheModel([slot])
        cache_bytes = sum(p.numel() * p.element_size() for p in cache.parameters())
        if not (patcher.get_free_memory(device) > 4 * cache_bytes
                or mm.ensure_pin_budget(cache_bytes, evict_active=False)):
            return None

        stack = contextlib.ExitStack()
        with mm.cuda_device_context(patcher.load_device):
            stack.enter_context(patcher.writable_model(cache))

        # CONDKVCache.concat (conds.py:127-128): one slot per cond, as a list
        prefix_cache = {device: [slot]}
        self._writable[self._context] = (prefix_cache, stack)
        return prefix_cache

    def forward(self, hidden_states, timestep, encoder_hidden_states, img_shapes=None,
                img_mask=None, **kwargs):
        dtype = self._dtype
        batch = hidden_states.shape[0]

        # The packed tokens are ComfyUI's latent, row by row: a (1, h, w) grid per image, the
        # references first and the target last.
        latents = []
        for (_, h, w), t in zip(img_shapes[0],
                                hidden_states.split([h * w for _, h, w in img_shapes[0]], dim=1)):
            latents.append(t.transpose(1, 2).reshape(batch, -1, h, w).to(dtype))
        x = latents.pop()

        # TEModel.encode_token_weights (text_encoders/qwen_image21.py:62-76): a reference's vision
        # tokens leave the text, a slot recording how much text comes before each image. Each slot
        # of the pipeline's mask stands for 2x2 latent tokens.
        context, image_slots = encoder_hidden_states, []
        if latents:
            vision = img_mask[0][:context.shape[1]]
            positions = vision.nonzero().flatten().tolist()
            start = 0
            for ref in latents:
                image_slots.append(positions[start] - start)
                start += ref.shape[-2] * ref.shape[-1] // 4
            context = context[:, ~vision]

        context = context.to(dtype)

        transformer_options = self.transformer_options()
        prefix_cache = self._get_writable(context, latents, batch, hidden_states.device,
                                          transformer_options)

        out = self.diffusion_model(x, timestep.float(), context=context, ref_latents=latents,
                                   image_slots=image_slots,
                                   transformer_options=transformer_options,
                                   prefix_cache=prefix_cache)

        return (out.float().flatten(2).transpose(1, 2),)


class TextEncoder(adapter.CLIP):
    """CLIPLoader of type qwen_image on ComfyUI's Qwen3-VL-8B file (sd.py:2108-2111), and the
    encode TextEncodeQwenImage21 runs (nodes_qwen.py:144-180)."""

    def __init__(self, path):
        sd = adapter.qwen3vl_sd(adapter.load_clip_sd(path))

        super().__init__(comfy.text_encoders.qwen_image21.te(**adapter.llama_detect(sd)),
                         comfy.text_encoders.qwen_image21.QwenImage21Tokenizer, sd)

    def encode(self, prompt, images=None):
        """TextEncodeQwenImage21's positive for `prompt` and its reference images, at their size
        on the 32 grid, as the pipeline takes it: (embeds, mask, image_pad_mask), each
        reference's vision tokens back in the text as rows of zeros, where its latents go
        (QwenImage21.forward takes them out again)."""
        images = [adapter.load_image(image) for image in images or []]

        tokens = self.tokenize(prompt, images=images, keep_vision=not images,
                               prevent_empty_text=True)
        cond, extra = self.encode_from_tokens(tokens)

        # A vision token per 32x32 pixels (16 pixel patches, merged 2x2).
        parts, pad, start = [], [], 0
        for image, slot in zip(images, extra.get("image_slots", [])):
            n = (image.shape[1] // 32) * (image.shape[2] // 32)
            parts += [cond[:, start:slot], cond.new_zeros(cond.shape[0], n, cond.shape[2])]
            pad += [False] * (slot - start) + [True] * n
            start = slot
        parts.append(cond[:, start:])
        pad += [False] * (cond.shape[1] - start)

        image_pad_mask = torch.tensor(pad, dtype=torch.bool)[None].expand(cond.shape[0], -1)
        return torch.cat(parts, dim=1), extra.get("attention_mask"), image_pad_mask


def load(weight_file, config, load_device, dtype, **options):
    """ComfyUI's transformer for its file, behind the call diffusers' pipeline makes."""

    def build(sd, operations):
        return qwen_image21.QwenImage21Transformer2DModel(**detect(sd), dtype=dtype,
                                                          operations=operations)

    return QwenImage21.load(build, weight_file, config, load_device, dtype, **options)
