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
#   ComfyUI's FLUX.3, behind the call diffusers' FLUX.3 modular pipeline makes. The transformer is
#   ComfyUI's own (comfy/ldm/flux3/model.py, vendored), so its fused kernels and its per-block
#   prefetch come with each re-sync. What ComfyUI keeps around it is mirrored here, citing its
#   lines (v0.45.4):
#
#     - model_detection.py:263-272, the settings read off the file (detect);
#     - Flux.process_unet_state_dict, supported_models.py:756-763 (flux2.process);
#     - model_base.Flux3 (model_base.py:1129-1159), the model ComfyUI samples (Flux3, over
#       adapter.BaseModel);
#     - model_lora_keys_unet's entries for it (lora.py:352-358, lora_keys).
#
#==================================================================================================

import torch

from . import adapter, flux2

import comfy.ldm.flux3.model as flux3
import comfy.text_encoders.flux


def detect(sd):
    """model_detection.py:263-272 (no key prefix): the transformer's settings off its file."""

    hidden_size = sd["content_mode_blocks.image.0.attn_out.weight"].shape[0]

    return {"hidden_size": hidden_size,
            "num_heads": hidden_size // 128,
            "fused_qkv": "single_blocks.0.qkv.weight" in sd,
            "flow_map": "ref_time_in.in_layer.weight" in sd,
            "depth": adapter.count_blocks(sd, "txt_mode_blocks.{}."),
            "depth_single_blocks": adapter.count_blocks(sd, "single_blocks.{}.")}


class Flux3(adapter.BaseModel):
    """model_base.Flux3 around ComfyUI's transformer, called the way diffusers' FLUX.3 modular
    pipeline calls its own: packed latent tokens with their (t, h, w, l) ids, one grid per l (the
    image, then each reference), and the text encoder's output. Its forward is _apply_model's
    (model_base.py:221-263): the latents and conditioning cast to the compute dtype, ComfyUI's
    transformer run on them, its output in float32 as ComfyUI samples in. A flow map's step
    target comes as the pipeline's target_timestep, the next sigma ComfyUI reads off the
    schedule. A flow map runs its text and references every step, as model_base.py:1141 has it."""

    def __init__(self, diffusion_model, config, dtype):
        super().__init__(diffusion_model, config, dtype)
        self._shapes = ((None, None), None)  # the grids of the last ids, read once per run

    def _get_shapes(self, img_ids, image_cond_ids):
        """Each grid's (h, w), the image's then each reference's, off their ids (image_ids,
        flux3/model.py:106-112: a new grid starts a new l). The pipeline hands the same ids every
        step."""
        (img, cond), shapes = self._shapes
        if img is not img_ids or cond is not image_cond_ids:
            ids = img_ids if image_cond_ids is None else torch.cat([img_ids, image_cond_ids])
            shapes = adapter.id_grids(ids, 3, self.diffusion_model.image_rope_scale)
            self._shapes = ((img_ids, image_cond_ids), shapes)
        return shapes

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids,
                image_cond_hidden_states=None, image_cond_ids=None, target_timestep=None,
                attention_kwargs=None, return_dict=True):
        dtype = self._dtype
        batch = hidden_states.shape[0]

        def grid(tokens, h, w):
            return tokens.transpose(1, 2).reshape(batch, -1, h, w).to(dtype)

        shapes = self._get_shapes(img_ids, image_cond_ids)

        x = grid(hidden_states, *shapes[0])

        ref_latents = None
        if image_cond_hidden_states is not None:
            tokens = image_cond_hidden_states.split([h * w for h, w in shapes[1:]], dim=1)
            ref_latents = [grid(t, h, w) for t, (h, w) in zip(tokens, shapes[1:])]

        transformer_options = self.transformer_options()
        if target_timestep is not None:
            transformer_options["sample_sigmas"] = torch.stack([timestep[0],
                                                                target_timestep[0]]).float()

        out = self.diffusion_model(x, timestep.float(), context=encoder_hidden_states.to(dtype),
                                   ref_latents=ref_latents,
                                   transformer_options=transformer_options)

        return (out.float().flatten(2).transpose(1, 2),)

    def lora_keys(self):
        """model_lora_keys_unet's entries for this model (lora.py:352-358) over the generic ones:
        the fused qkv addressed by its three projections."""
        key_map = super().lora_keys()
        for k, w in self.named_parameters():
            if k.endswith(".qkv.weight"):
                third = w.shape[0] // 3
                for i, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                    key_lora = "{}.{}".format(k[:-len(".qkv.weight")], proj)
                    to = (k, (0, i * third, third))
                    key_map[key_lora] = to
                    key_map["lora_unet_{}".format(key_lora[len("diffusion_model."):]
                                                  .replace(".", "_"))] = to
        return key_map


class TextEncoder(adapter.CLIP):
    """CLIPLoader of type flux3 on ComfyUI's Qwen3-VL-4B file, or a repo's shards in the same
    layout (sd.py:2088-2091); the pipeline encodes through CLIPTextEncode's `encode`."""

    def __init__(self, path):
        sd = adapter.qwen3vl_sd(adapter.load_clip_sd(path))

        super().__init__(comfy.text_encoders.flux.flux3_te(**adapter.llama_detect(sd)),
                         comfy.text_encoders.flux.Flux3Tokenizer, sd)


def load(weight_file, config, load_device, dtype, **options):
    """ComfyUI's transformer for its file, behind the call diffusers' pipeline makes."""

    def build(sd, operations):
        return flux3.Flux3(**detect(sd), dtype=dtype, operations=operations)

    return Flux3.load(build, weight_file, config, load_device, dtype,
                      **options, process=flux2.process)
