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
#   ComfyUI's Z-Image, behind the call diffusers' ZImagePipeline makes. The transformer is
#   ComfyUI's own (comfy/ldm/lumina/model.py, vendored), and so is the text encoder. What ComfyUI
#   keeps around them is mirrored here, citing its lines (v0.45.4):
#
#     - model_detection.py:594-604, :616-640, the settings of a Z-Image file (detect);
#     - model_base.Lumina2 (model_base.py:1569-1622), the model ComfyUI samples (Lumina2, over
#       adapter.BaseModel);
#     - model_lora_keys_unet's entries for it (lora.py:384-393, lora_keys);
#     - CLIPLoader of type lumina2 on a Qwen3 4B file (sd.py:2046-2052, TextEncoder).
#
#==================================================================================================

import torch

from . import adapter

import comfy.ldm.lumina.model as lumina
import comfy.text_encoders.z_image
import comfy.utils


def detect(sd):
    """model_detection.py:594-640 (no key prefix) for a Z-Image file: its settings."""

    w = sd["cap_embedder.1.weight"]
    if w.shape[0] != 3840:
        raise ValueError("not a Z-Image transformer (dim {})".format(w.shape[0]))

    config = {"image_model": "lumina2", "patch_size": 2, "in_channels": 16, "dim": w.shape[0],
              "cap_feat_dim": w.shape[1], "n_layers": adapter.count_blocks(sd, "layers.{}."),
              "qk_norm": True,
              "n_heads": 30, "n_kv_heads": 30, "axes_dims": [32, 48, 48],
              "axes_lens": [1536, 512, 512], "rope_theta": 256.0,
              "ffn_dim_multiplier": 8.0 / 3.0, "z_image_modulation": True, "time_scale": 1000.0}
    if "cap_pad_token" in sd:
        config["pad_tokens_multiple"] = 32
    if "siglip_embedder.0.weight" in sd:
        config["siglip_feat_dim"] = sd["siglip_embedder.0.weight"].shape[0]
    return config


class Lumina2(adapter.BaseModel):
    """model_base.Lumina2 around ComfyUI's transformer, called the way diffusers' ZImagePipeline
    calls its own: a list of (c, 1, h, w) latents, the timestep as 1 - sigma, and a list of the
    text encoder's outputs. Its forward is _apply_model's (model_base.py:221-263) with Lumina2's
    conditions (model_base.py:1574-1587): the text's length as num_tokens. Conds of different
    lengths run apart, as ComfyUI's batching can't concat them. ComfyUI's model already returns
    the velocity diffusers' pipeline negates its model's output into."""

    @property
    def in_channels(self):
        return self.diffusion_model.in_channels

    def forward(self, x, t, cap_feats, **kwargs):
        dtype = self._dtype

        x = torch.stack(x).squeeze(2).to(dtype)
        sigma = (1.0 - t).float()

        if len({c.shape for c in cap_feats}) == 1:
            groups = [(slice(None), torch.stack(cap_feats))]
        else:
            groups = [(slice(i, i + 1), c[None]) for i, c in enumerate(cap_feats)]

        out = []
        for rows, context in groups:
            out.append(self.diffusion_model(x[rows], sigma[rows], context.to(dtype),
                                            context.shape[1],
                                            transformer_options=self.transformer_options()))

        return (list(-torch.cat(out).float().unsqueeze(2)),)

    def lora_keys(self):
        """model_lora_keys_unet's entries for this model (lora.py:384-393) over the generic ones:
        its diffusers names."""
        key_map = super().lora_keys()
        diffusers_keys = comfy.utils.z_image_to_diffusers(self.diffusion_model.unet_config,
                                                          output_prefix="diffusion_model.")
        for k, to in diffusers_keys.items():
            if k.endswith(".weight"):
                key = k[:-len(".weight")]
                key_map["diffusion_model.{}".format(key)] = to
                key_map["transformer.{}".format(key)] = to
                key_map["lycoris_{}".format(key.replace(".", "_"))] = to
                key_map[key] = to
        return key_map


class TextEncoder(adapter.CLIP):
    """CLIPLoader of type lumina2 on ComfyUI's Qwen3 4B file (sd.py:2046-2052); the pipeline
    encodes through CLIPTextEncode's `encode`."""

    def __init__(self, path):
        sd = adapter.load_clip_sd(path)

        super().__init__(comfy.text_encoders.z_image.te(**adapter.llama_detect(sd)),
                         comfy.text_encoders.z_image.ZImageTokenizer, sd)


def load(weight_file, config, load_device, dtype, **options):
    """ComfyUI's transformer for its file, behind the call diffusers' pipeline makes."""

    def build(sd, operations):
        unet_config = detect(sd)
        model = lumina.NextDiT(**unet_config, dtype=dtype, operations=operations)
        model.unet_config = unet_config  # model_config.unet_config, for lora_keys
        return model

    return Lumina2.load(build, weight_file, config, load_device, dtype, **options)
