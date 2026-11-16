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
#   ComfyUI's FLUX.2, behind the call diffusers' Flux2KleinPipeline makes. The transformer is
#   ComfyUI's own (comfy/ldm/flux/model.py, vendored), and so is the text encoder. What ComfyUI
#   keeps around them is mirrored here, citing its lines (v0.45.4):
#
#     - model_detection.py:273-329, :359-362, the settings read off the file (detect);
#     - Flux.process_unet_state_dict, supported_models.py:756-763 (process);
#     - model_base.Flux2 and Flux (model_base.py:996-1126), the model ComfyUI samples (Flux2,
#       over adapter.BaseModel);
#     - model_lora_keys_unet's entries for it (lora.py:276-290, lora_keys);
#     - CLIPLoader of type flux2 on a Qwen3 file (sd.py:1844-1853, :2046-2062, TextEncoder).
#
#==================================================================================================

import torch
import torch.nn.functional as F

from . import adapter

import comfy.ldm.flux.model as flux
import comfy.text_encoders.flux
import comfy.utils


def detect(sd):
    """model_detection.py:273-329, :359-362 (no key prefix), FLUX.2's branch: the transformer's
    settings off its file."""

    def count_blocks(prefix):
        # model_detection.py:9-20
        count = 0
        while any(k.startswith("{}{}.".format(prefix, count)) for k in sd):
            count += 1
        return count

    config = {"image_model": "flux2", "axes_dim": [32, 32, 32, 32], "num_heads": 48,
              "mlp_ratio": 3.0, "theta": 2000, "out_channels": 128, "global_modulation": True,
              "mlp_silu_act": True, "qkv_bias": False, "ops_bias": False,
              "default_ref_method": "index", "ref_index_scale": 10.0, "txt_ids_dims": [3],
              "patch_size": 1}

    w = sd["img_in.weight"]
    config["in_channels"] = w.shape[1]
    config["hidden_size"] = w.shape[0]
    config["context_in_dim"] = sd["txt_in.weight"].shape[1]
    config["vec_in_dim"] = (sd["vector_in.in_layer.weight"].shape[1]
                            if "vector_in.in_layer.weight" in sd else None)
    config["num_heads"] = config["hidden_size"] // sum(config["axes_dim"])
    config["depth"] = count_blocks("double_blocks.")
    config["depth_single_blocks"] = count_blocks("single_blocks.")
    config["guidance_embed"] = "guidance_in.in_layer.weight" in sd
    config["yak_mlp"] = "double_blocks.0.img_mlp.gate_proj.weight" in sd
    config["txt_norm"] = "txt_norm.weight" in sd or "txt_norm.scale" in sd
    return config


def process(sd):
    """Flux.process_unet_state_dict (supported_models.py:756-763): the QK norms' scales are
    weights."""
    out = {}
    for k, v in sd.items():
        if k.endswith("_norm.scale"):
            k = "{}.weight".format(k[:-len(".scale")])
        out[k] = v
    return out


class Flux2(adapter.BaseModel):
    """model_base.Flux2 around ComfyUI's transformer, called the way diffusers' Flux2KleinPipeline
    calls its own: packed latent tokens with their (t, h, w, l) ids, the target at t 0 and each
    reference after it, and the text encoder's output. Its forward is _apply_model's
    (model_base.py:221-263) with Flux2's conditions: the text left-padded to 512 tokens
    (model_base.py:1118-1126) and Flux's guidance of 3.5 (:1057-1059), which a model without a
    guidance embed ignores."""

    def __init__(self, diffusion_model, config, dtype):
        super().__init__(diffusion_model, config, dtype)
        self._shapes = (None, None)  # the grids of the last ids, read once per run

    def forward(self, hidden_states, timestep, encoder_hidden_states, img_ids, guidance=None,
                **kwargs):
        dtype = self._dtype
        batch = hidden_states.shape[0]

        ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        if self._shapes[0] is not img_ids:
            self._shapes = (img_ids, adapter.id_grids(ids, 0))
        shapes = self._shapes[1]

        # The target's tokens first, then each reference's, a grid each.
        grids = [t.transpose(1, 2).reshape(batch, -1, h, w).to(dtype)
                 for t, (h, w) in zip(hidden_states.split([h * w for h, w in shapes], dim=1),
                                      shapes)]
        x, refs = grids[0], grids[1:]

        context = encoder_hidden_states
        if context.shape[1] < 512:
            context = F.pad(context, (0, 0, 512 - context.shape[1], 0))

        out = self.diffusion_model(x, timestep.float(), context.to(dtype),
                                   guidance=torch.full((batch,), 3.5, dtype=dtype,
                                                       device=x.device),
                                   ref_latents=refs or None,
                                   transformer_options=self.transformer_options())

        return (out.float().flatten(2).transpose(1, 2),)

    def lora_keys(self):
        """model_lora_keys_unet's entries for this model (lora.py:276-290) over the generic ones:
        its diffusers names, and the fused linear1's qkv."""
        key_map = super().lora_keys()
        config = self.diffusion_model.unet_config
        diffusers_keys = comfy.utils.flux_to_diffusers(config, output_prefix="diffusion_model.")
        for k, to in diffusers_keys.items():
            if k.endswith(".weight"):
                key = k[:-len(".weight")]
                key_map["transformer.{}".format(key)] = to
                key_map["lycoris_{}".format(key.replace(".", "_"))] = to
                key_map["lora_transformer_{}".format(key.replace(".", "_"))] = to
                key_map[key] = to
        qkv = (0, 0, config.get("hidden_size", 0) * 3)
        for k, _ in self.named_parameters():
            if k.endswith(".weight") and ".linear1." in k:
                key_map[k.replace(".linear1.weight", ".linear1_qkv")] = (k, qkv)
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key = k[len("diffusion_model."):-len(".weight")]
                key_map["base_model.model.{}".format(key)] = k  # PEFT on BFL's modules
        return key_map


class TextEncoder(adapter.CLIP):
    """CLIPLoader of type flux2 on a Qwen3 4B or 8B file, told apart by its width as
    detect_te_model does (sd.py:1844-1853, :2046-2062); the pipeline encodes through
    CLIPTextEncode's `encode`."""

    def __init__(self, path):
        sd = adapter.load_clip_sd(path)

        if sd["model.layers.0.post_attention_layernorm.weight"].shape[0] == 4096:
            te = comfy.text_encoders.flux.klein_te(**adapter.llama_detect(sd),
                                                   model_type="qwen3_8b")
            tokenizer = comfy.text_encoders.flux.KleinTokenizer8B
        else:
            te = comfy.text_encoders.flux.klein_te(**adapter.llama_detect(sd),
                                                   model_type="qwen3_4b")
            tokenizer = comfy.text_encoders.flux.KleinTokenizer

        super().__init__(te, tokenizer, sd)


def load(weight_file, config, load_device, dtype, convert=None):
    """ComfyUI's transformer for its file, behind the call diffusers' pipeline makes."""

    def build(sd, operations):
        unet_config = detect(sd)
        model = flux.Flux(**unet_config, dtype=dtype, operations=operations)
        model.unet_config = unet_config  # model_config.unet_config, for lora_keys
        return model

    return Flux2.load(build, weight_file, config, load_device, dtype,
                      convert=lambda sd: process(convert(sd) if convert else sd))
