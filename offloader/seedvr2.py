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
#   ComfyUI's SeedVR2, behind the calls a pipeline makes. The transformer is ComfyUI's own
#   (comfy/ldm/seedvr/model.py, vendored), and so is the VAE. There is no text encoder: the text is
#   two embeddings the transformer's file holds. What ComfyUI keeps around them is mirrored here,
#   citing its lines (v0.45.4):
#
#     - model_detection.py:714-750, the settings read off the file (detect);
#     - supported_models.SeedVR2 (supported_models.py:1865-1897): flow sampling at shift 1 with
#       ModelSamplingDiscreteFlow's multiplier (sampling), BASE's memory_usage_factor;
#     - model_base.SeedVR2 and _apply_model (model_base.py:221-263, 966-975), the model ComfyUI
#       samples, with SeedVR2Conditioning's positive conditioning (nodes_seedvr.py:395-414) at
#       cfg 1, one branch (SeedVR2, over adapter.BaseModel).
#
#==================================================================================================

from . import adapter

import comfy.ldm.seedvr.model as seedvr


def detect(sd):
    """model_detection.py:714-750 (no key prefix): the 7B with separate or shared MM blocks, or
    the 3B."""
    key = "blocks.35.mlp.vid.proj_out.weight"

    if key in sd and sd[key].shape[0] == 3072:
        return {"image_model": "seedvr2", "vid_dim": 3072, "heads": 24, "num_layers": 36,
                "mm_layers": 36, "norm_eps": 1e-5, "rope_type": "rope3d", "rope_dim": 64,
                "mlp_type": "normal"}
    if "blocks.35.mlp.all.proj_in_gate.weight" in sd:
        return {"image_model": "seedvr2", "vid_dim": 3072, "heads": 24, "num_layers": 36,
                "mm_layers": 10, "norm_eps": 1e-5, "rope_type": "rope3d", "rope_dim": 64,
                "mlp_type": "swiglu"}
    if "blocks.31.mlp.all.proj_in_gate.weight" in sd:
        return {"image_model": "seedvr2", "vid_dim": 2560, "heads": 20, "num_layers": 32,
                "norm_eps": 1.0e-05, "mlp_type": "swiglu", "vid_out_norm": True}

    raise ValueError("seedvr2: not a SeedVR2 3B or 7B transformer")


def sampling(weight_file):
    """ModelSamplingDiscreteFlow for SeedVR2's settings (model_sampling.py:294-306): shift 1 and
    the default multiplier, the model's timestep sigma × 1000."""
    return {"multiplier": 1000.0}


class SeedVR2(adapter.BaseModel):
    """model_base.SeedVR2 around ComfyUI's NaDiT: the (1, 16, t, h, w) latent, the sigma × 1000 as
    its timestep, the 17-channel condition, and the positive conditioning the file holds, as
    SeedVR2Conditioning gives KSampler at cfg 1 (one branch, no negative). Its forward is
    _apply_model's: the latent and conditioning cast to the compute dtype, the output in float32
    as ComfyUI samples in."""

    memory_usage_factor = 2.0  # supported_models_base.BASE's, which SeedVR2 keeps

    def forward(self, hidden_states, timestep, condition, return_dict=False):
        options = self.transformer_options()
        options["cond_or_uncond"] = [0]  # the positive branch alone

        device = hidden_states.device
        context = self.diffusion_model.positive_conditioning.unsqueeze(0)

        t = timestep.float().reshape(-1).expand(hidden_states.shape[0])

        out = self.diffusion_model(hidden_states.to(self._dtype), t,
                                   context.to(device, self._dtype),
                                   condition=condition.to(device, self._dtype),
                                   transformer_options=options)

        return (out.float(),)


def load(weight_file, config, load_device, dtype, **options):
    """ComfyUI's transformer for its file, behind the call the pipeline makes."""
    def build(sd, operations):
        return seedvr.NaDiT(**detect(sd), dtype=dtype, operations=operations)

    return SeedVR2.load(build, weight_file, config, load_device, dtype, **options)
