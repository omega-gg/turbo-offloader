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
#   ComfyUI's MOSS-Image, behind the call the official MOSS-Image pipeline makes. The transformer
#   is ComfyUI's own (comfy/ldm/moss_image/model.py, vendored), and so is the text encoder. What
#   ComfyUI keeps around them is mirrored here, citing its lines (v0.45.4):
#
#     - model_detection.py:1050-1058, the settings read off the file (detect);
#     - supported_models.MOSSImage (supported_models.py:2062-2093): its process_unet_state_dict
#       (_process), its sampling settings (SHIFT);
#     - model_base.MOSSImage (model_base.py:2947-2976), the model ComfyUI samples (MOSSImage, over
#       adapter.BaseModel), with SkipLayerGuidanceDiTSimple's skipped blocks
#       (nodes_slg.py:116-151);
#     - model_lora_keys_unet's entries for it (lora.py:361-371, lora_keys);
#     - CLIPLoader of type moss_image on a Qwen3.5 9B file (sd.py:2066-2070, TextEncoder);
#     - TextEncodeMOSSImageEdit in its Moss-Image-Edit format (nodes_moss_image.py:95-149) on a
#       source ImageScaleToTotalPixels sized (nodes_post_processing.py:249-258): MOSSImage.edit.
#
#==================================================================================================

import contextlib
import math

from . import adapter

import comfy.ldm.moss_image.model as moss_image
import comfy.text_encoders.moss_image
import comfy.utils

# supported_models.py:2067-2070: the flow shift ComfyUI samples MOSS-Image with (multiplier 1.0,
# the model takes the sigma as its timestep).
SHIFT = 5.0

# EDIT_FORMATS["Moss-Image-Edit"] (nodes_moss_image.py:98): the references at the text's start,
# their keys and values computed once.
_METHOD = "index_timestep_zero_text_start"


def detect(sd):
    """model_detection.py:1050-1058 (no key prefix): the transformer's settings off its file."""
    dim = sd["shared_block_modulation.weight"].shape[0] // 6

    return {"image_model": "moss_image", "dim": dim,
            "ffn_dim": sd["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2,
            "num_heads": dim // 128, "num_layers": adapter.count_blocks(sd, "layers.{}."),
            "fused_qkv": "layers.0.proj.qkv.weight" in sd}


def _process(sd):
    """process_unet_state_dict (supported_models.py:2082-2084)."""
    sd.pop("rope.bands", None)
    return sd


def _scale(image, megapixels, steps):
    """ImageScaleToTotalPixels (nodes_post_processing.py:249-258), lanczos: an IMAGE to about
    `megapixels`, each side rounded to `steps`."""
    samples = image.movedim(-1, 1)
    scale_by = math.sqrt(megapixels * 1024 * 1024 / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by / steps) * steps
    height = round(samples.shape[2] * scale_by / steps) * steps
    return comfy.utils.common_upscale(samples, int(width), int(height), "lanczos",
                                      "disabled").movedim(1, -1)


def _encode(vae, image):
    """VAE.encode (sd.py:1513-1562) on an IMAGE, through ComfyUI's own FLUX.2 VAE (comfy_vae): its
    latent, the 2x2 patches batch-normalized, as Flux2's process_latent_in leaves it."""
    return vae.encode(image.movedim(-1, 1).mul(2).sub(1)).latent_dist.mode()


def _skip(args, extra_args):
    """SkipLayerGuidanceDiTSimple's replacement for a skipped block (nodes_slg.py:116-117)."""
    return args


class MOSSImage(adapter.BaseModel):
    """model_base.MOSSImage around ComfyUI's transformer, called the way the official pipeline
    calls its own: the (b, 128, h, w) latent, the sigma as its timestep, the text encoder's
    output and the blocks its uncond pass skips. Its forward is _apply_model's
    (model_base.py:221-263): the latents and conditioning cast to the compute dtype, ComfyUI's
    transformer run on them with the references of `edit`, its output in float32 as ComfyUI
    samples in."""

    def __init__(self, diffusion_model, config, dtype):
        super().__init__(diffusion_model, config, dtype)
        self._refs = None
        self._ref_kv = None

    @contextlib.contextmanager
    def edit(self, images, vae):
        """TextEncodeMOSSImageEdit's reference latents (nodes_moss_image.py:146-149) for one
        generation: each image at about 1 MP on the 16 grid, encoded by `vae` before sampling.
        Their keys and values come from extra_conds."""
        self._refs = [_encode(vae, _scale(adapter.load_image(image), 1.0, 16)[..., :3])
                      for image in images]
        try:
            yield
        finally:
            self._refs = self._ref_kv = None

    def extra_conds(self, device):
        """extra_conds' ref_kv (model_base.py:2966-2974): the references' keys and values, once
        for cond and uncond, before sampling as ComfyUI computes them (the first forward
        otherwise)."""
        if self._refs is not None and self._ref_kv is None:
            self._ref_kv = self.diffusion_model.preprocess_refs(
                [r.to(device=device, dtype=self._dtype) for r in self._refs])

    def forward(self, hidden_states, timestep, encoder_hidden_states, skip_layers=(),
                return_dict=False):
        dtype = self._dtype
        x = hidden_states.to(dtype)

        options = self.transformer_options()
        if skip_layers:
            options["patches_replace"] = {"dit": {("single_block", i): _skip
                                                  for i in skip_layers}}

        refs = {}
        if self._refs is not None:
            self.extra_conds(x.device)
            refs = {"ref_kv": self._ref_kv, "ref_latents_method": _METHOD}

        sigma = timestep.float().reshape(-1).expand(x.shape[0])

        out = self.diffusion_model(x, sigma, encoder_hidden_states.to(dtype),
                                   transformer_options=options, **refs)

        return (out.float(),)

    def lora_keys(self):
        """model_lora_keys_unet's entries for this model (lora.py:361-371) over the generic ones:
        bare module paths, and a fused q/k/v addressed by its three projections."""
        key_map = super().lora_keys()
        for k, w in self.named_parameters():
            if not k.startswith("diffusion_model.") or not k.endswith(".weight"):
                continue
            key_map[k[len("diffusion_model."):-len(".weight")]] = k
            if k.endswith(".proj.qkv.weight"):
                third = w.shape[0] // 3
                for i, proj in enumerate(("q", "k", "v")):
                    key_lora = "{}.{}".format(k[:-len(".qkv.weight")], proj)
                    bare = key_lora[len("diffusion_model."):]
                    to = (k, (0, i * third, third))
                    key_map[key_lora] = to
                    key_map[bare] = to
                    key_map["lora_unet_{}".format(bare.replace(".", "_"))] = to
        return key_map


class TextEncoder(adapter.CLIP):
    """CLIPLoader of type moss_image on ComfyUI's Qwen3.5 9B file (sd.py:2066-2070); the pipeline
    encodes through CLIPTextEncode's `encode` (TextEncodeMOSSImageEdit's too, its images going to
    the VAE only)."""

    def __init__(self, path):
        sd = adapter.qwen3vl_sd(adapter.load_clip_sd(path))

        super().__init__(comfy.text_encoders.moss_image.te(**adapter.llama_detect(sd),
                                                           mtp="mtp.fc.weight" in sd),
                         comfy.text_encoders.moss_image.MOSSImageTokenizer, sd)


def load(weight_file, config, load_device, dtype, **options):
    """ComfyUI's transformer for its file, behind the call the pipeline makes."""

    def build(sd, operations):
        return moss_image.MOSSImage(**detect(sd), dtype=dtype, operations=operations)

    return MOSSImage.load(build, weight_file, config, load_device, dtype, process=_process,
                          **options)
