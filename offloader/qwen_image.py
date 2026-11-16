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
#   ComfyUI's Qwen-Image (Qwen-Image-Edit 2511 among them), behind the call diffusers'
#   QwenImagePipeline makes. The transformer is ComfyUI's own (comfy/ldm/qwen_image/model.py,
#   vendored), and so is the text encoder. What ComfyUI keeps around them is mirrored here, citing
#   its lines (v0.45.4):
#
#     - model_detection.py:1012-1021, the settings read off the file (detect);
#     - model_base.QwenImage (model_base.py:2784-2815), the model ComfyUI samples (QwenImage,
#       over adapter.BaseModel);
#     - model_lora_keys_unet's entries for it (lora.py:337-350, lora_keys);
#     - CLIPLoader of type qwen_image on a Qwen2.5-VL-7B file (sd.py:2023-2032, TextEncoder);
#     - TextEncodeQwenImageEditPlus (nodes_qwen.py:75-109): its text in TextEncoder.encode, its
#       reference latents in QwenImage.edit.
#
#==================================================================================================

import contextlib
import math

import torch
from einops import rearrange

from . import adapter

import comfy.ldm.qwen_image.model as qwen_image
import comfy.text_encoders.qwen_image
import comfy.utils

# TextEncodeQwenImageEditPlus's template (nodes_qwen.py:79)
_TEMPLATE = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, "
             "size, texture, objects, background), then explain how the user's text instruction "
             "should alter or modify the image. Generate a new image that meets the user's "
             "requirements while maintaining consistency with the original input where "
             "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")


def detect(sd):
    """model_detection.py:1012-1021 (no key prefix): the transformer's settings off its file."""
    # count_blocks (model_detection.py:9-20)
    num_layers = 0
    while any(k.startswith("transformer_blocks.{}.".format(num_layers)) for k in sd):
        num_layers += 1

    config = {"image_model": "qwen_image", "in_channels": sd["img_in.weight"].shape[1],
              "num_layers": num_layers}
    if "__index_timestep_zero__" in sd:  # 2511
        config["default_ref_method"] = "index_timestep_zero"
    if "time_text_embed.addition_t_embedding.weight" in sd:  # Layered
        config["use_additional_t_cond"] = True
        config["default_ref_method"] = "negative_index"
    return config


def _scale(image, area, multiple):
    """TextEncodeQwenImageEditPlus's resize (nodes_qwen.py:84-97): an IMAGE to about `area`
    pixels, each side rounded to a `multiple`, by area."""
    samples = image.movedim(-1, 1)
    scale_by = math.sqrt(area / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by / multiple) * multiple
    height = round(samples.shape[2] * scale_by / multiple) * multiple
    return comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)


class QwenImage(adapter.BaseModel):
    """model_base.QwenImage around ComfyUI's transformer, called the way diffusers'
    QwenImagePipeline calls its own: packed latent tokens, (c, 2, 2) per token, the sigma as its
    timestep and the text encoder's output. Its forward is _apply_model's
    (model_base.py:221-263): the latents and conditioning cast to the compute dtype, ComfyUI's
    transformer run on them with the references of `edit`, its output in float32 as ComfyUI
    samples in."""

    def __init__(self, diffusion_model, config, dtype):
        super().__init__(diffusion_model, config, dtype)
        self._refs = None

    @contextlib.contextmanager
    def edit(self, images, vae):
        """TextEncodeQwenImageEditPlus's reference latents (nodes_qwen.py:93-99) for one
        generation: each image at about 1024x1024 on the 8 grid, encoded by `vae` before sampling,
        into the model's latent space (process_latent_in, model_base.py:2797-2802: Wan21's, the
        statistics of the VAE's config)."""
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

        refs = []
        for image in images:
            pixels = _scale(adapter.load_image(image), 1024 * 1024, 8)[..., :3]
            latent = vae.encode(pixels.movedim(-1, 1)[:, :, None].mul(2).sub(1))
            refs.append((latent.latent_dist.mode().float().cpu() - mean) / std)

        self._refs = refs
        try:
            yield
        finally:
            self._refs = None

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_mask=None, img_shapes=None, **kwargs):
        dtype = self._dtype
        _, h, w = img_shapes[0][0]

        x = rearrange(hidden_states, "b (h w) (c ph pw) -> b c 1 (h ph) (w pw)", h=h, w=w, ph=2,
                      pw=2).to(dtype)

        refs = None if self._refs is None else [r.to(x) for r in self._refs]
        mask = encoder_hidden_states_mask
        if mask is not None:
            mask = mask.long()  # as the text encoder hands it over

        out = self.diffusion_model(x, timestep.float(), encoder_hidden_states.to(dtype),
                                   attention_mask=mask, ref_latents=refs,
                                   transformer_options=self.transformer_options())

        return (rearrange(out.float(), "b c 1 (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2),)

    def lora_keys(self):
        """model_lora_keys_unet's entries for this model (lora.py:337-350) over the generic ones:
        a fused gate_up (Qwen-Image 2.1's) addressed by its two halves."""
        key_map = super().lora_keys()
        for k, w in self.named_parameters():
            if not k.startswith("diffusion_model.") or not k.endswith(".weight"):
                continue
            key_lora = k[len("diffusion_model."):-len(".weight")]
            targets = [(key_lora, k)]
            if key_lora.endswith(".img_mlp.gate_up"):
                half = w.shape[0] // 2
                targets = [(key_lora.replace(".gate_up", ".gate_layer"), (k, (0, 0, half))),
                           (key_lora.replace(".gate_up", ".proj"), (k, (0, half, half)))]
            for key_lora, to in targets:
                key_map[key_lora] = to
                key_map["transformer.{}".format(key_lora)] = to
                key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = to
        return key_map


class TextEncoder(adapter.CLIP):
    """CLIPLoader of type qwen_image on ComfyUI's Qwen2.5-VL-7B file (sd.py:2023-2032), and the
    encode TextEncodeQwenImageEditPlus runs (nodes_qwen.py:75-109)."""

    def __init__(self, path):
        sd = adapter.load_clip_sd(path)

        super().__init__(comfy.text_encoders.qwen_image.te(**adapter.llama_detect(sd)),
                         comfy.text_encoders.qwen_image.QwenImageTokenizer, sd)

    def encode(self, prompt, images=()):
        """TextEncodeQwenImageEditPlus's conditioning for `prompt` and its images, each seen at
        about 384x384: (embeds, attention mask or None)."""
        images = [_scale(adapter.load_image(image), 384 * 384, 1) for image in images]

        image_prompt = "".join("Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
                               .format(i + 1) for i in range(len(images)))

        tokens = self.tokenize(image_prompt + prompt, images=images, llama_template=_TEMPLATE)
        cond, extra = self.encode_from_tokens(tokens)
        return cond, extra.get("attention_mask")


def load(weight_file, config, load_device, dtype, convert=None):
    """ComfyUI's transformer for its file, behind the call diffusers' pipeline makes."""

    def build(sd, operations):
        return qwen_image.QwenImageTransformer2DModel(**detect(sd), dtype=dtype,
                                                      operations=operations)

    return QwenImage.load(build, weight_file, config, load_device, dtype, convert=convert)
