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
#   ComfyUI's MiniMax H3, the audio-video model, behind the calls a pipeline makes. The
#   transformer is ComfyUI's own (comfy/ldm/minimax/model.py, vendored), and so are the text
#   encoder and the two VAEs. What ComfyUI keeps around them is mirrored here, citing its lines
#   (v0.45.4):
#
#     - model_detection.py:401-434, the settings read off the file, its audio branch (detect);
#     - supported_models.MiniMaxH3 (supported_models.py:986-1012) with ModelSamplingAV
#       (model_sampling.py:289-357): flow sampling at shift 12, ModelSamplingDiscreteFlow's
#       multiplier, the audio stream carried at audio_scale = shift / audio_shift (sampling);
#     - model_base.MiniMaxH3 and _apply_model (model_base.py:221-263, 2226-2307), the model ComfyUI
#       samples on the flat pack of the video and audio latents (samplers.py:1281-1284), with
#       MiniMaxH3ImageToVideo's keyframes (nodes_minimax_h3.py:150-179) at cfg 1, one branch
#       (MiniMaxH3, over adapter.BaseModel);
#     - MiniMaxH3ReferenceToVideo's references and MiniMaxH3AddGuide's frames and audio
#       (nodes_minimax_h3.py:87-95, 182-381), loaded as LoadImage and LoadAudio load them
#       (MiniMaxH3.reference_to_video, add_guide);
#     - model_lora_keys_unet's entries for it (lora.py:421-432, lora_keys);
#     - the experimental H3's cudnn_attention.py node, cuDNN SDPA ahead of flash in
#       comfy.ops.SDPA_BACKEND_PRIORITY (ops.py:68-74), for a generation that asks for it
#       (MiniMaxH3.attention);
#     - CLIPLoader of type minimax on the Qwen3-VL 32B file, as MiniMaxH3ImageToVideo encodes
#       with it (nodes_minimax_h3.py:172-173, TextEncoder).
#
#==================================================================================================

import contextlib
import math

import torch

from . import adapter
from . import minimax_h3_image as h3_image

import comfy.audio
import comfy.ldm.minimax.model as h3
import comfy.ops
import comfy.utils

# supported_models.MiniMaxH3's sampling_settings: the video stream's flow shift and the audio's.
SHIFT       = 12.0
AUDIO_SHIFT = 3.0

# nodes_minimax_h3.py's grid of the reference images, and the frames the audio latent counts per
# pixel frame on the shared time axis.
_GRID = 32
_FRAME_RESCALE = h3.FRAME_RESCALE


def detect(sd, config):
    """model_detection.py:401-434 (no key prefix), the audio branch: the image branch's settings
    (minimax_h3_image.detect), the audio stream's on top, then the file's metadata `config`."""
    out = h3_image.detect(sd, config)

    out["image_model"] = "minimax_h3"
    out["audio_latents_dim"] = (sd["final_layer.audio_out.weight"].shape[0]
                                // config.get("pdd_steps", 1))

    out.update(config)
    return out


def _resize(image, width, height, crop):
    """_resize (nodes_minimax_h3.py:80-84): the IMAGE through common_upscale, lanczos."""
    samples = image[..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(samples, width, height, "lanczos", crop).movedim(1, -1)


def _encode(vae, image):
    """VAE.encode of an IMAGE (sd.py:1513-1562): in [-1, 1], one frame, its latent in float32 where
    the IMAGE is."""
    return vae.encode(image.movedim(-1, 1).unsqueeze(2) * 2.0 - 1.0).latent_dist.mode()


def _encode_audio(audio_vae, audio):
    """_encode_ref_audio (nodes_minimax_h3.py:87-95): the AUDIO resampled to the VAE's rate, then
    encoded as VAE.encode takes it, vae_encode_crop_pixels (sd.py:1245-1267) cutting it to a
    whole number of latent frames, centered, and to the VAE's 2 channels, a mono one repeated;
    its latent and length."""
    waveform, rate = audio["waveform"], audio["sample_rate"]
    if rate != audio_vae.audio_sample_rate:
        waveform = comfy.audio.resample(waveform, rate, audio_vae.audio_sample_rate)
    samples = waveform[:1].movedim(1, -1)
    length, ratio = samples.shape[1], audio_vae.downscale_ratio
    if length % ratio:
        samples = samples.narrow(1, (length % ratio) // 2, length // ratio * ratio)
    channels = audio_vae.output_channels
    if samples.shape[-1] > channels:
        samples = samples[..., :channels]
    elif samples.shape[-1] < channels:
        samples = torch.nn.functional.pad(samples, (0, channels - samples.shape[-1]),
                                          mode="replicate")
    z = audio_vae.encode(samples.movedim(-1, 1)).latent_dist.mode()
    return z, z.shape[-1]


def sampling(weight_file):
    """ModelSamplingAV (model_sampling.py:289-357) for adapter.FlowSampler: the sigma of
    ModelSamplingDiscreteFlow at shift 12 (time_snr_shift) and its default multiplier, so the
    model takes sigma × 1000 as its timestep. The audio stream rides the same schedule, scaled by
    MiniMaxH3.audio_scale."""
    multiplier = 1000.0

    def sigma(timestep):
        t = timestep / multiplier
        return SHIFT * t / (1 + (SHIFT - 1) * t)

    return {"multiplier": multiplier, "sigma": sigma}


class MiniMaxH3(adapter.BaseModel):
    """model_base.MiniMaxH3 around ComfyUI's transformer: the flat pack of the video
    (1, 24, t, h, w) and audio (1, 32, 2, t) latents, the sigma × 1000 as its timestep, and the
    conditions extra_conds made once for the generation. Its forward is _apply_model's
    (model_base.py:221-263): the pack cast to the compute dtype and split into its two latents,
    ComfyUI's transformer run on them, its two outputs packed back in float32 as ComfyUI samples
    in."""

    memory_usage_factor = 0.114  # supported_models.MiniMaxH3's

    # The sampler carries the audio stream at this scale (MiniMaxH3.audio_scale, ModelSamplingAV).
    audio_scale = SHIFT / AUDIO_SHIFT

    @contextlib.contextmanager
    def attention(self, kind=None):
        """The SDPA priority around a generation: `kind` "cudnn" puts cuDNN ahead of flash, as the
        experimental H3's cudnn_attention.py node sets ComfyUI's (the text encoder and the VAEs
        too: the priority is the process's), restored after, so another engine in the same
        server keeps ComfyUI's. None keeps ComfyUI's."""
        priority = getattr(comfy.ops, "SDPA_BACKEND_PRIORITY", None)

        if kind != "cudnn" or priority is None:
            yield
            return

        from torch.nn.attention import SDPBackend

        saved = list(priority)
        priority[:] = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                       SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        try:
            yield
        finally:
            priority[:] = saved

    def reference_to_video(self, references, vae, audio_vae, width, height):
        """MiniMaxH3ReferenceToVideo (nodes_minimax_h3.py:259-381) on `references`, PIL images and
        audio paths in order: the images first, each down to the output's area on the 32 grid
        ("match") and encoded, then each audio loaded, resampled and encoded. Returns the items the
        tokenizer presents (<Picture i>, <Audio j>) and the reference blocks the DiT reads."""
        items, refs = [], []

        for image in (r for r in references if not isinstance(r, str)):
            image = adapter.load_image(image)
            h, w = image.shape[1], image.shape[2]
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
            tw = max(_GRID, round(w * scale / _GRID) * _GRID)
            th = max(_GRID, round(h * scale / _GRID) * _GRID)
            resized = _resize(image[:1], tw, th, "disabled")
            items.append({"type": "image", "data": resized})
            refs.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                         "latent": _encode(vae, resized)})

        for path in (r for r in references if isinstance(r, str)):
            items.append({"type": "audio"})
            z, length = _encode_audio(audio_vae, adapter.load_audio(path))
            refs.append({"kind": "audio", "ref_audio_t": length, "audio_latent": z})

        return items, refs

    def add_guide(self, keyframes, shapes, frame_idx, vae=None, image=None, audio_vae=None,
                  audio=None):
        """MiniMaxH3AddGuide (nodes_minimax_h3.py:182-256) for latents of `shapes`: a PIL image
        center-cropped to the canvas and encoded, or an audio path encoded and cut to what is left
        of the soundtrack, anchored at `frame_idx` (negative: from the end), appended to
        `keyframes`."""
        video = shapes[0]
        frame_count = sum(h3.FRAME_PER_TOKEN[k % 5] for k in range(video[2]))
        index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if index < 0 or index + 1 > frame_count:
            raise ValueError("frame_idx {} is outside the video's {} frames".format(
                frame_idx, frame_count))

        keyframe = {"resolved_frame_index": index}
        if image is not None:
            frames = _resize(adapter.load_image(image), video[4] * 16, video[3] * 16, "center")
            keyframe["latent"] = _encode(vae, frames)
        if audio is not None:
            z, length = _encode_audio(audio_vae, adapter.load_audio(audio))
            # the streams share one time axis: _FRAME_RESCALE per pixel frame
            limit = math.floor(shapes[1][-1] - _FRAME_RESCALE * index)
            if limit < 1:
                raise ValueError("frame_idx {} is past the end of the video's audio track".format(
                    frame_idx))
            keyframe["audio_latent"] = z[..., :limit].clone() if length > limit else z
        keyframes.append(keyframe)

    def extra_conds(self, context, tags, keyframes, shapes, seed, device, refs=None):
        """MiniMaxH3.extra_conds (model_base.py:2254-2307) for latents of `shapes` (the video's and
        the audio's): the text through the condition projection and the token refiner once, and
        the payload (the token tags, the keyframes, the references, their latents, KSampler's seed,
        the audio scale, the packed layout). Before sampling, as ComfyUI makes them."""
        context = self.diffusion_model.preprocess_text_embeds(
            context.to(device=device, dtype=self._dtype))

        payload = {}
        if tags is not None:
            payload["text_token_tags"] = tags.cpu()
        if keyframes:
            payload["keyframes"] = keyframes
        if refs:
            payload["refs"] = refs
        # _cond_latents (model_base.py:2302-2307): the keyframes' latents, then the references'
        refs = refs or []
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes if "latent" in kf] \
            + [r["latent"] for r in refs if "latent" in r]
        payload["cond_audio_latents"] = [kf["audio_latent"] for kf in keyframes
                                         if "audio_latent" in kf] \
            + [r["audio_latent"] for r in refs if "audio_latent" in r]
        payload["seed"] = seed
        payload["audio_scale"] = self.audio_scale

        video = shapes[0]
        payload["layout"] = h3.PackedLayout(context.shape[1], video[2], (video[3] + 1) // 2 * 2,
                                            (video[4] + 1) // 2 * 2, shapes[1][-1],
                                            keyframes=payload.get("keyframes"),
                                            refs=payload.get("refs"))

        return {"context": context, "minimax_payload": payload, "latent_shapes": shapes}

    def forward(self, hidden_states, timestep, context, minimax_payload, latent_shapes,
                sample_sigmas, return_dict=False):
        options = self.transformer_options()
        options["sample_sigmas"] = sample_sigmas  # samplers.py:1228-1229

        t = timestep.float().reshape(-1).expand(hidden_states.shape[0])

        x = comfy.utils.unpack_latents(hidden_states.to(self._dtype), latent_shapes)

        out = self.diffusion_model(x, t, context.to(self._dtype), transformer_options=options,
                                   minimax_payload=minimax_payload)

        return (comfy.utils.pack_latents(out)[0].float(),)

    def lora_keys(self):
        """model_lora_keys_unet's entries for this model (lora.py:421-432) over the generic ones:
        bare module paths. The video files have one output head (pdd_steps 1), so no PDD bank."""
        key_map = super().lora_keys()
        for k, _ in self.named_parameters():
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key_map[k[len("diffusion_model."):-len(".weight")]] = k
        return key_map


class TextEncoder(h3_image.TextEncoder):
    """CLIPLoader of type minimax on ComfyUI's Qwen3-VL 32B file, as minimax_h3_image's;
    `encode` is MiniMaxH3ImageToVideo's (nodes_minimax_h3.py:172-173)."""

    def encode(self, text, images=(), items=None):
        """The conditioning and its token tags: the keyframes as the node sized them, each
        labeled "<Picture n>: " before the prompt (the tokenizer's default), or
        MiniMaxH3ReferenceToVideo's reference `items` (nodes_minimax_h3.py:374)."""
        cond, extra = self.encode_from_tokens(self.tokenize(text, images=list(images),
                                                            minimax_ref_items=items))

        return cond, extra.get("minimax_token_tags")


def load(weight_file, config, load_device, dtype, **options):
    """ComfyUI's transformer for its file, behind the call the pipeline makes."""
    file_config = h3_image._config(weight_file)

    def build(sd, operations):
        return h3.MiniMaxH3Model(**detect(sd, file_config), dtype=dtype, operations=operations)

    return MiniMaxH3.load(build, weight_file, config, load_device, dtype, **options)
