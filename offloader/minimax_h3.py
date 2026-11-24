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
#     - CLIPLoader of type minimax on the Qwen3-VL 32B file, as MiniMaxH3ImageToVideo encodes
#       with it (nodes_minimax_h3.py:172-173, TextEncoder).
#
#   And Open-H3-Regenerate-2K IC's model side (NVlabs/Sana, sol-engine,
#   models/Open-H3-Regenerate-2K/IC, ref2va_ic/hyperflow.py and sharp.py): HyperFlowModel, a
#   video reference already in latents (MiniMaxH3.video_reference), audio rows held clean
#   (extra_conds' fixed_audio) and Diffusers' LoRA names for the transformer (lora_keys).
#
#==================================================================================================

import functools
import math
from types import SimpleNamespace

import torch

from . import adapter
from . import minimax_h3_image as h3_image

import comfy.audio
import comfy.ldm.minimax.model as h3
import comfy.utils

# supported_models.MiniMaxH3's sampling_settings: the video stream's flow shift and the audio's.
SHIFT       = 12.0
AUDIO_SHIFT = 3.0


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
                                          mode=audio_vae.pad_channel_value)
    z = audio_vae.encode(samples.movedim(-1, 1)).latent_dist.mode()
    return z, z.shape[-1]


def sampling(weight_file):
    """ModelSamplingAV (model_sampling.py:289-357) for adapter.FlowSampler: flow at SHIFT
    (comfy_sampler passes it) with the default multiplier, the model's timestep sigma × 1000. The
    audio stream rides the same schedule, scaled by MiniMaxH3.audio_scale."""
    return {"multiplier": 1000.0}


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
            grid = h3_image._GRID  # nodes_minimax_h3.py's, the reference images'
            tw = max(grid, round(w * scale / grid) * grid)
            th = max(grid, round(h * scale / grid) * grid)
            resized = _resize(image[:1], tw, th, "disabled")
            items.append({"type": "image", "data": resized})
            refs.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                         "latent": _encode(vae, resized)})

        for path in (r for r in references if isinstance(r, str)):
            items.append({"type": "audio"})
            z, length = _encode_audio(audio_vae, adapter.load_audio(path))
            refs.append({"kind": "audio", "ref_audio_t": length, "audio_latent": z})

        return items, refs

    def video_reference(self, latent, frames):
        """MiniMaxH3ReferenceToVideo's video reference (ref_videos) on a video already in
        latents: `latent` [1, 24, t, h, w], read by the reference rows as it is, and its `frames`
        (an IMAGE [T, H, W, 3] at 24 fps), every 12th (2 fps) presented to Qwen3-VL as
        <Video 1>. Returns the items and the reference blocks, as reference_to_video's."""
        items = [{"type": "video", "data": frames[::12]}]
        refs = [{"kind": "video", "ref_audio_t": 0, "latent_t": latent.shape[2],
                 "latent_h": latent.shape[3], "latent_w": latent.shape[4], "latent": latent}]
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
            # the streams share one time axis: FRAME_RESCALE audio frames per pixel frame
            limit = math.floor(shapes[1][-1] - h3.FRAME_RESCALE * index)
            if limit < 1:
                raise ValueError("frame_idx {} is past the end of the video's audio track".format(
                    frame_idx))
            keyframe["audio_latent"] = z[..., :limit].clone() if length > limit else z
        keyframes.append(keyframe)

    def extra_conds(self, context, tags, keyframes, shapes, seed, device, refs=None,
                    fixed_audio=False):
        """MiniMaxH3.extra_conds (model_base.py:2254-2307) for latents of `shapes` (the video's and
        the audio's): the text through the condition projection and the token refiner once, and
        the payload (the token tags, the keyframes, the references, their latents, KSampler's seed,
        the audio scale, the packed layout). Before sampling, as ComfyUI makes them.
        `fixed_audio` holds the audio rows clean (IC's silence): carried as they are, at t = 1,
        their velocity zero (the DiT's audio_denoise_mask of zeros)."""
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
        payload["audio_scale"] = 1.0 if fixed_audio else self.audio_scale

        video = shapes[0]
        payload["layout"] = h3.PackedLayout(context.shape[1], video[2], (video[3] + 1) // 2 * 2,
                                            (video[4] + 1) // 2 * 2, shapes[1][-1],
                                            keyframes=payload.get("keyframes"),
                                            refs=payload.get("refs"))

        conds = {"context": context, "minimax_payload": payload, "latent_shapes": shapes}
        if fixed_audio:
            conds["audio_denoise_mask"] = torch.zeros(1, 1, *shapes[1][2:], device=device)
        return conds

    def forward(self, hidden_states, timestep, context, minimax_payload, latent_shapes,
                sample_sigmas, audio_denoise_mask=None, return_dict=False):
        options = self.transformer_options()
        options["sample_sigmas"] = sample_sigmas  # samplers.py:1228-1229

        t = timestep.float().reshape(-1).expand(hidden_states.shape[0])

        x = comfy.utils.unpack_latents(hidden_states.to(self._dtype), latent_shapes)

        out = self.diffusion_model(x, t, context.to(self._dtype), transformer_options=options,
                                   minimax_payload=minimax_payload,
                                   audio_denoise_mask=audio_denoise_mask)

        return (comfy.utils.pack_latents(out)[0].float(),)

    def lora_keys(self):
        """ComfyUI's names (lora.py), and Diffusers' (transformer.<module>, as HyperFlow and IC
        publish theirs) onto the fused weights: to_q / to_k / to_v as slices of qkv_proj, and
        ff.net.0.proj onto fc1 with its halves swapped (Diffusers' SwiGLU is [value, gate],
        ComfyUI's [gate, value]). On a HyperFlowModel, the LoRA pieces its own install consumes
        (HyperFlow's endpoint embedder, Sharp's curve branches) are named off the weights."""
        key_map = h3_image.lora_keys(self, super().lora_keys())

        dm = self.diffusion_model
        attn, mlp = dm.blocks[0].attn, dm.blocks[0].mlp
        inner, half = attn.heads * attn.head_dim, mlp.fc2.in_features
        swap = functools.partial(torch.roll, shifts=half, dims=0)

        for source, target, count in (("transformer_blocks", "blocks", len(dm.blocks)),
                                      ("token_refiner.refiner_blocks", "token_refiner.blocks",
                                       len(dm.token_refiner.blocks))):
            for i in range(count):
                s = "transformer.%s.%d." % (source, i)
                t = "diffusion_model.%s.%d." % (target, i)
                for j, name in enumerate(("to_q", "to_k", "to_v")):
                    key_map[s + "attn." + name] = (t + "attn.qkv_proj.weight",
                                                   (0, j * inner, inner))
                key_map[s + "attn.to_out.0"] = t + "attn.out_proj.weight"
                key_map[s + "ff.net.0.proj"] = (t + "mlp.fc1.weight", None, swap)
                key_map[s + "ff.net.2"] = t + "mlp.fc2.weight"

        if hasattr(dm, "time_embedder"):
            for name, layer in (("linear_1", "proj_in"), ("linear_2", "proj_out")):
                key_map["transformer.time_embedder." + name] = \
                    "diffusion_model.time_embedder.%s.weight" % layer

        if isinstance(dm, HyperFlowModel):
            for name in ("linear_1", "linear_2"):
                key_map["transformer.endpoint_time_embedder." + name] = "hyperflow." + name
            for i in range(len(dm.blocks)):
                key_map["diffusion_model.blocks.%d.adaln_proj.linear" % i] = "sharp.%d" % i
        return key_map


class HyperFlowModel(h3.MiniMaxH3Model):
    """ComfyUI's transformer with Open-H3-Regenerate-2K IC's model-side pieces, on a full-form file
    (its time embedder and full-width adaln projections):
      - HyperFlow's two-time embedder (hyperflow.py's TwoTimeEmbedder): the rows at the step's
        timestep, the target's and the text's, are embedded on the interval to the next sigma,
        emb(t) + gate * (endpoint(r) - emb(t)) with r = 1 - sigma_next; the others keep r = t.
        `endpoint` is the file's time embedder with HyperFlow's endpoint LoRA merged in float32;
        the base one takes its LoRA from the patcher, as every other linear;
      - LMS Sharp's curve branches (sharp.py's CurveLinear): each block's adaln projection plus
        strength * B @ A of the [1025, 8] adaln_t_table read at the row's timestep.
    The pieces are kept off the module tree (a plain namespace), out of the loader's and the
    patcher's way."""

    def install(self, sd, hyperflow, sharp, curve, strength):
        """HyperFlow's file (its endpoint pairs and gate), Sharp's (its curve branches, folded
        once in float32 at `strength`; None leaves Sharp out) and `curve`'s adaln_t_table, `sd`
        the file's tensors."""
        from safetensors import safe_open

        def lora(f, name):
            return [f.get_tensor("%s.lora_%s.weight" % (name, m)).float() for m in "AB"]

        with safe_open(hyperflow, framework="pt") as f:
            meta = f.metadata()
            scale = float(meta["lora_alpha"]) / float(meta["lora_rank"])
            weights = {}
            for layer, name in (("proj_in", "linear_1"), ("proj_out", "linear_2")):
                a, b = lora(f, "transformer.endpoint_time_embedder." + name)
                base = "time_embedder." + layer
                weights[layer + ".weight"] = sd[base + ".weight"].float() + scale * (b @ a)
                weights[layer + ".bias"] = sd[base + ".bias"].float()

        endpoint = h3.TimeEmbedder(weights["proj_in.weight"].shape[1],
                                   weights["proj_in.weight"].shape[0],
                                   weights["proj_out.weight"].shape[0], dtype=torch.float32,
                                   operations=torch.nn)
        endpoint.load_state_dict(weights)

        # NOTE: Sharp is optional (IC's --sharp 0): no file, no table, no branch, no hook.
        branches, table = [], None
        if sharp:
            with safe_open(sharp, framework="pt") as f:
                for i in range(len(self.blocks)):
                    a, b = lora(f, "diffusion_model.blocks.%d.adaln_proj.linear" % i)
                    branches.append(strength * (b @ a))
            with safe_open(curve, framework="pt") as f:
                table = f.get_tensor("adaln_t_table").float()

            for i, block in enumerate(self.blocks):
                block.adaln_proj.linear.register_forward_hook(functools.partial(self._sharp, i))

        self._ic = SimpleNamespace(endpoint=endpoint, gate=float(meta["hyperflow_gate"]),
                                   branches=branches, table=table, t=None, r=None, coords=None)

    def place(self, device):
        """The pieces where the model computes (the forward's device, not the pipeline's), once."""
        ic = self._ic
        if ic.endpoint.proj_in.weight.device != device:
            ic.endpoint.to(device)
            ic.branches = [b.to(device) for b in ic.branches]
            ic.table = ic.table.to(device) if ic.table is not None else None

    def _sharp(self, i, module, args, out):
        return out + (self._ic.coords @ self._ic.branches[i].T).to(out.dtype)

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        # the step's interval, as FinalLayer finds it for PDD (model.py:358-359)
        sigmas = transformer_options["sample_sigmas"]
        sigma = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        i = int((sigmas.to(sigma.device) - sigma).abs().argmin())
        self._ic.t = float(1.0 - sigma)
        self._ic.r = float(1.0 - sigmas[min(i + 1, sigmas.shape[0] - 1)])
        return super()._forward(x, timestep, context, transformer_options, **kwargs)

    def _time_embedding(self, unique_t, device, dtype):
        self.place(torch.device(device))
        ic = self._ic
        t = torch.tensor(unique_t, dtype=torch.float32, device=device)

        emb = self.time_embedder(t)
        r = torch.where((t - ic.t).abs() <= 1e-6, torch.full_like(t, ic.r), t)
        emb = emb + ic.gate * (ic.endpoint(r) - emb)

        if ic.table is not None:  # Sharp's curve coordinates, at each row's t
            pos = t.clamp(0.0, 1.0) * (ic.table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=ic.table.shape[0] - 2)
            ic.coords = torch.lerp(ic.table[i0], ic.table[i0 + 1], (pos - i0).unsqueeze(1))

        return emb.to(dtype)


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

    # NOTE: IC's model side, on its files' paths: {"hyperflow", "sharp", "curve", "strength"}.
    ic = config.get("hyperflow")

    def build(sd, operations):
        model = (HyperFlowModel if ic else h3.MiniMaxH3Model)(
            **detect(sd, file_config), dtype=dtype, operations=operations)
        if ic:
            model.install(sd, ic, config["sharp"], config["curve"], config.get("strength", 1.0))
        return model

    return MiniMaxH3.load(build, weight_file, config, load_device, dtype, **options)
