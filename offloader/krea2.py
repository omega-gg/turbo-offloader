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
#   ComfyUI's Krea 2, behind the call diffusers' Krea2Pipeline makes. The transformer is ComfyUI's
#   own (comfy/ldm/krea2/model.py, vendored), its file loaded as it is. What ComfyUI keeps around
#   it is mirrored here, citing its lines (v0.45.4):
#
#     - model_detection.py:1060-1073, the settings read off the file (detect);
#     - model_base.Krea2 (model_base.py:2918-2945), the model ComfyUI samples (Krea2, over
#       adapter.BaseModel);
#     - model_lora_keys_unet's entries for it (lora.py:373-382, lora_keys).
#
#   An edit runs the comfyui-krea2edit node pack's forward (v2.0.0, Apache-2.0, its __init__.py),
#   on the path its fit mode takes, the references fitted to the target in pixels (_edit_forward).
#
#==================================================================================================

import contextlib
import math

import torch
import torch.nn.functional as F
from einops import rearrange

from . import adapter

import comfy.ldm.common_dit
import comfy.ldm.krea2.model as krea2
import comfy.text_encoders.krea2
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding


def detect(sd):
    """model_detection.py:1060-1073 (no key prefix): the transformer's settings off its file."""
    head_dim = 128
    first_w = sd["first.weight"]  # (features, channels*patch^2)


    return {"image_model": "krea2",
            "features": first_w.shape[0],
            "channels": first_w.shape[1] // (2 * 2),  # patch=2
            "patch": 2,
            "layers": adapter.count_blocks(sd, "blocks.{}."),
            "heads": sd["blocks.0.attn.wq.weight"].shape[0] // head_dim,
            "kvheads": sd["blocks.0.attn.wk.weight"].shape[0] // head_dim,
            "txtlayers": sd["txtfusion.projector.weight"].shape[1],
            "txtdim": sd["txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0]}


# The node pack's reference positions (__init__.py:30-81).

def _imgids(bs, frame, h_, w_, device):
    # __init__.py:30-35
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    # __init__.py:38-50: a fitted reference centered on the target's grid, at fractional offsets
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def _place_refs(ref_ids, modes, w_):
    # __init__.py:56-81, "aligned" and "beside": a reference beside the target is shifted right of
    # it, and of the references beside it before.
    out, x0 = [], float(w_)
    for ids, mode in zip(ref_ids, modes):
        if mode == "beside":
            ids = ids.clone()
            ids[..., 2] = ids[..., 2] - ids[..., 2].min() + x0
            x0 = float(ids[..., 2].max()) + 1.0
        out.append(ids)
    return out


def _fit_encode_image(image, vae, H, W):
    # __init__.py:107-151, the fit mode: a reference near the target's shape center-cropped to
    # fill it, any other fitted inside on the 16 grid, pre-cropped so its scale is exact; then
    # VAE-encoded. `image` is an IMAGE, (H, W) the target's latent size.
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)
    ih, iw = img.shape[-2:]
    sc = min(px_h / ih, px_w / iw)
    CROP_TOL = 0.08
    if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
        s = max(px_h / ih, px_w / iw)
        ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img[..., y0:y0 + ch, x0:x0 + cw]
        nh, nw = px_h, px_w
    else:
        nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
        nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
        ch2, cw2 = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
        y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
        img = img[..., y0:y0 + ch2, x0:x0 + cw2]
    img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
    # VAE.encode: its process_input (to [-1, 1]), one frame
    return vae.encode(img.clamp(0, 1).mul(2).sub(1)[:, :, None]).latent_dist.mode()[:, :, 0]


def _ref_attn_bias(boosts, txtlen, slens, tgtlen, device, dtype):
    # __init__.py:170-199, without a boost mask: log(boost) where the target attends to a reference
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        bias[:, :, rows0:, offs[i]:offs[i] + slens[i]] = math.log(max(b, 1e-4))
    return bias


class Krea2(adapter.BaseModel):
    """model_base.Krea2 around ComfyUI's transformer, called the way diffusers' Krea2Pipeline calls
    its own: packed latent tokens, the text encoder's layer stack and the sequence's position ids,
    which give the target's grid. Its forward is _apply_model's (model_base.py:221-263): the
    latents and conditioning cast to the compute dtype, ComfyUI's transformer run on them, its
    output in float32 as ComfyUI samples in. Within `edit`, references join the sequence as the
    node pack's Krea2EditModelPatch adds them (__init__.py:295-427)."""

    def __init__(self, diffusion_model, config, dtype):
        super().__init__(diffusion_model, config, dtype)
        self._grid = (None, None)  # the target's grid of the last position ids
        self._edit = None

    @contextlib.contextmanager
    def edit(self, images, vae, height, width, boosts, positions):
        """Krea2EditModelPatch (__init__.py:328-427) for one generation, on its pixel path with
        the target latent wired: each reference image, the scene first and the subject last,
        fitted to the target's (height, width) and encoded by `vae` before sampling, into the
        model's latent space (process_latent_in: Wan21's, the statistics of the VAE's config).
        `boosts` multiply the target's attention to each, `positions` place each "aligned" on
        the target's grid or "beside" it."""
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)

        references = [(_fit_encode_image(adapter.load_image(image), vae, height // 8,
                                         width // 8).float().cpu() - mean) / std
                      for image in images]

        self._edit = (references, boosts, positions)
        try:
            yield
        finally:
            self._edit = None

    def _get_grid(self, position_ids):
        """The target's (h, w) in tokens: its last position id, (0, h - 1, w - 1). The pipeline
        hands the same ids every step."""
        ids, grid = self._grid
        if ids is not position_ids:
            _, h, w = position_ids[-1].tolist()
            grid = (int(h) + 1, int(w) + 1)
            self._grid = (position_ids, grid)
        return grid

    def forward(self, hidden_states, encoder_hidden_states, timestep, position_ids, **kwargs):
        m = self.diffusion_model
        p = m.patch
        h, w = self._get_grid(position_ids)

        # The packed tokens are ComfyUI's latent in patches, (c, ph, pw) per token.
        x = rearrange(hidden_states, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=p,
                      pw=p).to(self._dtype)
        context = encoder_hidden_states.flatten(2).to(self._dtype)

        if self._edit is None:
            out = m(x, timestep.float(), context, transformer_options=self.transformer_options())
        else:
            out = self._edit_forward(x, timestep.float(), context)

        out = rearrange(out.float(), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=p, pw=p)
        return (out,)

    def _edit_forward(self, x, timesteps, context):
        """krea2_edit_forward (__init__.py:202-292) on the pixel fit path (ref_native, stride-1
        positions): [text | references | target], the references on the target's timestep, the
        target's tokens out."""
        m = self.diffusion_model
        references, boosts, positions = self._edit
        transformer_options = self.transformer_options()

        patch = m.patch
        bs, _, H_orig, W_orig = x.shape

        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        srcs = []
        for src in references:
            src = src.to(x.device, x.dtype)
            if src.shape[0] != bs:
                src = src[:1].expand(bs, *src.shape[1:])
            srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch),
                                                               padding_mode="replicate"))
        src_grids = [(s.shape[-2] // patch, s.shape[-1] // patch) for s in srcs]

        context = m._unpack_context(context)

        tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch,
                                    pw=patch))
        src_imgs = [m.first(rearrange(s, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch,
                                      pw=patch)) for s in srcs]

        t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
        tvec = m.tproj(t)

        context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = m.txtmlp(context)

        txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
        srclen = sum(s.shape[1] for s in src_imgs)
        combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)

        device = combined.device
        ref_ids = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
                   for i, (gh, gw) in enumerate(src_grids)]
        ref_ids = _place_refs(ref_ids, positions, w_)
        pos = torch.cat([torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]
                        + ref_ids + [_imgids(bs, 0, h_, w_, device)], dim=1)
        freqs = m.pe_embedder(pos)

        attn_bias = None
        if any(b != 1.0 for b in boosts):
            attn_bias = _ref_attn_bias(boosts, txtlen, [s.shape[1] for s in src_imgs], tgtlen,
                                       device, combined.dtype)

        for block in m.blocks:
            combined = block(combined, tvec, freqs, attn_bias,
                             transformer_options=transformer_options)

        final = m.last(combined, t)
        out = final[:, txtlen + srclen:txtlen + srclen + tgtlen, :]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch,
                        pw=patch, c=m.channels)
        return out[:, :, :H_orig, :W_orig]

    def lora_keys(self):
        """model_lora_keys_unet's entries for this model (lora.py:373-382) over the generic ones:
        the diffusers names of each weight, bare, under transformer or diffusion_model."""
        key_map = super().lora_keys()
        diffusers_keys = comfy.utils.krea2_to_diffusers(self.diffusion_model.unet_config,
                                                        output_prefix="diffusion_model.")
        for k, to in diffusers_keys.items():
            if k.endswith(".weight"):
                key_lora = k[:-len(".weight")]
                key_map["diffusion_model.{}".format(key_lora)] = to
                key_map["transformer.{}".format(key_lora)] = to
                key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = to
                key_map[key_lora] = to
        return key_map


class TextEncoder(adapter.CLIP):
    """CLIPLoader of type krea2 on ComfyUI's Qwen3-VL-4B file (sd.py:2084-2087), and
    the encode its graphs run: CLIPTextEncode, or with reference images the node pack's
    Krea2EditGroundedEncode (__init__.py:428-524), the images seen by the vision encoder."""

    # Krea2EditGroundedEncode.DEFAULT_SYSTEM (__init__.py:443-446)
    SYSTEM = ("Describe the image by detailing the color, shape, size, "
              "texture, quantity, text, spatial relationships of the objects and background:")

    def __init__(self, path):
        sd = adapter.qwen3vl_sd(adapter.load_clip_sd(path))

        super().__init__(comfy.text_encoders.krea2.te(**adapter.llama_detect(sd)),
                         comfy.text_encoders.krea2.Krea2Tokenizer, sd)

    def encode(self, prompt, images=None, grounding_px=768):
        """(conditioning, mask): the conditioning as (B, seq, 12 * 2560), its mask all valid
        unless the encoder padded it."""
        if images:
            # _prep and _template (__init__.py:456-462, :492-498)
            imgs = []
            for image in images:
                samples = adapter.load_image(image).movedim(-1, 1)
                h, w = samples.shape[2], samples.shape[3]
                if grounding_px and max(h, w) > grounding_px:
                    s = grounding_px / max(h, w)
                    samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s),
                                                         "area", "disabled")
                imgs.append(samples.movedim(1, -1)[:, :, :, :3])

            vis = "<|vision_start|><|image_pad|><|vision_end|>" * len(imgs)
            template = ("<|im_start|>system\n" + self.SYSTEM + "<|im_end|>\n<|im_start|>user\n"
                        + vis + "{}<|im_end|>\n<|im_start|>assistant\n")
            tokens = self.tokenize(prompt, images=imgs, llama_template=template)
        else:
            tokens = self.tokenize(prompt)

        cond, extra = self.encode_from_tokens(tokens)

        mask = extra.get("attention_mask")
        if mask is None:
            mask = torch.ones(cond.shape[:2], dtype=torch.bool)
        return cond, mask.bool()


def load(weight_file, config, load_device, dtype, **options):
    """ComfyUI's transformer for its file, behind the call diffusers' pipeline makes."""

    def build(sd, operations):
        unet_config = detect(sd)
        model = krea2.SingleStreamDiT(**unet_config, dtype=dtype, operations=operations)
        model.unet_config = unet_config  # model_config.unet_config, for lora_keys
        return model

    return Krea2.load(build, weight_file, config, load_device, dtype, **options)
