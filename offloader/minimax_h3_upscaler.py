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
#   MiniMax H3's latent upscaler: the community MiniMax H3 Latent Upscaler (3D) node, which
#   enlarges H3's video latent with a trained network for H3 to refine at the new size. Not
#   ComfyUI's: LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler @ 40316cf (MIT),
#   nodes/minimax_h3_latent_upscaler_3d.py, mirrored as the experimental H3's --hires runs it
#   (one pass, enable_temporal_chunking off, fp16, the model back on CPU after, force_unload on):
#
#     - LATENTS_MEAN / LATENTS_STD (:83-98), the statistics it normalizes H3's latent with;
#     - LatentResizer3D and its blocks (:134-343), the module tree as the node's so its file
#       loads strict; the attention blocks left out, as _detect_arch forces them off (:414);
#     - _detect_arch (:379-415), the network's size read off the file;
#     - MinimaxH3LatentUpscaler3D.execute (:520-602), its target dimensions mode on sizes of
#       the 32 grid: the effective scale, the normalized pass, the latent back in float32.
#
#==================================================================================================

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils

VAE_DOWNSAMPLE = 16

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]


def _norm(channels):
    return nn.GroupNorm(32, channels)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout):
        super().__init__()
        self.in_layers = nn.Sequential(_norm(channels), nn.SiLU(),
                                       nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * channels))
        self.out_norm = _norm(channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=dropout),
                                        nn.Conv3d(channels, channels, 3, padding=1))
        self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = torch.chunk(self.emb_layers(emb).type(h.dtype)[..., None, None, None], 2,
                                   dim=1)
        h = self.out_layers(self.out_norm(h) * (1 + scale) + shift)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1),
                                padding=(kernel_size // 2, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels, in_blocks, out_blocks, channels, temporal_every,
                 temporal_kernel, dropout=0.1):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))

        def blocks(count):
            out = nn.ModuleList()
            for b in range(count):
                out.append(ResBlockEmb3D(channels, 64, dropout))
                if temporal_every > 0 and b % temporal_every == 0:
                    out.append(TemporalConv(channels, temporal_kernel))
            return out

        self.in_blocks = blocks(in_blocks)
        self.out_blocks = blocks(out_blocks)
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale, size):
        """_forward_seg (:313-343), the whole clip in one pass."""
        emb = self.embed(torch.tensor([scale - 1], dtype=x.dtype, device=x.device).unsqueeze(0))

        def run(blocks, x):
            for b in blocks:
                x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
            return x

        x = run(self.in_blocks, self.conv_in(x))
        x = run(self.out_blocks, F.interpolate(x, size=size, mode="trilinear",
                                               align_corners=False))
        return self.conv_out(F.silu(self.norm_out(x)))


def _arch(sd):
    """_detect_arch (:379-415): the network's size from its keys, attention off."""
    def count(pattern):
        return len({m.group(1) for m in map(re.compile(pattern).match, sd) if m})

    temporal = [v for k, v in sd.items() if k.endswith("dwconv.weight")]

    return {"in_channels": sd["conv_in.weight"].shape[1],
            "channels": sd["conv_in.weight"].shape[0],
            "in_blocks": count(r"in_blocks\.(\d+)\.in_layers\."),
            "out_blocks": count(r"out_blocks\.(\d+)\.in_layers\."),
            "temporal_every": 2 if temporal else 0,
            "temporal_kernel": temporal[0].shape[2] if temporal else 5}


class LatentUpscaler:
    """The node on the file at `path`: its network built and loaded once, kept on the CPU in
    fp16, on the compute device for a call only. It places itself, as a ComfyVAE does."""

    def __init__(self, path):
        sd = comfy.utils.load_torch_file(path)
        sd = sd.get("model", sd)
        if any(k.startswith("upscaler.") for k in sd):  # _extract_upscaler_sd (:374-377)
            sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}

        self.model = LatentResizer3D(**_arch(sd)).eval().requires_grad_(False)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(torch.float16)

    def __call__(self, latent, width, height):
        """`latent` [B, C, T, H, W] (float32) enlarged to `width` x `height` pixels, on the 32
        grid, as execute does: the effective scale the mean of the two axes', the pass in fp16
        on H3's normalized latent, the result in float32 on the CPU."""
        b, c, t, h_in, w_in = latent.shape
        size = (t, height // VAE_DOWNSAMPLE, width // VAE_DOWNSAMPLE)

        if size[1:] == (h_in, w_in):
            return latent
        if size[1] < h_in or size[2] < w_in:
            raise ValueError("the latent upscaler only enlarges")

        scale = (width / (w_in * VAE_DOWNSAMPLE) + height / (h_in * VAE_DOWNSAMPLE)) / 2.0
        device, dtype = mm.get_torch_device(), torch.float16

        mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)

        self.model.to(device)
        try:
            with torch.inference_mode():
                x = (latent.to(device=device, dtype=dtype) - mean) / std
                out = self.model(x, scale, size) * std + mean
            return out.to(device="cpu", dtype=latent.dtype)
        finally:
            self.model.to("cpu")
            mm.soft_empty_cache()
