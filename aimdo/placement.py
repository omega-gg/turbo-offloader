#==================================================================================================
#
#   Copyright (C) 2026-2026 turbo-comfy authors. <https://omega.gg/turbo-comfy>
#
#   Author: Benjamin Arnaud. <https://bunjee.me> <bunjee@omega.gg>
#
#   This file is part of turbo-comfy.
#
#   - GNU General Public License Usage:
#   This file may be used under the terms of the GNU General Public License version 3 as published
#   by the Free Software Foundation and appearing in the LICENSE.md file included in the packaging
#   of this file. Please review the following information to ensure the GNU General Public License
#   requirements will be met: https://www.gnu.org/licenses/gpl.html.
#
#==================================================================================================

# =================================================================================================
#  placement.py -- resource-driven offloader placement, mirroring ComfyUI's measured decision.
#
#  ComfyUI never assumes a config: it MEASURES free VRAM, subtracts reserves, and keeps a resident
#  budget on the GPU while streaming the rest -- the host RAM<->disk tier self-manages via the page
#  cache / RAM-pressure cache (see PLAN-unified-offloader.md). This module reproduces the ONE
#  explicit, measured decision ComfyUI makes -- the GPU resident budget -- so the diffusion server
#  can stop hardcoding the offloader path per engine.
#
#  Pure + side-effect free (besides reading file sizes / CUDA mem info). Wired in "shadow" mode
#  first: get_pipe logs what this WOULD choose without changing behaviour, so the choice can be
#  validated against the current hardcoded path before the switch (IMPL-unified-offloader.md, Steps
#  1-2).
#
#  Pinned ComfyUI ref: C:\dev\test\ComfyUI @ 5955ddff52a2eda2ba0cf7f3fb0927c93fb2fbb8
# =================================================================================================
import os
import glob

import torch
import psutil

GB = 1024 ** 3

# ComfyUI pins streamed weights into a HostBuffer up to a measured RAM budget; below it keeps a ~2
# GB headroom floor. [CU model_management.py ensure_pin_budget L645-L656]:
#   shortfall = size + max(RAM_CACHE_HEADROOM/2, 2 GB) - psutil.virtual_memory().available
# i.e. pinnable bytes = available_RAM - max(RAM_CACHE_HEADROOM/2, 2 GB). We use the 2 GB floor.
PIN_HEADROOM = 2 * GB

# ComfyUI's resident-weight ratio: keep up to 40% of free VRAM as weights before streaming the
# rest. [CU model_management.py MIN_WEIGHT_MEMORY_RATIO L453].
MIN_WEIGHT_MEMORY_RATIO = 0.4

_WINDOWS = os.name == "nt"


def extra_reserved_memory(total_vram_bytes):
    # VRAM held back for everything that is not streamed weights.
    # [CU model_management.py L789-L793]: 400 MB, 600 MB on Windows (shared-VRAM), +100 MB on 16
    # GB+ cards.
    extra = 400 * 1024 * 1024
    if _WINDOWS:
        extra = 600 * 1024 * 1024
        if total_vram_bytes > (15 * 1024 * 1024 * 1024):
            extra += 100 * 1024 * 1024
    return extra


def minimum_inference_memory(total_vram_bytes):
    # Reserve for activations/inference. [CU model_management.py minimum_inference_memory L802].
    return int(0.8 * GB + extra_reserved_memory(total_vram_bytes))


def gpu_mem(dev=0):
    # (free, total) driver-reported bytes. [CU model_management.py get_free_memory L1653]
    # (torch.cuda.mem_get_info). We use raw mem_get_info; the torch reserved/active delta ComfyUI
    # adds back is small and conservative to omit here.
    free, total = torch.cuda.mem_get_info(dev)
    return int(free), int(total)


def model_bytes(model):
    # Streamed-weight footprint = total bytes of the transformer .safetensors shards (the component
    # that streams; the VAE/TE are placed separately). File size ~= weight bytes; a slight
    # over-count of non-streamed small params is fine for a budget estimate.
    tdir = os.path.join(model, "transformer")
    shards = glob.glob(os.path.join(tdir, "*.safetensors"))
    return int(sum(os.path.getsize(p) for p in shards))


def text_encoder_bytes(model):
    tedir = os.path.join(model, "text_encoder")
    shards = glob.glob(os.path.join(tedir, "*.safetensors"))
    return int(sum(os.path.getsize(p) for p in shards))


def resident_budget(free_vram, reserve):
    # ComfyUI's lowvram_model_memory, [CU model_management.py L935]:
    #   max(0, free - minimum_memory_required, min(free*RATIO, free - minimum_inference_memory()))
    # We use `reserve` (== minimum_inference_memory) for both reserve terms.
    return int(max(0,
                   free_vram - reserve,
                   min(free_vram * MIN_WEIGHT_MEMORY_RATIO, free_vram - reserve)))


def pin_budget():
    # Measured RAM budget for pinning streamed weights into a HostBuffer (truly-async H2D),
    # mirroring ComfyUI ensure_pin_budget [CU model_management.py L645-L656]: available RAM minus a
    # ~2 GB headroom floor. No hardcoded sizes -- adapts to whatever RAM the box has.
    return int(max(0, psutil.virtual_memory().available - PIN_HEADROOM))


def placement(model_size, free_vram, reserve):
    # The one measured decision. full_resident iff the whole transformer + activation reserve fits
    # VRAM (== ComfyUI NORMAL/HIGH_VRAM full_load); otherwise stream with a measured resident
    # budget. There is deliberately NO RAM-vs-disk branch -- the mmap + RAM-pressure cache handles
    # that tier (see PLAN).
    if model_size <= 0:
        # model_bytes() found no transformer shards (e.g. an HF-cache snapshot dir of blob
        # symlinks, or a non-standard layout): we cannot prove it fits, so stream rather than risk
        # a full_resident OOM.
        return ("stream", resident_budget(free_vram, reserve))
    if model_size + reserve <= free_vram:
        return ("full_resident", int(model_size))
    return ("stream", resident_budget(free_vram, reserve))


def probe(model, dev=0):
    # Gather the measured numbers + the placement they imply. Returns a dict; never raises for
    # missing files (model_bytes just returns 0). Call AFTER VAE/TE placement so free_vram reflects
    # the real remaining ceiling for the transformer.
    free, total = gpu_mem(dev)
    reserve = minimum_inference_memory(total)
    msize = model_bytes(model)
    mode, budget = placement(msize, free, reserve)
    return {
        "free_vram": free,
        "total_vram": total,
        "reserve": reserve,
        "model_bytes": msize,
        "te_bytes": text_encoder_bytes(model),
        "mode": mode,
        "budget": budget,
        "pin_budget": pin_budget(),
    }


def describe(p):
    return ("[placement] free_vram=%.2fGB total=%.2fGB reserve=%.2fGB transformer=%.2fGB "
            "te=%.2fGB pin_budget=%.2fGB -> mode=%s budget=%.2fGB"
            % (p["free_vram"] / GB, p["total_vram"] / GB, p["reserve"] / GB,
               p["model_bytes"] / GB, p["te_bytes"] / GB, p.get("pin_budget", 0) / GB,
               p["mode"], p["budget"] / GB))
