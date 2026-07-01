#==================================================================================================
#
#   Copyright (C) 2026-2026 turbo-aimdo authors. <https://omega.gg/turbo-aimdo>
#
#   Author: Benjamin Arnaud. <https://bunjee.me> <bunjee@omega.gg>
#
#   This file is part of turbo-aimdo.
#
#   - GNU General Public License Usage:
#   This file may be used under the terms of the GNU General Public License version 3 as published
#   by the Free Software Foundation and appearing in the LICENSE.md file included in the packaging
#   of this file. Please review the following information to ensure the GNU General Public License
#   requirements will be met: https://www.gnu.org/licenses/gpl.html.
#
#==================================================================================================
#
#   End-to-end driver for qwen-image-edit (an IMAGE-input / edit engine) through the v2 seam. Drives
#   the Lightning 4-step speed LoRA on-cast. The transformer (~39GB) + text encoder (~16GB) stream
#   disk->VRAM via VBAR, so it runs on a tiny GPU (very slow -- 55GB streams from disk).
#
#   AIMDO_MODEL points at the Qwen-Image-Edit-2511 diffusers dir (which also holds the LoRA files).
#
#   Run:
#       AIMDO_MODEL=/path/to/Qwen-Image-Edit-2511  python tests/drive_qwen.py cuda 512 512 4
#       args: <device=cuda> <width=512> <height=512> <steps=4>
#
#==================================================================================================

import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 512
HEIGHT = int(sys.argv[3]) if len(sys.argv) > 3 else 512
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 4

MODEL = os.environ.get("AIMDO_MODEL")
if not MODEL:
    sys.exit("set AIMDO_MODEL to the Qwen-Image-Edit-2511 directory")

# Lightning 4-step speed LoRA -> apply on-cast so 4 steps suffice.
LIGHTNING = os.path.join(MODEL, "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
lora_files = [(LIGHTNING, 1.0)] if os.path.exists(LIGHTNING) else None

import aimdo

aimdo.pre_torch_init()
print("available:", aimdo.available(), "| lora:", "yes" if lora_files else "no")

import torch
from PIL import Image
import numpy as np

# Input image to edit (a simple synthetic scene so no asset is needed).
xs = np.linspace(0, 255, WIDTH, dtype=np.uint8)
grad = np.stack([np.tile(xs, (HEIGHT, 1)),
                 np.tile(xs[::-1], (HEIGHT, 1)),
                 np.full((HEIGHT, WIDTH), 128, np.uint8)], axis=-1)
in_img = Image.fromarray(grad, "RGB")

dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()
    print("GPU:", torch.cuda.get_device_name(0))

t0 = time.time()
pipe = aimdo.load_pipe(model=MODEL, dtype=dtype, engine="qwen-image-edit", device=DEVICE,
                       lora_files=lora_files)
print("load_pipe: %.1fs" % (time.time() - t0))

aimdo.prepare(pipe)
print("prepared; execution_device:", getattr(pipe, "_execution_device", "?"))

gen = torch.Generator(device="cpu").manual_seed(42)
t1 = time.time()
marks = []
with torch.inference_mode():
    img = pipe(image=in_img, prompt="turn the sky into a starry night",
               num_inference_steps=STEPS, true_cfg_scale=1.0, generator=gen,
               callback_on_step_end=lambda p, s, t, k: (marks.append(time.time()), k)[1]).images[0]
print("generate: %.1fs" % (time.time() - t1))
if marks:
    steps = [b - a for a, b in zip(marks[:-1], marks[1:])]
    if steps:
        print("  denoise avg/step: %.1fs" % (sum(steps) / len(steps)))
if DEVICE == "cuda":
    print("peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_qwen_%s_%dx%d.png" % (DEVICE, WIDTH, HEIGHT))
img.save(out)
print("Saved:", out)

aimdo.reclaim(pipe)
aimdo.release(pipe)
print("done")
