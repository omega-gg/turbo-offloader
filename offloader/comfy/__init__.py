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
#   Vendored ComfyUI offloading subsystem -- a byte-for-byte snapshot of ComfyUI's memory manager /
#   model patcher / ops, driven through offloader/adapter.py so turboCLI's diffusers pipelines
#   reuse ComfyUI's device-agnostic (CPU/CUDA/MPS) partial-offload path with 1:1 parity.
#
#   The ONLY edits applied over the upstream files are documented in resync.md and fall in three
#   categories: (1) this sys.modules alias so the vendored `import comfy.X` resolves here, (2)
#   try/except around the optional comfy_aimdo imports, (3) `# [offloader] disabled for turboCLI:`
#   comment-outs. Re-syncing a newer ComfyUI = re-copy the files, re-apply that short list.
#
#   Source snapshots (bump together with the files):
#       ComfyUI     bb131be9e83d2f773c90f1d6f1e4b248a498c8c5  (v0.27.0)
#       comfy-aimdo afa70d91ec9f6e1ab6758089d1b551f0269b6457
#
#==================================================================================================

# The vendored files import each other with absolute names (`import comfy.model_management`,
# `from comfy.cli_args import args`). Alias this package as top-level `comfy` so those resolve to the
# vendored copies without editing a single import line. Import `offloader.comfy` once to
# establish the alias, then always use `comfy.*` (never `offloader.comfy.*`) so there is exactly
# one module object per vendored file.
import sys as _sys

_sys.modules.setdefault("comfy", _sys.modules[__name__])

# comfy_aimdo is the CUDA-only VBAR accelerator (cuMemAddressReserve etc.); it is absent on CPU/MPS
# builds. Every vendored `import comfy_aimdo.X` line is kept verbatim; when the real package is
# missing we register empty stand-in submodules here so those imports resolve. All actual
# comfy_aimdo *usage* in the vendored files is gated on `comfy.memory_management.aimdo_enabled`
# (default False), so the stand-ins are never dereferenced on CPU/MPS -- only the native
# device-agnostic cast_to path runs. Phase D flips aimdo_enabled True when the real package loads.
try:
    import comfy_aimdo  # noqa: F401  (real accelerator present -- nothing to stub)
except ImportError:
    import types as _types

    _stub = _types.ModuleType("comfy_aimdo")
    _stub.__path__ = []  # mark as a package so `import comfy_aimdo.X` is well-formed
    _sys.modules["comfy_aimdo"] = _stub
    for _sub in ("host_buffer", "vram_buffer", "model_vbar", "torch", "model_mmap", "control"):
        _m = _types.ModuleType("comfy_aimdo." + _sub)
        _sys.modules["comfy_aimdo." + _sub] = _m
        setattr(_stub, _sub, _m)
