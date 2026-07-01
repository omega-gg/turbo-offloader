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
#  aimdo offload backend (v2) -- the GPL "custom block" the (LGPL) runner discovers as
#  backend/<mode>/ (here backend/aimdo/ for cuda_offload="aimdo") and drives through this seam only:
#
#      pre_torch_init()                       - one-time setup; MUST run before `import torch`
#      available()                            - True once the vendored offloader imports (any device)
#      supports(engine)                       - True if this backend can place `engine`
#      load_pipe(model, dtype, engine, ...)   - build a fully-placed diffusers pipeline
#      prepare/reclaim/release(pipe)          - per-generation / teardown hooks
#
#  v2 delegates all offloading to a byte-for-byte vendored ComfyUI snapshot (aimdo/comfy/) via the
#  thin bridge in aimdo/adapter.py, so the SAME device-agnostic path serves CPU / CUDA / MPS.
#  comfy-aimdo's CUDA-only VBAR is an optional accelerator (aimdo_enabled), off by default. The v1
#  VBAR streamer (offload.py + placement.py) is kept as reference until Phase D re-homes it.
#
# =================================================================================================

import os
import traceback

# True once pre_torch_init() has run. Unlike v1, the native path needs no comfy-aimdo, so this is
# True on CPU/MPS too.
_available = False


# Per-engine diffusers classes: (PipelineCls, TransformerCls). Add engines as scripts migrate.
def _classes(engine):
    if engine == "flux2":
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        return Flux2KleinPipeline, Flux2Transformer2DModel

    if engine == "z-image":
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        return ZImagePipeline, ZImageTransformer2DModel

    if engine == "qwen-image-edit":
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
        return QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    raise ValueError("aimdo backend: unsupported engine %r" % (engine,))


def supports(engine):
    """True if this backend can place `engine`."""
    return engine in ("flux2", "z-image", "qwen-image-edit")


def pre_torch_init():
    """Bring up the vendored offloader. The native (device-agnostic) path needs NO comfy-aimdo, so
    this never raises and never touches CUDA -- importing the adapter just establishes the `comfy`
    alias and validates the vendored package.

    We deliberately do NOT call comfy_aimdo.control.init() here. Its CUDA allocator / VBAR hooks map
    physical VRAM through a separate CUDA VMM that competes with torch's pool (v1 aimdo.md), and they
    also reserve lockable host RAM -- both of which starve the native cast path (measured: stole
    enough VRAM to OOM z-image on a 4.3GB GPU, and pushed pinning over the RAM ceiling). comfy-aimdo
    is only initialised in Phase D, behind an explicit opt-in, when the VBAR accelerator is enabled
    (aimdo_enabled=True)."""
    global _available

    # Importing the adapter establishes the `comfy` alias and validates the vendored package.
    import aimdo.adapter  # noqa: F401
    _available = True


def available():
    """True once pre_torch_init() has brought up the vendored offloader (any device)."""
    return _available


def load_pipe(model, dtype, engine, device="cuda:0", lora_files=None):
    """Build a diffusers pipeline whose big models (transformer, text encoder) are offloaded through
    ComfyUI's ModelPatcher and streamed to the compute device per forward. Device-agnostic: `device`
    selects CPU / CUDA / MPS via the adapter. lora_files kept for parity (wired in Phase C)."""
    import torch  # noqa: F401
    import aimdo.adapter as adapter

    load_dev = adapter.set_device(device)

    PipelineCls, _Transformer = _classes(engine)

    # Load every module onto the offload device (CPU) with mmap-backed safetensors. ModelPatcher
    # then streams weights to `load_dev` per forward -- exactly like ComfyUI offloads a UNet.
    p = PipelineCls.from_pretrained(model, torch_dtype=dtype, use_safetensors=True,
                                    low_cpu_mem_usage=True, local_files_only=True)

    patchers = []

    # Transformer: the big model -> comfy-ize its standard leaves, keep custom param leaves (norms
    # etc.) resident so they aren't stranded on the offload device, then wrap in a ModelPatcher.
    adapter.comfy_ize(p.transformer)
    adapter.keep_uncastable_resident(p.transformer, load_dev)
    transformer_patcher = adapter.build_patcher(p.transformer)
    patchers.append(transformer_patcher)

    # Text encoder: also large for flux2/qwen -> its own managed patcher. Its custom norms (e.g.
    # Qwen3RMSNorm) can't be comfy-ized, so keep them resident too.
    if getattr(p, "text_encoder", None) is not None:
        adapter.comfy_ize(p.text_encoder)
        adapter.keep_uncastable_resident(p.text_encoder, load_dev)
        encoder_patcher = adapter.build_patcher(p.text_encoder)
        patchers.append(encoder_patcher)
        p._aimdo_encoder = encoder_patcher

    # Small resident modules (VAE) go straight to the compute device; the offloader handles the
    # heavy ones. VAE tiling/slicing is left to the caller.
    if getattr(p, "vae", None) is not None:
        p.vae.to(load_dev)

    p._aimdo_patchers = patchers
    p._aimdo_device = load_dev

    # Diffusers reads _execution_device from module placement; pin it to the compute device so inputs
    # land there while offloaded weights stream in.
    try:
        p._execution_device = load_dev
    except Exception:
        pass

    # NOTE: This might improve performances.
    p.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    return p


def prepare(pipe):
    """Per-generation load boundary: ask ComfyUI to place the managed models on the compute device
    (partial load + cast-path flags) before the pipeline reads _execution_device / runs a forward."""
    patchers = getattr(pipe, "_aimdo_patchers", None)
    if not patchers:
        return

    import comfy.model_management as mm
    mm.load_models_gpu(patchers)


def reclaim(pipe):
    """Per-generation housekeeping: free non-resident models + return the allocator pool."""
    if not getattr(pipe, "_aimdo_patchers", None):
        return

    import comfy.model_management as mm
    dev = getattr(pipe, "_aimdo_device", None)
    if dev is not None:
        mm.free_memory(mm.minimum_inference_memory(), dev)
    mm.soft_empty_cache()


def release(pipe):
    """Tear down the offloaders before the pipe is dropped (detach unpatches weights and lets the
    current_loaded_models finalizers fire)."""
    for patcher in getattr(pipe, "_aimdo_patchers", []):
        try:
            patcher.detach(unpatch_all=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
