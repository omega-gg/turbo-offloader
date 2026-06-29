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

# =================================================================================================
#  aimdo offload backend -- the GPL "custom block" the (LGPL) runner discovers as backend/<mode>/
#  (here backend/aimdo/ for cuda_offload="aimdo") and drives through this seam only:
#
#      pre_torch_init()                       - one-time setup; MUST run before `import torch`
#      available()                            - True once comfy-aimdo initialised (CUDA-only)
#      supports(engine)                       - True if this backend can place `engine`
#      load_pipe(model, dtype, engine, ...)   - build a fully-placed diffusers pipeline
#      prepare/reclaim/release(pipe)          - optional per-generation / teardown hooks (no-ops
#                                               unless the host caches + reuses the pipe)
#
#  All GPL-derived code lives in this package (__init__.py + offload.py + placement.py); the
#  calling shell scripts stay GPL-free. Design + upstream line references: aimdo.md.
# =================================================================================================

import os

# Set by pre_torch_init(): True once comfy-aimdo brought up a device. cpu/mps builds don't ship
# comfy-aimdo, so pre_torch_init() raises there and the caller leaves the backend out.
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
    """Install the comfy-aimdo CUDA hooks. MUST be called before torch is imported.

    Raises if comfy-aimdo is not installed (cpu/mps builds), so a host that wants to degrade
    gracefully should call this inside try/except and gate on available().
    """
    global _available

    import comfy_aimdo.control as ctl

    _available = bool(ctl.init())
    ctl.set_log_warning()


def available():
    """True once pre_torch_init() has brought up comfy-aimdo (CUDA-only; absent on cpu/mps
    builds)."""
    return _available


def load_pipe(model, dtype, engine, device="cuda:0", lora_files=None):
    """Resource-driven placement of `engine` (all budgets MEASURED -- placement.py): full_resident
    when the transformer fits VRAM, else streamed through the VBAR offloader. lora_files: on-cast
    LoRA specs (path or (path, weight)); None for engines without LoRA. See aimdo.md."""
    from . import placement

    plan = placement.probe(model)
    print(placement.describe(plan), flush=True)

    if plan["mode"] == "full_resident":
        # Whole transformer fits VRAM: load resident, no offloader. Reached only by small models on
        # big GPUs (qwen never fits).
        PipelineCls, _ = _classes(engine)

        p = PipelineCls.from_pretrained(model, torch_dtype=dtype, use_safetensors=True,
                                        low_cpu_mem_usage=True, local_files_only=True)
        p.to("cuda")
    else:
        p = _load_streamed(model, dtype, engine, lora_files)   # unified merged VBAR offloader

    # NOTE: This might improve performances.
    p.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    return p


def _load_streamed(model, dtype, engine, lora_files):
    # Unified loader: meta-load the transformer (no pageable weight copy), then stream its Linear
    # weights to the GPU per forward via the VBAR offloader; the text encoder is a second managed
    # offloader streamed from disk. Both coexist as managed dynamic VBARs (ComfyUI-style); only the
    # class table + qwen's on-cast LoRA differ per engine. See aimdo.md.
    from . import offload
    from . import placement

    from accelerate import init_empty_weights

    PipelineCls, Transformer = _classes(engine)

    tdir = os.path.join(model, "transformer")
    cfg = Transformer.load_config(tdir)

    with init_empty_weights():
        meta_transformer = Transformer.from_config(cfg).to(dtype)

    # Build the pipe FIRST so placement.pin_budget() is measured AFTER the text encoder is in host
    # RAM (conservative). from_pretrained accepts the meta transformer as-is (never loads weights).
    p = PipelineCls.from_pretrained(
        model,
        transformer=meta_transformer,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )

    p.vae.to("cuda")

    # VAE stays full-frame (slicing is a no-op at batch=1; full-frame fits <=1024x768 with
    # cudaMallocAsync). The caller can enable VAE tiling (slicing="slice") for higher resolutions.
    # See aimdo.md.

    # On-cast LoRA (qwen): the caller resolves the specs and passes them as lora_files (each spec
    # is (path, weight); deltas accumulate). None for flux2/z-image.

    # Transformer = a managed dynamic VBAR. manage=True everywhere mirrors ComfyUI (registration is
    # unconditional); the manager reclaims the inactive model's GPU footprint at the active one's
    # load boundary -- made explicit here rather than via comfy-aimdo's on-demand eviction.
    # aimdo.md.
    transformer_offloader = offload.Offloader(
        meta_transformer, tdir, "cuda:0", lora_files=lora_files or None,
        pin_budget=placement.pin_budget(), manage=True)

    # Text encoder: a managed dynamic VBAR for every engine, streamed from disk via from_module
    # (_match_disk_keys maps live names -> disk keys model-agnostically, so it works whether the
    # loader rewrote the keys or not). Big Linears stream (host RAM = page cache); small params
    # (embedding, norms, qwen's conv3d) stay resident on CUDA. The tied embedding/lm_head is never
    # streamed, so the tie is preserved. manage=True -> released during denoise, reloaded per
    # encode. See aimdo.md.
    encoder_offloader = offload.Offloader(
        p.text_encoder, tdir=os.path.join(model, "text_encoder"), from_module=True,
        device="cuda:0", manage=True)

    # prepare() reloads the TE to GPU BEFORE the pipeline reads _execution_device (else input lands
    # on CPU while the weight is on CUDA). A fresh single-shot pipe runs without it; it matters
    # only when the pipe is reused. See aimdo.md.
    p._aimdo_encoder = encoder_offloader

    # Keep both offloaders reachable so release() can free() them (else they leak via the
    # offloader<->module hook cycle).
    p._aimdo_offloaders = [transformer_offloader, encoder_offloader]

    return p


def prepare(pipe):
    """Per-generation load boundary: reload the managed text encoder to GPU before pipe() reads
    _execution_device. No-op for a pipe with no managed encoder, or a fresh single-shot pipe.
    See aimdo.md."""
    enc = getattr(pipe, "_aimdo_encoder", None)

    if enc is not None:
        enc.activate()


def reclaim(pipe):
    """Per-generation housekeeping: return torch's retained allocator pool + drop the VBAR resident
    floors so the persistent VBAR weights don't re-stream every layer. No-op without offloaders.
    See aimdo.md (offload.reclaim_between_runs)."""
    if getattr(pipe, "_aimdo_offloaders", None):
        from . import offload

        offload.reclaim_between_runs()


def release(pipe):
    """Tear down any offloaders before the pipe is dropped (free() unregisters pinned regions +
    decommits the host buffer; a plain del would leak via the offloader<->module cycle). See
    aimdo.md (offload.Offloader.free)."""
    import traceback

    for offloader in getattr(pipe, "_aimdo_offloaders", []):
        try:
            offloader.free()
        except Exception:
            print(traceback.format_exc(), flush=True)
