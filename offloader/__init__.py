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
#  offloader backend (v2) -- the GPL "custom block" the (LGPL) runner discovers as
# backend/<mode>/ (here backend/offloader/ for offload="offloader") and drives through this
# seam only:
#
#      pre_torch_init()                       - one-time setup; MUST run before `import torch`
# available()                            - True once the vendored offloader imports (any device)
# supports(engine)                       - True if this backend can place `engine` load_pipe(model,
# dtype, engine, ...)   - build a fully-placed diffusers pipeline prepare/reclaim/release(pipe)
# - per-generation / teardown hooks
#
#  v2 delegates all offloading to a byte-for-byte vendored ComfyUI snapshot (offloader/comfy/) via
#  the thin bridge in offloader/adapter.py, so the SAME device-agnostic path serves CPU/CUDA/MPS.
#  comfy-aimdo's CUDA-only VBAR is an optional accelerator (aimdo_enabled) that runs ComfyUI's
# ModelPatcherDynamic (partial GPU residency) instead of the plain patcher -- streaming weights
# from host RAM, or disk->VRAM for a component too big for RAM.
#
# =================================================================================================

import os
import glob
import traceback

# True once pre_torch_init() has run. Unlike v1, the native path needs no comfy-aimdo, so this is
# True on CPU/MPS too.
_available = False


# Per-engine diffusers classes: (PipelineCls, TransformerCls). The transformer class is used to
# meta-load an oversized transformer for the disk-stream path. Add engines as scripts migrate.
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

    raise ValueError("offloader backend: unsupported engine %r" % (engine,))


def supports(engine):
    """True if this backend can place `engine`."""
    return engine in ("flux2", "z-image", "qwen-image-edit")


_vbar_ready = False


def pre_torch_init():
    """One-time setup; MUST run before `import torch` (the runner calls it there). Its ONLY job
    that truly needs to precede torch is installing comfy-aimdo's CUDA allocator hooks -- so this
    initialises comfy-aimdo here (and nowhere else) and imports NOTHING that pulls in torch.

    comfy-aimdo is optional: absent (cpu/mps builds) or a failed init just leaves `_vbar_ready`
    False and the portable native cast path runs. When it succeeds, load_pipe flips aimdo_enabled
    and the transformer streams through the VBAR dynamic path. The vendored `comfy` package (which
    imports
    torch) is imported lazily in load_pipe, AFTER the runner's own `import torch`."""
    global _available, _vbar_ready

    try:
        import comfy_aimdo.control as ctl  # must not import torch at module load
        _vbar_ready = bool(ctl.init())
        if _vbar_ready:
            ctl.set_log_warning()
    except Exception:
        _vbar_ready = False

    _available = True


def available():
    """True once pre_torch_init() has brought up the vendored offloader (any device)."""
    return _available


def _direct_load(module, component_dir, device):
    """Materialise `module`'s weights straight into `device` memory via ComfyUI's load_torch_file
    (safetensors safe_open ON the device), then bind them with torch's
    load_state_dict(assign=True). This is exactly how ComfyUI loads a model on MPS -- read direct
    to the compute device -- and it halves placement vs diffusers' CPU load followed by
    load_models_gpu's per-leaf CPU->device copy.

    Model-agnostic: globs every *.safetensors in the diffusers component dir and merges them, so a
    single-file component (flux2, z-image) and a sharded one (qwen-image-edit's index+shards) both
    work with no per-model filename knowledge. load_state_dict is torch's, not ComfyUI's: ComfyUI
    only applies state dicts to its own model classes, so there is no comfy helper for a foreign
    diffusers/transformers module. assign=True rebinds the params to the device-resident tensors (a
    plain copy_ would first need them on device, defeating the point); that breaks tied-weight
    sharing (e.g. Qwen3 tie_word_embeddings, whose lm_head.weight is omitted from the file), so
    re-tie afterwards. On any failure fall back silently: the weights stay as from_pretrained left
    them and
    load_models_gpu places them the normal way."""
    try:
        from .comfy.utils import load_torch_file
        shards = sorted(glob.glob(os.path.join(component_dir, "*.safetensors")))
        if not shards:
            return False
        sd = {}
        for shard in shards:
            sd.update(load_torch_file(shard, device=device))
        module.load_state_dict(sd, assign=True, strict=False)
        if hasattr(module, "tie_weights"):
            module.tie_weights()
        return True
    except Exception:
        print("offloader: direct load failed for %s, using normal placement:\n%s"
              % (component_dir, traceback.format_exc()), flush=True)
        return False


def load_pipe(model, dtype, engine, device="cuda:0", lora_files=None):
    """Build a diffusers pipeline whose big models (transformer, text encoder) are offloaded via
    ComfyUI's ModelPatcher and streamed from host RAM to the compute device per forward -- exactly
    how ComfyUI offloads a UNet. Device-agnostic: `device` selects CPU / CUDA / MPS via the
    adapter. On CUDA with comfy-aimdo present it uses ComfyUI's ModelPatcherDynamic (partial GPU
    residency sized from live free VRAM), streaming each component from host RAM or, when it's too
    big for RAM, from
    disk; otherwise the portable native cast path. lora_files: [(path, strength)]."""
    import torch  # noqa: F401
    from . import adapter

    load_dev = adapter.set_device(device)

    import comfy.model_management as mm

    # CPU: match ComfyUI's storage/compute split. turboCLI hands us fp32 on CPU, which would DOUBLE
    # RAM (store every weight fp32) and skip manual_cast entirely. Instead store the checkpoint's
    # OWN dtype (model-agnostic: bf16 for flux2/z-image, whatever a future model ships) so the mmap
    # slices need no upcast, and let manual_cast_dtype pick the compute dtype (unet_manual_cast:
    # bf16/fp16 aren't natively fast on CPU -> fp32) -- exactly how ComfyUI runs a low-precision
    # checkpoint on CPU. A genuinely fp32 checkpoint stays fp32 (nothing to gain).
    if mm.is_device_cpu(load_dev) and dtype == torch.float32:
        weight_dtype = adapter.weight_dtype(os.path.join(model, "transformer"))
        if weight_dtype is not None and weight_dtype != torch.float32:
            dtype = weight_dtype

    # ComfyUI's storage-vs-compute dtype split: on a card without bf16 tensor cores (e.g. Turing) a
    # bf16 checkpoint computes in fp16 (manual_cast) -- weights stay bf16 (mmap, no RAM blow-up),
    # every matmul runs on fp16 tensor cores (~5x on such cards). None on Ampere+ (compute in
    # bf16).
    manual_cast = adapter.manual_cast_dtype(dtype, load_dev)
    if manual_cast is not None:
        print("offloader: manual_cast compute dtype %s (storage %s)"
              % (manual_cast, dtype), flush=True)

    # ComfyUI's own op-namespace pick (pick_operations): comfy.ops.manual_cast when compute !=
    # weight (every leaf casts, incl. resident ones -- how ComfyUI runs a bf16 model in fp16 on a
    # card without bf16 tensor cores), else disable_weight_init (plain forward unless offloaded).
    # comfy_ize / load_streamed re-class each leaf into this namespace.
    operations = adapter.pick_operations(dtype, manual_cast, load_dev)

    PipelineCls, Transformer = _classes(engine)

    # VBAR: CUDA + comfy-aimdo (initialised in pre_torch_init) -> flip aimdo_enabled so ComfyUI's
    # ModelPatcherDynamic engages (the "dynamic VRAM loading" path). Must run before loading so the
    # vendored device-selection / lazy-load branches see the flag. Off CUDA (or without
    # comfy-aimdo) this is False and the plain-ModelPatcher native cast path runs instead.
    use_vbar = _vbar_ready and adapter.enable_vbar(device)

    # Direct-to-device load, MPS only. On MPS (unified memory) we read the big models straight into
    # device memory (below) instead of diffusers' CPU load + load_models_gpu's per-leaf CPU->device
    # copy. Safe only here: MPS has no separate VRAM pool, so full residency is always chosen and
    # can never OOM -- whereas on CUDA load_models_gpu's partial-residency / VBAR streaming
    # decisions must stand (a full direct load would OOM a smaller-than-model card), and on CPU the
    # weights already load there so there is nothing to gain. LoRA is fine: direct-load only places
    # the base weights; load_models_gpu still merges each patch on top via patch_weight_to_device
    # (verified bit-identical to the normal path). Gated off manual_cast, whose
    # bf16-storage/fp16-compute split relies on the ModelPatcher cast path a direct assign would
    # bypass (never hit on MPS+fp16 anyway -> None).
    direct_load = mm.is_device_mps(load_dev) and manual_cast is None

    # CPU has no VRAM to stream into, but the mmap slice path still applies: keep weights file-backed
    # (safetensors-native mmap) so RAM never holds the whole model -- ComfyUI's low-RAM CPU load.
    # VBAR (dynamic disk->VRAM faulting, comfy-aimdo) stays CUDA-only; on CPU we use the static
    # ModelPatcher + manual_cast, so the same components stream without VBAR.
    is_cpu = mm.is_device_cpu(load_dev)

    patchers = []

    # Big models stream to VRAM per-forward through ComfyUI's ModelPatcherDynamic (comfy-aimdo
    # VBAR): partial GPU residency sized from live free VRAM, so a model larger than VRAM runs
    # -- exactly ComfyUI's dynamic offload. The static ModelPatcher lowvram path can't: on a small
    # card its per-layer peak OOMs a big model (z-image's 12GB DiT on 4GB), which is the whole
    # reason comfy-aimdo's VBAR exists. Off CUDA / without comfy-aimdo -> plain ModelPatcher native
    # cast path.
    build = adapter.build_dynamic_patcher if use_vbar else adapter.build_patcher

    # Transformer: meta-loaded + mmap file-sliced (load_streamed) so its weights never materialise.
    # On CUDA the slices are comfy_aimdo-pinned host buffers streamed to VRAM per-forward (VBAR); on
    # CPU they are safetensors-native mmap, paged from disk on demand. Same load_streamed either way
    # -- ComfyUI's load_torch_file picks the mmap source per aimdo_enabled. Slices come from the page
    # cache when the model fits RAM, or straight from disk when it doesn't (qwen-image-edit ~55GB).
    stream_transformer = use_vbar or is_cpu

    if stream_transformer:
        transformer, missing = adapter.load_streamed(
            Transformer, os.path.join(model, "transformer"), dtype, operations)
        if missing:
            print("offloader: %d transformer weights had no matching module (skipped)"
                  % len(missing), flush=True)
        p = PipelineCls.from_pretrained(model, transformer=transformer, torch_dtype=dtype,
                                        use_safetensors=True, low_cpu_mem_usage=True,
                                        local_files_only=True)
    else:
        p = PipelineCls.from_pretrained(model, torch_dtype=dtype, use_safetensors=True,
                                        low_cpu_mem_usage=True, local_files_only=True)
        adapter.comfy_ize(p.transformer, operations)
        if direct_load:
            _direct_load(p.transformer, os.path.join(model, "transformer"), load_dev)

    adapter.keep_uncastable_resident(p.transformer, load_dev, manual_cast)
    if stream_transformer and use_vbar:
        adapter.install_prefetch(p.transformer)  # overlap disk->VRAM streaming with compute (VBAR)
    # ComfyUI's exact SDPA (comfy.ops, cuDNN-first + size gate)
    adapter.use_comfy_attention(p.transformer)
    adapter.use_kitchen_rope(p.transformer)  # ComfyUI's comfy_kitchen fused RoPE
    if manual_cast is not None:
        # bf16 storage, manual_cast compute (fp16 on Turing GPUs, fp32 on CPU)
        adapter.install_manual_cast(p.transformer, manual_cast, dtype)
    transformer_patcher = build(p.transformer)

    # On-cast LoRA (e.g. qwen's Lightning 4-step speed LoRA): applied to the transformer while its
    # weights stream in, via ComfyUI's add_patches -> calculate_weight. lora_files: [(path,
    # strength)].
    if lora_files:
        n = adapter.add_lora(transformer_patcher, lora_files)
        print("offloader: applied %d LoRA patches" % n, flush=True)

    patchers.append(transformer_patcher)

    # Text encoder: same as the transformer -- weights swapped for mmap file-slices and streamed
    # pinned via VBAR. Custom norms (e.g. Qwen3RMSNorm) are kept resident by
    # keep_uncastable_resident since they can't be comfy-ized.
    if getattr(p, "text_encoder", None) is not None:
        adapter.comfy_ize(p.text_encoder, operations)
        adapter.use_comfy_attention(p.text_encoder)
        if direct_load:
            _direct_load(p.text_encoder, os.path.join(model, "text_encoder"), load_dev)
        stream_text_encoder = use_vbar or is_cpu
        if stream_text_encoder:
            te_missing = adapter.assign_streamed_weights(p.text_encoder,
                                                         os.path.join(model, "text_encoder"))
            if te_missing:
                print("offloader: %d text-encoder weights had no matching param (skipped)"
                      % len(te_missing), flush=True)
        adapter.keep_uncastable_resident(p.text_encoder, load_dev, manual_cast)
        if manual_cast is not None:
            # ComfyUI casts the TE the same way (manual_cast compute)
            adapter.install_manual_cast(p.text_encoder, manual_cast, dtype)
        encoder_patcher = build(p.text_encoder)
        patchers.append(encoder_patcher)
        p._offloader_encoder = encoder_patcher

    # Small resident modules (VAE) go straight to the compute device; the offloader handles the
    # heavy ones. VAE tiling/slicing is left to the caller.
    if getattr(p, "vae", None) is not None:
        # ComfyUI runs the VAE in fp32 on CPU (vae_dtype: bf16/fp16 conv is unsupported / emulated
        # ~10-30x slower there, so it falls through to fp32). Our CPU storage dtype is bf16, which
        # would leave the VAE in bf16 -- a multi-minute decode. Upcast it to fp32 on CPU; on GPU keep
        # the pipeline dtype.
        if is_cpu:
            p.vae.to(device=load_dev, dtype=torch.float32)
            # diffusers' flux2 pipeline hands the VAE bf16 latents; ComfyUI upcasts the latent to the
            # VAE's own dtype before decode (VAE.decode: samples.to(self.vae_dtype)). Mirror that so
            # the fp32 VAE never sees bf16 input (else conv raises a dtype mismatch).
            _vae = p.vae

            def _cast_in(fn):
                def wrapped(x, *a, **k):
                    return fn(x.to(torch.float32) if hasattr(x, "to") else x, *a, **k)
                return wrapped

            _vae.decode = _cast_in(_vae.decode)
            _vae.encode = _cast_in(_vae.encode)
        else:
            p.vae.to(load_dev)

    p._offloader_patchers = patchers
    p._offloader_device = load_dev

    # Diffusers reads _execution_device from module placement; pin it to the compute device so
    # inputs land there while offloaded weights stream in.
    try:
        p._execution_device = load_dev
    except Exception:
        pass

    # NOTE: This might improve performances.
    p.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # Cache the text-encoder output by prompt (ComfyUI's node cache): a repeated prompt then skips
    # the encoder's forward, so the streamed encoder is never loaded and the transformer keeps the
    # VRAM.
    adapter.install_encode_cache(p)

    return p


def prepare(pipe):
    """Per-generation load boundary: hand the managed models to ComfyUI's load_models_gpu, which
    partial-loads / streams them to the compute device (its dynamic path streams weights
    per-forward, so this marks them loaded without pinning the whole set). When
    install_encode_cache serves a repeated prompt the text encoder's forward never runs, so its
    weights never stream and the transformer keeps
    the throughput."""
    patchers = getattr(pipe, "_offloader_patchers", None)
    if not patchers:
        return

    import comfy.model_management as mm
    mm.load_models_gpu(patchers)


def reclaim(pipe):
    """Per-generation housekeeping: run ComfyUI's per-execution offloader teardown, then free
    non-resident models + return the allocator pool.

    The teardown copies what ComfyUI runs in the `finally` after EVERY node execution when
    offloader is on (comfy/execution.py:544-549): reset_cast_buffers + cleanup_prefetch_queues +
    vbars_reset_watermark_limits. It resets the VBAR streaming state (the globally-cached cast
    buffers, the block prefetch queues, the VBAR watermarks) between generations. reclaim() is our
    analog of that per-execution boundary, so we do the same thing ComfyUI does. (This is faithful
    teardown / hygiene; it is NOT what fixed the flux2->z-image first-gen nondeterminism -- that
    was copying
    ComfyUI's exact SDPA size gate in adapter.use_comfy_attention.)"""
    if not getattr(pipe, "_offloader_patchers", None):
        return

    import comfy.model_management as mm
    import comfy.memory_management as memm

    if getattr(memm, "aimdo_enabled", False):
        try:
            mm.reset_cast_buffers()

            import comfy.model_prefetch as mp
            mp.cleanup_prefetch_queues()

            import comfy_aimdo.model_vbar as mv
            mv.vbars_reset_watermark_limits()
        except Exception:
            print(traceback.format_exc(), flush=True)

    dev = getattr(pipe, "_offloader_device", None)
    if dev is not None:
        mm.free_memory(mm.minimum_inference_memory(), dev)
    mm.soft_empty_cache()


def release(pipe):
    """Tear down the offloaders before the pipe is dropped (detach unpatches weights and lets the
    current_loaded_models finalizers fire)."""
    for patcher in getattr(pipe, "_offloader_patchers", []):
        try:
            patcher.detach(unpatch_all=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
