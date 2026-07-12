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
#  backend/<mode>/ (here backend/offloader/ for offload="offloader") and drives through this
#  seam only:
#
#      pre_torch_init()                     - one-time setup; MUST run before `import torch`
#      available()                          - True once the vendored offloader imports (any device)
#      supports(engine)                     - True if this backend can place `engine`
#      load_pipe(model, dtype, ...)         - build a fully-placed diffusers pipeline
#      prepare/reclaim/release(pipe)        - per-generation / teardown hooks
#
#  v2 delegates all offloading to a byte-for-byte vendored ComfyUI snapshot (offloader/comfy/) via
#  the thin bridge in offloader/adapter.py, so the SAME device-agnostic path serves CPU/CUDA/MPS.
#  comfy-aimdo's CUDA-only VBAR is an optional accelerator (aimdo_enabled) that runs ComfyUI's
#  ModelPatcherDynamic (partial GPU residency) instead of the plain patcher -- streaming weights
#  from host RAM, or disk->VRAM for a component too big for RAM.
#
# =================================================================================================

import os
import glob
import traceback

# True once pre_torch_init() has run. Unlike v1, the native path needs no comfy-aimdo, so this is
# True on CPU/MPS too.
_available = False


def supports(engine):
    """Model-agnostic: the offloader places any diffusers pipeline the runner wires with
    PIPELINE/TRANSFORMER -- its mechanics key off module structure (comfy_ize by leaf type,
    rope/attention by detected symbol, file-sliced streaming), never a model name -- so it claims
    every engine. Offload eligibility (which engines are wired for it) is a turboCLI-side decision,
    gated on that declaration; the backend keeps no model list."""
    return True


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


def _prepare_offload(device, dtype, fits_full_load):
    """Shared offloader setup for load_pipe / load_pipe_single_file. Resolves the compute device,
    then -- exactly as ComfyUI decides -- the CPU comfy/stream mode + storage dtype, the manual_cast
    compute dtype, the comfy op namespace, whether comfy-aimdo VBAR engages, and the patcher builder.
    `fits_full_load` is a thunk giving the caller's 'does the fp32 model fit RAM' verdict (consulted
    only on CPU with no OFFLOADER_CPU_MODE override). Returns
    (load_device, dtype, cpu_stream, manual_cast, operations, use_vbar, build)."""
    from . import adapter

    import comfy.model_management as mm

    load_device = adapter.set_device(device)

    # CPU placement mode. ComfyUI's default CPU path is fp32 storage, fully materialised -- faster
    # when the model fits RAM (cast once at load, then plain matmuls), an OOM when it doesn't; the
    # stream fallback is bf16 storage (--bf16-unet), mmap-kept (never materialised, RAM bounded),
    # fp32 compute cast per forward. Auto-picks by fit -- ComfyUI's full-load-when-it-fits, applied
    # to CPU; OFFLOADER_CPU_MODE=comfy|stream forces one.
    cpu_stream = False
    if mm.is_device_cpu(load_device):
        mode = os.environ.get("OFFLOADER_CPU_MODE")
        if mode not in ("comfy", "stream"):
            mode = "comfy" if fits_full_load() else "stream"
        cpu_stream = mode == "stream"
        # Drive the storage dtype through ComfyUI's own unet_dtype(): --bf16-unet (bf16) when
        # streaming, else its CPU default (fp32). manual_cast_dtype (below) then picks fp32 compute.
        from comfy.cli_args import args as _args
        _args.bf16_unet = cpu_stream
        dtype = mm.unet_dtype()
        print("offloader: CPU mode=%s (storage %s)" % (mode, dtype), flush=True)

    # ComfyUI's storage-vs-compute dtype split: on a card without bf16 tensor cores (e.g. Turing) a
    # bf16 checkpoint computes in fp16 (manual_cast) -- weights stay bf16 (mmap, no RAM blow-up),
    # every matmul runs on fp16 tensor cores (~5x on such cards). None on Ampere+ (compute in bf16).
    manual_cast = adapter.manual_cast_dtype(dtype, load_device)
    if manual_cast is not None:
        print("offloader: manual_cast compute dtype %s (storage %s)"
              % (manual_cast, dtype), flush=True)

    # ComfyUI's own op-namespace pick (pick_operations): comfy.ops.manual_cast when compute != weight
    # (every leaf casts, incl. resident ones), else disable_weight_init (plain forward unless
    # offloaded). comfy_ize / the streamers re-class each leaf into this namespace.
    operations = adapter.pick_operations(dtype, manual_cast, load_device)

    # VBAR: CUDA + comfy-aimdo (initialised in pre_torch_init) -> flip aimdo_enabled so ComfyUI's
    # ModelPatcherDynamic engages (partial GPU residency sized from live free VRAM, streaming the
    # rest per forward). Off CUDA / without comfy-aimdo -> plain ModelPatcher native cast path.
    use_vbar = _vbar_ready and adapter.enable_vbar(device)
    build = adapter.build_dynamic_patcher if use_vbar else adapter.build_patcher

    return load_device, dtype, cpu_stream, manual_cast, operations, use_vbar, build


def load_pipe(model, dtype, pipeline_cls, transformer_cls, device="cuda:0", lora_files=None):
    """Build a diffusers pipeline whose big models (transformer, text encoder) are offloaded via
    ComfyUI's ModelPatcher and streamed from host RAM to the compute device per forward -- exactly
    how ComfyUI offloads a UNet. Device-agnostic: `device` selects CPU / CUDA / MPS via the
    adapter. On CUDA with comfy-aimdo present it uses ComfyUI's ModelPatcherDynamic (partial GPU
    residency sized from live free VRAM), streaming each component from host RAM or, when it's too
    big for RAM, from
    disk; otherwise the portable native cast path. pipeline_cls / transformer_cls are the diffusers
    classes supplied by the runner from its engine declaration (the transformer class meta-loads an
    oversized transformer for the disk-stream path). lora_files: [(path, strength)]."""
    import torch  # noqa: F401
    from . import adapter

    import comfy.model_management as mm

    load_device, dtype, cpu_stream, manual_cast, operations, use_vbar, build = _prepare_offload(
        device, dtype, lambda: adapter.cpu_fits_full_load(model))

    # Pipeline + transformer classes come from the runner (turboCLI engine/<name>.py PIPELINE /
    # TRANSFORMER); the transformer meta-loads an oversized transformer for the disk-stream path.
    PipelineCls, Transformer = pipeline_cls, transformer_cls

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
    direct_load = mm.is_device_mps(load_device) and manual_cast is None

    patchers = []

    # Transformer: meta-loaded + mmap file-sliced (load_streamed) so its weights never materialise.
    # On CUDA the slices are comfy_aimdo-pinned host buffers streamed to VRAM per-forward (VBAR); on
    # CPU they are safetensors-native mmap, paged from disk on demand. Same load_streamed either way
    # -- ComfyUI's load_torch_file picks the mmap source per aimdo_enabled. Slices come from the page
    # cache when the model fits RAM, or straight from disk when it doesn't (qwen-image-edit ~55GB).
    stream_transformer = use_vbar or cpu_stream

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
            _direct_load(p.transformer, os.path.join(model, "transformer"), load_device)

    adapter.keep_uncastable_resident(p.transformer, load_device, manual_cast)
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
        stream_text_encoder = use_vbar or cpu_stream
        # ComfyUI runs the text encoder on ComfyUI's own text_encoder_device() -- CPU under
        # vram_state SHARED (Apple Silicon), the compute device otherwise -- and moves only the
        # conditioning to the compute device. Honor that for the resident path (device-agnostic via
        # comfy's own selector): when it lands the TE off the compute device (MPS -> CPU), residency
        # is the transformer alone and encode runs on te_dev via the bridge below. The streaming
        # paths (VBAR / CPU-stream) place the TE their own way, so keep load_device there.
        te_dev = load_device if stream_text_encoder else mm.text_encoder_device()
        adapter.comfy_ize(p.text_encoder, operations)
        adapter.use_comfy_attention(p.text_encoder)
        if direct_load and te_dev == load_device:
            _direct_load(p.text_encoder, os.path.join(model, "text_encoder"), load_device)
        if stream_text_encoder:
            te_missing = adapter.assign_streamed_weights(p.text_encoder,
                                                         os.path.join(model, "text_encoder"))
            if te_missing:
                print("offloader: %d text-encoder weights had no matching param (skipped)"
                      % len(te_missing), flush=True)
        adapter.keep_uncastable_resident(p.text_encoder, te_dev, manual_cast)
        if manual_cast is not None:
            # ComfyUI casts the TE the same way (manual_cast compute)
            adapter.install_manual_cast(p.text_encoder, manual_cast, dtype)
        encoder_patcher = (build(p.text_encoder) if te_dev == load_device
                           else adapter.build_patcher(p.text_encoder, load_device=te_dev,
                                                      offload_device=te_dev))
        patchers.append(encoder_patcher)
        p._offloader_encoder = encoder_patcher
        p._offloader_te_device = te_dev

    return _finalize_pipe(p, patchers, load_device, cpu_stream)


def _finalize_pipe(p, patchers, load_device, cpu_stream):
    """Shared tail for the offloader pipes (dir-based load_pipe and single-file
    load_pipe_single_file). Places the small VAE, records the patchers, pins the execution device,
    wires the ComfyUI-faithful encode bridge, and installs the prompt-encode cache. The heavy models
    (transformer, text encoder) are already comfy-ized / patched by the caller; p._offloader_te_device
    tells us where the encoder lives."""
    import torch
    from . import adapter

    # Small resident modules (VAE) go straight to the compute device; the offloader handles the
    # heavy ones. VAE tiling/slicing is left to the caller.
    if getattr(p, "vae", None) is not None:
        # ComfyUI runs the VAE in fp32 on CPU (vae_dtype: bf16/fp16 conv is unsupported / emulated
        # ~10-30x slower there, so it falls through to fp32). In stream mode our storage dtype is
        # bf16, which would leave the VAE in bf16 -- a multi-minute decode; upcast it to fp32. In
        # comfy mode the pipeline is already fp32, and on GPU we keep the pipeline dtype.
        if cpu_stream:
            p.vae.to(device=load_device, dtype=torch.float32)
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
            p.vae.to(load_device)

    p._offloader_patchers = patchers
    p._offloader_device = load_device

    # Diffusers infers _execution_device from module placement. Normally every managed module sits
    # on the compute device, so the historical best-effort hint below suffices (and CUDA / CPU keep
    # exactly that path). Only when the text encoder is placed OFF the compute device (te_dev !=
    # load_device, e.g. MPS -> CPU) would the property resolve to that other device and build the
    # timesteps/latents there, mismatching the compute-device transformer -- so in that case pin it
    # to the compute device via a per-instance subclass (ComfyUI runs the sampler on the compute
    # device regardless of where the text encoder lives). The encode still runs on te_dev because the
    # bridge below overrides encode_prompt's device argument explicitly.
    te_off_device = getattr(p, "_offloader_te_device", load_device) != load_device
    if te_off_device:
        _cls = type(p)
        p.__class__ = type(_cls.__name__, (_cls,),
                           {"_execution_device": property(lambda self: load_device)})
    else:
        try:
            p._execution_device = load_device
        except Exception:
            pass

    # NOTE: This might improve performances.
    p.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # ComfyUI-faithful encode placement: when the TE lives off the compute device (CPU on MPS),
    # diffusers' __call__ would still pass device=_execution_device to encode_prompt and stream the
    # TE to MPS per leaf. Run the encode on the TE's own device instead (native forward, no stream),
    # then move only the (small) embeddings to the compute device -- ComfyUI's "encode on CPU,
    # conditioning to GPU". Wrapped BEFORE install_encode_cache so the cache stores the CPU result.
    te_dev = getattr(p, "_offloader_te_device", load_device)
    if te_dev != load_device and getattr(p, "encode_prompt", None) is not None:
        _real_encode = p.encode_prompt

        def _encode_on_te(*args, **kwargs):
            kwargs["device"] = te_dev
            out = _real_encode(*args, **kwargs)
            if isinstance(out, (list, tuple)):
                return type(out)(o.to(load_device) if isinstance(o, torch.Tensor) else o for o in out)
            return out.to(load_device) if isinstance(out, torch.Tensor) else out

        p.encode_prompt = _encode_on_te

    # Cache the text-encoder output by prompt (ComfyUI's node cache): a repeated prompt then skips
    # the encoder's forward, so the streamed encoder is never loaded and the transformer keeps the
    # VRAM.
    adapter.install_encode_cache(p)

    return p


def load_pipe_single_file(scaffold, files, dtype, pipeline_cls, transformer_cls,
                          device="cuda:0", lora_files=None):
    """load_pipe's ComfyUI-reuse sibling: same disk-stream offload, but the big models are meta-
    loaded straight from ComfyUI's split single files (files[role]) rather than a diffusers component
    dir. The tiny configs/tokenizer/scheduler come from `scaffold` (the engine's turbo-model dir).
    Only the weight source differs -- transformer via the diffusers single-file remap, text encoder
    with the ComfyUI `model.` prefix stripped -- everything after (patchers, placement, encode cache)
    is the shared _finalize_pipe tail."""
    import os
    import torch  # noqa: F401
    from . import adapter
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.loaders.single_file_utils import (
        convert_z_image_transformer_checkpoint_to_diffusers as convert_transformer)
    from transformers import AutoConfig, AutoTokenizer, Qwen3Model

    import comfy.model_management as mm

    # Same offloader setup as load_pipe, but the CPU fit is sized from the single files (no component
    # dir for cpu_fits_full_load to glob): fp32 (~2x on-disk bf16) of the two big models vs RAM.
    def _fits_full_load():
        big = (files["transformer"], files["text_encoder"])
        cpu = torch.device("cpu")
        return 2 * sum(os.path.getsize(f) for f in big) <= mm.get_total_memory(cpu) * 0.85

    load_device, dtype, cpu_stream, manual_cast, operations, use_vbar, build = _prepare_offload(
        device, dtype, _fits_full_load)

    stream = use_vbar or cpu_stream

    patchers = []

    # Transformer: meta-load + stream from the ComfyUI single file (diffusers rename + qkv-chunk
    # converter, mmap-preserving), or materialise via from_single_file when not streaming.
    if stream:
        def _meta_transformer():
            cfg = transformer_cls.load_config(os.path.join(scaffold, "transformer"))
            with init_empty_weights():
                return transformer_cls.from_config(cfg).to(dtype)

        transformer, missing = adapter.stream_single_file(
            _meta_transformer, files["transformer"], operations, convert=convert_transformer)
        if missing:
            print("offloader: %d transformer weights had no matching module (skipped)"
                  % len(missing), flush=True)
    else:
        transformer = transformer_cls.from_single_file(
            files["transformer"], config=scaffold, subfolder="transformer",
            torch_dtype=dtype, local_files_only=True)
        adapter.comfy_ize(transformer, operations)

    adapter.keep_uncastable_resident(transformer, load_device, manual_cast)
    if stream and use_vbar:
        adapter.install_prefetch(transformer)
    adapter.use_comfy_attention(transformer)
    adapter.use_kitchen_rope(transformer)
    if manual_cast is not None:
        adapter.install_manual_cast(transformer, manual_cast, dtype)
    transformer_patcher = build(transformer)
    if lora_files:
        n = adapter.add_lora(transformer_patcher, lora_files)
        print("offloader: applied %d LoRA patches" % n, flush=True)
    patchers.append(transformer_patcher)

    # Text encoder (Qwen3): ComfyUI prefixes every key with `model.`; strip it so the bare Qwen3Model
    # binds. Streamed like the transformer, else materialised.
    def _strip_model(sd):
        return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}

    te_config = os.path.join(scaffold, "text_encoder")
    te_dev = load_device if stream else mm.text_encoder_device()
    if stream:
        def _meta_te():
            cfg = AutoConfig.from_pretrained(te_config)
            with init_empty_weights():
                return Qwen3Model(cfg).to(dtype)

        text_encoder, te_missing = adapter.stream_single_file(
            _meta_te, files["text_encoder"], operations, convert=_strip_model)
        if te_missing:
            print("offloader: %d text-encoder weights had no matching param (skipped)"
                  % len(te_missing), flush=True)
    else:
        import comfy.utils as cu
        text_encoder = Qwen3Model(AutoConfig.from_pretrained(te_config))
        text_encoder.load_state_dict(
            _strip_model(cu.load_torch_file(files["text_encoder"], device=torch.device("cpu"))),
            strict=False)
        text_encoder = text_encoder.to(dtype)
        adapter.comfy_ize(text_encoder, operations)

    adapter.use_comfy_attention(text_encoder)
    adapter.keep_uncastable_resident(text_encoder, te_dev, manual_cast)
    if manual_cast is not None:
        adapter.install_manual_cast(text_encoder, manual_cast, dtype)
    encoder_patcher = (build(text_encoder) if te_dev == load_device
                       else adapter.build_patcher(text_encoder, load_device=te_dev,
                                                  offload_device=te_dev))
    patchers.append(encoder_patcher)

    # VAE (small) + scheduler + tokenizer come straight from the scaffold; assemble the pipeline.
    vae = AutoencoderKL.from_single_file(files["vae"], config=scaffold, subfolder="vae",
                                         torch_dtype=dtype, local_files_only=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scaffold, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(scaffold, "tokenizer"))

    p = pipeline_cls(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                     tokenizer=tokenizer, transformer=transformer)

    p._offloader_te_device = te_dev
    p._offloader_encoder = encoder_patcher

    return _finalize_pipe(p, patchers, load_device, cpu_stream)


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
