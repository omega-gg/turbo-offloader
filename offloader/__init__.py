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
        # NVML memory pressure: ComfyUI's default (main.py:83, disable-nvml-pressure off).
        _vbar_ready = bool(ctl.init(nvml_pressure=True))
        if _vbar_ready:
            ctl.set_log_warning()
    except Exception:
        _vbar_ready = False

    _available = True


def available():
    """True once pre_torch_init() has brought up the vendored offloader (any device)."""
    return _available


def _direct_load(module, component_dir, device):
    """Materialise `module`'s weights in `device` memory: ComfyUI's load_torch_file maps the file
    on the CPU, each tensor is copied once onto the device, then bound with torch's
    load_state_dict(assign=True). This is how ComfyUI places a model it builds on MPS -- the file
    read on the CPU, one copy into the device -- and it halves placement vs diffusers' CPU load
    followed by load_models_gpu's per-leaf CPU->device copy. The file is never read onto the
    device itself: on MPS (torch 2.14.1) such tensors come out zeroed once GPU work is queued on
    them.

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
            sd.update(load_torch_file(shard))
        sd = {k: v.to(device) for k, v in sd.items()}
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
    then -- exactly as ComfyUI decides -- the CPU comfy/stream mode + storage dtype, the
    manual_cast compute dtype, the comfy op namespace, whether comfy-aimdo VBAR engages, and the
    patcher builder. `fits_full_load` is a thunk giving the caller's 'does the fp32 model fit RAM'
    verdict (consulted only on CPU with no OFFLOADER_CPU_MODE override). Returns
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
        # streaming, else its CPU default (fp32). manual_cast_dtype (below) picks fp32 compute.
        from comfy.cli_args import args as _args
        _args.bf16_unet = cpu_stream
        dtype = mm.unet_dtype()
        print("offloader: CPU mode=%s (storage %s)" % (mode, dtype), flush=True)

    # ComfyUI's storage-vs-compute dtype split: on a card without bf16 tensor cores (e.g. Turing) a
    # bf16 checkpoint computes in fp16 (manual_cast) -- weights stay bf16 (mmap, no RAM blow-up),
    # every matmul runs on fp16 tensor cores (~5x there). None on Ampere+ (compute in bf16).
    manual_cast = adapter.manual_cast_dtype(dtype, load_device)
    if manual_cast is not None:
        print("offloader: manual_cast compute dtype %s (storage %s)"
              % (manual_cast, dtype), flush=True)

    # ComfyUI's own op-namespace pick (pick_operations): comfy.ops.manual_cast when compute !=
    # weight (every leaf casts, incl. resident ones), else disable_weight_init (plain forward
    # unless offloaded). comfy_ize / the streamers re-class each leaf into this namespace.
    operations = adapter.pick_operations(dtype, manual_cast, load_device)

    # VBAR: CUDA + comfy-aimdo (initialised in pre_torch_init) -> flip aimdo_enabled so ComfyUI's
    # ModelPatcherDynamic engages (partial GPU residency sized from live free VRAM, streaming the
    # rest per forward). Off CUDA / without comfy-aimdo -> plain ModelPatcher native cast path.
    use_vbar = _vbar_ready and adapter.enable_vbar(device)
    if use_vbar:
        # transient HostBuffer.truncate under RAM pressure -> pageable, not a crash
        adapter.install_pin_rollback_guard()
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

    # Direct-to-device load, MPS only. On MPS (unified memory) we copy the big models into device
    # memory at load (below) instead of diffusers' CPU load + load_models_gpu's per-leaf
    # CPU->device copy. Safe only here: MPS has no separate VRAM pool, so full residency is always
    # chosen and can never OOM -- whereas on CUDA load_models_gpu's partial-residency / VBAR
    # streaming decisions must stand (a full direct load would OOM a smaller-than-model card), and
    # on CPU the weights already load there so there is nothing to gain. LoRA is fine: direct-load
    # only places the base weights; load_models_gpu still merges each patch on top via
    # patch_weight_to_device (verified bit-identical to the normal path). Gated off manual_cast,
    # whose bf16-storage/fp16-compute split relies on the ModelPatcher cast path a direct assign
    # would bypass (never hit on MPS+fp16 anyway -> None).
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
    # ComfyUI's exact SDPA (comfy.ops, its backend priority + size gate)
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

    return _finalize_pipe(p, patchers, load_device, cpu_stream, manual_cast or dtype)


def _finalize_pipe(p, patchers, load_device, cpu_stream, compute_dtype):
    """Shared tail for the offloader pipes (dir-based load_pipe and single-file
    load_pipe_single_file). Places the small VAE, records the patchers, pins the execution device,
    wires the ComfyUI-faithful encode bridge, and installs the prompt-encode cache. The heavy
    models (transformer, text encoder) are already comfy-ized / patched by the caller;
    p._offloader_te_device tells us where the encoder lives."""
    import torch
    from . import adapter
    import comfy.model_management as mm

    # Small resident modules (VAE) go straight to the compute device; the offloader handles the
    # heavy ones. VAE tiling/slicing is left to the caller.
    if getattr(p, "vae", None) is not None:
        # ComfyUI runs the VAE in fp32 on CPU (vae_dtype: bf16/fp16 conv is unsupported / emulated
        # ~10-30x slower there, so it falls through to fp32). In stream mode our storage dtype is
        # bf16, which would leave the VAE in bf16 -- a multi-minute decode; upcast it to fp32. In
        # comfy mode the pipeline is already fp32, and on GPU we keep the pipeline dtype.
        if cpu_stream:
            p.vae.to(device=load_device, dtype=torch.float32)
        else:
            p.vae.to(load_device)

        # ComfyUI hands a VAE its input in the VAE's own dtype (VAE.decode / encode:
        # samples.to(self.vae_dtype)), whatever the pipeline samples in: float32 latents from
        # ComfyUI's models, bf16 ones from a diffusers transformer. A ComfyVAE does it itself.
        if not isinstance(p.vae, adapter.ComfyVAE):
            _vae = p.vae

            def _cast_in(fn):
                def wrapped(x, *a, **k):
                    return fn(x.to(_vae.dtype) if hasattr(x, "to") else x, *a, **k)
                return wrapped

            _vae.decode = _cast_in(_vae.decode)
            _vae.encode = _cast_in(_vae.encode)

    p._offloader_patchers = patchers
    p._offloader_device = load_device

    # Diffusers infers _execution_device from module placement. Normally every managed module sits
    # on the compute device, so the historical best-effort hint below suffices (and CUDA / CPU keep
    # exactly that path). Only when the text encoder is placed OFF the compute device (te_dev !=
    # load_device, e.g. MPS -> CPU) would the property resolve to that other device and build the
    # timesteps/latents there, mismatching the compute-device transformer -- so in that case pin it
    # to the compute device via a per-instance subclass (ComfyUI runs the sampler on the compute
    # device regardless of where the text encoder lives). The encode still runs on te_dev because
    # the bridge below overrides encode_prompt's device argument explicitly.
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
    # TE to MPS per leaf. Run the encode on the TE's device instead (native forward, no stream),
    # then move only the (small) embeddings to the compute device -- ComfyUI's "encode on CPU,
    # conditioning to GPU". Wrapped BEFORE install_encode_cache so the cache stores the CPU result.
    te_dev = getattr(p, "_offloader_te_device", load_device)
    if te_dev != load_device and getattr(p, "encode_prompt", None) is not None:
        _real_encode = p.encode_prompt

        # ComfyUI casts the conditioning to the diffusion model's compute dtype on the way in
        # (model_base._apply_model: manual_cast_dtype, else the model dtype) -- a TE kept in its
        # file dtype (bf16) would otherwise set the latent dtype (flux2: prompt_embeds.dtype)
        # against an fp16 VAE. The compute dtype, not the storage one: fp8 / manual_cast weights
        # store another dtype than they compute in.
        def _to_compute(o):
            if not isinstance(o, torch.Tensor):
                return o
            if o.is_floating_point():
                return o.to(load_device, dtype=compute_dtype)
            return o.to(load_device)

        def _encode_on_te(*args, **kwargs):
            kwargs["device"] = te_dev
            out = _real_encode(*args, **kwargs)
            if isinstance(out, (list, tuple)):
                return type(out)(_to_compute(o) for o in out)
            return _to_compute(out)

        p.encode_prompt = _encode_on_te

    # ComfyUI's sampler->VAE node boundary (DiT VBAR residency becomes evictable before the VAE
    # needs VRAM) + its OOM fallback: regular decode/encode, retry tiled (comfy/sd.py:1403).
    # KSampler ends before VAEDecode; a reference's VAEEncode runs before sampling starts.
    adapter.install_tiled_vae_fallback(
        p, node_boundary=lambda kind: _end_sampling(p) if kind == "decoding" else node_teardown())

    # The TE->transformer node boundary ComfyUI has and a diffusers pipeline doesn't: run comfy's
    # per-node teardown once the encoder is done, before the denoise loop, exactly as
    # comfy/execution.py:559-563 does between CLIPTextEncode and KSampler. Installed BEFORE the
    # cache so it is the encoder's own boundary -- a cache hit never runs the encoder, so (like a
    # cached comfy node) it needs no teardown. Each node loads its own model: CLIPTextEncode the
    # encoder, then KSampler the sampling models (prepare_sampling). The teardown marks every
    # model's pins inactive and only those loads mark them active again, so while sampling, pin
    # pressure takes the encoder's pins, not the transformer's.
    if getattr(p, "encode_prompt", None) is not None:
        _encode = p.encode_prompt
        encoder = getattr(p, "_offloader_encoder", None)

        def _encode_then_teardown(*args, **kwargs):
            if encoder is not None:
                mm.load_models_gpu([encoder])
            out = _encode(*args, **kwargs)
            node_teardown()
            mm.load_models_gpu(_sampling_patchers(p))
            return out

        p.encode_prompt = _encode_then_teardown

    # Drop padded text tokens (ComfyUI never pads, and drops an all-ones mask): a pipeline padding
    # to a fixed width would otherwise run those dead tokens through every block. Wrapped BEFORE
    # install_encode_cache so the cache stores the unpadded result.
    adapter.install_unpadded_encode(p)

    # Cache the text-encoder output by prompt (ComfyUI's node cache): a repeated prompt then skips
    # the encoder's forward, so the streamed encoder is never loaded and the transformer keeps the
    # VRAM.
    adapter.install_encode_cache(p)

    return p


# ComfyUI's models with an adapter here, each in the module named after it (its model_detection
# image_model): offloader/qwen_image.py, qwen_image21.py, flux2.py, flux3.py, krea2.py,
# lumina2.py.
_COMFY_MODELS = ("qwen_image", "qwen_image21", "flux2", "flux3", "krea2", "lumina2")


def _comfy_model(name):
    """The adapter running ComfyUI's model `name` behind the calls a diffusers pipeline makes."""
    import importlib

    if name not in _COMFY_MODELS:
        raise ValueError("offloader: no adapter for ComfyUI's model %r" % name)

    return importlib.import_module("." + name, __name__)


def load_pipe_comfy(pipeline_cls, transformer, text_encoder, components, dtype, device="cuda:0",
                    lora_files=None):
    """A pipeline whose transformer and text encoder are ComfyUI's own models, each run by its
    adapter in offloader/ (_COMFY_MODELS) behind the calls the pipeline makes. The engine
    (engine/<id>.py) names them as data:
        transformer  = {"model": ComfyUI model name, "file": path,
                        "config": what the pipeline reads off its transformer's config,
                        "lora_names": fn(ComfyUI's LoRA key map)->dict}  # optional aliases
        text_encoder = {"model": ComfyUI model name, "file": path}  # its TextEncoder
        components   = {<pipeline __init__ arg>: prebuilt module, ...}  # vae/tokenizer/scheduler
    A "file" may also be a list, a model's shards (a diffusers repo's folder). The transformer is
    built with ComfyUI's ops and loads as sd.py loads it, so it prefetches, attends and ropes as
    ComfyUI's does; the text encoder loads and sits where its CLIPLoader puts it. Then
    pipeline_cls(transformer=, text_encoder=, **components) and the shared _finalize_pipe tail."""
    import os
    import torch  # noqa: F401
    from . import adapter

    import comfy.model_management as mm

    def _fits_full_load():
        big = adapter.as_files(transformer["file"]) + adapter.as_files(text_encoder["file"])
        cpu = torch.device("cpu")
        return 2 * sum(os.path.getsize(f) for f in big) <= mm.get_total_memory(cpu) * 0.85

    load_device, dtype, cpu_stream, manual_cast, _operations, _use_vbar, build = (
        _prepare_offload(device, dtype, _fits_full_load))

    tf, unexpected = _comfy_model(transformer["model"]).load(
        transformer["file"], transformer["config"], load_device, dtype)
    if unexpected:
        print("offloader: %d transformer keys unexpected" % len(unexpected), flush=True)
    tf_patcher = build(tf)
    if lora_files:
        # ComfyUI's keys for the model, plus the engine's other names for them (diffusers').
        key_map = tf.lora_keys()
        key_map.update(transformer.get("lora_names", lambda keys: {})(key_map))
        n = adapter.add_lora(tf_patcher, lora_files, key_map=key_map)
        print("offloader: applied %d LoRA patches" % n, flush=True)

    te = _comfy_model(text_encoder["model"]).TextEncoder(text_encoder["file"])

    p = pipeline_cls(transformer=tf, text_encoder=te, **components)

    p._offloader_te_device = te.patcher.load_device
    p._offloader_encoder = te.patcher

    return _finalize_pipe(p, [tf_patcher, te.patcher], load_device, cpu_stream,
                          manual_cast or dtype)


def comfy_vae(path, config):
    """Opt in to ComfyUI's own VAE: the model comfy/sd.py builds for the VAE file at `path` (its
    code vendored under comfy/ldm/), read and run the way ComfyUI does, behind the calls a
    diffusers pipeline makes. `config` is what the pipeline reads off the VAE's config (z_dim,
    latent statistics). See adapter.ComfyVAE and adapter._comfy_vae_model."""
    from . import adapter

    return adapter.ComfyVAE(path, config)


def prepare(pipe):
    """Per-generation load boundary: hand the managed models to ComfyUI's load_models_gpu, which
    partial-loads / streams them to the compute device (its dynamic path streams weights
    per-forward, so this marks them loaded without pinning the whole set). When
    install_encode_cache serves a repeated prompt the text encoder's forward never runs, so its
    weights never stream and the transformer keeps
    the throughput.

    A generation is ComfyUI's prompt: mark its models in use by the current prompt
    (PromptModelTracker, as comfy/execution.py does per node output), so pin eviction under host
    RAM pressure takes other models' pins before the running ones'. reclaim() ends it. As a
    sampling run does (comfy/samplers.py:1260), pre_run hands a model declaring `current_patcher`
    (as ComfyUI's BaseModel does) its patcher, for get_free_memory. Its cleanup() is left out: from
    reclaim(), its eject of the dynamic model trips aimdo's VBAR warnings."""
    patchers = getattr(pipe, "_offloader_patchers", None)
    if not patchers:
        return

    import comfy.model_management as mm
    from comfy.model_patcher import PromptModelTracker

    pipe._offloader_prompt = PromptModelTracker()
    pipe._offloader_prompt.add(patchers)

    mm.load_models_gpu(_sampling_patchers(pipe))  # the encoder loads at encode, see _finalize_pipe
    for patcher in patchers:
        patcher.pre_run()


def _sampling_patchers(pipe):
    # What KSampler loads: every managed model but the text encoder.
    encoder = getattr(pipe, "_offloader_encoder", None)
    return [patcher for patcher in pipe._offloader_patchers if patcher is not encoder]


def node_teardown():
    """ComfyUI's per-NODE offloader teardown, verbatim: cleanup_prefetch_queues +
    reset_cast_buffers + vbars_reset_watermark_limits, in that order: what comfy runs in the
    `finally` after EVERY node execution when the offloader is on (comfy/execution.py:559-563). It
    resets the block prefetch queues (and aborts a malloc graph), the globally-cached per-stream
    cast buffers and the VBAR watermarks.

    ComfyUI gets these boundaries for free because its graph is nodes: CLIPTextEncode ends, the
    teardown runs, THEN KSampler starts. A diffusers pipeline has no such boundary -- encode_prompt
    flows straight into the denoise loop -- so without calling this ourselves the text encoder's
    cast buffers and stream state leak into the transformer's first forwards. That is not just
    untidy: with 2 async offload streams it made z-image NON-DETERMINISTIC at a fixed seed (the
    drift appears at forward 3, the first VBAR eviction) and left the DiT starved of the VRAM the
    encoder still held (z-image 512^2: 1.83 -> 1.00 s/it once the boundary is honoured).
    No-op unless comfy-aimdo is active."""
    import comfy.model_management as mm
    import comfy.memory_management as memm

    if not getattr(memm, "aimdo_enabled", False):
        return

    try:
        import comfy.model_prefetch as mp
        mp.cleanup_prefetch_queues()

        mm.reset_cast_buffers()

        import comfy_aimdo.model_vbar as mv
        mv.vbars_reset_watermark_limits()
    except Exception:
        print(traceback.format_exc(), flush=True)


def _end_sampling(pipe):
    """KSampler's end: the run's end as ComfyUI's cleanup() marks it (model_patcher.py:1337-1338),
    then the node's teardown. A model declaring current_patcher drops it, and with it what it
    keeps for one run (Qwen-Image 2.1's prefix cache), before the VAE needs the memory. The rest
    of cleanup() stays out, see prepare."""
    for patcher in pipe._offloader_patchers:
        if hasattr(patcher.model, "current_patcher"):
            patcher.model.current_patcher = None

    node_teardown()


def reclaim(pipe):
    """Per-generation housekeeping: run ComfyUI's per-node offloader teardown (node_teardown --
    the same one we run at the encode boundary), then free non-resident models + return the
    allocator pool."""
    if not getattr(pipe, "_offloader_patchers", None):
        return

    import comfy.model_management as mm

    # A run the VAE never ended (an error, a latent output) ends here.
    _end_sampling(pipe)

    # After a hard CUDA OOM even mem_get_info raises; reclaim runs on the error path too, so a
    # failure here must not bury the real exception under a cleanup cascade.
    try:
        dev = getattr(pipe, "_offloader_device", None)
        if dev is not None:
            mm.free_memory(mm.minimum_inference_memory(), dev)
        mm.soft_empty_cache()
    except Exception:
        print(traceback.format_exc(), flush=True)

    prompt = getattr(pipe, "_offloader_prompt", None)
    if prompt is not None:
        prompt.end()


def release(pipe):
    """Tear down the offloaders before the pipe is dropped (detach unpatches weights and lets the
    current_loaded_models finalizers fire)."""
    for patcher in getattr(pipe, "_offloader_patchers", []):
        try:
            patcher.detach(unpatch_all=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
