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
#   The thin middleman between turboCLI's diffusers pipelines and the vendored ComfyUI offloading
#   subsystem (offloader/comfy/). It contains the ONLY real logic in the package; everything under
#   offloader/comfy/ is a verbatim ComfyUI snapshot. The bridge is two ideas:
#
#     1. comfy_ize(model): re-class each vanilla torch leaf module (Linear/Conv/Norm/Embedding) to
#        its ComfyUI `disable_weight_init.*` counterpart. Those classes ARE `torch.nn.X` +
#        CastWeightBiasOp, so their forward routes through ComfyUI's device-agnostic cast path
# (forward_comfy_cast_weights -> cast_bias_weight -> cast_to) exactly when ModelPatcher flags the
# module for offload -- and computes identically to the original op otherwise.
#
#     2. build_patcher(model): wrap the comfy-ized module in a ComfyUI ModelPatcher. From there the
#        seam drives ComfyUI's own load_models_gpu / free_memory to stream weights CPU<->device per
#        forward, the same way ComfyUI offloads a UNet -- on CPU, CUDA or MPS.
#
#   comfy_aimdo's CUDA-only VBAR path is left dormant (aimdo_enabled False) until Phase D.
#
#==================================================================================================

# Establishes the top-level `comfy` alias (see offloader/comfy/__init__.py).
from . import comfy  # noqa: F401

import contextlib
import math
import os
from types import SimpleNamespace

import torch

# ComfyUI's model_management runs get_torch_device() at IMPORT (module level); with cpu_state's
# default of GPU that calls torch.cuda.current_device(), which asserts "Torch not compiled with
# CUDA enabled" on a CPU-only build. ComfyUI avoids this because a CPU user passes `--cpu` (->
# args.cpu -> cpu_state=CPU, model_management.py:158). We parse no argv, so set args.cpu ourselves
# when there is no GPU at all, BEFORE model_management imports -- then it initialises cpu_state=CPU
# and never touches torch.cuda.
from comfy.cli_args import args as _cli_args  # noqa: E402

_mps = getattr(torch.backends, "mps", None)
if not torch.cuda.is_available() and not (_mps is not None and _mps.is_available()):
    _cli_args.cpu = True

# ComfyUI's --disable-pinned-memory (cli_args.py:207, model_management.py:1639-1652), read when
# model_management imports: no pinned host copy of the weights. For a GPU that holds the whole
# model: half the RAM, the same output. OFFLOADER_DISABLE_PINNED_MEMORY=1 sets it.
if os.environ.get("OFFLOADER_DISABLE_PINNED_MEMORY") == "1":
    _cli_args.disable_pinned_memory = True

import comfy.model_management as mm
import comfy.model_patcher as model_patcher
import comfy.ops as ops
import comfy.utils


# -------------------------------------------------------------------------------------------------
# Op mapping: torch leaf type -> ComfyUI op class NAME (a torch.nn.X subclass that also mixes in
# CastWeightBiasOp). The name is resolved against a chosen ComfyUI op namespace at re-class time,
# so the SAME map serves both `disable_weight_init` (plain forward unless offloaded) and
# `manual_cast` (every leaf casts) -- exactly as ComfyUI's pick_operations picks one namespace for
# the whole model. Checked in order; the first isinstance() match wins. Every diffuser
# transformer/text-encoder leaf that carries an offloadable weight is one of these.
# -------------------------------------------------------------------------------------------------
_OP_MAP = [
    (torch.nn.Linear,    "Linear"),
    (torch.nn.Conv1d,    "Conv1d"),
    (torch.nn.Conv2d,    "Conv2d"),
    (torch.nn.Conv3d,    "Conv3d"),
    (torch.nn.GroupNorm, "GroupNorm"),
    (torch.nn.LayerNorm, "LayerNorm"),
    (torch.nn.Embedding, "Embedding"),
]

# torch.nn.RMSNorm exists only on recent torch; add it when present (diffusers uses it widely).
if hasattr(torch.nn, "RMSNorm"):
    _OP_MAP.append((torch.nn.RMSNorm, "RMSNorm"))


def _comfy_class_for(module, operations):
    """The `operations` (disable_weight_init or manual_cast) class to re-class `module` into, or
    None if it isn't an offloadable leaf op. Skips CastWeightBiasOp modules (idempotent)."""
    if isinstance(module, ops.CastWeightBiasOp):
        return None
    for torch_cls, name in _OP_MAP:
        if isinstance(module, torch_cls):
            return getattr(operations, name, None)
    # Custom RMSNorm classes (diffusers/transformers write their own, e.g. Qwen3RMSNorm, not
    # torch.nn.RMSNorm) run an eager mul/rsqrt that is ~3.5x slower than ComfyUI's fused path.
    # ComfyUI's own RMSNorm calls torch.nn.functional.rms_norm; route these through the SAME
    # vendored class so the kernel (and behavior) matches ComfyUI exactly.
    if hasattr(operations, "RMSNorm") and type(module).__name__.endswith("RMSNorm") \
            and getattr(module, "weight", None) is not None:
        return operations.RMSNorm
    return None


def _prep_rmsnorm(module):
    """Give a re-classed custom RMSNorm the attributes torch.nn.RMSNorm / F.rms_norm read
    (normalized_shape, eps) that its original class didn't expose under those names.

    elementwise_affine is set too: it isn't read by F.rms_norm, but torch.nn.RMSNorm.extra_repr()
    reads it, and ModelPatcher.load() reprs each module in a debug-log line -- that .format() runs
    eagerly regardless of log level, so a missing key raises KeyError mid-load (native path). The
    module always carries a weight here, so affine is True."""
    if not hasattr(module, "normalized_shape"):
        module.normalized_shape = tuple(module.weight.shape)
    if not hasattr(module, "eps") or module.eps is None:
        module.eps = (getattr(module, "variance_epsilon", None)
                      or getattr(module, "epsilon", None) or 1e-6)
    if not hasattr(module, "elementwise_affine"):
        module.elementwise_affine = True


def _rmsnorm_matches_fused(module):
    """True when re-classing this custom RMSNorm to comfy's fused RMSNorm would preserve its math.

    comfy's RMSNorm runs `F.rms_norm(input, normalized_shape, weight, eps)` -- the weight applied
    verbatim. A module whose forward transforms the weight first (diffusers' Krea2RMSNorm
    normalizes by `1 + weight`, its scale stored zero-centered) would silently lose that, so it has
    to keep its OWN forward -- which is what ComfyUI runs anyway: comfy hands a model `operations`
    for its Linear/Conv leaves, it never swaps out the model's norm.

    Probed structurally rather than by class name: hand the module a real random weight (its own is
    still meta here) and compare its forward against the exact fused call the re-class would make.
    That also guards the eps/normalized_shape _prep_rmsnorm just inferred. Anything that does not
    match -- unknown convention, wrong eps, a probe that raises -- returns False and keeps the
    module's forward, so the re-class can only ever be a no-op speedup."""
    import torch.nn.functional as F
    saved = module.weight
    try:
        weight = torch.randn(tuple(module.weight.shape))
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
        probe = torch.randn(2, weight.shape[-1])
        with torch.no_grad():
            own = module(probe).float()
        fused = F.rms_norm(probe, module.normalized_shape, weight, module.eps)
        return bool(torch.allclose(own, fused, atol=1e-4))
    except Exception:
        return False
    finally:
        module.weight = saved


def pick_operations(weight_dtype, compute_dtype, load_device=None):
    """ComfyUI's own op-namespace selector (comfy.ops.pick_operations): returns `manual_cast` when
    the compute dtype differs from the weight dtype (every leaf casts, comfy_cast_weights=True at
    the class level), else `disable_weight_init`. This is the exact call ComfyUI uses to build a
    model; hand the result to comfy_ize / load_streamed. Its fp8/cublas branches are inert here (we
    parse no argv, pass no quant_config), so it returns only those two namespaces."""
    return ops.pick_operations(weight_dtype, compute_dtype, load_device)


def comfy_ize(model, operations=None):
    """Re-class every offloadable leaf of `model` in place so its forward routes through ComfyUI's
    cast path. `operations` is the ComfyUI op namespace to re-class into (from pick_operations):
    `disable_weight_init` (default -- plain forward unless ModelPatcher offloads the leaf) or
    `manual_cast` (every leaf casts, comfy_cast_weights=True at the class level -- so under
    manual_cast a RESIDENT leaf casts too, exactly as ComfyUI). Injects the attributes
    ModelPatcher.load()/cast_bias_weight read. Returns the number of modules converted.

    Re-classing to ComfyUI's own class (rather than a synthesized mixin) keeps true 1:1 parity: the
    running forward IS <operations>.<Op>.forward. The lazy-init __init__ of those classes is
    bypassed here (we mutate existing instances that already hold their weights)."""
    if operations is None:
        operations = ops.disable_weight_init
    rms_cls = getattr(operations, "RMSNorm", None)
    count = 0
    for _name, m in model.named_modules():
        target = _comfy_class_for(m, operations)
        if target is None:
            continue

        # Custom RMSNorm needs normalized_shape/eps before re-classing so F.rms_norm can read them.
        # Only re-class when the fused op provably reproduces the module's own forward: a norm with
        # its own weight convention (Krea2RMSNorm's `1 + weight`) keeps its forward, exactly as
        # ComfyUI runs it -- keep_uncastable_resident then places it (a norm's weight is tiny).
        if target is rms_cls and not isinstance(m, torch.nn.RMSNorm):
            _prep_rmsnorm(m)
            if not _rmsnorm_matches_fused(m):
                continue

        # Re-class the live instance. torch.nn.X subclasses share a compatible object layout, so
        # __class__ assignment is valid; diffusers custom subclasses that add only config (no
        # forward override that matters for weight application) keep working through the base op.
        m.__class__ = target

        # cast_bias_weight always reads s.bias; weight-only ops (Embedding/RMSNorm) have none.
        # Mirror what the op's __init__ would have set.
        if not hasattr(m, "bias"):
            m.bias = None

        # comfy_cast_weights: `manual_cast` ops set it True at the CLASS level (cast every leaf --
        # ComfyUI's manual_cast path, and it survives ModelPatcher.load for resident leaves). Only
        # pin an instance False for the `disable_weight_init` path, so a manual_cast leaf keeps its
        # class-level True. ModelPatcher.load() overwrites this to True for any leaf it offloads
        # either way. weight_function/bias_function are per-instance lists so per-module offload
        # patches never alias the shared CastWeightBiasOp class lists.
        if not getattr(type(m), "comfy_cast_weights", False):
            m.comfy_cast_weights = False
        m.weight_function = []
        m.bias_function = []
        count += 1
    return count


def make_patchable(model):
    """ComfyUI's ModelPatcher assigns `self.model.device = ...`, but diffusers ModelMixin and HF
    PreTrainedModel expose `device` as a READ-ONLY property. Shadow it with a plain, settable class
    attribute via a one-off subclass so the assignment sticks (its value -- the load device -- is
    what those models would report anyway once loaded). No-op if `device` is already settable."""
    cls = type(model)
    if isinstance(getattr(cls, "device", None), property):
        # Masquerade as the parent for str(cls): transformers 5.x keys its output-capture registry
        # on `str(self.__class__)` (_CAN_RECORD_REGISTRY.get(str(self.__class__))), so a subclass
        # with a new class string silently loses output_hidden_states. Copying
        # __module__/__qualname__ makes str(subclass) == str(cls), so the lookup (and hidden-state
        # capture) keeps working.
        ns = {"device": None, "__module__": cls.__module__, "__qualname__": cls.__qualname__}
        model.__class__ = type(cls.__name__, (cls,), ns)
    return model


def keep_uncastable_resident(model, device, compute_dtype=None):
    """Move every parameter/buffer that ComfyUI's offloader will NOT stream onto `device`, so it
    stays resident. `compute_dtype` (manual_cast): when set, float stragglers are also cast to it.
    The offloader only streams the weight/bias of CastWeightBiasOp leaves (the ones comfy_ize
    converted); anything else -- a custom leaf norm (transformers' Qwen3RMSNorm, forward
    `self.weight * hidden_states`), or a bare parameter/buffer hung directly on a container module
    (diffusers' `x_pad_token`, rotary caches, class tokens) -- would otherwise be left on the
    offload device by ModelPatcher.load() and blow up mid-forward with a cuda-vs-cpu mismatch.

    We walk ALL modules and move only their DIRECT (recurse=False) params/buffers, skipping
    CastWeightBiasOp modules entirely so their big weights keep streaming. These stragglers are
    almost always tiny; returns (count, bytes) for logging so a large one is visible."""
    # When manual_cast is active (compute_dtype set), float stragglers must also move to the
    # compute dtype so they don't mismatch the fp16 activations -- ComfyUI's model runs uniformly
    # in the compute dtype. Complex/integer buffers (rope caches, indices) keep their dtype.
    def _cast(t):
        if compute_dtype is not None and t.is_floating_point() and t.dtype != compute_dtype:
            return t.to(device=device, dtype=compute_dtype)
        return t.to(device)

    moved = 0
    nbytes = 0
    for _name, m in model.named_modules():
        if isinstance(m, ops.CastWeightBiasOp):
            continue  # its weight/bias stream via the cast path -- leave on the offload device
        for _n, p in m.named_parameters(recurse=False):
            if p is not None and (p.device != device or (compute_dtype is not None
                                                         and p.is_floating_point()
                                                         and p.dtype != compute_dtype)):
                p.data = _cast(p.data)
                moved += 1
                nbytes += p.numel() * p.element_size()
        for bn, b in m.named_buffers(recurse=False):
            if b is not None and (b.device != device or (compute_dtype is not None
                                                        and b.is_floating_point()
                                                        and b.dtype != compute_dtype)):
                m._buffers[bn] = _cast(b)
                moved += 1
                nbytes += b.numel() * b.element_size()
    return moved, nbytes


def manual_cast_dtype(weight_dtype, inference_device):
    """ComfyUI's storage-vs-compute dtype split -- comfy/model_management.py::unet_manual_cast
    (vendored; same arg names). Returns the COMPUTE dtype to run in when `weight_dtype` is not
    natively fast on `inference_device`, else None (compute stays in the weight dtype). On a card
    without bf16 tensor cores (e.g. Turing sm_75) a bf16 checkpoint -> torch.float16; on Ampere+
    bf16 -> None. The decision is comfy's own, unchanged -- GPU-agnostic."""
    return mm.unet_manual_cast(weight_dtype, inference_device)


def install_manual_cast(model, compute_dtype, storage_dtype):
    """Copy ComfyUI's manual_cast (should_use_bf16 False path): keep weights in their storage dtype
    (bf16, mmap-backed -- no load-time materialisation, so RAM use is unchanged from the working
    path) but run the forward in `compute_dtype` (fp16). The comfy-ized leaves' cast_bias_weight
    already casts each weight to the INPUT activation dtype at forward (comfy/ops.py), so casting
    the transformer's float inputs to `compute_dtype` makes every streamed weight cast bf16->fp16
    on the GPU per layer and every matmul run on fp16 tensor cores -- exactly what ComfyUI does
    when the card has no bf16 tensor cores. Complex (rope freqs_cis) and integer inputs are left
    untouched. The output is cast back to `storage_dtype` so the diffusers pipeline's latent dtype
    contract holds.
    Sets model.manual_cast_dtype to mirror comfy's model attribute (read by ModelPatcher)."""
    model.manual_cast_dtype = compute_dtype

    def _to(v, dt):
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != dt:
            return v.to(dt)
        return v

    def _cast_inputs(_m, args, kwargs):
        return (tuple(_to(a, compute_dtype) for a in args),
                {k: _to(v, compute_dtype) for k, v in kwargs.items()})

    def _cast_output(_m, _args, _kwargs, out):
        if isinstance(out, torch.Tensor):
            return _to(out, storage_dtype)
        if isinstance(out, tuple):
            return tuple(_to(o, storage_dtype) for o in out)
        s = getattr(out, "sample", None)
        if s is not None:
            out.sample = _to(s, storage_dtype)
        return out

    model.register_forward_pre_hook(_cast_inputs, with_kwargs=True, prepend=True)
    model.register_forward_hook(_cast_output, with_kwargs=True)

    # Per-leaf input cast -- the crux of manual_cast. ComfyUI's native model computes EVERY layer
    # in the manual_cast dtype; the diffusers model computes its time-embedding / adaLN modulation
    # in fp32, so `norm(x) * scale` promotes fp16 -> fp32 and cast_bias_weight then casts weights
    # to fp32, dropping every matmul onto the FP32 (volta_sgemm) path instead of fp16 tensor cores
    # (measured: ~63% of the step). Casting each weighted op's float input to compute_dtype at its
    # forward forces weight (cast_bias_weight follows input.dtype) and matmul back to fp16 --
    # exactly ComfyUI's "compute in the manual_cast dtype" invariant. Int inputs (Embedding
    # indices) are left untouched.
    def _leaf_cast(_m, args):
        if args and isinstance(args[0], torch.Tensor) and args[0].is_floating_point() \
                and args[0].dtype != compute_dtype:
            return (args[0].to(compute_dtype),) + tuple(args[1:])
        return None

    for _leaf in model.modules():
        if isinstance(_leaf, ops.CastWeightBiasOp):
            _leaf.register_forward_pre_hook(_leaf_cast)

    # ComfyUI clamps activations to the fp16 range after each transformer block (clamp_fp16,
    # comfy/ldm/lumina/model.py:71-74): without it fp16 overflow -> inf -> NaN -> black image. The
    # diffusers model has no such guard, so replicate it when computing in fp16. Model-agnostic:
    # hook the output of every nn.ModuleList child (the block stacks) rather than a named class.
    # nan_to_num is a near-no-op on in-range values, so over-applying it (e.g. to a norm list) is
    # harmless.
    if compute_dtype == torch.float16:
        _MAX = 65504.0

        def _clamp(_m, _a, out):
            if isinstance(out, torch.Tensor) and out.dtype == torch.float16:
                return torch.nan_to_num(out, nan=0.0, posinf=_MAX, neginf=-_MAX)
            return out

        seen = set()
        for _mod in model.modules():
            if isinstance(_mod, torch.nn.ModuleList):
                for _blk in _mod:
                    if id(_blk) not in seen:
                        _blk.register_forward_hook(_clamp)
                        seen.add(id(_blk))


def build_patcher(model, load_device=None, offload_device=None, size=0):
    """Wrap a comfy-ized `model` in a ComfyUI ModelPatcher. Defaults load/offload devices from
    ComfyUI's own device selection so one code path serves CPU/CUDA/MPS. `size` (bytes) seeds
    model_size(); 0 lets ModelPatcher measure it via module_size()."""
    if load_device is None:
        load_device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()
    make_patchable(model)
    return model_patcher.ModelPatcher(model, load_device=load_device,
                                      offload_device=offload_device, size=size)


# -------------------------------------------------------------------------------------------------
# VBAR (CUDA only). Flipping comfy.memory_management.aimdo_enabled turns on ComfyUI's own dynamic
# path: ModelPatcherDynamic streams each RAM-resident weight to VRAM per forward and keeps only
# what fits GPU-resident (partial residency, sized by load_models_gpu from live free VRAM), so a
# model larger than VRAM still runs -- exactly as ComfyUI offloads a UNet. On CPU/MPS this stays
# off and the native cast_to path runs.
# -------------------------------------------------------------------------------------------------
def enable_vbar(device):
    """Flip aimdo_enabled so the vendored VBAR path engages. CUDA only. comfy-aimdo's global
    init (allocator hooks) must have run in the seam's pre_torch_init() BEFORE torch was imported;
    here -- after torch, with the device known -- we do the per-device init (init_device, like v1)
    that ModelVBAR needs, then set the flag. Returns True when VBAR is active, False (-> native
    path)
    off CUDA or if the device init fails."""
    dev = str(device).lower()
    if dev.split(":")[0] != "cuda":
        return False
    index = int(dev.split(":")[1]) if ":" in dev else 0
    try:
        import comfy_aimdo.control as ctl
        if not ctl.devctxs:
            ctl.init_device(index)
    except Exception:
        return False
    import comfy.memory_management as memm
    # As main.py:318-319 when DynamicVRAM engages: what ComfyUI builds itself (its VAE, ...) gets
    # the dynamic patcher too.
    model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic
    memm.aimdo_enabled = True
    return True


def install_pin_rollback_guard():
    """Survive a transient `HostBuffer.truncate failed` inside comfy.pinned_memory.pin_memory,
    preserving ComfyUI's own recovery (steal a lower-priority pin).

    When the host cudaHostRegister for a weight's pin fails, pin_memory rolls the hostbuf back
    with truncate() and returns `_steal_pin(...)` -- reusing a lower-priority registered pin.
    Under host-RAM pressure that rollback truncate can ITSELF fail; the RuntimeError escapes and
    aborts the whole diffusion run mid-step (seen on a 4GB card at 1080p+).

    We catch ONLY that truncate RuntimeError and finish exactly what pin_memory was about to do:
    call `_steal_pin` with the same args it computes (pinned_memory.py). The rollback truncate is
    the only truncate that can escape pin_memory (its `except RuntimeError` rollback truncates
    again, unguarded; the budget/registration paths evict without truncating), so the message
    match is unambiguous. The module then
    still gets a real stolen pin (or, if no victim, _steal_pin returns False and the weight
    transfers pageable via comfy/ops.py handle_pin) -- identical to the in-file rollback. Anything
    other than the truncate failure re-raises unchanged.

    Idempotent, and leaves the vendored pinned_memory.py byte-for-byte (see resync.md).
    Installed on the CUDA/VBAR path where comfy-aimdo pins host memory; unused elsewhere."""
    import comfy.pinned_memory as pm
    import comfy.memory_management as memm
    import comfy.utils as cu

    if getattr(pm, "_turbo_pin_rollback_guard", False):
        return

    _orig_pin_memory = pm.pin_memory

    def pin_memory(module, subset="weights", size=None):
        try:
            return _orig_pin_memory(module, subset=subset, size=size)
        except RuntimeError as exc:
            if "HostBuffer.truncate failed" not in str(exc):
                raise
            # Mirror pin_memory's own tail (pinned_memory.py): recompute the _steal_pin args it
            # uses and steal a pin instead of aborting. priority was already set before the crash.
            pin_state = module._pin_state
            _hostbuf, stack, _stack_split, _pinned_size, counter, buckets = pin_state[subset]
            if size is None:
                size = memm.vram_aligned_size([ module.weight, module.bias ])
            module_pin = module.__dict__.setdefault("_pins", {}).setdefault(subset, {})
            priority = module_pin.get("balancer_priority")
            if priority is None:
                priority = cu.bit_reverse_range(counter[0], 16)
                counter[0] += 1
                module_pin["balancer_priority"] = priority
            return pm._steal_pin(module, stack, buckets, size, priority, subset)

    pm.pin_memory = pin_memory
    pm._turbo_pin_rollback_guard = True


def _shards(model_dir):
    """The .safetensors shard paths for a diffusers component dir, honouring a .index.json
    if present (mirrors v1's _offsets shard discovery)."""
    import json
    idx = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(model_dir, n) for n in names]
    return [os.path.join(model_dir, n) for n in sorted(os.listdir(model_dir))
            if n.endswith(".safetensors")]


def _match_by_suffix(live_name, live_tensor, disk_by_shape):
    """Pick the disk key for `live_name` among same-shape/dtype candidates by longest common dotted
    suffix; return it only if that best match is unique (mirrors v1's _match_disk_keys). Handles
    checkpoints whose keys were renamed by a prefix (e.g. transformers' `model.` ->
    `language_model.`
    for Qwen2.5-VL) where a direct name lookup misses."""
    cands = disk_by_shape.get((tuple(live_tensor.shape), live_tensor.dtype))
    if not cands:
        return None
    live_parts = live_name.split(".")

    def common_suffix(disk_name):
        dp = disk_name.split(".")
        n = 0
        while n < len(dp) and n < len(live_parts) and dp[-1 - n] == live_parts[-1 - n]:
            n += 1
        return n

    scored = sorted(((common_suffix(dk), dk) for dk in cands), reverse=True)
    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def _assign_sd(model, sd):
    """Rebind `model`'s params/buffers to the tensors in `sd` (kept as-is, so mmap/file-sliced
    storage survives). Direct name match first, then structural shape+suffix fallback for renamed
    prefixes. Returns the list of disk keys that found no home."""
    import comfy.utils as cu

    own = {n: p for n, p in model.named_parameters()}
    own.update({n: b for n, b in model.named_buffers()})

    used = set()
    # 1) direct name matches.
    for name, tensor in sd.items():
        if name in own:
            cu.set_attr_param(model, name, tensor)  # keeps the file-sliced storage
            used.add(name)

    # 2) structural fallback for the live params still holding their from_pretrained (materialized)
    #    weights: match by shape + longest-common dotted suffix among the leftover disk keys.
    leftover = {k: v for k, v in sd.items() if k not in used}
    if leftover:
        disk_by_shape = {}
        for k, v in leftover.items():
            disk_by_shape.setdefault((tuple(v.shape), v.dtype), []).append(k)
        for name, param in own.items():
            if name in used:
                continue
            dk = _match_by_suffix(name, param, disk_by_shape)
            if dk is not None and dk in leftover:
                cu.set_attr_param(model, name, leftover[dk])
                used.add(dk)
                disk_by_shape[(tuple(param.shape), param.dtype)].remove(dk)

    return [k for k in sd if k not in used]


def cpu_fits_full_load(model, transformer="transformer"):
    """True when ComfyUI's plain CPU full-load (fp32, materialised) would fit host RAM -- so the
    caller can use that faster path instead of streaming. The fp32 model is ~2x the on-disk bf16
    size; require it under 85% of total RAM (headroom for activations + the OS). Compared against
    ComfyUI's own get_total_memory(cpu). This is ComfyUI's full-load-when-it-fits call, which
    ComfyUI applies on GPU (via free VRAM) but not on CPU; here we apply it to CPU.
    Model-agnostic: sums the on-disk bytes of each big component (single-file or index+shards),
    the transformer in its folder `transformer`."""
    total = 0
    for comp in (transformer, "text_encoder"):
        comp_dir = os.path.join(model, comp)
        if os.path.isdir(comp_dir):
            total += sum(os.path.getsize(s) for s in _shards(comp_dir))
    return 2 * total <= mm.get_total_memory(torch.device("cpu")) * 0.85


def assign_streamed_weights(model, model_dir):
    """Replace `model`'s weights with mmap-backed, file-sliced tensors from its .safetensors
    shard(s) via ComfyUI's own load_torch_file, which memory-maps each shard: comfy-aimdo's
    ModelMMAP when the VBAR path is active (CUDA), else safetensors' native mmap (CPU/MPS, no
    comfy-aimdo). Either way the tensors stay file-backed, so RAM never holds the whole model.
    Returns the list of disk keys that found no home."""
    import comfy.utils as cu

    sd = {}
    shards = _shards(model_dir)
    for shard in shards:
        sd.update(cu.load_torch_file(shard, device=torch.device("cpu")))
    model._offloader_files = shards  # build_dynamic_patcher's fast_disk
    missing = _assign_sd(model, sd)
    # Rebinding by name skips a tied weight (Qwen3's lm_head, absent from the file): it would keep
    # from_pretrained's tensor, and that one view holds the whole ~8GB load buffer alive. Re-tie,
    # as _direct_load does, so the head shares the file slice.
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return missing


def load_streamed(model_cls, model_dir, dtype, operations=None):
    """Build a diffusers model whose weights stream from mmap file-slices: meta-load the module (no
    weight RAM), comfy-ize it (into `operations`; see comfy_ize), then assign the file-sliced
    weights (ComfyUI's load_torch_file -- VBAR disk->VRAM fault on CUDA, native mmap on CPU/MPS).
    Returns (model, missing)."""
    from accelerate import init_empty_weights

    cfg = model_cls.load_config(model_dir)
    with init_empty_weights():
        model = model_cls.from_config(cfg)

    # In `dtype`, but for the modules the model keeps in float32 (_keep_in_fp32_modules), as
    # from_pretrained loads them: their buffers made at build (MiniMax-H3-Image's rope) stay exact.
    # The weights keep their file's dtype either way (_assign_sd).
    keep = set(getattr(model, "_keep_in_fp32_modules", None) or ())
    for name, module in model.named_modules():
        if keep & set(name.split(".")):
            continue
        for key, tensor in module.named_parameters(recurse=False):
            if tensor.is_floating_point():
                module._parameters[key] = torch.nn.Parameter(tensor.to(dtype),
                                                             requires_grad=tensor.requires_grad)
        for key, tensor in module.named_buffers(recurse=False):
            if tensor.is_floating_point():
                module._buffers[key] = tensor.to(dtype)

    comfy_ize(model, operations)
    missing = assign_streamed_weights(model, model_dir)
    return model, missing


def as_files(weight_file):
    """A spec's "file" as a list: one single file, or a model's shards."""
    return list(weight_file) if isinstance(weight_file, (list, tuple)) else [weight_file]


def mixed_precision_operations(compute_dtype, load_device, quant_config):
    """ComfyUI's mixed-precision op namespace (comfy.ops.mixed_precision_ops) for a scaled-fp8 (or
    other quantized) checkpoint -- the branch ops.pick_operations takes only when a model_config
    carries quant_config, which adapter.pick_operations never reaches. Formats the compute device
    can't run natively are `disabled` -> emulated dequant, exactly as ComfyUI: on Ampere (no fp8
    tensor cores) float8 runs emulated (dequant to compute_dtype per forward). The disabled set is
    comfy's own (ops.get_disabled_quant_formats, as pick_operations computes it). Reused by
    load_comfy_single_file."""
    disabled = ops.get_disabled_quant_formats(load_device)

    return ops.mixed_precision_ops(quant_config, compute_dtype, disabled=disabled)


def load_comfy_single_file(build, weight_file, load_device, dtype, compute_dtype=None,
                           dynamic=True, process=None):
    """A model of ComfyUI's own (vendored under comfy/ldm/) from its single file (or a model's
    shards, a list of files), as sd.py's load_diffusion_model_state_dict does (sd.py:2460-2555,
    v0.45.4): mmap the file, inject its quant markers (convert_old_quants), detect its
    quantization, pick ComfyUI's ops for it, build the model through `build(sd, operations)`
    (model_base.py:184-187) and load the file's weights (load_model_weights, sd.py:2548-2551):
    a dynamic patcher's model takes the file's tensors as they are, any other copies them into
    its `dtype`, the storage one (sd.py:2529-2538: fp32 on a CPU, unless streamed), computing in
    `compute_dtype` (manual_cast) where it differs. `process` is the model's
    process_unet_state_dict, run after the quant markers. Returns (model, unexpected_keys)."""
    files = as_files(weight_file)

    sd, metadata = {}, None
    for f in files:
        part, part_metadata = comfy.utils.load_torch_file(f, device=torch.device("cpu"),
                                                          return_metadata=True)
        sd.update(part)
        metadata = metadata or part_metadata

    sd, metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)

    if process is not None:
        sd = process(sd)

    quant_config = comfy.utils.detect_layer_quantization(sd, "")

    compute_dtype = compute_dtype or dtype

    if quant_config is not None:
        operations = mixed_precision_operations(compute_dtype, load_device, quant_config)
    else:
        operations = pick_operations(dtype, compute_dtype, load_device)

    model = build(sd, operations)
    model.requires_grad_(False)
    model.eval()
    # model_base.py:197: each tensor's dtype as built, which the patcher casts the file's to when
    # it loads it. A float32 file tensor in a layer built in the model's dtype (MiniMax-H3-Image's
    # float32 biases) is loaded in that dtype, as ComfyUI does, not kept in float32.
    mm.archive_model_dtypes(model)
    model._offloader_files = files  # build_dynamic_patcher's fast_disk

    _missing, unexpected = model.load_state_dict(sd, strict=False, assign=dynamic)

    return model, unexpected


def count_blocks(state_dict_keys, prefix_string):
    """model_detection.py:9-20: how many blocks `prefix_string` numbers ({} the block)."""
    count = 0
    while any(k.startswith(prefix_string.format(count)) for k in state_dict_keys):
        count += 1
    return count


def llama_detect(state_dict, prefix=""):
    """hunyuan_video.llama_detect (text_encoders/hunyuan_video.py:11-23): a Llama-family text
    encoder's dtype and quantization off its file."""
    out = {}
    for key in ("{}model.norm.weight".format(prefix),
                "{}model.layers.0.input_layernorm.weight".format(prefix)):
        if key in state_dict:
            out["dtype_llama"] = state_dict[key].dtype
            break

    quant = comfy.utils.detect_layer_quantization(state_dict, prefix)
    if quant is not None:
        out["llama_quantization_metadata"] = quant
    return out


def load_clip_sd(path):
    """A text encoder's file (or shards) as load_clip reads it (sd.py:1717-1720), with
    load_text_encoder_state_dicts' renames that apply to it (sd.py:1898-1899)."""
    sd = {}
    for f in as_files(path):
        sd.update(comfy.utils.load_torch_file(f, safe_load=True))

    if "lm_head.weight" in sd:
        sd["model.lm_head.weight"] = sd.pop("lm_head.weight")  # prefix missing in some models
    return sd


def qwen3vl_sd(sd):
    """The renames every Qwen3-VL branch of load_text_encoder_state_dicts makes
    (sd.py:2067-2111)."""
    return comfy.utils.state_dict_prefix_replace(sd, {"model.language_model.": "model.",
                                                      "model.visual.": "visual.",
                                                      "lm_head.": "model.lm_head."})


def load_image(image):
    """LoadImage's IMAGE for a PIL image (nodes.py:1782-1789): upright, RGB, (1, H, W, 3) in
    [0, 1]."""
    import numpy
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0)[None]


def load_audio(path):
    """LoadAudio's AUDIO for a file (nodes_audio.py:360-396, 398-430): its first stream decoded
    with PyAV to float32 PCM, a (1, channels, samples) waveform and its rate."""
    import av

    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream found in the file.")
        stream = container.streams.audio[0]
        rate, channels = stream.codec_context.sample_rate, stream.channels
        frames = []
        for frame in container.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != channels:
                buf = buf.view(-1, channels).t()
            frames.append(buf)
    if not frames:
        raise ValueError("No audio frames decoded.")
    wav = torch.cat(frames, dim=1)
    if wav.dtype == torch.int16:  # f32_pcm
        wav = wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        wav = wav.float() / (2 ** 31)
    elif not wav.dtype.is_floating_point:
        raise ValueError(f"Unsupported wav dtype: {wav.dtype}")
    return {"waveform": wav[None], "sample_rate": rate}


class CLIP:
    """sd.py's CLIP (sd.py:246-310, :332-343, :408-439, :441-455, :471-476, v0.45.4) for a text
    encoder of ComfyUI's own: its tokenizer, its model behind a patcher, and the encode its
    CLIPTextEncode runs. `clip` and `tokenizer` are the model's (its sd.py branch), `state_dict`
    the file's, as load_text_encoder_state_dicts hands them over (sd.py:1887-2232)."""

    def __init__(self, clip, tokenizer, state_dict, dtype=None):
        import comfy.hooks
        import comfy.storage

        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()
        # text_encoder_dtype, or the dtype a --*-text-enc flag would set (model_management.py:1249)
        dtype = dtype or mm.text_encoder_dtype(load_device)
        parameters = comfy.utils.calculate_parameters(state_dict)

        params = {"dtype": dtype, "model_options": {},
                  "device": mm.text_encoder_initial_device(load_device, offload_device,
                                                           parameters * mm.dtype_size(dtype))}

        self.cond_stage_model = clip(**params)

        for dt in self.cond_stage_model.dtypes:
            if not mm.supports_cast(load_device, dt):
                load_device = offload_device
                if params["device"] != offload_device:
                    self.cond_stage_model.to(offload_device)

        mm.archive_model_dtypes(self.cond_stage_model)

        self.tokenizer = tokenizer(embedding_directory=None, tokenizer_data={})

        dynamic = not getattr(self.cond_stage_model, "disable_offload", False)
        patcher = model_patcher.CoreModelPatcher if dynamic else model_patcher.ModelPatcher
        self.patcher = patcher(self.cond_stage_model, load_device=load_device,
                               offload_device=offload_device,
                               fast_disk=comfy.storage.state_dict_fast_disk(state_dict))
        # Match torch.float32 hardcode upcast in TE implemention
        self.patcher.set_model_compute_dtype(torch.float32)
        self.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        self.patcher.is_clip = True

        self.load_sd(state_dict)

        if params["device"] == load_device:
            mm.load_models_gpu([self.patcher], force_full_load=True)

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, False, **kwargs)

    def encode(self, text):
        """CLIPTextEncode's conditioning for `text`."""
        return self.encode_from_tokens(self.tokenize(text))[0]

    def encode_from_tokens(self, tokens):
        """encode_from_tokens with return_dict, as encode_from_tokens_scheduled runs it
        unscheduled: the conditioning and what the model returns beside it."""
        self.cond_stage_model.reset_clip_options()

        self.load_model(tokens)
        device = self.patcher.load_device
        self.cond_stage_model.set_clip_options({"execution_device": device})

        with mm.cuda_device_context(device):
            o = self.cond_stage_model.encode_token_weights(tokens)

        return o[0], (o[2] if len(o) > 2 else {})

    def load_sd(self, sd):
        can_assign = self.patcher.is_dynamic()
        self.cond_stage_model.can_assign_sd = can_assign
        for m in self.cond_stage_model.modules():
            m.can_assign_sd = can_assign
        return self.cond_stage_model.load_sd(sd)

    def load_model(self, tokens={}):
        memory_used = 0
        if hasattr(self.cond_stage_model, "memory_estimation_function"):
            memory_used = self.cond_stage_model.memory_estimation_function(
                tokens, device=self.patcher.load_device)
        mm.load_models_gpu([self.patcher], memory_required=memory_used)
        return self.patcher


def id_grids(ids, axis, scale=1.0):
    """Each grid's (h, w) off a pipeline's (seq, 4) position ids, the grids one after another,
    one per value on `axis`: a grid ends on (h - 1, w - 1), times `scale`, in its columns 1 and 2,
    as ComfyUI builds them. It reads the ids on the host, so once per run."""
    rows = ids.tolist()
    return [(round(row[1] / scale) + 1, round(row[2] / scale) + 1)
            for i, row in enumerate(rows)
            if i + 1 == len(rows) or rows[i + 1][axis] != row[axis]]


class BaseModel(torch.nn.Module):
    """ComfyUI's BaseModel (model_base.py:174-276) around a transformer of its own, for one run
    behind the call a diffusers pipeline makes: what the patcher reads off it (device, get_dtype,
    current_patcher), what the pipeline reads (config, dtype), and _apply_model's options. A
    model's adapter subclasses it with its forward (offloader/qwen_image21.py, flux3.py)."""

    current_patcher = None  # set by pre_run, as BaseModel's

    # The model's memory_usage_factor in supported_models.py, for prepare_sampling's estimate. None
    # leaves the load to prepare(), with no estimate.
    memory_usage_factor = None

    def __init__(self, diffusion_model, config, dtype):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.config = SimpleNamespace(**config)
        self.device = None  # set by the patcher
        self._dtype = dtype

    def prepare_sampling(self, shape):
        """sampler_helpers._prepare_sampling's load (v0.45.4, :188-201) for a latent of `shape`,
        with model_base.memory_required's estimate (:430-452), called right after the pipeline
        prepares its latents (_finalize_pipe), before the first step, as KSampler loads its model
        once the noise exists. The estimate makes room for the activations, the text encoder's
        pages going first, so a big latent's model stays resident instead of streaming a part of
        itself every step."""
        patcher = self.current_patcher

        if self.memory_usage_factor is None or patcher is None:
            return

        # memory_required's batch counts cond and uncond, minimum_memory_required's one of them.
        area = shape[0] * math.prod(shape[2:])

        if mm.xformers_enabled() or mm.pytorch_attention_flash_attention():
            unit = mm.dtype_size(self._dtype) * 0.01 * self.memory_usage_factor * 1024 ** 2
        else:
            unit = 0.15 * self.memory_usage_factor * 1024 ** 2

        mm.load_models_gpu([patcher], memory_required=2 * area * unit,
                           minimum_memory_required=area * unit)

    @classmethod
    def load(cls, build, weight_file, config, load_device, dtype, compute_dtype=None,
             dynamic=True, process=None):
        """The model of `build` from its file (load_comfy_single_file, which takes the dtypes,
        `dynamic` and `process`) behind the pipeline's call, its inputs cast to the compute dtype
        as _apply_model casts them; `config` is what the pipeline reads off its transformer's
        config. Returns (model, unexpected_keys)."""
        model, unexpected = load_comfy_single_file(build, weight_file, load_device, dtype,
                                                   compute_dtype, dynamic, process)
        out = cls(model, config, compute_dtype or dtype)
        out._offloader_files = model._offloader_files  # build_dynamic_patcher's fast_disk
        return out, unexpected

    @contextlib.contextmanager
    def attention(self, kind=None):
        """The SDPA priority around a generation: `kind` "cudnn" puts cuDNN ahead of flash
        (`ops.SDPA_BACKEND_PRIORITY`, ops.py:68-73), as the experimental H3's cudnn_attention.py
        node sets ComfyUI's, restored after, so another engine in the same server keeps ComfyUI's.
        The priority is the process's: the text encoder and the VAEs too, and diffusers'
        attention through use_comfy_attention. None keeps ComfyUI's."""
        priority = getattr(ops, "SDPA_BACKEND_PRIORITY", None)

        if kind != "cudnn" or priority is None:
            yield
            return

        from torch.nn.attention import SDPBackend

        saved = list(priority)
        priority[:] = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                       SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        try:
            yield
        finally:
            priority[:] = saved

    @property
    def dtype(self):
        return self._dtype

    def get_dtype(self):
        return self._dtype

    @contextlib.contextmanager
    def cache_context(self, name):
        """The context a pipeline names each call by ("cond", "uncond"), as a diffusers model takes
        it; a model that keeps something per cond reads it."""
        yield

    def transformer_options(self):
        """_apply_model's (model_base.py:254-257)."""
        patcher = self.current_patcher
        return {"prefetch_dynamic_vbars": patcher is not None and patcher.is_dynamic()}

    def lora_keys(self):
        """model_lora_keys_unet's generic entries (lora.py:200-207), as {LoRA key: target}."""
        key_map = {}
        for k, _ in self.named_parameters():
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")]
                key_map["lora_unet_{}".format(key_lora.replace(".", "_"))] = k
                key_map[k[:-len(".weight")]] = k
        return key_map


class FlowSampler:
    """ComfyUI's KSampler for a flow model (ModelSamplingDiscreteFlow, CONST) behind the calls a
    pipeline makes on its scheduler: set_timesteps, then `timesteps` (what the model takes),
    `sigmas`, and a `step` per model output (the velocity), with the KSampler's `seed`. It mirrors,
    citing v0.45.4:

      - ModelSamplingDiscreteFlow (model_sampling.py:289-336): the sigma table, percent_to_sigma;
      - the simple, normal and beta schedulers (samplers.py:645-708), the sigmas on the model's
        device (samplers.py:1257);
      - calculate_denoised for CONST (model_sampling.py:35-37);
      - sample_euler (k_diffusion/sampling.py:190-212), sample_res_multistep (res_multistep at
        eta 0, :1462-1529) and sample_er_sde (:1593-1656), its first sigma offset (:168-176) and
        its noise (default_noise_sampler, :78-88), one step a call.

    The initial noise is the pipeline's, as ComfyUI's (sample.py:22-38): the seed's CPU float32
    draw, scaled by a first sigma of 1.

    A model with its own model sampling passes its `sigma` (a timestep to its sigma) in place of
    the shift, as MiniMax-H3-Image's grid does: the table, the schedulers and percent_to_sigma run
    on it, as ComfyUI's run on any model sampling."""

    def __init__(self, sampler, scheduler, shift, multiplier=1.0, sigma=None):
        if sampler not in ("euler", "res_multistep", "er_sde") \
                or scheduler not in ("simple", "normal", "beta"):
            raise ValueError("FlowSampler: no %s / %s" % (sampler, scheduler))

        self.sampler = sampler
        self.scheduler = scheduler
        self.shift = shift
        self.multiplier = multiplier
        self.seed = None

        self._model_sigma = sigma

        self._table = self._sigma((torch.arange(1, 1001, 1) / 1000) * multiplier)

    def _shifted(self, t):
        if self.shift == 1.0:
            return t
        return self.shift * t / (1 + (self.shift - 1) * t)

    def _sigma(self, timestep):
        if self._model_sigma is not None:
            return self._model_sigma(timestep)
        return self._shifted(timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        if self._model_sigma is not None:
            return float(self._model_sigma(torch.tensor((1.0 - percent) * self.multiplier)))
        return self._shifted(1.0 - percent)

    def _schedule(self, steps):
        s = self._table
        if self.scheduler == "simple":
            ss = len(s) / steps
            return torch.FloatTensor([float(s[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])
        if self.scheduler == "beta":  # alpha = beta = 0.6
            import numpy
            import scipy.stats

            ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
            ts = numpy.rint(scipy.stats.beta.ppf(ts, 0.6, 0.6) * (len(s) - 1))
            sigs = [float(s[int(t)]) for i, t in enumerate(ts) if i == 0 or t != ts[i - 1]]
            return torch.FloatTensor(sigs + [0.0])

        start, end = s[-1] * self.multiplier, s[0] * self.multiplier
        append_zero = True
        if math.isclose(float(self._sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        sigs = [float(self._sigma(t)) for t in torch.linspace(start, end, steps)]
        return torch.FloatTensor(sigs + ([0.0] if append_zero else []))

    def set_timesteps(self, num_inference_steps, device=None, denoise=1.0):
        # BasicScheduler's denoise (nodes_custom_sampler.py:39-46): a longer schedule's last
        # steps, for a sampling that starts from a latent.
        sigmas = self._schedule(num_inference_steps if denoise >= 1.0
                                else int(num_inference_steps / denoise))
        sigmas = sigmas[-(num_inference_steps + 1):].to(device)

        if self.sampler == "er_sde":
            if sigmas[0] >= 1:
                sigmas = sigmas.clone()
                sigmas[0] = self.percent_to_sigma(1e-4)
            self._lambdas = sigmas.logit().neg().neg().exp()  # er_lambda_t = sigma_t / alpha_t

        self.sigmas = sigmas
        self.timesteps = sigmas[:-1] * self.multiplier
        self._i = 0
        self._old = self._old_d = self._old_sigma_down = self._generator = None

    def _noise(self, x):
        if self._generator is None and self.seed is not None:
            self._generator = torch.Generator(device=x.device)
            self._generator.manual_seed(self.seed + (1 if x.device.type == "cpu" else 0))
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device,
                           generator=self._generator)

    def step(self, model_output, timestep, sample, return_dict=False):
        i, s, x = self._i, self.sigmas, sample
        denoised = x - model_output * s[i]

        if self.sampler == "euler":
            x = x + (x - denoised) / s[i] * (s[i + 1] - s[i])
        elif self.sampler == "res_multistep":
            x = self._res_multistep(i, x, denoised)
        else:
            x = self._er_sde(i, x, denoised)

        self._i += 1
        return (x,)

    def _res_multistep(self, i, x, denoised):
        # res_multistep at eta 0 (k_diffusion/sampling.py:1462-1529): sigma_down is the next sigma
        # and no noise is added; Euler on the first step and onto sigma 0, then the second order
        # multistep on the previous denoised.
        s = self.sigmas
        sigma_down = s[i + 1]

        if sigma_down == 0 or self._old is None:
            x = x + (x - denoised) / s[i] * (sigma_down - s[i])
        else:
            def t_fn(sigma):
                return sigma.log().neg()

            def phi1_fn(t):
                return torch.expm1(t) / t

            def phi2_fn(t):
                return (phi1_fn(t) - 1.0) / t

            t, t_old, t_next, t_prev = t_fn(s[i]), t_fn(self._old_sigma_down), t_fn(sigma_down), \
                t_fn(s[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            x = h.neg().exp() * x + h * (b1 * denoised + b2 * self._old)

        self._old = denoised
        self._old_sigma_down = sigma_down
        return x

    def _er_sde(self, i, x, denoised):
        s, lam = self.sigmas, self._lambdas

        def scaler(v):
            return v * ((v ** 0.3).exp() + 10.0)

        if s[i + 1] == 0:
            x = denoised
        else:
            lam_s, lam_t = lam[i], lam[i + 1]
            alpha_t = s[i + 1] / lam_t
            r = scaler(lam_t) / scaler(lam_s)

            # Stage 1 Euler
            x = alpha_t / (s[i] / lam_s) * r * x + alpha_t * (1 - r) * denoised

            if i >= 1:
                dt = lam_t - lam_s
                step = -dt / 200.0
                pos = lam_t + torch.arange(0, 200.0, dtype=torch.float32, device=x.device) * step
                scaled = scaler(pos)

                # Stage 2
                d = (denoised - self._old) / (lam_s - lam[i - 1])
                x = x + alpha_t * (dt + torch.sum(1 / scaled) * step * scaler(lam_t)) * d

                if i >= 2:
                    # Stage 3
                    s_u = torch.sum((pos - lam_s) / scaled) * step
                    u = (d - self._old_d) / ((lam_s - lam[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * scaler(lam_t)) * u
                self._old_d = d

            spread = (lam_t ** 2 - lam_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
            x = x + alpha_t * self._noise(x) * spread

        self._old = denoised
        return x


_sdpa_patched = False


def use_comfy_attention(model=None):
    """Route diffusers' attention through a copy of ComfyUI's own
    `comfy.ops.scaled_dot_product_attention` (comfy/ops.py:58-109), so our SDPA behaves EXACTLY
    like ComfyUI's, not an approximation.

    ComfyUI, on CUDA with a torch that has `sdpa_kernel(set_priority=)`, forces its SDPA backend
    priority (`ops.SDPA_BACKEND_PRIORITY`: [FLASH, CUDNN, EFFICIENT, MATH] since v0.39), but
    ONLY per call and ONLY for large inputs (`q.nelement() >= 1024*128`); small attentions fall
    through to torch's default backend. It also repeats K/V for a masked GQA call when the backend
    can't do GQA natively, and (since v0.41) makes a strided K/V contiguous where it would spill
    the L2 cache (AMD gfx1151). We reproduce that body verbatim, reading the priority list, the
    L2 size and the GQA helper from the vendored ops (so a re-sync carries them over). We can't
    call the vendored function itself: it calls `torch.nn.functional.scaled_dot_product_attention`,
    the very name we patch. diffusers calls that name by attribute
    (diffusers/models/attention_dispatch.py), so we patch it once, process-global (idempotent),
    delegating to the saved original. `model` is accepted only for signature parity with the other
    patchers; the patch is global, as ComfyUI's is. No-op wherever ComfyUI doesn't define the
    priority (no CUDA, or a torch without set_priority), exactly ComfyUI's own guard."""
    global _sdpa_patched
    if _sdpa_patched:
        return True

    import torch.nn.functional as F
    import comfy.model_management as mm

    priority = getattr(ops, "SDPA_BACKEND_PRIORITY", None)
    if priority is None:
        return False

    from torch.nn.attention import sdpa_kernel

    l2_size = ops.SDPA_CONTIGUOUS_KV_L2_SIZE

    orig = F.scaled_dot_product_attention

    def scaled_dot_product_attention(*args, **kwargs):
        # ComfyUI's models call comfy.ops.scaled_dot_product_attention(q, k, v, ...) POSITIONALLY,
        # but diffusers calls torch's F.sdpa by KEYWORD (query=/key=/value=,
        # attention_dispatch.py), so we pull q, k, v out of either form, then run comfy's body.
        q, k, v = args[:3] + tuple(kwargs.pop(n) for n in ("query", "key", "value")[len(args):])
        args = args[3:]
        # manual_cast (fp16 compute on a bf16 checkpoint) can leave the attention inputs
        # mismatched: transformers' qwen3 RoPE promotes q,k to fp32 (fp16 * fp32 cos/sin) while v
        # stays fp16, and torch's SDPA requires one dtype. Coerce to the lowest-precision float
        # present (the compute dtype), exactly what a uniform native fp16 model would run. Only
        # fires on an actual mismatch.
        if not (q.dtype == k.dtype == v.dtype):
            floats = [t.dtype for t in (q, k, v) if t.dtype.is_floating_point]
            if floats:
                tgt = min(floats, key=lambda d: torch.finfo(d).bits)
                q, k, v = (t.to(tgt) for t in (q, k, v))
        # comfy/ops.py:83-106 verbatim from here.
        if q.nelement() < 1024 * 128:  # comfy: small inputs -> default backend
            return orig(q, k, v, *args, **kwargs)
        attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
        if kwargs.get("enable_gqa", False) and attn_mask is not None and not mm.is_nvidia():
            k, v = ops.repeat_kv_for_gqa(k, v, q.shape[-3], -3)
            kwargs["enable_gqa"] = False
        # comfy: a strided K/V whose head outgrows the L2 cache gets copied first (AMD gfx1151).
        if (l2_size
                and sum(t.shape[-2] * t.shape[-1]
                        for t in (k, v) if not t.is_contiguous()) * k.element_size() > l2_size):
            k, v = k.contiguous(), v.contiguous()
        with sdpa_kernel(priority, set_priority=True):
            if (kwargs.get("enable_gqa", False)
                    and attn_mask is not None
                    and q.shape[-3] != k.shape[-3]):
                dropout_p = args[1] if len(args) > 1 else kwargs.get("dropout_p", 0.0)
                is_causal = args[2] if len(args) > 2 else kwargs.get("is_causal", False)
                params = torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal,
                                                        True)
                supports_native_gqa = (
                    torch.backends.cuda.can_use_flash_attention(params)
                    or torch.backends.cuda.can_use_cudnn_attention(params)
                    or torch.backends.cuda.can_use_efficient_attention(params)
                )
                if not supports_native_gqa:
                    k, v = ops.repeat_kv_for_gqa(k, v, q.shape[-3], -3)
                    kwargs["enable_gqa"] = False
            return orig(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = scaled_dot_product_attention
    _sdpa_patched = True
    return True


# -------------------------------------------------------------------------------------------------
# comfy-kitchen fused RoPE. ComfyUI runs its flux RoPE through comfy_kitchen's fused kernel
# (comfy/ldm/flux/math.py:apply_rope1 -> comfy.quant_ops.ck.apply_rope1) on the normal bf16 path.
# We use diffusers models, so we route diffusers' rope through the SAME kernel. No kernel is
# reimplemented: we import comfy_kitchen (via the vendored, already-backend-configured
# comfy.quant_ops.ck) and only add a tiny format shim between diffusers' (cos, sin) and ck's
# freqs_cis. Like ComfyUI, we do NOT device-gate -- ck's own registry picks the backend per device
# (CUDA kernel on cu130+, eager on cpu/mps).
# -------------------------------------------------------------------------------------------------

_KITCHEN_ROPE = os.environ.get("OFFLOADER_KITCHEN_ROPE", "1") != "0"


def _build_freqs_cis(cos, sin):
    """Pack diffusers' (cos, sin) -- each frequency duplicated across its adjacent pair -- into
    comfy_kitchen / ComfyUI's rope() layout [[cos, -sin], [sin, cos]] of shape [..., D/2, 2, 2], in
    float32 (matching diffusers' float accumulation). Reproduces comfy/ldm/flux/math.py:rope() from
    precomputed cos/sin. cos/sin arrive already broadcast to the input rank (via the sequence_dim
    None-insertion), so freqs_cis broadcasts against ck.apply_rope1's reshaped x directly."""
    c = cos[..., 0::2].float()
    s = sin[..., 0::2].float()
    return torch.stack([c, -s, s, c], dim=-1).reshape(*c.shape, 2, 2)


def _make_kitchen_rope(orig):
    """Build the replacement for a diffusers `apply_rotary_emb`. `orig` is the module's own binding
    (the fall-through target). comfy_kitchen is resolved LAZILY on the first *matching* call, so a
    module whose calls never match (or a model we patch but never run) imports nothing.

    The packed freqs_cis is cached by (cos, sin) tensor identity: ComfyUI builds `freqs` ONCE per
    forward and hands it to every block (comfy/ldm/krea2/model.py:364), and diffusers likewise
    passes one (cos, sin) tuple to every block's apply_rotary_emb -- so the pack is loop-invariant.
    Rebuilding it per call was 2 x 28 = 56 builds/step on krea2, each allocating a ~1.6 MB fp32
    tensor (~9 ms/step, ~90 MB of allocator churn). The cache's strong refs keep the ids valid."""
    ck_state = {"ck": None, "resolved": False}
    cache = {"cos": None, "sin": None, "dim": None, "freqs_cis": None}

    def _ck():
        if not ck_state["resolved"]:
            ck_state["resolved"] = True
            try:
                import comfy.quant_ops as qo  # configures ck backends exactly as ComfyUI does
                if getattr(qo, "_CK_AVAILABLE", False):
                    ck_state["ck"] = qo.ck
                    print("offloader: kitchen rope engaged; comfy_kitchen backends=%s"
                          % qo.ck.list_backends(), flush=True)
            except Exception:
                ck_state["ck"] = None
        return ck_state["ck"]

    def apply_rotary_emb(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
        # Only the flux interleaved convention on a (cos, sin) pair, at inference; else native.
        if (not use_real or use_real_unbind_dim != -1 or torch.is_grad_enabled()
                or sequence_dim not in (1, 2) or not isinstance(freqs_cis, (tuple, list))
                or x.dtype not in (torch.bfloat16, torch.float16)):
            return orig(x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim)

        ck = _ck()
        if ck is None:
            return orig(x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim)

        cos, sin = freqs_cis  # [S, D]
        if (cache["cos"] is not cos or cache["sin"] is not sin or cache["dim"] != sequence_dim
                or cache["freqs_cis"].device != x.device):
            # Same broadcast diffusers applies (embeddings.py), so freqs_cis lands on the right
            # axis.
            if sequence_dim == 2:
                cos_b, sin_b = cos[None, None, :, :], sin[None, None, :, :]
            else:
                cos_b, sin_b = cos[None, :, None, :], sin[None, :, None, :]
            cache.update(cos=cos, sin=sin, dim=sequence_dim,
                         freqs_cis=_build_freqs_cis(cos_b.to(x.device),
                                                    sin_b.to(x.device)))
        return ck.apply_rope1(x, cache["freqs_cis"])

    apply_rotary_emb._offloader_kitchen = True
    return apply_rotary_emb


def use_kitchen_rope(model):
    """Route a diffusers transformer's `apply_rotary_emb` through comfy_kitchen's `apply_rope1`
    -- the same kernel ComfyUI uses for flux RoPE (comfy/ldm/flux/math.py). ComfyUI wires this into
    each model's own code; the diffusers analogue is that each transformer does
    `from ..embeddings import apply_rotary_emb`, binding the name BY VALUE in its own module at
    import. So we patch that name in the TRANSFORMER'S module (`type(model).__module__`) --
    patching `embeddings` alone would not reach an already-imported binding.

    Fully model-scoped and safe:
      * Only the model's own module is touched -- other engines' modules are never affected, so a
        long-lived server that runs flux2 then z-image is fine.
      * A model whose module has no such symbol (e.g. z-image, whose rope is a complex-valued
        nested function, not a `from ..embeddings import apply_rotary_emb`) is skipped --
        nothing is patched, numerics byte-for-byte unchanged.
      * Per call, anything outside the flux interleaved convention falls through to the original;
        ck is imported lazily only on a matching call. No device gate (ck's registry picks per
        device).
    No-op when OFFLOADER_KITCHEN_ROPE=0. Idempotent (no double-wrap). Returns True if patched."""
    if not _KITCHEN_ROPE:
        return False

    import sys

    mod = sys.modules.get(type(model).__module__)
    orig = getattr(mod, "apply_rotary_emb", None)

    if orig is None or getattr(orig, "_offloader_kitchen", False):
        return False  # no module-level rope symbol here (e.g. z-image), or already patched

    mod.apply_rotary_emb = _make_kitchen_rope(orig)
    return True


def install_prefetch(model):
    """Overlap weight streaming with compute by copying ComfyUI's model_prefetch mechanism onto a
    diffusers transformer. ComfyUI's own models call prefetch_queue_pop() between transformer
    blocks so block N+1's weights stream on the offload stream while block N computes
    (av_model.py:935). A diffusers forward doesn't, so every layer stalls waiting for its weights
    (~30% of a VBAR step).

    We reproduce it with forward hooks: for each ModuleList of repeated blocks, build a prefetch
    queue at the transformer's forward-start and pop it before each block runs; tear the queues
    down at the transformer's forward-end. No-op unless comfy-aimdo/VBAR is active
    (make_prefetch_queue
    returns None off the dynamic path). Returns the number of block sequences instrumented."""
    import torch.nn as nn
    import comfy.model_prefetch as mp

    # Block sequences = top-level ModuleLists of >=2 blocks (layers + refiners), each running its
    # own layers in turn. A list of bare layers is indexed, one a call, not run in order
    # (MiniMax-H3-Image's per-step output heads): prefetched as a sequence, the head a step runs
    # is not the one the queue streamed, and its weights race the copy.
    sequences = []
    for _name, child in model.named_children():
        if (isinstance(child, nn.ModuleList)
                and len(child) >= 2
                and all(any(True for _ in block.children()) for block in child)):
            sequences.append(child)
    if not sequences:
        return 0

    state = {}  # id(block_list) -> live queue for this forward

    def _device_of(args, kwargs):
        for t in list(args) + list(kwargs.values()):
            if hasattr(t, "device"):
                return t.device
        return None

    def _start(_m, args, kwargs):
        dev = _device_of(args, kwargs)
        if dev is None:
            return
        opts = {"prefetch_dynamic_vbars": True}
        for seq in sequences:
            state[id(seq)] = mp.make_prefetch_queue(list(seq), dev, opts)

    def _block_pre(seq):
        def hook(block, args, kwargs):
            q = state.get(id(seq))
            if q is not None:
                dev = _device_of(args, kwargs)
                if dev is not None:
                    mp.prefetch_queue_pop(q, dev, block)
        return hook

    def _end(_m, args, kwargs, output):
        dev = _device_of(args, kwargs)
        for seq in sequences:
            q = state.pop(id(seq), None)
            if q is not None and dev is not None:
                mp.prefetch_queue_pop(q, dev, None)  # drain/cleanup the last prefetched block
        return output

    model.register_forward_pre_hook(_start, with_kwargs=True)
    for seq in sequences:
        for block in seq:
            block.register_forward_pre_hook(_block_pre(seq), with_kwargs=True)
    model.register_forward_hook(_end, with_kwargs=True)
    return len(sequences)


def add_lora(patcher, lora_specs, key_map=None):
    """Apply LoRAs on-cast, ComfyUI-style: build the {lora_key -> model_weight_key} map exactly as
    comfy.lora.model_lora_keys_unet does, hand it whole to comfy.lora.load_lora (which uses the
    vendored weight_adapter classes to read lora_up/down, lora_A/B, lokr/loha, diff + alpha --
    only bases whose tensors exist in the file are picked), and register the result via
    ModelPatcher.add_patches. The patches become LowVramPatch weight_functions that
    comfy.lora.calculate_weight applies while each weight streams in -- no fusing, no extra
    resident copy. lora_specs is [(path, strength), ...]. key_map: engine-supplied entries for
    published namings that differ from the model's own (e.g. ComfyUI-native Krea2 keys on our
    diffusers module tree); the generic entries below are comfy's model_lora_keys_unet generic
    branch applied to the model's named_parameters (our model IS the target layout)."""
    import comfy.utils as cu
    import comfy.lora as clora

    key_map = dict(key_map) if key_map else {}
    for k, _ in patcher.model.named_parameters():
        if k.endswith(".weight"):
            key_lora = k[:-len(".weight")]
            key_map.setdefault(key_lora, k)  # generic lora format without any weird key names
            key_map.setdefault("transformer.{}".format(key_lora), k)
            key_map.setdefault("diffusion_model.{}".format(key_lora), k)
            key_map.setdefault("lora_unet_{}".format(key_lora.replace(".", "_")), k)

    applied = 0
    for path, strength in lora_specs:
        sd = cu.load_torch_file(path, safe_load=True)
        patch_dict = clora.load_lora(sd, key_map, log_missing=False)
        patcher.add_patches(patch_dict, strength)
        applied += len(patch_dict)
    return applied


def build_dynamic_patcher(model, load_device=None, offload_device=None, size=0):
    """Wrap a streamed model in ModelPatcherDynamic (the VBAR-aware patcher). On a CPU load_device
    it transparently reroutes to a plain ModelPatcher (VBAR is GPU-only).

    fast_disk is ComfyUI's per-model storage policy: it passes
    `comfy.storage.state_dict_fast_disk(sd)` (sd.py:2408), i.e. model_fast_disk over the files the
    weights come from. True (a fast NVMe) streams VBAR weights from the file instead of pinning
    them in host RAM. Our loaders record those files as `_offloader_files`."""
    import comfy.model_patcher as model_patcher
    import comfy.storage as storage
    if load_device is None:
        load_device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()
    make_patchable(model)
    fast_disk = storage.model_fast_disk(getattr(model, "_offloader_files", []))
    return model_patcher.ModelPatcherDynamic(model, load_device=load_device,
                                             offload_device=offload_device, size=size,
                                             fast_disk=fast_disk)


def _vae_memory_used(kind, shape, dtype):
    """ComfyUI's VAE working-memory estimates, copied verbatim from comfy/sd.py: the Wan 2.1 VAE
    pair for video-style (B,C,T,H,W) tensors (:757-758 -- the Qwen-Image / Krea 2 VAE) and the
    AutoencoderKL pair for classic (B,C,H,W) tensors (:481-482, VAE_KL_MEM_RATIO=1.0). The
    constants embed comfy's empirical safety margins; an under-estimate only means the OOM
    fallback fires, an over-estimate only evicts more of the streamed models."""
    if len(shape) >= 5:
        if kind == "decoding":
            return (2200 if shape[2] <= 4 else 7000) * shape[3] * shape[4] * (8 * 8) \
                * mm.dtype_size(dtype)
        return (1500 if shape[2] <= 4 else 6000) * shape[3] * shape[4] * mm.dtype_size(dtype)
    if kind == "decoding":
        return (2178 * shape[2] * shape[3] * 64) * mm.dtype_size(dtype)
    return (1767 * shape[2] * shape[3]) * mm.dtype_size(dtype)


# -------------------------------------------------------------------------------------------------
# ComfyUI's VAE, mirrored from comfy/sd.py (v0.45.4): the model sd.py builds for a file and the way
# its VAE class runs it. The model code itself is vendored (comfy/ldm/); what sd.py does around it
# is copied here, each piece citing its sd.py lines, so a re-sync diffs those lines between the
# two tags and ports any change (resync.md, category 3).
# -------------------------------------------------------------------------------------------------
def _comfy_vae_model(sd, metadata=None):
    """sd.py's VAE.__init__ for the files an engine opts in with: the model, from the file's keys
    and metadata, and the settings sd.py gives it. Another file needs its sd.py branch added
    here."""
    if "decoder.middle.0.residual.0.gamma" in sd:  # sd.py:846-887, the Wan VAEs
        wan22_layout = "decoder.upsamples.0.upsamples.0.residual.2.weight" in sd
        head = sd.get("decoder.head.2.weight", None)
        if wan22_layout and head is not None and head.ndim == 5 and head.shape[2] == 1:
            # sd.py:849-859, the Qwen Image 2.1 VAE: Wan 2.2 layout, temporal kernel 1, RGBA.
            import comfy.ldm.wan.vae2_2

            output_channels = sd["decoder.head.2.weight"].shape[0]
            model = comfy.ldm.wan.vae2_2.WanVAE(
                dim=sd["encoder.conv1.weight"].shape[0],
                dec_dim=sd["decoder.head.0.gamma"].shape[0], z_dim=64, dim_mult=[1, 2, 4, 8, 8],
                num_res_blocks=2, attn_scales=[],
                temperal_downsample=[False, True, True, True], dropout=0.0,
                image_channels=output_channels, patch_size=1, temporal_kernel=1)
            return model, dict(
                latent_dim=2, latent_channels=64, output_channels=output_channels,
                working_dtypes=[torch.bfloat16, torch.float16, torch.float32],
                upscale_ratio=16, downscale_ratio=16,
                upscale_index_formula=None, downscale_index_formula=None,
                memory_used_encode=lambda shape, dtype: (600 * shape[2] * shape[3]
                                                         * mm.dtype_size(dtype)),
                memory_used_decode=lambda shape, dtype: (900 * shape[2] * shape[3] * (16 * 16)
                                                         * mm.dtype_size(dtype)))
        if not wan22_layout:
            # sd.py:872-887, the Wan 2.1 VAE (Qwen-Image, Krea 2).
            import comfy.ldm.wan.vae

            output_channels = sd["encoder.conv1.weight"].shape[1]
            model = comfy.ldm.wan.vae.WanVAE(
                dim=sd["decoder.head.0.gamma"].shape[0], z_dim=16, dim_mult=[1, 2, 4, 4],
                num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True],
                image_channels=output_channels,
                conv_out_channels=sd["decoder.head.2.weight"].shape[0], dropout=0.0)
            return model, dict(
                latent_dim=3, latent_channels=16, output_channels=output_channels,
                working_dtypes=[torch.bfloat16, torch.float16, torch.float32],
                upscale_ratio=(lambda a: max(0, a * 4 - 3), 8, 8), upscale_index_formula=(4, 8, 8),
                downscale_ratio=(lambda a: max(0, math.floor((a + 3) / 4)), 8, 8),
                downscale_index_formula=(4, 8, 8),
                memory_used_encode=lambda shape, dtype: ((1500 if shape[2] <= 4 else 6000)
                                                         * shape[3] * shape[4]
                                                         * mm.dtype_size(dtype)),
                memory_used_decode=lambda shape, dtype: ((2200 if shape[2] <= 4 else 7000)
                                                         * shape[3] * shape[4] * (8 * 8)
                                                         * mm.dtype_size(dtype)))
    if "bn.running_mean" in sd and "decoder.conv_in.weight" in sd:
        # sd.py:678-716, the FLUX.2 VAE (its small decoder too) on sd.py's AutoencoderKL defaults
        # (:509-519): the latent's 2x2 patches batch-normalized, 128 channels at 1/16.
        from comfy.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
        from comfy.ldm.modules.diffusionmodules.model import fused_norm_served
        from comfy.utils import state_dict_prefix_replace

        ratio = 2.73 if mm.is_amd() else 1.0
        ddconfig = {"double_z": True, "z_channels": sd["decoder.conv_in.weight"].shape[1],
                    "resolution": 256, "in_channels": 3, "out_ch": 3, "ch": 128,
                    "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_resolutions": [],
                    "dropout": 0.0, "batch_norm_latent": True, "fused_norm": True}
        if "decoder.post_quant_conv.weight" in sd:
            state_dict_prefix_replace(sd, {"decoder.post_quant_conv.": "post_quant_conv.",
                                           "encoder.quant_conv.": "quant_conv."})
        decoder = None
        decoder_ch = sd["decoder.conv_in.weight"].shape[0] // ddconfig["ch_mult"][-1]
        if decoder_ch != ddconfig["ch"]:
            decoder = dict(ddconfig, ch=decoder_ch)
        if "post_quant_conv.weight" in sd:
            model = AutoencoderKL(ddconfig=ddconfig,
                                  embed_dim=sd["post_quant_conv.weight"].shape[1],
                                  **({"decoder_ddconfig": decoder} if decoder is not None else {}))
        else:
            model = AutoencodingEngine(
                regularizer_config={
                    "target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                encoder_config={"target": "comfy.ldm.modules.diffusionmodules.model.Encoder",
                                "params": ddconfig},
                decoder_config={"target": "comfy.ldm.modules.diffusionmodules.model.Decoder",
                                "params": decoder or ddconfig})

        def memory_used_decode(shape, dtype):
            fused = fused_norm_served(mm.vae_device(), dtype)
            return (2178 * shape[2] * shape[3] * 64 * mm.dtype_size(dtype) * ratio
                    * (1.0 if fused else 4.0))

        return model, dict(
            latent_dim=2, latent_channels=ddconfig["z_channels"] * 4, output_channels=3,
            working_dtypes=[torch.float16, torch.bfloat16, torch.float32],
            upscale_ratio=16, downscale_ratio=16,
            upscale_index_formula=None, downscale_index_formula=None,
            memory_used_encode=lambda shape, dtype: (1767 * shape[2] * shape[3]
                                                     * mm.dtype_size(dtype) * ratio),
            memory_used_decode=memory_used_decode)
    if ("decoder.transformer_blocks.0.scale1" in sd and "decoder.proj_out.weight" in sd
            and sd["decoder.proj_out.weight"].shape[0] == 768
            and sd["decoder.x_embedder.weight"].shape[0] == 1024):
        # sd.py:1050-1124, the MiniMax-H3-Image VAE: the H3 encoder with a single-frame 2D ViT
        # decoder, its latent statistics from the release, one frame of 24 channels at 1/16. It
        # tiles itself (handles_tiling) and decodes straight to [0, 1] (process_output is the
        # identity).
        import comfy.ldm.minimax.vae
        import comfy.ops

        sd.setdefault("latents_mean", torch.tensor(comfy.ldm.minimax.vae.LATENTS_MEAN))
        sd.setdefault("latents_std", torch.tensor(comfy.ldm.minimax.vae.LATENTS_STD))
        layers = sum(k.startswith("decoder.transformer_blocks.") and k.endswith(".scale1")
                     for k in sd)
        model = comfy.ldm.minimax.vae.MiniMaxH3ImageVAE(operations=comfy.ops.disable_weight_init,
                                                        num_layers=layers)
        upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
        chunk_frames = (model.tokens_chunk_size + model.token_overlap) * model.vae_ratio_t

        def memory_used_encode(shape, dtype):
            frames = min(shape[2], model.clip_length)
            fixed, per_pixel = (110_000_000, 7) if shape[2] == 1 else (1_300_000_000, 9.5)
            return ((per_pixel * frames * shape[3] * shape[4] + fixed) * mm.dtype_size(dtype)
                    * 1.03)

        def memory_used_decode(shape, dtype):
            # still_frame is 0: a single-frame decoder
            frames = upscale_ratio[0](shape[2])
            height, width = shape[3] * 16, shape[4] * 16
            fixed = 110_000_000 if frames <= 22 else 270_000_000
            frames = min(frames, chunk_frames + 2)
            return (9.5 * frames * height * width + fixed) * mm.dtype_size(dtype) * 1.03

        return model, dict(
            latent_dim=3, latent_channels=24, output_channels=3,
            working_dtypes=[torch.float16, torch.float32], handles_tiling=True,
            process_output=lambda image: image,
            upscale_ratio=upscale_ratio, upscale_index_formula=(4, 16, 16),
            downscale_ratio=(lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16),
            downscale_index_formula=(4, 16, 16),
            memory_used_encode=memory_used_encode, memory_used_decode=memory_used_decode)
    if "decoder.transformer_blocks.0.scale1" in sd and not (
            "decoder.proj_out.weight" in sd and sd["decoder.proj_out.weight"].shape[0] == 768):
        # sd.py:1050-1125, the MiniMax H3 video VAE with its stock decoder: a causal 3D encoder and
        # a ViT decoder, 24 channels at 1/16, 17k+5 frames to 5k+2 latents, an int8 file through
        # ComfyUI's mixed-precision ops. It tiles itself (handles_tiling), streams its chunks into
        # the caller's buffer (comfy_has_chunked_io) and decodes straight to [0, 1]
        # (process_output is the identity). The Sol and LightVAE decoders are not mirrored.
        import comfy.ldm.minimax.vae
        import comfy.ops

        ops = comfy.ops.disable_weight_init
        quant = comfy.utils.detect_layer_quantization(sd, "")
        if quant is not None:
            ops = comfy.ops.mixed_precision_ops(quant, torch.float16)
        layers = sum(k.startswith("decoder.transformer_blocks.") and k.endswith(".scale1")
                     for k in sd)
        still_frame = None
        if metadata is not None and metadata.get("h3_t1_direct") == "true":
            still_frame = int(metadata.get("h3_t1_output_slice", 3))
        kept = {(int(k.split(".")[2]), int(k.split(".")[4])) for k in sd
                if k.startswith("encoder.down.") and k.endswith(".conv1.weight")}
        model = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE(
            operations=ops, num_layers=layers, still_frame=still_frame, encoder_kept_blocks=kept,
            encoder="encoder.conv_in.weight" in sd, decoder=layers > 0, noncausal_decoder=False)
        upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
        chunk_frames = (model.tokens_chunk_size + model.token_overlap) * model.vae_ratio_t

        def memory_used_encode(shape, dtype):
            fixed, per_pixel = (110_000_000, 7) if shape[2] == 1 else (1_300_000_000, 9.5)
            frames = min(shape[2], model.clip_length)
            return ((per_pixel * frames * shape[3] * shape[4] + fixed) * mm.dtype_size(dtype)
                    * 1.03)

        def memory_used_decode(shape, dtype):
            frames = upscale_ratio[0](shape[2])
            height, width = shape[3] * 16, shape[4] * 16
            fixed = (150_000_000 if frames == 1 and still_frame is None
                     else 110_000_000 if frames <= 22 else 270_000_000)
            frames = min(frames, chunk_frames + 2)
            return (9.5 * frames * height * width + fixed) * mm.dtype_size(dtype) * 1.03

        return model, dict(
            latent_dim=3, latent_channels=24, output_channels=3,
            working_dtypes=[torch.float16, torch.float32], handles_tiling=True,
            process_output=lambda image: image,
            upscale_ratio=upscale_ratio, upscale_index_formula=(4, 16, 16),
            downscale_ratio=(lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16),
            downscale_index_formula=(4, 16, 16),
            memory_used_encode=memory_used_encode, memory_used_decode=memory_used_decode)
    if "pre_block.attn.zero_k_bias" in sd:
        # sd.py:1127-1147, the MiniMax H3 audio VAE: a DAC encoder and a BigVGAN decoder, stereo at
        # 32 kHz, 32 channels at 1/800, float32 only; the waveform as it comes (process_output is
        # the identity).
        import comfy.ldm.minimax.audio_vae

        return comfy.ldm.minimax.audio_vae.MiniMaxH3AudioVAE(), dict(
            latent_dim=2, latent_channels=32, output_channels=2, pad_channel_value="replicate",
            audio_sample_rate=32000, upscale_ratio=800, downscale_ratio=800,
            working_dtypes=[torch.float32], process_output=lambda audio: audio,
            memory_used_encode=lambda shape, dtype: ((900 * shape[2] + 105_000_000)
                                                     * mm.dtype_size(dtype) * 1.03),
            memory_used_decode=lambda shape, dtype: (max(42_000_000,
                                                         220 * shape[-1] * 800 + 20_000_000)
                                                     * mm.dtype_size(dtype) * 1.03))
    if "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight" in sd:
        # sd.py:609-624, the SeedVR2 VAE: a causal 3D VAE, 16 channels at 1/8, its latent the
        # posterior mean scaled by format_encoded, loaded whole (disable_offload). It tiles itself
        # (handles_tiling) and estimates its own memory.
        import comfy.ldm.seedvr.vae

        model = comfy.ldm.seedvr.vae.VideoAutoencoderKLWrapper()
        return model, dict(
            latent_dim=3, latent_channels=comfy.ldm.seedvr.vae.SEEDVR2_LATENT_CHANNELS,
            output_channels=3, disable_offload=True, handles_tiling=True,
            working_dtypes=[torch.float16, torch.bfloat16, torch.float32],
            format_encoded=model.comfy_format_encoded,
            upscale_ratio=(lambda a: max(0, a * 4 - 3), 8, 8), upscale_index_formula=(4, 8, 8),
            downscale_ratio=(lambda a: max(0, math.floor((a + 3) / 4)), 8, 8),
            downscale_index_formula=(4, 8, 8),
            memory_used_encode=model.comfy_memory_used_encode,
            memory_used_decode=model.comfy_memory_used_decode)
    raise ValueError("comfy_vae: no comfy/sd.py VAE branch is mirrored for this file, add it to "
                     "adapter._comfy_vae_model")


class ComfyVAE:
    """ComfyUI's VAE as sd.py's VAE class runs it, behind the calls a diffusers pipeline makes:
    the model sd.py builds for the file, loaded through ComfyUI's model management (a
    CoreModelPatcher, dynamic when VBAR is on), its encode / decode with sd.py's memory steps and
    out-of-memory fallback to its tiles, the output clamped (process_output). sd.py builds its
    models once ComfyUI's startup picked the patcher; here that is enable_vbar in load_pipe, so the
    file is read and the model built on first use. Any other attribute is the model's (e.g.
    temperal_downsample). `tiled`, a (tile_size, overlap) pair in pixels, runs every encode and
    decode tiled at it, as the VAEEncodeTiled and VAEDecodeTiled nodes do (nodes.py:348-410)."""

    def __init__(self, path, config, tiled=None):
        self._path = path
        self._tiled = tiled
        self.config = SimpleNamespace(**config)  # what the pipeline reads (z_dim, ...)

    @staticmethod
    def process_output(x):
        # sd.py's process_output for the [-1, 1] a diffusers pipeline takes: clamped. A model
        # that decodes to its own range sets the identity in its _comfy_vae_model settings.
        return x.clamp(-1.0, 1.0)

    def __getattr__(self, name):
        # Only reached for what the instance lacks: build it, then read it off the model.
        if name.startswith("_"):
            raise AttributeError(name)
        self._build()
        return self.__dict__[name] if name in self.__dict__ else getattr(self.model, name)

    def _build(self):
        # sd.py:1166-1188, the end of VAE.__init__.
        if "model" in self.__dict__:
            return
        # As VAELoader reads it (nodes.py:857), with its metadata: under VBAR the weights stay
        # file slices, not RAM.
        sd, metadata = comfy.utils.load_torch_file(self.__dict__.pop("_path"),
                                                   return_metadata=True)
        # sd.py:503-505: a file in diffusers' layout, converted to ComfyUI's.
        if ("decoder.up_blocks.2.upsamplers.0.upscale_conv.weight" not in sd
                and "decoder.up_blocks.0.resnets.0.norm1.weight" in sd):
            from comfy.diffusers_convert import convert_vae_state_dict
            sd = convert_vae_state_dict(sd)
        model, settings = _comfy_vae_model(sd, metadata)
        self.__dict__.update(settings)
        self.device = mm.vae_device()
        offload_device = mm.vae_offload_device()
        self.vae_dtype = mm.vae_dtype(self.device, self.working_dtypes)
        model.eval().to(self.vae_dtype)
        mm.archive_model_dtypes(model)
        self.output_device = mm.intermediate_device()
        self.patcher = model_patcher.CoreModelPatcher(model, load_device=self.device,
                                                      offload_device=offload_device)
        model.current_patcher = self.patcher
        missing, _ = model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())
        if not self.patcher.is_dynamic():
            model.to(self.vae_dtype)
        if missing:
            print("Missing VAE keys %s" % missing, flush=True)
        self.model = model

    @property
    def dtype(self):
        self._build()
        return self.vae_dtype

    def to(self, *args, **kwargs):
        return self  # ComfyUI's model management places the model

    def decode(self, z, return_dict=True):
        # sd.py:1360-1410: load through load_models_gpu with the estimate, tile straight away only
        # for a model declaring its estimate reliable, else try whole and tile on an OOM.
        from diffusers.models.autoencoders.vae import DecoderOutput

        import comfy.model_prefetch

        self._build()
        framed = self.latent_dim == 2 and z.dim() == 5  # sd.py:1364 drops a 2D model's frame axis
        z = (z[:, :, 0] if framed else z).to(self.device, self.vae_dtype)
        do_tile = False
        with mm.cuda_device_context(self.device):
            if self._tiled is not None:
                x = self._decode_tiled(z)
            else:
                try:
                    memory_used = self.memory_used_decode(z.shape, self.vae_dtype)
                    with comfy.model_prefetch.pause_malloc_graph():
                        mm.load_models_gpu([self.patcher], memory_required=memory_used,
                                           force_full_load=self._full_load())
                    if getattr(self.model, "comfy_decode_estimate_is_reliable", False) \
                            and memory_used > self.patcher.get_free_memory(self.device):
                        raise mm.OOM_EXCEPTION("decode estimate exceeds free memory")
                    if getattr(self.model, "comfy_has_chunked_io", False):
                        # sd.py:1384-1394: the model writes each chunk into the output, there
                        with comfy.model_prefetch.pause_malloc_graph():
                            x = torch.empty(self.model.decode_output_shape(z.shape),
                                            device=self.output_device,
                                            dtype=mm.intermediate_dtype())
                        self.model.decode(z, output_buffer=x)
                    else:
                        x = self.model.decode(z)
                except Exception as e:
                    mm.raise_non_oom(e)
                    print("Warning: Ran out of memory when regular VAE decoding, retrying with "
                          "tiled VAE decoding.", flush=True)
                    do_tile = True
            if do_tile:
                mm.soft_empty_cache()
                if getattr(self, "handles_tiling", False):  # _decode_tiled_owned, sd.py:1301-1303
                    x = self.model.decode_tiled(z)
                elif self.latent_dim == 3:
                    x = self.decode_tiled_3d(z)
                else:
                    x = self.decode_tiled_(z)
        x = self.process_output(x.to(self.output_device, mm.intermediate_dtype()))
        x = x.unsqueeze(2) if framed else x
        return DecoderOutput(sample=x) if return_dict else (x,)

    def encode(self, x, return_dict=True):
        # sd.py:1513-1562: the same memory steps; it encodes to the mean.
        from diffusers.models.modeling_outputs import AutoencoderKLOutput

        self._build()
        framed = self.latent_dim == 2 and x.dim() == 5
        device, dtype = x.device, x.dtype
        x = (x[:, :, 0] if framed else x).to(self.device, self.vae_dtype)
        do_tile = False
        with mm.cuda_device_context(self.device):
            if self._tiled is not None:
                mu = self._encode_tiled(x)
            else:
                try:
                    memory_used = self.memory_used_encode(x.shape, self.vae_dtype)
                    mm.load_models_gpu([self.patcher], memory_required=memory_used,
                                       force_full_load=self._full_load())
                    mu = self.model.encode(x)
                except Exception as e:
                    mm.raise_non_oom(e)
                    print("Warning: Ran out of memory when regular VAE encoding, retrying with "
                          "tiled VAE encoding.", flush=True)
                    do_tile = True
            if do_tile:
                mm.soft_empty_cache()
                if getattr(self, "handles_tiling", False):  # _encode_tiled_owned, sd.py:1341-1344
                    mu = self.model.encode_tiled(x)
                elif self.latent_dim == 3:
                    mu = self.encode_tiled_3d(x)
                else:
                    mu = self.encode_tiled_(x)
        if getattr(self, "format_encoded", None) is not None:  # in vae_output_dtype, sd.py:1536,
            mu = self.format_encoded(mu.to(dtype=mm.intermediate_dtype()))  # 1567-1568, 1619-1620
        # Back where and as the pipeline passed it, as a diffusers VAE returns it: the pipeline
        # feeds the latent to its transformer as is (ComfyUI's model casts its input itself).
        mu = mu.to(device, dtype)
        mu = mu.unsqueeze(2) if framed else mu
        dist = SimpleNamespace(mode=lambda: mu, sample=lambda generator=None: mu)
        return AutoencoderKLOutput(latent_dist=dist) if return_dict else (dist,)

    def _full_load(self):
        # sd.py's force_full_load=self.disable_offload: a VAE loaded whole.
        return getattr(self, "disable_offload", False)

    def _tiles(self):
        # The (tile_size, overlap) pair, for the VAEs whose own tiling is mirrored.
        if not getattr(self, "handles_tiling", False):
            raise ValueError("comfy_vae: tiled is mirrored for a VAE that tiles itself")
        return self._tiled

    def _encode_tiled(self, x):
        # VAEEncodeTiled (nodes.py:399-415) through sd.py's encode_tiled (:1571-1621), for a model
        # that tiles itself: the whole input's estimate, then _encode_tiled_owned at the pixel
        # tile and overlap.
        tile, overlap = self._tiles()
        mm.load_models_gpu([self.patcher],
                           memory_required=self.memory_used_encode(x.shape, self.vae_dtype),
                           force_full_load=self._full_load())
        return self.model.encode_tiled(x, tile_x=tile, tile_y=tile, overlap=overlap)

    def _decode_tiled(self, z):
        # VAEDecodeTiled (nodes.py:348-397) through sd.py's decode_tiled (:1481-1511): the tile
        # and overlap in latent units, the estimate one tile's (_tile_bounded_shape, :1462-1471),
        # then _decode_tiled_owned (:1301-1303).
        tile, overlap = self._tiles()
        if tile < overlap * 4:
            overlap = tile // 4
        ratio = self.upscale_ratio[-1] if isinstance(self.upscale_ratio, tuple) \
            else self.upscale_ratio
        tile, overlap = tile // ratio, overlap // ratio
        shape = list(z.shape)
        if len(shape) == 5:
            shape[3], shape[4] = min(shape[3], tile), min(shape[4], tile)
        mm.load_models_gpu([self.patcher],
                           memory_required=self.memory_used_decode(shape, self.vae_dtype),
                           force_full_load=self._full_load())
        return self.model.decode_tiled(z, tile_x=tile, tile_y=tile, overlap=overlap)

    def decode_tiled_(self, z, tile_x=64, tile_y=64, overlap=16):
        # sd.py:1273-1295: three tilings, averaged.
        return sum(comfy.utils.tiled_scale(z, self.model.decode, w, h, overlap,
                                           upscale_amount=self.upscale_ratio,
                                           out_channels=self.output_channels,
                                           output_device=z.device)
                   for w, h in ((tile_x // 2, tile_y * 2), (tile_x * 2, tile_y // 2),
                                (tile_x, tile_y))) / 3.0

    def decode_tiled_3d(self, z):
        # sd.py:1425-1448: reserve what a whole decode would use, capped at 80% of the device,
        # shrink the temporal tile until one fits, then grow the spatial tile while it still fits.
        ratio = self.upscale_ratio  # spacial_compression_decode, sd.py:1626-1630
        tile = 256 // (ratio[-1] if isinstance(ratio, tuple) else ratio)
        budget = min(self.memory_used_decode(z.shape, self.vae_dtype),
                     int(mm.get_total_memory(self.device) * 0.8))
        mm.load_models_gpu([self.patcher], memory_required=budget)

        def est(tile_t, tile_xy):  # _tile_bounded_shape, sd.py:1462-1470
            s = list(z.shape)
            s[2], s[3], s[4] = min(s[2], tile_t), min(s[3], tile_xy), min(s[4], tile_xy)
            return self.memory_used_decode(s, self.vae_dtype)

        tile_t = z.shape[2]
        while tile_t > 2 and est(tile_t, tile) > budget:
            tile_t = -(-tile_t // 2)
        while tile * 2 <= max(z.shape[3], z.shape[4]) and est(tile_t, tile * 2) <= budget:
            tile *= 2
        overlap = tile // 4
        # decode_tiled_3d, sd.py:1297-1299
        return comfy.utils.tiled_scale_multidim(
            z, self.model.decode, tile=(tile_t, tile, tile), overlap=(1, overlap, overlap),
            upscale_amount=self.upscale_ratio, out_channels=self.output_channels,
            index_formulas=self.upscale_index_formula, output_device=z.device)

    def encode_tiled_(self, x, tile_x=512, tile_y=512, overlap=64):
        # sd.py:1305-1315: three tilings, averaged.
        return sum(comfy.utils.tiled_scale(x, self.model.encode, w, h, overlap,
                                           upscale_amount=1 / self.downscale_ratio,
                                           out_channels=self.latent_channels,
                                           output_device=x.device)
                   for w, h in ((tile_x, tile_y), (tile_x * 2, tile_y // 2),
                                (tile_x // 2, tile_y * 2))) / 3.0

    def encode_tiled_3d(self, x):
        # sd.py:1553-1560 picks 256-pixel tiles and 64 of overlap; encode_tiled_3d is
        # sd.py:1337-1339.
        return comfy.utils.tiled_scale_multidim(
            x, self.model.encode, tile=(9999, 256, 256), overlap=(1, 64, 64),
            upscale_amount=self.downscale_ratio, out_channels=self.latent_channels,
            downscale=True, index_formulas=self.downscale_index_formula, output_device=x.device)


def install_tiled_vae_fallback(pipe, node_boundary=None):
    """ComfyUI's sampler->VAE node boundary + its VAE out-of-memory fallback.

    `node_boundary(kind)` (the seam's node boundary, "decoding" or "encoding") runs first: in
    ComfyUI the KSampler node ENDS before VAEDecode begins, so the teardown has already reset the
    VBAR watermarks and the DiT's residency is evictable when the VAE needs VRAM. A diffusers
    pipeline calls vae.decode inside the same __call__, so without this boundary the DiT still pins
    its VRAM (measured: 2.66GB held, 0 free) and a 1600x1200 decode dies on one ~1.4GB fp32
    upsample after all 8 steps succeeded.

    Then comfy/sd.py:1369-1371: estimate the call's working memory (`memory_used_decode`) and hand
    it to load_models_gpu as memory_required BEFORE the first kernel. For an already-resident VAE
    load_models_gpu's guts are free_memory(max(inference_memory, memory_required +
    extra_reserved_memory())) (model_management.py:944,1001): the CONTROLLED eviction of the
    streamed models. Skipping this and letting the first big cudnn workspace request evict on
    demand inside the allocator dies with an uncatchable C++ abort instead of a catchable OOM
    (measured: the abort fires on the VAE's first conv3d).

    Then the fallback itself (comfy/sd.py:1403-1410 decode, :1448-1455 encode): try the regular
    call; on OOM, `raise_non_oom` anything else, warn with comfy's own wording, and only set a
    flag -- comfy deliberately retries OUTSIDE the except block because "the exception itself refs
    them all until we get out of this except block", so the tensors can gc first -- then retry
    tiled. Comfy retries with its own tiler; a diffusers VAE ships the equivalent (`enable_tiling`,
    seam-blended; the qwen VAE's 256px tile / 64px overlap equal the tile comfy starts from,
    sd.py:1426-1427; since v0.39 comfy then resizes a 3D tile to its VRAM budget, sd.py:1437-1448),
    and the regular path is restored after, matching comfy's per-call semantics.
    Model-agnostic: keyed only off the vae exposing enable_tiling. A ComfyVAE runs sd.py's own
    memory steps and fallback, so it only gets the node boundary. No-op without a vae, or if
    already installed."""
    vae = getattr(pipe, "vae", None)

    if vae is None or getattr(vae, "_tiled_fallback_installed", False):
        return

    if isinstance(vae, ComfyVAE):
        def bounded(real, kind):
            def call(*args, **kwargs):
                if node_boundary is not None:
                    node_boundary(kind)
                return real(*args, **kwargs)
            return call

        vae.decode = bounded(vae.decode, "decoding")
        vae.encode = bounded(vae.encode, "encoding")
        vae._tiled_fallback_installed = True
        return

    if not hasattr(vae, "enable_tiling"):
        return

    def wrap(real, kind):
        def call(*args, **kwargs):
            if node_boundary is not None:
                node_boundary(kind)
            samples_in = args[0] if args else next(iter(kwargs.values()), None)
            do_tile = False
            if hasattr(samples_in, "shape") and samples_in.device.type != "cpu":
                # A VAE may carry its own estimates, as comfy's VAE class does
                # (memory_used_decode / memory_used_encode(shape, dtype) -> bytes); else the
                # generic pairs, which fit comfy's 8x VAEs.
                dtype = getattr(vae, "dtype", samples_in.dtype)
                own = getattr(vae, {"decoding": "memory_used_decode",
                                    "encoding": "memory_used_encode"}[kind], None)
                memory_used = (own(samples_in.shape, dtype) if own is not None
                               else _vae_memory_used(kind, samples_in.shape, dtype))
                extra_mem = max(mm.minimum_inference_memory(),
                                memory_used + mm.extra_reserved_memory())
                mm.free_memory(extra_mem, samples_in.device)
                # Still short after freeing: tile up front (comfy only tiles after an OOM; a
                # diffusers VAE run into a sure OOM can end in an uncatchable CUDA abort instead,
                # measured: a 4 GB card decoding 1024² with a 7 GB working set).
                do_tile = mm.get_free_memory(samples_in.device) < memory_used
            if not do_tile:
                try:
                    return real(*args, **kwargs)
                except Exception as e:
                    mm.raise_non_oom(e)
                    print("Warning: Ran out of memory when regular VAE %s, retrying with tiled "
                          "VAE %s." % (kind, kind), flush=True)
                    do_tile = True
            if do_tile:
                mm.soft_empty_cache()
                vae.enable_tiling()
                try:
                    return real(*args, **kwargs)
                finally:
                    vae.disable_tiling()
        return call

    vae.decode = wrap(vae.decode, "decoding")
    if hasattr(vae, "encode"):
        vae.encode = wrap(vae.encode, "encoding")
    vae._tiled_fallback_installed = True


def install_unpadded_encode(pipe):
    """Drop padded text tokens from the conditioning before it ever reaches the transformer.

    ComfyUI never pads: its Qwen3-VL tokenizer runs `pad_to_max_length=False`
    (comfy/text_encoders/qwen3vl.py:133), and its krea2 text encoder drops an all-ones mask
    outright (comfy/text_encoders/krea2.py:65-66), so every attention call gets mask=None.
    diffusers' own qwen pipeline already agrees (`padding=True`, then `if prompt_embeds_mask.all():
    prompt_embeds_mask = None` -- pipeline_qwenimage_edit_plus.py:327). Its krea2 pipeline is the
    outlier: it pads every prompt to a fixed 512 tokens (`padding="max_length"`,
    pipeline_krea2.py:229-235) and keeps the mask, so the DiT drags ~500 dead tokens through all 28
    blocks -- 1536 tokens at 512x512 where ComfyUI runs ~1054, which is the whole per-step gap.

    Model-agnostic: keyed only off the mask the pipeline itself returns, it names nothing. Compacts
    each sequence to its valid tokens -- krea2 pads mid-template (`[prefix | prompt | PAD |
    suffix]`, see pipeline_krea2.py:243-247), so this GATHERS rather than truncates -- re-pads to
    the longest in the batch, and returns mask=None once nothing is masked, which is the comfy/qwen
    idiom. A pipeline that already emits unpadded conditioning is untouched: its mask is all-ones,
    the gather is an identity, and the mask was already dropped. No-op without encode_prompt, or if
    already installed."""
    real = getattr(pipe, "encode_prompt", None)

    if real is None or getattr(pipe, "_unpadded_encode_installed", False):
        return

    def encode_prompt(*args, **kwargs):
        out = real(*args, **kwargs)

        if not (isinstance(out, tuple) and len(out) == 2):
            return out
        prompt_embeds, prompt_embeds_mask = out
        if not (isinstance(prompt_embeds, torch.Tensor)
                and isinstance(prompt_embeds_mask, torch.Tensor)):
            return out
        # Only touch a real key-padding mask: (batch, seq) of bool/int over the embeds' leading
        # dims. A pipeline whose encode_prompt returns a second EMBEDS tensor instead (z-image's
        # `return prompt_embeds, negative_prompt_embeds`) must fall straight through.
        if (prompt_embeds_mask.dtype.is_floating_point or prompt_embeds_mask.ndim != 2
                or prompt_embeds_mask.shape != prompt_embeds.shape[:2]):
            return out
        if bool(prompt_embeds_mask.all()):
            return prompt_embeds, None

        kept = [e[m.bool()] for e, m in zip(prompt_embeds, prompt_embeds_mask)]
        seq_len = max(k.shape[0] for k in kept)
        if all(k.shape[0] == seq_len for k in kept):
            return torch.stack(kept), None

        # Ragged batch: re-pad to the longest and keep a mask for what is still padding.
        padded = prompt_embeds.new_zeros((len(kept), seq_len) + tuple(prompt_embeds.shape[2:]))
        padded_mask = prompt_embeds_mask.new_zeros((len(kept), seq_len))
        for i, k in enumerate(kept):
            padded[i, :k.shape[0]] = k
            padded_mask[i, :k.shape[0]] = True
        return padded, padded_mask

    pipe.encode_prompt = encode_prompt
    pipe._unpadded_encode_installed = True


def install_encode_cache(pipe):
    """Cache the text encoder's output by prompt -- a copy of ComfyUI's default node cache
    (comfy_execution/caching.py::RAMPressureCache): keep every result keyed by the encode call's
    full inputs, HOLD IT ON CPU, and evict under host-RAM pressure. Reuses ComfyUI's own memory
    reading: mm.get_free_memory(cpu) below min(10GB, max(2GB, 10% of RAM)) triggers eviction, worst
    entry first by 1.3**generation_age * cached_bytes (LRU tiebreak) --
    RAMPressureCache.ram_release verbatim minus the node-graph bookkeeping. ComfyUI holds
    CONDITIONING on the CPU for exactly this; a diffusers pipeline encodes on the compute device,
    so we stash a CPU copy and move it back on a hit (which also frees the VRAM the embeddings
    would otherwise pin, and lets them count as RAM for eviction).

    diffusers encodes inside __call__ via self.encode_prompt; wrapping that one method means a
    repeated prompt skips the encoder's forward entirely -- and since load_models_gpu is idempotent
    the un-run encoder is never streamed in, leaving the diffusion model resident (why ComfyUI's
    warm runs spend ~0s on a cached encode). The key is the call's FULL (args, kwargs), like
    ComfyUI keying a node on its whole input signature; an image, array or tensor argument (an
    edit's conditioning images) is keyed by its content, as ComfyUI's LoadImage keys its output by
    the file's hash, so a repeated edit with a new seed reuses its encode as ComfyUI's does. Any
    other unhashable argument makes the call uncacheable, so it never gets a false hit.

    The VAE encode of an edit's reference images goes through the same cache (ComfyUI's edit
    node caches its reference latents along with the prompt): vae.encode takes no generator, so
    its output depends on the image alone. One store for both, as ComfyUI keeps one for all node
    outputs. No-op if the pipe has no encode_prompt or the cache is already installed."""
    real = getattr(pipe, "encode_prompt", None)

    if real is None or getattr(pipe, "_encode_cache_installed", False):
        return

    import copy
    import time
    import bisect
    import hashlib

    import numpy as np
    from PIL import Image

    OLD_OOM_MULT = 1.3   # comfy RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER
    BASE_USAGE = 0.05    # comfy RAM_CACHE_DEFAULT_RAM_USAGE (keeps zero-size entries LRU-ordered)
    cpu = torch.device("cpu")
    target = min(10 * 1024 ** 3, max(2 * 1024 ** 3, mm.get_total_memory(cpu) * 0.10))

    uncacheable = object()

    def norm(v):
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, (torch.device, torch.dtype)):
            return str(v)
        if isinstance(v, dict):  # MiniMax H3's reference items
            v = tuple(sorted(v.items()))
        if isinstance(v, (list, tuple)):
            parts = tuple(norm(x) for x in v)
            return uncacheable if any(p is uncacheable for p in parts) else parts
        if isinstance(v, Image.Image):
            return ("image", v.mode, v.size, hashlib.sha256(v.tobytes()).hexdigest())
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            return ("tensor", str(t.dtype), tuple(t.shape),
                    hashlib.sha256(t.view(-1).view(torch.uint8).numpy()).hexdigest())
        return uncacheable

    def key_of(args, kwargs):
        a = tuple(norm(v) for v in args)
        kw = tuple((k, norm(kwargs[k])) for k in sorted(kwargs))
        if any(x is uncacheable for x in a) or any(v is uncacheable for _, v in kw):
            return None
        return (a, kw)

    def move(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(move(o, device) for o in obj)
        if hasattr(obj, "__dict__"):           # an output object, e.g. a VAE's latent distribution
            new = copy.copy(obj)
            for k, v in vars(obj).items():
                setattr(new, k, move(v, device))
            return new
        return obj

    def nbytes(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(nbytes(o) for o in obj)
        if hasattr(obj, "__dict__"):
            return sum(nbytes(v) for v in vars(obj).values())
        return 0

    cache = {}   # key -> [cpu_value, bytes, timestamp, generation]
    gen = [0]

    def ram_release():
        if mm.get_free_memory(cpu) >= target:
            return
        scored = []
        for k, (_, sz, ts, g) in cache.items():
            if g == gen[0]:                    # comfy: never evict this generation's own entries
                continue
            bisect.insort(scored, ((OLD_OOM_MULT ** (gen[0] - g)) * (sz + BASE_USAGE), ts, k))
        while scored and mm.get_free_memory(cpu) < target:
            cache.pop(scored.pop()[2], None)   # highest oom_score first

    def cached(name, call, device_of):
        def wrapped(*args, **kwargs):
            key = key_of(args, kwargs)

            if key is None:                    # an unhashable arg -> not safely cacheable
                return call(*args, **kwargs)

            key = (name, key)
            gen[0] += 1
            hit = cache.get(key)

            if hit is not None:
                cpu_value, sz, _, _ = hit
                cache[key] = [cpu_value, sz, time.time(), gen[0]]
                return move(cpu_value, device_of(args, kwargs))

            out = call(*args, **kwargs)
            cache[key] = [move(out, cpu), nbytes(out), time.time(), gen[0]]
            ram_release()

            return out
        return wrapped

    pipe.encode_prompt = cached(
        "encode_prompt", real,
        lambda args, kwargs: getattr(pipe, "_execution_device", None) or pipe.device)

    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "encode"):
        vae.encode = cached(
            "vae.encode", vae.encode,
            lambda args, kwargs: (args[0] if args else next(iter(kwargs.values()))).device)

    pipe._encode_cache_installed = True


# -------------------------------------------------------------------------------------------------
# Device state: one knob switches ComfyUI's whole device stack. get_torch_device() / offload-device
# helpers read cpu_state, so setting it from the seam's device= arg routes load/offload to CPU, MPS
# or the GPU without touching any other code.
# -------------------------------------------------------------------------------------------------
def set_device(device):
    """Force ComfyUI's cpu_state from a turboCLI renderer string ('cpu'/'mps'/'cuda[:N]'). Returns
    the resolved torch load device."""
    dev = str(device).split(":")[0].lower()
    if dev == "cpu":
        mm.cpu_state = mm.CPUState.CPU
    elif dev == "mps":
        mm.cpu_state = mm.CPUState.MPS
    else:
        mm.cpu_state = mm.CPUState.GPU
    return mm.get_torch_device()
