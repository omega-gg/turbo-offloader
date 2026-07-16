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

import os

import torch

# ComfyUI's model_management runs get_torch_device() at IMPORT (module level); with cpu_state's
# default of GPU that calls torch.cuda.current_device(), which asserts "Torch not compiled with
# CUDA enabled" on a CPU-only build. ComfyUI avoids this because a CPU user passes `--cpu` (->
# args.cpu -> cpu_state=CPU, model_management.py:156). We parse no argv, so set args.cpu ourselves
# when there is no GPU at all, BEFORE model_management imports -- then it initialises cpu_state=CPU
# and never touches torch.cuda.
_mps = getattr(torch.backends, "mps", None)
if not torch.cuda.is_available() and not (_mps is not None and _mps.is_available()):
    from comfy.cli_args import args as _cli_args
    _cli_args.cpu = True

import comfy.model_management as mm
import comfy.model_patcher as model_patcher
import comfy.ops as ops


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


def keep_declared_fp32(model):
    """Honour a model's own `_keep_in_fp32_modules` after a meta-build + assign load.

    diffusers keeps those submodules in fp32 (modeling_utils.py:493) -- but only through
    from_pretrained. Our meta-build (`from_config(...).to(dtype)`) and
    `load_state_dict(assign=True)` both bypass it, so they land in the checkpoint dtype instead.
    That silently costs the fused
    kernel wherever the module upcasts its input: diffusers' Krea2RMSNorm runs
    `F.rms_norm(hidden_states.float(), ..., weight=self.weight + 1.0)`, so a bf16 weight makes
    torch refuse to fuse ("Mismatch dtype between input and weight ... Cannot dispatch to fused
    implementation") -- measured 1.653 vs 0.679 ms/call on the A1000. ComfyUI casts the scale to
    fp32 for exactly this reason (comfy/ldm/krea2/model.py:33), so this is also what restores
    numeric parity with it (a bf16 `+ 1.0` rounds a zero-centered scale).

    Model-agnostic: reads the model's own declaration and matches names diffusers' own way
    (`any(m in name for m in _keep_in_fp32_modules)`, modeling_utils.py:178). No-op when the model
    declares none. Returns the number of parameters promoted."""
    names = getattr(model, "_keep_in_fp32_modules", None)

    if not names:
        return 0

    count = 0
    for name, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != torch.float32 \
                and any(m in name for m in names):
            param.data = param.data.float()
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
    # comfy/ldm/lumina/model.py:68-71): without it fp16 overflow -> inf -> NaN -> black image. The
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
    memm.aimdo_enabled = True
    return True


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


def cpu_fits_full_load(model):
    """True when ComfyUI's plain CPU full-load (fp32, materialised) would fit host RAM -- so the
    caller can use that faster path instead of streaming. The fp32 model is ~2x the on-disk bf16
    size; require it under 85% of total RAM (headroom for activations + the OS). Compared against
    ComfyUI's own get_total_memory(cpu). This is ComfyUI's full-load-when-it-fits call, which
    ComfyUI applies on GPU (via free VRAM) but not on CPU -- here we apply it to CPU. Model-agnostic:
    sums the on-disk bytes of each big component (single-file or index+shards)."""
    total = 0
    for comp in ("transformer", "text_encoder"):
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
    for shard in _shards(model_dir):
        sd.update(cu.load_torch_file(shard, device=torch.device("cpu")))
    return _assign_sd(model, sd)


def load_streamed(model_cls, model_dir, dtype, operations=None):
    """Build a diffusers model whose weights stream from mmap file-slices: meta-load the module (no
    weight RAM), comfy-ize it (into `operations`; see comfy_ize), then assign the file-sliced
    weights (ComfyUI's load_torch_file -- VBAR disk->VRAM fault on CUDA, native mmap on CPU/MPS).
    Returns (model, missing)."""
    from accelerate import init_empty_weights

    cfg = model_cls.load_config(model_dir)
    with init_empty_weights():
        model = model_cls.from_config(cfg).to(dtype)

    comfy_ize(model, operations)
    missing = assign_streamed_weights(model, model_dir)
    return model, missing


def stream_single_file(build_meta, weight_file, operations=None, convert=None):
    """load_streamed's single-file sibling (ComfyUI-reuse engines): meta-load the module via
    build_meta() (accelerate init_empty_weights, so no weight RAM), comfy-ize it, mmap the ONE
    ComfyUI safetensors via load_torch_file, optionally run `convert` on the state dict -- the
    diffusers single-file key remap (renames + a fused-qkv chunk that returns views, so mmap file-
    slices survive) -- then rebind by name. Returns (model, missing)."""
    import comfy.utils as cu

    model = build_meta()

    comfy_ize(model, operations)

    sd = cu.load_torch_file(weight_file, device=torch.device("cpu"))

    if convert is not None:
        sd = convert(sd)

    missing = _assign_sd(model, sd)
    # The slices carry the file's dtype, so honour _keep_in_fp32_modules afterwards.
    keep_declared_fp32(model)

    return model, missing


def mixed_precision_operations(compute_dtype, load_device, quant_config):
    """ComfyUI's mixed-precision op namespace (comfy.ops.mixed_precision_ops) for a scaled-fp8 (or
    other quantized) checkpoint -- the branch ops.pick_operations takes only when a model_config
    carries quant_config, which adapter.pick_operations never reaches. Formats the compute device
    can't run natively are `disabled` -> emulated dequant, exactly as ComfyUI: on Ampere (no fp8
    tensor cores) float8 runs emulated (dequant to compute_dtype per forward). Reused by
    load_quant_single_file."""
    disabled = set()

    if not mm.supports_fp8_compute(load_device):
        disabled |= {"float8_e4m3fn", "float8_e5m2"}
    if not mm.supports_nvfp4_compute(load_device):
        disabled.add("nvfp4")
    if not mm.supports_mxfp8_compute(load_device):
        disabled.add("mxfp8")

    return ops.mixed_precision_ops(quant_config, compute_dtype, disabled=disabled)


def _quant_ize(model, operations, compute_dtype):
    """Re-class every Linear of `model` to the mixed-precision op namespace and inject the attrs
    its __init__ would have set -- factory_kwargs / _orig_shape / tensor_class / _full_precision_mm
    (read by comfy.ops._load_quantized_module) plus the per-instance weight/bias_function lists.
    Unlike comfy_ize (which re-classes over live weights for cast-only ops), a mixed-precision
    Linear then LOADS its weight via _load_from_state_dict -> _load_quantized_module (a
    comfy_kitchen QuantizedTensor); a non-quant Linear (e.g. lm_head) falls through its plain
    branch. Returns the count re-classed."""
    count = 0

    for _name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and not isinstance(m, operations.Linear):
            out_features, in_features = m.weight.shape

            m.__class__ = operations.Linear
            m.factory_kwargs = {"device": None, "dtype": compute_dtype}
            m._orig_shape = (out_features, in_features)
            m.tensor_class = None
            m._full_precision_mm = operations._full_precision_mm
            m._full_precision_mm_config = False
            m.weight_function = []
            m.bias_function = []
            count += 1

    return count


def load_quant_single_file(build_meta, weight_file, load_device, compute_dtype, convert=None):
    """stream_single_file's fp8 sibling for a ComfyUI scaled-fp8 single file (transformer OR text
    encoder), kept fp8 (mmap, dequantized per forward -- "what comfy does"). Meta-build via
    build_meta() (no weight RAM), mmap the file (load_torch_file, with metadata), run comfy's
    convert_old_quants -- which injects the .comfy_quant markers from either the classic
    scaled_fp8 + scale_weight keys OR the file's `_quantization_metadata` -- detect the quant,
    re-class Linears to the mixed-precision namespace (+ comfy_ize the rest), optionally `convert`
    the state-dict keys (e.g. a flat->nested rename), then load_state_dict(assign=True) so each
    quant Linear builds its QuantizedTensor. Returns (model, unexpected_keys)."""
    import comfy.utils as cu

    model = build_meta()

    sd, metadata = cu.load_torch_file(weight_file, device=torch.device("cpu"),
                                      return_metadata=True)
    sd, metadata = cu.convert_old_quants(sd, model_prefix="", metadata=metadata)
    quant_config = cu.detect_layer_quantization(sd, "")

    if convert is not None:
        sd = convert(sd)

    operations = mixed_precision_operations(compute_dtype, load_device, quant_config)
    _quant_ize(model, operations, compute_dtype)
    # _quant_ize only re-classes Linears (they need the quant _load_from_state_dict); comfy_ize the
    # REST of the leaves (Embedding, Conv, norms) into the same namespace so they stream too, else
    # keep_uncastable_resident pins them (e.g. the ~1GB input Embedding), starving the transformer
    # VRAM stream budget. comfy_ize skips the already-quant Linears (CastWeightBiasOp; see
    # _comfy_class_for), so it converts only the non-Linear leaves.
    comfy_ize(model, operations)

    _missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
    # assign=True installs the checkpoint's own dtype, so honour _keep_in_fp32_modules afterwards.
    keep_declared_fp32(model)

    return model, unexpected


_sdpa_patched = False


def use_comfy_attention(model=None):
    """Route diffusers' attention through a copy of ComfyUI's own
    `comfy.ops.scaled_dot_product_attention` (comfy/ops.py:39-64), so our SDPA behaves EXACTLY like
    ComfyUI's -- not an approximation.

    ComfyUI, on Windows+CUDA with a recent torch, forces the SDPA backend priority
    [CUDNN, FLASH, EFFICIENT, MATH] -- but ONLY per call and ONLY for large inputs
    (`q.nelement() >= 1024*128`); small attentions fall through to torch's default backend (cuDNN
    is slower there). We reproduce that verbatim. diffusers calls
    `torch.nn.functional.scaled_dot_product_attention` by attribute
    (diffusers/models/attention_dispatch.py), so we patch that one name, once and process-global
    (idempotent), delegating to the saved original -- no self-recursion. `model` is accepted only
    for signature parity with the other patchers; the patch is global, as ComfyUI's is.
    No-op off Windows/CUDA or on a torch without set_priority (exactly ComfyUI's own guard)."""
    global _sdpa_patched
    if _sdpa_patched:
        return True

    import torch.nn.functional as F
    import comfy.model_management as mm

    try:
        if not (torch.cuda.is_available() and getattr(mm, "WINDOWS", False)):
            return False
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import inspect
        if "set_priority" not in inspect.signature(sdpa_kernel).parameters:
            return False
    except Exception:
        return False

    # comfy/ops.py builds [FLASH, EFFICIENT, MATH] then inserts CUDNN at the front (ops.py:48-54).
    priority = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    orig = F.scaled_dot_product_attention

    def scaled_dot_product_attention(*args, **kwargs):
        # comfy/ops.py:56-60 verbatim (the branch + sdpa_kernel), only the arg handling differs:
        # ComfyUI's models call comfy.ops.scaled_dot_product_attention(q, k, v, ...) POSITIONALLY,
        # but diffusers calls torch's F.sdpa by KEYWORD (query=/key=/value=,
        # attention_dispatch.py), so we take the query tensor from args[0] or kwargs["query"] and
        # forward the call untouched.
        q = args[0] if args else kwargs.get("query")
        # manual_cast (fp16 compute on a bf16 checkpoint) can leave the attention inputs
        # mismatched: transformers' qwen3 RoPE promotes q,k to fp32 (fp16 * fp32 cos/sin) while v
        # stays fp16, and torch's SDPA requires one dtype. Coerce to the lowest-precision float
        # present -- the compute dtype -- exactly what a uniform native fp16 model would run. Only
        # fires on an actual mismatch.
        k = args[1] if len(args) > 1 else kwargs.get("key")
        v = args[2] if len(args) > 2 else kwargs.get("value")
        if (q is not None and k is not None and v is not None
                and not (q.dtype == k.dtype == v.dtype)):
            floats = [t.dtype for t in (q, k, v) if t.dtype.is_floating_point]
            if floats:
                tgt = min(floats, key=lambda d: torch.finfo(d).bits)
                args = list(args)
                for i, name in ((0, "query"), (1, "key"), (2, "value")):
                    if i < len(args) and args[i] is not None:
                        args[i] = args[i].to(tgt)
                    elif name in kwargs and kwargs[name] is not None:
                        kwargs[name] = kwargs[name].to(tgt)
                q = args[0] if args else kwargs.get("query")
        if q is None or q.nelement() < 1024 * 128:  # comfy: small inputs -> default backend
            return orig(*args, **kwargs)
        with sdpa_kernel(priority, set_priority=True):
            return orig(*args, **kwargs)

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
    module whose calls never match (or a model we patch but never run) imports nothing."""
    ck_state = {"ck": None, "resolved": False}

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
        # Same broadcast diffusers applies (embeddings.py), so freqs_cis lands on the right axis.
        if sequence_dim == 2:
            cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        else:
            cos, sin = cos[None, :, None, :], sin[None, :, None, :]

        fc = _build_freqs_cis(cos.to(x.device), sin.to(x.device))
        return ck.apply_rope1(x, fc)

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
    (av_model.py:913). A diffusers forward doesn't, so every layer stalls waiting for its weights
    (~30% of a VBAR step).

    We reproduce it with forward hooks: for each ModuleList of repeated blocks, build a prefetch
    queue at the transformer's forward-start and pop it before each block runs; tear the queues
    down at the transformer's forward-end. No-op unless comfy-aimdo/VBAR is active
    (make_prefetch_queue
    returns None off the dynamic path). Returns the number of block sequences instrumented."""
    import torch.nn as nn
    import comfy.model_prefetch as mp

    # Block sequences = top-level ModuleLists of >=2 identical-typed modules (layers + refiners).
    sequences = []
    for _name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) >= 2:
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


_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight",
                  ".alpha", ".dora_scale", ".diff", ".diff_b")


def add_lora(patcher, lora_specs):
    """Apply LoRAs on-cast, ComfyUI-style: build the {lora_base_key -> model_weight_key} map, hand
    it to comfy.lora.load_lora (which uses the vendored weight_adapter classes to read lora_up/down
    or lora_A/B + alpha), and register the result via ModelPatcher.add_patches. The patches become
    LowVramPatch weight_functions that comfy.lora.calculate_weight applies while each weight
    streams
    in -- no fusing, no extra resident copy. lora_specs is [(path, strength), ...]."""
    import comfy.utils as cu
    import comfy.lora as clora

    model_keys = set(n for n, _ in patcher.model.named_parameters())
    applied = 0
    for path, strength in lora_specs:
        sd = cu.load_torch_file(path, safe_load=True)
        bases = set()
        for k in sd:
            for suf in _LORA_SUFFIXES:
                if k.endswith(suf):
                    bases.add(k[:-len(suf)])
                    break
        to_load = {}
        for base in bases:
            mk = base
            for pre in ("transformer.", "diffusion_model.", "lora_unet_"):
                if mk.startswith(pre):
                    mk = mk[len(pre):]
                    break
            mk = mk + ".weight"
            if mk in model_keys:
                to_load[base] = mk
        patch_dict = clora.load_lora(sd, to_load, log_missing=False)
        patcher.add_patches(patch_dict, strength)
        applied += len(patch_dict)
    return applied


def build_dynamic_patcher(model, load_device=None, offload_device=None, size=0):
    """Wrap a streamed model in ModelPatcherDynamic (the VBAR-aware patcher). On a CPU load_device
    it transparently reroutes to a plain ModelPatcher (VBAR is GPU-only)."""
    import comfy.model_patcher as model_patcher
    if load_device is None:
        load_device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()
    make_patchable(model)
    return model_patcher.ModelPatcherDynamic(model, load_device=load_device,
                                             offload_device=offload_device, size=size)


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


def install_tiled_vae_fallback(pipe, node_boundary=None):
    """ComfyUI's sampler->VAE node boundary + its VAE out-of-memory fallback.

    `node_boundary` (the seam's node_teardown) runs first: in ComfyUI the KSampler node ENDS before
    VAEDecode begins, so the teardown has already reset the VBAR watermarks and the DiT's residency
    is evictable when the VAE needs VRAM. A diffusers pipeline calls vae.decode inside the same
    __call__, so without this boundary the DiT still pins its VRAM (measured: 2.66GB held, 0 free)
    and a 1600x1200 decode dies on one ~1.4GB fp32 upsample after all 8 steps succeeded.

    Then comfy/sd.py:1057-1058: estimate the call's working memory (`memory_used_decode`) and hand
    it to load_models_gpu as memory_required BEFORE the first kernel. For an already-resident VAE
    load_models_gpu's guts are free_memory(max(inference_memory, memory_required +
    extra_reserved_memory())) (model_management.py:854,911) -- the CONTROLLED eviction of the
    streamed models. Skipping this and letting the first big cudnn workspace request evict on
    demand inside the allocator dies with an uncatchable C++ abort instead of a catchable OOM
    (measured: the abort fires on the VAE's first conv3d).

    Then the fallback itself (comfy/sd.py:1080-1087 decode, :1165-1168 encode): try the regular
    call; on OOM, `raise_non_oom` anything else, warn with comfy's own wording, and only set a
    flag -- comfy deliberately retries OUTSIDE the except block because "the exception itself refs
    them all until we get out of this except block", so the tensors can gc first -- then retry
    tiled. Comfy retries with its own tiler; a diffusers VAE ships the equivalent (`enable_tiling`,
    seam-blended; the qwen VAE's 256px tile / 64px overlap equal comfy's decode_tiled_3d defaults,
    sd.py:1097-1098), and the regular path is restored after, matching comfy's per-call semantics.
    Model-agnostic: keyed only off the vae exposing enable_tiling. No-op without a vae, or if
    already installed."""
    vae = getattr(pipe, "vae", None)

    if vae is None or not hasattr(vae, "enable_tiling") \
            or getattr(vae, "_tiled_fallback_installed", False):
        return

    def wrap(real, kind):
        def call(*args, **kwargs):
            if node_boundary is not None:
                node_boundary()
            samples_in = args[0] if args else next(iter(kwargs.values()), None)
            if hasattr(samples_in, "shape") and samples_in.device.type != "cpu":
                memory_used = _vae_memory_used(
                    kind, samples_in.shape, getattr(vae, "dtype", samples_in.dtype))
                extra_mem = max(mm.minimum_inference_memory(),
                                memory_used + mm.extra_reserved_memory())
                mm.free_memory(extra_mem, samples_in.device)
            do_tile = False
            try:
                return real(*args, **kwargs)
            except Exception as e:
                mm.raise_non_oom(e)
                print("Warning: Ran out of memory when regular VAE %s, retrying with tiled VAE "
                      "%s." % (kind, kind), flush=True)
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
    (comfy/text_encoders/qwen3vl.py:129), and its krea2 text encoder drops an all-ones mask
    outright (comfy/text_encoders/krea2.py:69-70), so every attention call gets mask=None.
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
    ComfyUI keying a node on its whole input signature: an argument that isn't cleanly hashable --
    a tensor/image, e.g. an image-conditioned edit encode -- makes the call uncacheable, so it
    never gets a false hit. No-op if
    the pipe has no encode_prompt or the cache is already installed."""
    real = getattr(pipe, "encode_prompt", None)

    if real is None or getattr(pipe, "_encode_cache_installed", False):
        return

    import time
    import bisect

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
        if isinstance(v, (list, tuple)):
            parts = tuple(norm(x) for x in v)
            return uncacheable if any(p is uncacheable for p in parts) else parts
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
        return obj

    def nbytes(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(nbytes(o) for o in obj)
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

    def cached_encode(*args, **kwargs):
        key = key_of(args, kwargs)

        if key is None:                        # tensor/image arg -> not safely cacheable
            return real(*args, **kwargs)

        gen[0] += 1
        hit = cache.get(key)

        if hit is not None:
            cpu_value, sz, _, _ = hit
            cache[key] = [cpu_value, sz, time.time(), gen[0]]
            device = getattr(pipe, "_execution_device", None) or pipe.device
            return move(cpu_value, device)

        out = real(*args, **kwargs)
        cache[key] = [move(out, cpu), nbytes(out), time.time(), gen[0]]
        ram_release()

        return out

    pipe.encode_prompt = cached_encode
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
