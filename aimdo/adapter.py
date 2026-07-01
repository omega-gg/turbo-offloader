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
#   The thin middleman between turboCLI's diffusers pipelines and the vendored ComfyUI offloading
#   subsystem (aimdo/comfy/). It contains the ONLY real logic in the package; everything under
#   aimdo/comfy/ is a verbatim ComfyUI snapshot. The bridge is two ideas:
#
#     1. comfy_ize(model): re-class each vanilla torch leaf module (Linear/Conv/Norm/Embedding) to
#        its ComfyUI `disable_weight_init.*` counterpart. Those classes ARE `torch.nn.X` +
#        CastWeightBiasOp, so their forward routes through ComfyUI's device-agnostic cast path
#        (forward_comfy_cast_weights -> cast_bias_weight -> cast_to) exactly when ModelPatcher flags
#        the module for offload -- and computes identically to the original op otherwise.
#
#     2. build_patcher(model): wrap the comfy-ized module in a ComfyUI ModelPatcher. From there the
#        seam drives ComfyUI's own load_models_gpu / free_memory to stream weights CPU<->device per
#        forward, the same way ComfyUI offloads a UNet -- on CPU, CUDA or MPS.
#
#   comfy_aimdo's CUDA-only VBAR path is left dormant (aimdo_enabled False) until Phase D.
#
#==================================================================================================

import aimdo.comfy  # noqa: F401  establishes the top-level `comfy` alias (see aimdo/comfy/__init__.py)

import torch

import comfy.model_management as mm
import comfy.model_patcher as model_patcher
import comfy.ops as ops


# --------------------------------------------------------------------------------------------------
# Op mapping: torch leaf type -> ComfyUI disable_weight_init counterpart (a torch.nn.X subclass that
# also mixes in CastWeightBiasOp). Checked in order; the first isinstance() match wins. Every diffuser
# transformer/text-encoder leaf that carries an offloadable weight is one of these.
# --------------------------------------------------------------------------------------------------
_dwi = ops.disable_weight_init

_OP_MAP = [
    (torch.nn.Linear,    _dwi.Linear),
    (torch.nn.Conv1d,    _dwi.Conv1d),
    (torch.nn.Conv2d,    _dwi.Conv2d),
    (torch.nn.Conv3d,    _dwi.Conv3d),
    (torch.nn.GroupNorm, _dwi.GroupNorm),
    (torch.nn.LayerNorm, _dwi.LayerNorm),
    (torch.nn.Embedding, _dwi.Embedding),
]

# torch.nn.RMSNorm exists only on recent torch; add it when present (diffusers uses it widely).
if hasattr(torch.nn, "RMSNorm") and hasattr(_dwi, "RMSNorm"):
    _OP_MAP.append((torch.nn.RMSNorm, _dwi.RMSNorm))


def _comfy_class_for(module):
    """The disable_weight_init class to re-class `module` into, or None if it isn't an offloadable
    leaf op. Skips modules that already are CastWeightBiasOp (idempotent)."""
    if isinstance(module, ops.CastWeightBiasOp):
        return None
    for torch_cls, dwi_cls in _OP_MAP:
        if isinstance(module, torch_cls):
            return dwi_cls
    return None


def comfy_ize(model):
    """Re-class every offloadable leaf of `model` in place so its forward routes through ComfyUI's
    cast path when flagged. Injects exactly the attributes ModelPatcher.load()/cast_bias_weight read
    (comfy_cast_weights, weight_function, bias_function, and bias=None for weight-only ops). Returns
    the number of modules converted.

    Re-classing to ComfyUI's own class (rather than a synthesized mixin) keeps true 1:1 parity: the
    running forward IS disable_weight_init.<Op>.forward. The lazy-init __init__ of those classes is
    bypassed here (we mutate existing instances that already hold their weights)."""
    count = 0
    for _name, m in model.named_modules():
        dwi_cls = _comfy_class_for(m)
        if dwi_cls is None:
            continue

        # Re-class the live instance. torch.nn.X subclasses share a compatible object layout, so
        # __class__ assignment is valid; diffusers custom subclasses that add only config (no forward
        # override that matters for weight application) keep working through the base op.
        m.__class__ = dwi_cls

        # cast_bias_weight always reads s.bias; weight-only ops (Embedding/RMSNorm) have none. Mirror
        # what disable_weight_init.__init__ would have set.
        if not hasattr(m, "bias"):
            m.bias = None

        # Instance-level (never the shared CastWeightBiasOp class lists) so per-module offload flags
        # don't alias. ModelPatcher.load() overwrites these when it decides to offload the module.
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
        # with a new class string silently loses output_hidden_states. Copying __module__/__qualname__
        # makes str(subclass) == str(cls), so the lookup (and hidden-state capture) keeps working.
        ns = {"device": None, "__module__": cls.__module__, "__qualname__": cls.__qualname__}
        model.__class__ = type(cls.__name__, (cls,), ns)
    return model


def keep_uncastable_resident(model, device):
    """Move every parameter/buffer that ComfyUI's offloader will NOT stream onto `device`, so it
    stays resident. The offloader only streams the weight/bias of CastWeightBiasOp leaves (the ones
    comfy_ize converted); anything else -- a custom leaf norm (transformers' Qwen3RMSNorm, forward
    `self.weight * hidden_states`), or a bare parameter/buffer hung directly on a container module
    (diffusers' `x_pad_token`, rotary caches, class tokens) -- would otherwise be left on the
    offload device by ModelPatcher.load() and blow up mid-forward with a cuda-vs-cpu mismatch.

    We walk ALL modules and move only their DIRECT (recurse=False) params/buffers, skipping
    CastWeightBiasOp modules entirely so their big weights keep streaming. These stragglers are
    almost always tiny; returns (count, bytes) for logging so a large one is visible."""
    moved = 0
    nbytes = 0
    for _name, m in model.named_modules():
        if isinstance(m, ops.CastWeightBiasOp):
            continue  # its weight/bias stream via the cast path -- leave on the offload device
        for _n, p in m.named_parameters(recurse=False):
            if p is not None and p.device != device:
                p.data = p.data.to(device)
                moved += 1
                nbytes += p.numel() * p.element_size()
        for bn, b in m.named_buffers(recurse=False):
            if b is not None and b.device != device:
                m._buffers[bn] = b.to(device)
                moved += 1
                nbytes += b.numel() * b.element_size()
    return moved, nbytes


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


# --------------------------------------------------------------------------------------------------
# Device state: one knob switches ComfyUI's whole device stack. get_torch_device() / offload-device
# helpers read cpu_state, so setting it from the seam's device= arg routes load/offload to CPU, MPS
# or the GPU without touching any other code.
# --------------------------------------------------------------------------------------------------
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
