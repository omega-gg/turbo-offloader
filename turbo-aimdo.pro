TEMPLATE = subdirs

OTHER_FILES += README.md  \
               LICENSE.md \
               aimdo.md   \
               doc/aimdo_v1.md      \
               doc/AIMDO_V2_PLAN.md \

# Backend seam + the thin diffusers <-> ComfyUI bridge (the only real logic).
OTHER_FILES += aimdo/__init__.py \
               aimdo/adapter.py  \

# Vendored ComfyUI offloading snapshot (byte-for-byte; see aimdo/comfy/resync.md).
OTHER_FILES += aimdo/comfy/__init__.py          \
               aimdo/comfy/resync.md            \
               aimdo/comfy/cli_args.py          \
               aimdo/comfy/options.py           \
               aimdo/comfy/model_management.py  \
               aimdo/comfy/model_patcher.py     \
               aimdo/comfy/memory_management.py \
               aimdo/comfy/model_prefetch.py    \
               aimdo/comfy/ops.py               \
               aimdo/comfy/quant_ops.py         \
               aimdo/comfy/float.py             \
               aimdo/comfy/lora.py              \
               aimdo/comfy/utils.py             \
               aimdo/comfy/hooks.py             \
               aimdo/comfy/patcher_extension.py \
               aimdo/comfy/pinned_memory.py     \

# Vendored comfy sub-packages.
OTHER_FILES += aimdo/comfy/comfy_types/__init__.py    \
               aimdo/comfy/comfy_types/node_typing.py \
               aimdo/comfy/weight_adapter/__init__.py \
               aimdo/comfy/weight_adapter/base.py     \
               aimdo/comfy/weight_adapter/lora.py     \
               aimdo/comfy/weight_adapter/loha.py     \
               aimdo/comfy/weight_adapter/lokr.py     \
               aimdo/comfy/weight_adapter/glora.py    \
               aimdo/comfy/weight_adapter/oft.py      \
               aimdo/comfy/weight_adapter/boft.py     \
               aimdo/comfy/weight_adapter/bypass.py   \

# Test / drive harnesses.
OTHER_FILES += tests/bridge_parity.py \
               tests/check_img.py     \
               tests/drive.py         \
               tests/drive_qwen.py    \
