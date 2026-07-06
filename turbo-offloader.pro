TEMPLATE = subdirs

OTHER_FILES += README.md  \
               LICENSE.md \
               offloader.md   \
               doc/aimdo_v1.md      \
               doc/AIMDO_V2_PLAN.md \
               doc/BENCHMARKING.md  \
               doc/COMFYUI_OFFLOAD_MAP.md \

# Backend seam + the thin diffusers <-> ComfyUI bridge (the only real logic).
OTHER_FILES += offloader/__init__.py \
               offloader/adapter.py  \

# Vendored ComfyUI offloading snapshot (byte-for-byte; see offloader/comfy/resync.md).
OTHER_FILES += offloader/comfy/__init__.py          \
               offloader/comfy/resync.md            \
               offloader/comfy/cli_args.py          \
               offloader/comfy/options.py           \
               offloader/comfy/model_management.py  \
               offloader/comfy/model_patcher.py     \
               offloader/comfy/memory_management.py \
               offloader/comfy/model_prefetch.py    \
               offloader/comfy/ops.py               \
               offloader/comfy/quant_ops.py         \
               offloader/comfy/float.py             \
               offloader/comfy/lora.py              \
               offloader/comfy/utils.py             \
               offloader/comfy/hooks.py             \
               offloader/comfy/patcher_extension.py \
               offloader/comfy/pinned_memory.py     \

# Vendored comfy sub-packages.
OTHER_FILES += offloader/comfy/comfy_types/__init__.py    \
               offloader/comfy/comfy_types/node_typing.py \
               offloader/comfy/weight_adapter/__init__.py \
               offloader/comfy/weight_adapter/base.py     \
               offloader/comfy/weight_adapter/lora.py     \
               offloader/comfy/weight_adapter/loha.py     \
               offloader/comfy/weight_adapter/lokr.py     \
               offloader/comfy/weight_adapter/glora.py    \
               offloader/comfy/weight_adapter/oft.py      \
               offloader/comfy/weight_adapter/boft.py     \
               offloader/comfy/weight_adapter/bypass.py   \

# Test / drive harnesses.
OTHER_FILES += tests/bridge_parity.py \
               tests/check_img.py     \
               tests/drive.py         \
               tests/drive_qwen.py    \
