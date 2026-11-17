TEMPLATE = subdirs

OTHER_FILES += README.md  \
               LICENSE.md \
               implementation.md \
               dummy.md \
               resync.md \
               doc/aimdo_v1.md      \
               doc/AIMDO_V2_PLAN.md \
               doc/AIMDO_KITCHEN_PLAN.md \
               doc/BENCHMARKING.md  \
               doc/COMFYUI_OFFLOAD_MAP.md \
               doc/DUMMY_PLAN.md \
               doc/COMFYUI_039_RESYNC_PLAN.md \

# Backend seam + the thin diffusers <-> ComfyUI bridge (the only real logic).
OTHER_FILES += offloader/__init__.py \
               offloader/adapter.py \
               offloader/qwen_image.py \
               offloader/qwen_image21.py \
               offloader/flux2.py \
               offloader/flux3.py \
               offloader/krea2.py \
               offloader/lumina2.py \
               offloader/moss_image.py \

# Vendored ComfyUI offloading snapshot (byte-for-byte; see resync.md).
OTHER_FILES += offloader/comfy/__init__.py          \
               offloader/comfy/anonymous_file.py    \
               offloader/comfy/cli_args.py          \
               offloader/comfy/diffusers_convert.py \
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
               offloader/comfy/system_memory.py     \
               offloader/comfy/internal_logging.py  \
               offloader/comfy/rmsnorm.py           \
               offloader/comfy/storage.py           \

# Vendored ComfyUI text encoders, run as ComfyUI's own.
OTHER_FILES += offloader/comfy/clip_model.py                                        \
               offloader/comfy/sd1_clip.py                                          \
               offloader/comfy/sdxl_clip.py                                         \
               offloader/comfy/text_encoders/bpe_tokenizer.py                       \
               offloader/comfy/text_encoders/flux.py                                \
               offloader/comfy/text_encoders/krea2.py                               \
               offloader/comfy/text_encoders/llama.py                               \
               offloader/comfy/text_encoders/moss_image.py                          \
               offloader/comfy/text_encoders/qwen25_tokenizer/merges.txt            \
               offloader/comfy/text_encoders/qwen25_tokenizer/tokenizer_config.json \
               offloader/comfy/text_encoders/qwen25_tokenizer/vocab.json            \
               offloader/comfy/text_encoders/qwen35.py                              \
               offloader/comfy/text_encoders/qwen35_tokenizer/merges.txt            \
               offloader/comfy/text_encoders/qwen35_tokenizer/tokenizer_config.json \
               offloader/comfy/text_encoders/qwen35_tokenizer/vocab.json            \
               offloader/comfy/text_encoders/qwen3vl.py                             \
               offloader/comfy/text_encoders/qwen_image.py                          \
               offloader/comfy/text_encoders/qwen_image21.py                        \
               offloader/comfy/text_encoders/qwen_vl.py                             \
               offloader/comfy/text_encoders/sd3_clip.py                            \
               offloader/comfy/text_encoders/t5.py                                  \
               offloader/comfy/text_encoders/z_image.py                             \

# Vendored ComfyUI model code: the VAEs and transformers run as ComfyUI's own.
OTHER_FILES += offloader/comfy/ldm/chroma_radiance/layers.py            \
               offloader/comfy/ldm/common_dit.py                        \
               offloader/comfy/ldm/flux/layers.py                       \
               offloader/comfy/ldm/flux/math.py                         \
               offloader/comfy/ldm/flux/model.py                        \
               offloader/comfy/ldm/flux3/model.py                       \
               offloader/comfy/ldm/krea2/model.py                       \
               offloader/comfy/ldm/lightricks/model.py                  \
               offloader/comfy/ldm/lightricks/symmetric_patchifier.py   \
               offloader/comfy/ldm/lumina/model.py                      \
               offloader/comfy/ldm/models/autoencoder.py                \
               offloader/comfy/ldm/modules/attention.py                 \
               offloader/comfy/ldm/modules/distributions/__init__.py    \
               offloader/comfy/ldm/modules/distributions/distributions.py \
               offloader/comfy/ldm/modules/ema.py                       \
               offloader/comfy/ldm/modules/diffusionmodules/__init__.py \
               offloader/comfy/ldm/modules/diffusionmodules/mmdit.py    \
               offloader/comfy/ldm/modules/diffusionmodules/model.py    \
               offloader/comfy/ldm/modules/diffusionmodules/util.py     \
               offloader/comfy/ldm/modules/sub_quadratic_attention.py   \
               offloader/comfy/ldm/moss_image/model.py                  \
               offloader/comfy/ldm/qwen_image/model.py                  \
               offloader/comfy/ldm/qwen_image21/model.py                \
               offloader/comfy/ldm/util.py                              \
               offloader/comfy/ldm/wan/model.py                         \
               offloader/comfy/ldm/wan/model_animate2.py                \
               offloader/comfy/ldm/wan/vae.py                           \
               offloader/comfy/ldm/wan/vae2_2.py                        \

# Vendored comfy sub-packages.
OTHER_FILES += offloader/comfy/comfy_types/README.md      \
               offloader/comfy/comfy_types/__init__.py    \
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
