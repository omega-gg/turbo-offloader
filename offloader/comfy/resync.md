# Re-syncing the vendored ComfyUI snapshot

`offloader/comfy/` is a **byte-for-byte snapshot** of ComfyUI's offloading subsystem, driven
through `offloader/adapter.py` so turboCLI's diffusers pipelines reuse ComfyUI's device-agnostic
(CPU/CUDA/MPS) partial-offload path with 1:1 parity. What the offloader takes from ComfyUI falls in
three categories, and each one has its own step in a re-sync:

1. **Copied verbatim**: re-copy the files from the new tag.
2. **Edited**: three one-line comment-outs, re-applied after the copy.
3. **Mirrored or called by name, not copied**: code in `offloader/` that reproduces a piece of
   ComfyUI or depends on one of its internals, each citing its upstream lines; diff those lines
   between the two tags and port any change.

Nothing else should differ from upstream.

## Source snapshots (bump together)

Pin tagged releases only, never a branch head: copy from the ComfyUI tag, and take comfy-aimdo /
comfy-kitchen at the versions that tag's `requirements.txt` pins.

| repo          | commit                                     | tag                    |
|---------------|--------------------------------------------|------------------------|
| ComfyUI       | `de0125d9aefc225ec1a6f787a3a806730b772678` | v0.39.1                |
| comfy-aimdo   | `3b8e8c162efeb9470d912609a7a6e7a2b1c693ec` | v0.5.5 (pip)           |
| comfy-kitchen | `be003b7c23c5b01328657955b8bc5d3f073d868e` | v0.2.37 (pip)          |

The runtime matches the reference ComfyUI install too, so turbo and ComfyUI are compared on the
same kernels: torch 2.14.1+cu130, torchvision 0.29.1, torchaudio 2.11.0 (pinned in turboCLI
`bash/turbo/build.sh`).

## 1. Copied verbatim from `ComfyUI/comfy/`

Flat modules: `model_management.py`, `model_patcher.py`, `ops.py`, `memory_management.py`,
`utils.py`, `lora.py`, `float.py`, `quant_ops.py`, `patcher_extension.py`, `hooks.py`,
`pinned_memory.py`, `model_prefetch.py`, `cli_args.py`, `options.py`, `system_memory.py`,
`internal_logging.py`, `rmsnorm.py`, `storage.py` (the last four are imported by the others since
v0.39; none pulls anything off-path).

Packages (whole dir): `comfy_types/`, `weight_adapter/`.

Model code, for the VAEs an engine runs as ComfyUI's own (`comfy_vae`, category 3):
`ldm/wan/vae.py` (Wan 2.1: Qwen-Image, Krea 2), `ldm/wan/vae2_2.py` (Wan 2.2 layout: Qwen-Image
2.1), and what they import, `ldm/modules/diffusionmodules/model.py` (with that package's empty
`__init__.py`).

`cli_args.py` + `options.py` are vendored as-is (not stubbed): `options.args_parsing` is `False`, so
`cli_args` does `parser.parse_args([])` and every flag gets its upstream default. This is more
faithful and lower-maintenance than a hand-written `args` stub (no field list to drift).

## 2. Edited: three one-line comment-outs

Each is a single commented import for a dependency that is **off the offloader's code path**, so
the offloader is unaffected; only unrelated helper functions would `NameError` if ever called (they
are not). Grep `# [turbo-offloader] disabled for turboCLI:` to find them all.

| file      | line   | commented import                              | why it's off-path |
|-----------|--------|-----------------------------------------------|-------------------|
| `utils.py`| ~34    | `from einops import rearrange`                | only used by an attention-reshape helper (~L1389), not the offloader; drops the `einops` dependency |
| `lora.py` | ~22    | `import comfy.model_base`                     | only used by `model_lora_keys_unet()` (LoRA key-mapping at load time), which the adapter bypasses; would otherwise pull the whole model zoo |
| `hooks.py`| ~17    | `from node_helpers import conditioning_set_values` | only used by the conditioning helpers at the bottom of the file (~L692-781); keeps every hook class/enum `ModelPatcher` needs verbatim |

## 3. Mirrored or called by name, not copied

Code in `offloader/` that reproduces a piece of ComfyUI, or relies on one of its internals by name.
The verbatim copy does not cover it, so a re-sync diffs the upstream lines listed here.

### ComfyUI's VAE, from `comfy/sd.py`

`sd.py` imports most of ComfyUI (every model, every text encoder), so it is not vendored: a
`comfy.sd` import would cost a cold start about 0.8 s and about 100 MB of RAM. What `comfy_vae`
needs from it is copied into `offloader/adapter.py`, each piece citing its `sd.py` lines (v0.39.1):

| upstream | in adapter.py | what |
|---|---|---|
| `VAE.__init__` branches, sd.py:837-878 | `_comfy_vae_model` | the model a file gets and its settings: Wan 2.1 (:863-878), Qwen-Image 2.1 (:840-850) |
| end of `VAE.__init__`, sd.py:1104-1125; `VAELoader.load_vae`, nodes.py:856 | `ComfyVAE._build` | the file read, device, dtype, the patcher, the weight load |
| `VAE.decode`, sd.py:1258-1308; `VAE.encode`, sd.py:1411-1460 | `ComfyVAE.decode` / `encode` | `load_models_gpu` with the estimate, the out-of-memory fallback; unlike sd.py, `encode` returns the caller's dtype, as the pipeline feeds it to its transformer |
| `decode_tiled_`, `decode_tiled_3d` (sized at sd.py:1323-1346), `encode_tiled_`, `encode_tiled_3d` | the same names on `ComfyVAE` | the tiled fallbacks |
| main.py:303-304 | `enable_vbar` | `CoreModelPatcher` becomes `ModelPatcherDynamic` |

Adding a VAE: vendor its `comfy/ldm/` model file and what it imports (category 1), add its `sd.py`
branch to `_comfy_vae_model`, then compare a decode against ComfyUI's on the same latent.

### Offloading internals

| upstream | in offloader/ | what |
|---|---|---|
| per-block `prefetch_queue_pop` loop (e.g. `comfy/ldm/lightricks/av_model.py`) | `adapter.install_prefetch` | the same loop, through forward hooks on the diffusers transformer's block ModuleLists |
| `pinned_memory` internals | `adapter.install_pin_rollback_guard` | survive a transient `HostBuffer.truncate` failure, keeping ComfyUI's own recovery |
| the SDPA body in `comfy/ops.py` (:58-101) | `adapter.use_comfy_attention` | diffusers' attention through a copy of it |
| `ComfyAttention._load_from_state_dict` and `attention_comfy_kitchen_int8` (`comfy/ldm/modules/attention.py`:82-95, :622-654) | `adapter.use_comfy_attention_config` | a file's per-module attention config: comfy-kitchen's int8 attention where it runs |
| `pre_run` before sampling (`comfy/samplers.py`:1259) | `prepare` | a model declaring `current_patcher` gets its patcher |
| each node's `load_models_gpu` (CLIPTextEncode, KSampler's `prepare_sampling`, `comfy/sampler_helpers.py`:201) after the teardown marks pins inactive (`model_management.py`:1484) | `prepare`, the encode boundary in `_finalize_pipe` | the encoder loads at encode, the sampling models before sampling |
| the per-node teardown in `execution.py` | `node_teardown` | the sampler to VAE node boundary |
| aimdo's `control.init` arguments in `main.py` | `pre_torch_init` | comfy-aimdo's allocator hooks before torch |
| the `ModelPatcher` constructor (`fast_disk`, as sd.py:2277 passes it) | `adapter.build_dynamic_patcher` | the storage policy of the streamed files |
| `PromptModelTracker` | `prepare` / `reclaim` | a generation is ComfyUI's prompt |
| `load_safetensors` storage tags `_comfy_tensor_file_slice` / `_comfy_tensor_mmap_refs` | `adapter._own_file_slice` | file slices read straight to the device |

## The offloader's own glue: `offloader/comfy/__init__.py`

Not an edit of a ComfyUI file:

- It aliases this package as top-level `comfy` so the vendored `import comfy.X` lines resolve here
  with zero per-file edits. Always import via `comfy.*`, never `offloader.comfy.*`.
- When `comfy_aimdo` (CUDA-only VBAR accelerator) is absent (CPU/MPS builds), it registers empty
  stand-in submodules in `sys.modules` so the verbatim `import comfy_aimdo.X` lines still resolve
  (`host_buffer`, `vram_buffer`, `model_vbar`, `torch`, `model_mmap`, `control`, `storage`,
  `malloc_graph`; add any new `comfy_aimdo.X` a re-sync brings in).
  All comfy_aimdo *usage* is gated on `memory_management.aimdo_enabled` (default False), so the
  stand-ins are never dereferenced off-CUDA. **No vendored file is edited for comfy_aimdo.**

## Dependency: `comfy_kitchen`, pip-installed, not vendored

Upstream `float.py` / `quant_ops.py` already wrap `import comfy_kitchen` in try/except and log a
warning when absent, so nothing is edited for it. comfy-kitchen is **pip-installed** into the
runtime venv (turboCLI `bash/turbo/build.sh`, pinned `0.2.37`, Apache-2.0): a native wheel, so it
is **not vendored** into `offloader/comfy/`. Present: `quant_ops.py` configures its backends (cuda
disabled unless torch cu130+, triton unless flagged) exactly as ComfyUI does, and
`offloader/adapter.py:use_kitchen_rope` routes diffusers' `apply_rotary_emb` via `ck.apply_rope1`
(the same fused kernel ComfyUI's `comfy/ldm/flux/math.py` uses). Absent: fp8/fp4 quant paths and
the rope routing both degrade to the native path.

Two caveats since v0.39. `model_prefetch.py` imports `comfy_kitchen` unconditionally; that is
safe because nothing vendored imports `model_prefetch` (only the adapter does, lazily, on the VBAR
path) and comfy-kitchen is installed on every build. And a version older than the tag's pin is
not safe: `quant_ops.py`'s import block then fails as a whole, silently disabling fp8 and the
kitchen RoPE.

## Re-sync procedure

1. Copy the files of category 1 from the ComfyUI tag over `offloader/comfy/`
   (`git show <tag>:comfy/<file>`, `git archive <tag> comfy/<pkg>` for the packages), then check
   `git diff --stat` matches upstream's own diff between the two tags.
2. Update the commits in this file, in `offloader/comfy/__init__.py` and in `README.md`, and the
   comfy-aimdo / comfy-kitchen versions in turboCLI `bash/turbo/build.sh`, along with the torch /
   torchvision / torchaudio the reference ComfyUI install runs.
3. Re-apply the 3 comment-outs of category 2 (grep the marker in the OLD tree first to relocate
   them if line numbers moved). If new upstream imports pull an off-path top-level module (like
   `node_helpers`), add a documented one-line comment-out there rather than vendoring the tail.
4. Smoke test: it must print `import OK` and `aimdo_enabled= False`, both as is and with the
   comfy_aimdo stand-ins forced (what a CPU/MPS build without comfy-aimdo runs):
   ```
   python -c "import offloader.comfy; import comfy.model_patcher, comfy.ops, comfy.utils, \
   comfy.memory_management as m; print('import OK; aimdo_enabled=', m.aimdo_enabled)"
   python -c "import sys; sys.modules['comfy_aimdo'] = None; import offloader.comfy; \
   import comfy.model_patcher, comfy.ops, comfy.utils, comfy.model_prefetch, \
   comfy.memory_management as m; print('import OK; aimdo_enabled=', m.aimdo_enabled)"
   ```
5. Diff the upstream code listed in category 3 between the two tags, port any change into
   `offloader/` and update the cited line numbers, here, in comments and in `implementation.md`.
6. Verify on CUDA at 512² (seed 42), keeping runs short: flux2-4b, comfy-krea2-turbo and (CPU)
   flux2-4b give the same md5 before and after, unless upstream changed numerics on purpose;
   then compare against the ComfyUI install at the same tag, same files and template graph. For
   the VAE, decode one latent with a comfy- engine and with ComfyUI and compare the images.
   The test laptop throttles: interleave the runs (A, B, A, B) and trust a delta only once a
   second pair reproduces it.
