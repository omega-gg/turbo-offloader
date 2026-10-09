# Re-syncing the vendored ComfyUI snapshot

`offloader/comfy/` is a **byte-for-byte snapshot** of ComfyUI's offloading subsystem, driven
through `offloader/adapter.py` so turboCLI's diffusers pipelines reuse ComfyUI's device-agnostic
(CPU/CUDA/MPS) partial-offload path with 1:1 parity. Keeping the files verbatim is what makes
upgrades cheap: to move to a newer ComfyUI you **re-copy the files below, then re-apply the short
edit list here**. Nothing else should differ from upstream.

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

## Files copied verbatim from `ComfyUI/comfy/`

Flat modules: `model_management.py`, `model_patcher.py`, `ops.py`, `memory_management.py`,
`utils.py`, `lora.py`, `float.py`, `quant_ops.py`, `patcher_extension.py`, `hooks.py`,
`pinned_memory.py`, `model_prefetch.py`, `cli_args.py`, `options.py`, `system_memory.py`,
`internal_logging.py`, `rmsnorm.py`, `storage.py` (the last four are imported by the others since
v0.39; none pulls anything off-path).

`model_prefetch.py` is copied verbatim, driven from `offloader/adapter.py:install_prefetch`, which
reproduces ComfyUI's per-block `prefetch_queue_pop` loop (its own models call it from their forward,
e.g. `comfy/ldm/lightricks/av_model.py`) via forward hooks on the diffusers transformer's block
ModuleLists.
Packages (whole dir): `comfy_types/`, `weight_adapter/`.

Model code, for an engine whose component runs as ComfyUI's own: `ldm/wan/vae2_2.py` (the Wan 2.2
VAE, which `comfy/sd.py` builds for the Qwen-Image 2.1 file), with what it imports,
`ldm/wan/vae.py` and `ldm/modules/diffusionmodules/model.py` (and that package's empty
`__init__.py`). An engine imports them as `comfy.ldm.*` once `comfy_api()` brought the package up.

`cli_args.py` + `options.py` are vendored as-is (not stubbed): `options.args_parsing` is `False`, so
`cli_args` does `parser.parse_args([])` and every flag gets its upstream default. This is more
faithful and lower-maintenance than a hand-written `args` stub (no field list to drift).

## The only edits over upstream — three categories

### (1) `sys.modules` alias + optional-comfy_aimdo shim — `offloader/comfy/__init__.py` (ours)
- Aliases this package as top-level `comfy` so the vendored `import comfy.X` lines resolve here
  with zero per-file edits. Always import via `comfy.*`, never `offloader.comfy.*`.
- When `comfy_aimdo` (CUDA-only VBAR accelerator) is absent (CPU/MPS builds), registers empty
  stand-in submodules in `sys.modules` so the verbatim `import comfy_aimdo.X` lines still resolve
  (`host_buffer`, `vram_buffer`, `model_vbar`, `torch`, `model_mmap`, `control`, `storage`,
  `malloc_graph`; add any new `comfy_aimdo.X` a re-sync brings in).
  All comfy_aimdo *usage* is gated on `memory_management.aimdo_enabled` (default False), so the
  stand-ins are never dereferenced off-CUDA. **No vendored file is edited for comfy_aimdo.**

### (2) optional `comfy_kitchen` — no vendored edit
Upstream `float.py` / `quant_ops.py` already wrap `import comfy_kitchen` in try/except and log a
warning when absent, so nothing is edited here. comfy-kitchen is **pip-installed** into the
runtime venv (turboCLI `bash/turbo/build.sh`, pinned `0.2.37`, Apache-2.0): a native wheel, so it
is **not vendored** into `offloader/comfy/`. Present → `quant_ops.py` configures its backends (cuda
disabled unless torch cu130+, triton unless flagged) exactly as ComfyUI does, and
`offloader/adapter.py:use_kitchen_rope` routes diffusers' `apply_rotary_emb` via `ck.apply_rope1`
(the same fused kernel ComfyUI's `comfy/ldm/flux/math.py` uses). Absent → fp8/fp4 quant paths and
the rope routing both degrade to the native path.

Two caveats since v0.39. `model_prefetch.py` imports `comfy_kitchen` unconditionally; that is
safe because nothing vendored imports `model_prefetch` (only the adapter does, lazily, on the VBAR
path) and comfy-kitchen is installed on every build. And a version older than the tag's pin is
not safe: `quant_ops.py`'s import block then fails as a whole, silently disabling fp8 and the
kitchen RoPE.

### (3) `# [turbo-offloader] disabled for turboCLI:` comment-outs — 3 one-line edits total
Each is a single commented import for a dependency that is **off the offloader's code path**, so the
offloader is unaffected; only unrelated helper functions would `NameError` if ever called (they are
not). Grep `# [turbo-offloader] disabled for turboCLI:` to find them all.

| file      | line   | commented import                              | why it's off-path |
|-----------|--------|-----------------------------------------------|-------------------|
| `utils.py`| ~34    | `from einops import rearrange`                | only used by an attention-reshape helper (~L1389), not the offloader; drops the `einops` dependency |
| `lora.py` | ~22    | `import comfy.model_base`                     | only used by `model_lora_keys_unet()` (LoRA key-mapping at load time), which the adapter bypasses; would otherwise pull the whole model zoo |
| `hooks.py`| ~17    | `from node_helpers import conditioning_set_values` | only used by the conditioning helpers at the bottom of the file (~L692-781); keeps every hook class/enum `ModelPatcher` needs verbatim |

## Re-sync procedure

1. Copy the files listed above from the ComfyUI tag over `offloader/comfy/`
   (`git show <tag>:comfy/<file>`, `git archive <tag> comfy/<pkg>` for the packages), then check
   `git diff --stat` matches upstream's own diff between the two tags.
2. Update the commits in this file, in `offloader/comfy/__init__.py` and in `README.md`, and the
   comfy-aimdo / comfy-kitchen versions in turboCLI `bash/turbo/build.sh`, along with the torch /
   torchvision / torchaudio the reference ComfyUI install runs.
3. Re-apply the 3 comment-outs in category (3) (grep the marker in the OLD tree first to relocate
   them if line numbers moved).
4. Smoke test — must print `import OK` and `aimdo_enabled= False`, both as is and with the
   comfy_aimdo stand-ins forced (what a CPU/MPS build without comfy-aimdo runs):
   ```
   python -c "import offloader.comfy; import comfy.model_patcher, comfy.ops, comfy.utils, \
   comfy.memory_management as m; print('import OK; aimdo_enabled=', m.aimdo_enabled)"
   python -c "import sys; sys.modules['comfy_aimdo'] = None; import offloader.comfy; \
   import comfy.model_patcher, comfy.ops, comfy.utils, comfy.model_prefetch, \
   comfy.memory_management as m; print('import OK; aimdo_enabled=', m.aimdo_enabled)"
   ```
5. If new upstream imports pull an off-path top-level module (like `node_helpers`), extend
   category (3) with a documented one-line comment-out rather than vendoring the extra tail.
6. Re-check what the adapter mirrors or calls by name, since those are not covered by the
   verbatim copy: `pinned_memory` internals (`install_pin_rollback_guard`), the SDPA body in
   `comfy/ops.py` (`use_comfy_attention`), the per-node teardown in `execution.py`
   (`node_teardown`), aimdo's `control.init` arguments in `main.py` (`pre_torch_init`), the
   `ModelPatcher` constructor (`fast_disk`), `PromptModelTracker` (`prepare`/`reclaim`), and the
   `load_safetensors` storage tags `_comfy_tensor_file_slice` / `_comfy_tensor_mmap_refs` that
   `_own_file_slice` rebuilds, plus
   the upstream line numbers cited in comments and `implementation.md`.
7. Verify on CUDA at 512² (seed 42), keeping runs short: flux2-4b, comfy-krea2-turbo and (CPU)
   flux2-4b give the same md5 before and after, unless upstream changed numerics on purpose;
   then compare against the ComfyUI install at the same tag, same files and template graph.
   The test laptop throttles: interleave the runs (A, B, A, B) and trust a delta only once a
   second pair reproduces it.
