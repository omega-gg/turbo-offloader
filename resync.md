# Re-syncing the vendored ComfyUI snapshot

`offloader/comfy/` is a **byte-for-byte snapshot** of ComfyUI's offloading subsystem, driven
through `offloader/adapter.py` so turboCLI's diffusers pipelines reuse ComfyUI's device-agnostic
(CPU/CUDA/MPS) partial-offload path with 1:1 parity. What the offloader takes from ComfyUI falls in
three categories, and each one has its own step in a re-sync:

1. **Copied verbatim**: re-copy the files from the new tag.
2. **Edited**: two one-line comment-outs, re-applied after the copy.
3. **Mirrored or called by name, not copied**: code in `offloader/` that reproduces a piece of
   ComfyUI or depends on one of its internals, each citing its upstream lines; diff those lines
   between the two tags and port any change.

Nothing else should differ from upstream.

## Source snapshots (bump together)

Pin tagged releases only, never a branch head: copy from the ComfyUI tag, and take comfy-aimdo /
comfy-kitchen at the versions that tag's `requirements.txt` pins.

| repo          | commit                                     | tag                    |
|---------------|--------------------------------------------|------------------------|
| ComfyUI       | `2aa4d535a76e4b6acf6a94f263bc482430c396dd` | v0.45.4                |
| comfy-aimdo   | `68088de9dce9f5e32231fccf230938769c5562da` | v0.6.0 (pip)           |
| comfy-kitchen | `cbc44a5b7cc3b8f910ef76ae5251c94971eb05d1` | v0.2.41 (pip)          |

The runtime matches the reference ComfyUI install too, so turbo and ComfyUI are compared on the
same kernels: torch 2.14.1+cu130, torchvision 0.29.1, torchaudio 2.11.0 (pinned in turboCLI
`bash/turbo/build.sh`).

## 1. Copied verbatim from `ComfyUI/comfy/`

Flat modules: `model_management.py`, `model_patcher.py`, `ops.py`, `memory_management.py`,
`utils.py`, `lora.py`, `float.py`, `quant_ops.py`, `patcher_extension.py`, `hooks.py`,
`pinned_memory.py`, `model_prefetch.py`, `cli_args.py`, `options.py`, `system_memory.py`,
`internal_logging.py`, `rmsnorm.py`, `storage.py` (these four are imported by the others since
v0.39; none pulls anything off-path), `configurable.py` (imported by `model_patcher.py` since
v0.41), `anonymous_file.py` (imported by `utils.py` since v0.45, standard library only).

Packages (whole dir): `comfy_types/`, `weight_adapter/`.

Model code, for the VAEs an engine runs as ComfyUI's own (`comfy_vae`, category 3):
`ldm/wan/vae.py` (Wan 2.1: Qwen-Image, Krea 2), `ldm/wan/vae2_2.py` (Wan 2.2 layout: Qwen-Image
2.1), `ldm/models/autoencoder.py` (the FLUX.2 VAE: MOSS-Image), and what they import,
`ldm/modules/diffusionmodules/model.py` (with that package's empty `__init__.py`),
`ldm/modules/distributions/` (whole dir), `ldm/modules/ema.py`; and `diffusers_convert.py`, which
sd.py runs on a VAE file in diffusers' layout.

Model code, for the transformers an engine runs as ComfyUI's own (category 3):
`ldm/qwen_image/model.py` (Qwen-Image, Qwen-Image-Edit), `ldm/qwen_image21/model.py` (Qwen-Image
2.1), `ldm/flux/model.py` (FLUX.2), `ldm/flux3/model.py` (FLUX.3), `ldm/krea2/model.py` (Krea 2),
`ldm/lumina/model.py` (Z-Image), `ldm/moss_image/model.py` (MOSS-Image), and what they import when
they load: `ldm/flux/layers.py`, `ldm/flux/math.py`, `ldm/lightricks/model.py`,
`ldm/lightricks/symmetric_patchifier.py`, `ldm/modules/attention.py`,
`ldm/modules/sub_quadratic_attention.py`, `ldm/modules/diffusionmodules/util.py`,
`ldm/modules/diffusionmodules/mmdit.py`, `ldm/chroma_radiance/layers.py`, `ldm/util.py`,
`ldm/common_dit.py`, `ldm/wan/model.py`, `ldm/wan/model_animate2.py`. Imports made inside a
function are left out: they only run on paths the offloader never takes.

Model code, for the text encoders an engine runs as ComfyUI's own (category 3):
`text_encoders/qwen_image.py` (Qwen2.5-VL-7B), `text_encoders/qwen_image21.py` (Qwen3-VL-8B),
`text_encoders/flux.py` (FLUX.2's Qwen3 4B and 8B, FLUX.3's Qwen3-VL-4B), `text_encoders/krea2.py`
(Krea 2's Qwen3-VL-4B), `text_encoders/z_image.py` (Qwen3 4B), `text_encoders/moss_image.py`
(MOSS-Image's Qwen3.5 9B), and what they import when they load: `text_encoders/qwen3vl.py`,
`text_encoders/qwen35.py`, `text_encoders/qwen_vl.py`, `text_encoders/llama.py`,
`text_encoders/sd3_clip.py`, `text_encoders/t5.py`, `text_encoders/bpe_tokenizer.py`,
`sd1_clip.py`, `sdxl_clip.py`, `clip_model.py`, with the tokenizer files they read,
`text_encoders/qwen25_tokenizer/` and `text_encoders/qwen35_tokenizer/` (data, byte-for-byte).

`cli_args.py` + `options.py` are vendored as-is (not stubbed): `options.args_parsing` is `False`, so
`cli_args` does `parser.parse_args([])` and every flag gets its upstream default. This is more
faithful and lower-maintenance than a hand-written `args` stub (no field list to drift).

## 2. Edited: two one-line comment-outs

Each is a single commented import for a dependency that is **off the offloader's code path**, so
the offloader is unaffected; only unrelated helper functions would `NameError` if ever called (they
are not). Grep `# [turbo-offloader] disabled for turboCLI:` to find them all.

| file      | line   | commented import                              | why it's off-path |
|-----------|--------|-----------------------------------------------|-------------------|
| `lora.py` | ~22    | `import comfy.model_base`                     | only used by `model_lora_keys_unet()` (LoRA key-mapping at load time), which the adapter bypasses; would otherwise pull the whole model zoo |
| `hooks.py`| ~17    | `from node_helpers import conditioning_set_values` | only used by the conditioning helpers at the bottom of the file (~L692-781); keeps every hook class/enum `ModelPatcher` needs verbatim |

## 3. Mirrored or called by name, not copied

Code in `offloader/` that reproduces a piece of ComfyUI, or relies on one of its internals by name.
The verbatim copy does not cover it, so a re-sync diffs the upstream lines listed here.

### ComfyUI's VAE, from `comfy/sd.py`

`sd.py` imports most of ComfyUI (every model, every text encoder), so it is not vendored: a
`comfy.sd` import would cost a cold start about 0.8 s and about 100 MB of RAM. What `comfy_vae`
needs from it is copied into `offloader/adapter.py`, each piece citing its `sd.py` lines (v0.45.4):

| upstream | in adapter.py | what |
|---|---|---|
| `VAE.__init__` branches, sd.py:846-887, :678-716 on the defaults of :509-519 | `_comfy_vae_model` | the model a file gets and its settings: Wan 2.1 (:872-887), Qwen-Image 2.1 (:849-859), the FLUX.2 VAE (:678-716, its small decoder too) |
| sd.py:503-505; end of `VAE.__init__`, sd.py:1166-1188; `VAELoader.load_vae`, nodes.py:857 | `ComfyVAE._build` | the file read (a diffusers layout converted), device, dtype, the patcher, the weight load |
| `VAE.decode`, sd.py:1360-1410; `VAE.encode`, sd.py:1513-1562 | `ComfyVAE.decode` / `encode` | `load_models_gpu` with the estimate, the out-of-memory fallback; unlike sd.py, `encode` returns the caller's dtype, as the pipeline feeds it to its transformer |
| `decode_tiled_`, `decode_tiled_3d` (sized at sd.py:1425-1448), `encode_tiled_`, `encode_tiled_3d` | the same names on `ComfyVAE` | the tiled fallbacks |
| main.py:318-319 | `enable_vbar` | `CoreModelPatcher` becomes `ModelPatcherDynamic` |

Adding a VAE: vendor its `comfy/ldm/` model file and what it imports (category 1), add its `sd.py`
branch to `_comfy_vae_model`, then compare a decode against ComfyUI's on the same latent.

### ComfyUI's transformers, behind a diffusers pipeline's call

An engine can name its transformer by ComfyUI's model (`load_pipe_comfy`'s `transformer["model"]`):
the offloader then builds ComfyUI's own model from its file and runs it behind the call the
diffusers pipeline makes, so its fused kernels, per-block prefetch and caches come with each
re-sync. Each model has an adapter in `offloader/` (`_comfy_model` picks it); what ComfyUI keeps
around the model is mirrored there (v0.45.4):

| upstream | in offloader/ | what |
|---|---|---|
| `load_diffusion_model_state_dict`, sd.py:2460-2555; model_base.py:184-187 | `adapter.load_comfy_single_file` | the file read, its quantization, ComfyUI's ops for the storage and compute dtypes (sd.py:2529-2538), the build and the weight load, assigned only under a dynamic patcher (sd.py:2551) |
| `BaseModel`, model_base.py:174-276 (`_apply_model`'s options, :254-257); `model_lora_keys_unet`'s generic entries, lora.py:200-207 | `adapter.BaseModel` | what every model shares: what the patcher and the pipeline read off it (a `cache_context` that keeps nothing), its LoRA keys |
| `count_blocks`, model_detection.py:9-20 | `adapter.count_blocks` | how many blocks a prefix numbers, for each model's `detect` |
| the grids `process_img` builds per image (e.g. ldm/flux/model.py, ldm/flux3/model.py) | `adapter.id_grids` | each grid's size off a pipeline's position ids, read once per run |
| model_detection.py:1012-1021 | `qwen_image.detect` | Qwen-Image's settings off its file (2511's `index_timestep_zero`) |
| `model_base.QwenImage`, model_base.py:2784-2815; `BaseModel._apply_model`, model_base.py:221-263 | `qwen_image.QwenImage` | the model ComfyUI samples, its references as `edit` hands them |
| `TextEncodeQwenImageEditPlus`, nodes_qwen.py:75-109 | `qwen_image.QwenImage.edit`, `qwen_image._scale` | each reference at about 1024x1024 on the 8 grid, VAE-encoded before sampling |
| `model_lora_keys_unet`, lora.py:337-350 | `qwen_image.QwenImage.lora_keys` | its names, a fused `gate_up` (Qwen-Image 2.1's) by its two halves |
| model_detection.py:991-1010 | `qwen_image21.detect` | Qwen-Image 2.1's settings off its file |
| `model_base.QwenImage21`, model_base.py:2825-2879; `BaseModel._apply_model`, model_base.py:221-263 | `qwen_image21.QwenImage21`, over `qwen_image.QwenImage` | the model ComfyUI samples: dynamic units, `current_patcher`, the writable prefix cache, the casts around the transformer |
| `ModelPatcher.writable_models` around sampling, samplers.py:1237, model_patcher.py:1408-1417 | `qwen_image21.QwenImage21._get_writable` | opened on a run's first forward, once its conditioning is known, closed at its end |
| `TEModel.encode_token_weights`, text_encoders/qwen_image21.py:62-76 | `qwen_image21.QwenImage21.forward` | the vision tokens leave the text, a slot recording where each image goes |
| model_detection.py:273-329, :359-362 | `flux2.detect` | FLUX.2's settings off its file |
| `Flux.process_unet_state_dict`, supported_models.py:756-763 | `flux2.process` | the QK norms' scales as weights (FLUX.2 and FLUX.3) |
| `model_base.Flux2` and `Flux`, model_base.py:996-1126; `BaseModel._apply_model`, model_base.py:221-263 | `flux2.Flux2` | the model ComfyUI samples: the text left-padded to 512, guidance 3.5, the references by index |
| `model_lora_keys_unet`, lora.py:276-290 | `flux2.Flux2.lora_keys` | its diffusers names, the fused `linear1`'s qkv |
| model_detection.py:263-272 | `flux3.detect` | FLUX.3's settings off its file |
| `model_base.Flux3`, model_base.py:1129-1159; `BaseModel._apply_model`, model_base.py:221-263 | `flux3.Flux3` | the model ComfyUI samples; a flow map's step target as `sample_sigmas`, and its text and references run every step (:1141) |
| `model_lora_keys_unet`, lora.py:352-358 | `flux3.Flux3.lora_keys` | the fused qkv by its three projections |
| model_detection.py:1060-1073 | `krea2.detect` | Krea 2's settings off its file |
| `model_base.Krea2`, model_base.py:2918-2945; `BaseModel._apply_model`, model_base.py:221-263 | `krea2.Krea2` | the model ComfyUI samples, the target's grid off the pipeline's position ids |
| `model_lora_keys_unet`, lora.py:373-382 | `krea2.Krea2.lora_keys` | the diffusers names of each weight (`krea2_to_diffusers`) |
| model_detection.py:594-604, :616-640 | `lumina2.detect` | Z-Image's settings off its file |
| `model_base.Lumina2`, model_base.py:1569-1622; `BaseModel._apply_model`, model_base.py:221-263 | `lumina2.Lumina2` | the model ComfyUI samples: the text's length as `num_tokens`, conds of different lengths apart |
| `model_lora_keys_unet`, lora.py:384-393 | `lumina2.Lumina2.lora_keys` | its diffusers names (`z_image_to_diffusers`) |
| model_detection.py:1050-1058; `MOSSImage.process_unet_state_dict`, supported_models.py:2082-2084 | `moss_image.detect`, `moss_image._process` | MOSS-Image's settings off its file, `rope.bands` dropped |
| `model_base.MOSSImage`, model_base.py:2947-2976; `BaseModel._apply_model`, model_base.py:221-263 | `moss_image.MOSSImage` | the model ComfyUI samples: the uncond's skipped blocks as `SkipLayerGuidanceDiTSimple` replaces them (nodes_slg.py:116-151), the references' keys and values computed once |
| `TextEncodeMOSSImageEdit` (Moss-Image-Edit), nodes_moss_image.py:95-149; `ImageScaleToTotalPixels`, nodes_post_processing.py:249-258 | `moss_image.MOSSImage.edit`, `_scale`, `_encode` | the source at about 1 MP on the 16 grid, encoded by ComfyUI's VAE (`comfy_vae`) |
| `MOSSImage.extra_conds`' `ref_kv`, model_base.py:2966-2974 | `moss_image.MOSSImage.extra_conds` | the references' keys and values once, before sampling |
| `model_lora_keys_unet`, lora.py:361-371 | `moss_image.MOSSImage.lora_keys` | bare module paths, a fused qkv by its three projections |

### ComfyUI's sampler, behind a pipeline's scheduler

An engine whose template samples otherwise than diffusers' schedulers do can take ComfyUI's
KSampler instead (`comfy_sampler`): `adapter.FlowSampler` answers the scheduler calls a pipeline
makes (`set_timesteps`, `timesteps`, `sigmas`, `step`), for a flow model. `samplers.py` and
`k_diffusion/` are not vendored (they import most of ComfyUI); what it needs is mirrored (v0.45.4),
and checked against them bit for bit:

| upstream | in adapter.py | what |
|---|---|---|
| `ModelSamplingDiscreteFlow`, model_sampling.py:289-336 | `FlowSampler._table`, `_sigma`, `percent_to_sigma` | the shifted sigma table |
| `simple_scheduler`, `normal_scheduler`, samplers.py:645-693; the sigmas on the device, samplers.py:1257 | `FlowSampler._schedule`, `set_timesteps` | the schedules |
| `calculate_denoised` (CONST), model_sampling.py:35-37 | `FlowSampler.step` | the denoised off the model's velocity |
| `sample_euler`, k_diffusion/sampling.py:190-212 | `FlowSampler.step` | one Euler step |
| `sample_er_sde`, k_diffusion/sampling.py:1593-1656; `offset_first_sigma_for_snr`, :168-176; `default_noise_sampler`, :78-88 | `FlowSampler._er_sde`, `_noise` | one ER-SDE step, its noise drawn from the KSampler's seed |

### ComfyUI's text encoders, behind a diffusers pipeline's encode

An engine can name its text encoder by ComfyUI's model too (`text_encoder["model"]`): the
offloader loads it as `CLIPLoader` does, and the pipeline encodes through it, as `CLIPTextEncode`
does. sd.py is not vendored (see above), so what it does for a text encoder is mirrored:

| upstream | in offloader/ | what |
|---|---|---|
| `CLIP`, sd.py:246-310, :332-343, :408-439, :441-455, :471-476 | `adapter.CLIP` | the tokenizer, the model behind its patcher, the encode |
| `llama_detect`, text_encoders/hunyuan_video.py:11-23 (sd.py:1877-1885 picks the file) | `adapter.llama_detect` | a Llama-family encoder's dtype and quantization |
| `LoadImage`, nodes.py:1782-1789 | `adapter.load_image` | a PIL image as an IMAGE |
| `load_text_encoder_state_dicts`, sd.py:1891-1899 | `adapter.load_clip_sd`, `adapter.qwen3vl_sd` | the file's renames, a Qwen3-VL file's prefixes |
| the krea2 branch, sd.py:2084-2087 | `krea2.TextEncoder` | Krea 2's model and tokenizer |
| the qwen_image branch, sd.py:2023-2032; `TextEncodeQwenImageEditPlus`, nodes_qwen.py:75-109 | `qwen_image.TextEncoder` | Qwen-Image's, the prompt with each image at about 384x384 |
| sd.py:2108-2111; `TextEncodeQwenImage21`, nodes_qwen.py:144-180 | `qwen_image21.TextEncoder` | Qwen-Image 2.1's, the prompt with its images |
| sd.py:1844-1853, :2046-2062 | `flux2.TextEncoder` | FLUX.2's, Qwen3 4B or 8B by its width |
| sd.py:2088-2091 | `flux3.TextEncoder` | FLUX.3's Qwen3-VL-4B |
| sd.py:2046-2052 | `lumina2.TextEncoder` | Z-Image's Qwen3 4B |
| sd.py:2066-2070 | `moss_image.TextEncoder` | MOSS-Image's Qwen3.5 9B |

A Krea 2 edit runs the forward of the comfyui-krea2edit node pack (v2.0.0, Apache-2.0,
`__init__.py`), what a ComfyUI user runs for it, on the path its `fit` mode takes. Re-sync it
like category 3 when the pack moves:

| upstream (comfyui-krea2edit `__init__.py`) | in offloader/ | what |
|---|---|---|
| `_imgids`, `_imgids_offset`, `_place_refs`, :30-81 | `krea2._imgids`, `_imgids_offset`, `_place_refs` | the references' positions, aligned on the target's grid or beside it |
| `_ref_attn_bias`, :170-199 | `krea2._ref_attn_bias` | the boost on the target's attention to a reference, no mask |
| `krea2_edit_forward`, :202-292; `Krea2EditModelPatch.patch`, :328-427 | `krea2.Krea2._edit_forward`, `Krea2.edit` | [text, references, target], the references fitted in pixels (`ref_native`, stride-1 positions) |
| `_fit_encode_image`, :107-151 | `krea2._fit_encode_image` | a reference fitted to the target in pixels and VAE-encoded, before sampling |
| `Krea2EditGroundedEncode`, :428-524 | `krea2.TextEncoder.encode` | the prompt encoded with the references in the user turn, their longest side capped |

Adding a text encoder: vendor its `comfy/text_encoders/` file and what it imports (category 1),
mirror its sd.py branch as the model's `TextEncoder` after `krea2.py`, then compare its
conditioning with ComfyUI's on the same prompt (and image).

Adding a transformer: vendor its `comfy/ldm/` model file and what it imports (category 1), write
its adapter on `adapter.BaseModel` after `flux3.py`, add it to `_comfy_model`, then compare its
images and speed against ComfyUI's on the same files and template graph.

### Offloading internals

| upstream | in offloader/ | what |
|---|---|---|
| per-block `prefetch_queue_pop` loop (e.g. `comfy/ldm/lightricks/av_model.py`) | `adapter.install_prefetch` | the same loop, through forward hooks on the diffusers transformer's block ModuleLists |
| `pinned_memory` internals | `adapter.install_pin_rollback_guard` | survive a transient `HostBuffer.truncate` failure, keeping ComfyUI's own recovery |
| the SDPA body in `comfy/ops.py` (:58-109) | `adapter.use_comfy_attention` | diffusers' attention through a copy of it |
| `pre_run` before sampling (`comfy/samplers.py`:1260); `cleanup`'s `current_patcher = None` after (`model_patcher.py`:1337-1338) | `prepare`; `_end_sampling` | a model declaring `current_patcher` gets its patcher for the run and drops it before the VAE decodes (Qwen-Image 2.1 frees its prefix cache) |
| each node's `load_models_gpu` (CLIPTextEncode, KSampler's `prepare_sampling`, `comfy/sampler_helpers.py`:201) after the teardown marks pins inactive (`model_management.py`:1484) | `prepare`, the encode boundary in `_finalize_pipe` | the encoder loads at encode, the sampling models before sampling |
| the per-node teardown in `execution.py` | `node_teardown`, `_end_sampling` | the sampler to VAE node boundary |
| `VAE.decode` / `encode` casting their input to the VAE's dtype (e.g. sd.py:1279) | `_finalize_pipe` | a diffusers VAE gets its input in its own dtype, float32 latents from ComfyUI's models included |
| aimdo's `control.init` arguments in `main.py` | `pre_torch_init` | comfy-aimdo's allocator hooks before torch |
| the `ModelPatcher` constructor (`fast_disk`, as sd.py:2408 passes it) | `adapter.build_dynamic_patcher` | the storage policy of the streamed files |
| `PromptModelTracker` | `prepare` / `reclaim` | a generation is ComfyUI's prompt |

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
runtime venv (turboCLI `bash/turbo/build.sh`, pinned `0.2.41`, Apache-2.0): a native wheel, so it
is **not vendored** into `offloader/comfy/`. Present: `quant_ops.py` configures its backends (cuda
disabled unless torch cu130+, triton unless flagged) exactly as ComfyUI does, and
`offloader/adapter.py:use_kitchen_rope` routes diffusers' `apply_rotary_emb` via `ck.apply_rope1`
(the same fused kernel ComfyUI's `comfy/ldm/flux/math.py` uses).

Two caveats. comfy-kitchen is required: `model_prefetch.py` imports it unconditionally, and since
v0.45 `model_patcher.py` imports `model_prefetch`, so turboCLI installs it on every build. And a
version older than the tag's pin is not safe: `quant_ops.py`'s import block then fails as a whole,
silently disabling fp8 and the kitchen RoPE.

## Re-sync procedure

1. Copy the files of category 1 from the ComfyUI tag over `offloader/comfy/`
   (`git show <tag>:comfy/<file>`, `git archive <tag> comfy/<pkg>` for the packages), then check
   `git diff --stat` matches upstream's own diff between the two tags.
2. Update the commits in this file, in `offloader/comfy/__init__.py` and in `README.md`, and the
   comfy-aimdo / comfy-kitchen versions in turboCLI `bash/turbo/build.sh`, along with the torch /
   torchvision / torchaudio the reference ComfyUI install runs.
3. Re-apply the 2 comment-outs of category 2 (grep the marker in the OLD tree first to relocate
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
   the VAE, decode one latent with a comfy- engine and with ComfyUI and compare the images. For
   `FlowSampler`, run ComfyUI's `simple` / `normal` schedulers and `sample_euler` /
   `sample_er_sde` on a stand-in model next to it: on CUDA they must be bit-identical.
   The test laptop throttles: interleave the runs (A, B, A, B) and trust a delta only once a
   second pair reproduces it.
