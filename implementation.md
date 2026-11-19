# offloader (v2)

GPL offload backend for the turboCLI diffusion runner. Instead of cherry-picking lines from ComfyUI
(v1, see `doc/aimdo_v1.md`), v2 **vendors ComfyUI's offloading subsystem verbatim** and adds a
**thin adapter** so turboCLI's diffusers pipelines reuse ComfyUI's own memory manager with 1:1
parity. The same code runs a model larger than VRAM by streaming its weights to VRAM per forward
with partial GPU residency — from host RAM when the model fits (ComfyUI's UNet offload), else
disk→VRAM via comfy-aimdo file-slices for a model larger than RAM.

**Style:** code and comments wrap at 99 columns.

**New here?** `dummy.md` is the plain-English introduction — what the offloader does and why, step
by step, no diffusion background required. It is kept up to date with each iteration of this doc.

## Layout

```
offloader/
  __init__.py   backend seam the runner drives (pre_torch_init/available/supports/load_pipe/...)
  adapter.py    the thin middleman -- the only real logic in the package
  <model>.py    one adapter per ComfyUI model an engine runs as ComfyUI's own (qwen_image,
                qwen_image21, flux2, flux3, krea2, lumina2, moss_image, minimax_h3_image)
  comfy/        byte-for-byte ComfyUI offloading snapshot (see resync.md)
```

`offloader/comfy/` is a pristine mirror of ComfyUI's `model_management.py`, `model_patcher.py`,
`ops.py`, `memory_management.py`, `model_prefetch.py`, `lora.py`, `utils.py`, etc. Aliased to the
top-level `comfy` package via a `sys.modules` entry in `comfy/__init__.py`, so the vendored
`import comfy.X` lines resolve unchanged. The **only** edits over upstream (all documented in
`resync.md`): three commented-out off-path imports, `try/except` shims for the optional
`comfy_aimdo` / `comfy_kitchen`, and a stubless `cli_args`+`options` (vendored, parses no argv).

## The seam (`__init__.py`)

The runner discovers `backend/<mode>/` and drives it through this interface only:

| fn | v2 behavior |
|---|---|
| `pre_torch_init()` | init comfy-aimdo's global CUDA hooks **before torch import** (its allocator can't hook afterwards); imports nothing that pulls torch. Optional -- absent/failed → native path. |
| `available()` | True once the vendored offloader imports (any device) |
| `supports(engine)` | `True` -- model-agnostic; offload eligibility is a turboCLI-side call |
| `load_pipe(model, dtype, pipeline_cls, transformer_cls, device, lora_files)` | build a fully-placed diffusers pipeline (below); runner supplies the classes |
| `load_pipe_comfy(pipeline_cls, transformer, text_encoder, components, dtype, device, lora_files)` | the pipeline of a ComfyUI-reuse engine, its transformer and text encoder ComfyUI's own models. **Model-agnostic**: the engine names each by its ComfyUI model (`{model, file, config}` for the transformer, `{model, file}` for the text encoder) plus prebuilt small `components` (vae/tokenizer/scheduler). Each model's adapter in `offloader/` (`qwen_image.py`, `qwen_image21.py`, `flux2.py`, `flux3.py`, `krea2.py`, `lumina2.py`, `moss_image.py`) runs ComfyUI's transformer behind the pipeline's call, and its `TextEncoder` loads ComfyUI's text encoder as CLIPLoader does, the pipeline encoding through it. |
| `prepare(pipe)` | `load_models_gpu` of the sampling models (the encoder loads at encode, as each ComfyUI node loads its own model): place them on the compute device; mark them the current prompt (`PromptModelTracker`); `pre_run` each patcher, as a sampling run does, so a model declaring `current_patcher` can ask it for `get_free_memory` |
| `reclaim(pipe)` | end a run the VAE decode did not end (`_end_sampling`: `current_patcher = None` as ComfyUI's `cleanup` does, then `node_teardown`), then `free_memory` + `soft_empty_cache` between generations; end the prompt |
| `release(pipe)` | `detach` each patcher |
| `comfy_vae(path, config)` | opt in to ComfyUI's own VAE: the model comfy/sd.py builds for the VAE file (its code vendored under `comfy/ldm/`), read as VAELoader reads it (`load_torch_file`, file slices under VBAR) and run as sd.py's VAE class runs it, behind the calls a diffusers pipeline makes. `adapter.ComfyVAE` mirrors sd.py: the model and settings a file gets (`_comfy_vae_model`), device, dtype and patcher (dynamic under VBAR), `decode` / `encode` through `load_models_gpu` with the out-of-memory fallback to its tiles; `resync.md` lists the mirrored lines. Used by `comfy-qwen-image-2-1` (Wan 2.2 layout), `comfy-krea2-turbo` and `comfy-qwen-image-edit-2511` (Wan 2.1), `comfy-moss-image` (FLUX.2, its small decoder, in fp16 as ComfyUI picks it), `comfy-minimax-h3-image` (the MiniMax-H3-Image VAE, a 2D ViT decoder that tiles itself and decodes to [0, 1]). |
| `comfy_sampler(model, sampler, scheduler, file=None)` | opt in to ComfyUI's own KSampler for a template that samples otherwise than diffusers' schedulers: `adapter.FlowSampler`, behind the scheduler calls a pipeline makes (`set_timesteps`, `timesteps`, `sigmas`, `step`), on ComfyUI's model `model`'s flow shift, or on the model sampling its adapter reads off the transformer's `file` (`sampling`: MiniMax-H3-Image's PDD grid); the pipeline sets its `seed`. The `simple` and `normal` schedulers, `euler` and `er_sde` with its seeded noise, bit-identical to ComfyUI's on CUDA; `resync.md` lists the mirrored lines. Used by `comfy-moss-image` and `comfy-minimax-h3-image`. |

All GPL-derived code lives in this package; the calling runner stays GPL-free.

## Two paths, one code

- **Native (device-agnostic: CPU / CUDA / MPS).** Weights load onto the offload device (CPU,
  mmap); a `ModelPatcher` streams them to the compute device per forward via ComfyUI's cast path
  (`cast_bias_weight` → `cast_to`). Needs the model to fit RAM. `comfy_aimdo` not required.
- **VBAR (optional, CUDA).** When comfy-aimdo is present, `load_pipe` flips
  `comfy.memory_management.aimdo_enabled` and uses `ModelPatcherDynamic`, which keeps as much of
  the model GPU-resident as fits (partial residency, sized from live free VRAM) and streams the
  rest per forward. Every component is meta-loaded and mmap file-sliced (`load_streamed` /
  `assign_streamed_weights`), so its weights fault straight from the file — from the **OS page
  cache** when the model fits host RAM, else **disk→VRAM** when it doesn't (no materialization, no
  explicit per-component gate; the page cache makes the choice). This is ComfyUI's "dynamic VRAM
  loading" path.

The seam picks VBAR automatically on CUDA-with-comfy-aimdo, native otherwise. `set_device()` maps
the seam's `device=` arg onto `comfy.model_management.cpu_state`, the single knob that switches the
whole device stack.

## CPU placement (`comfy` / `stream`)

On CPU there is no device boundary to manage, so ComfyUI itself just full-loads (its lowvram path
is `DISABLED` on CPU) at fp32 — fast when the model fits RAM, an OOM when it doesn't. `load_pipe`
offers both, and picks the way ComfyUI decides on GPU (full-load when it fits, stream otherwise),
applied to CPU:

- **`comfy`** — ComfyUI's default: fp32 storage, materialised. Faster (weights cast to fp32 once at
  load, then plain forwards), but needs the whole model resident.
- **`stream`** — bf16 storage (ComfyUI `--bf16-unet`) + mmap-keep (weights never materialise, RAM
  bounded) + fp32 compute cast per forward via `manual_cast`. Runs a model larger than RAM; slower
  per step (casts every forward instead of once).

The default auto-selects via `adapter.cpu_fits_full_load()` — the fp32 model (~2× the on-disk bf16
size) against ComfyUI's `get_total_memory(cpu)`, 85% headroom. `OFFLOADER_CPU_MODE=comfy|stream`
forces one. The dtype is ComfyUI's own call: flip `args.bf16_unet` for the chosen mode and read
`mm.unet_dtype()` (bf16 or fp32); `manual_cast_dtype` (`mm.unet_manual_cast`) then picks fp32
compute on CPU. The VAE runs fp32 either way (ComfyUI's `vae_dtype(cpu)`); `stream` upcasts its
input latents to match.

## The adapter bridge (`adapter.py`)

turboCLI's models are vanilla diffusers/transformers `nn.Module`s; ComfyUI's offloader expects its
own `comfy.ops` modules and a `ModelPatcher`. The adapter closes that gap, minimally:

- **`comfy_ize(model)`** — re-class each offloadable leaf (`Linear`/`Conv`/`LayerNorm`/`GroupNorm`/
  `Embedding`/`RMSNorm`, incl. custom `*RMSNorm`) to its `comfy.ops.disable_weight_init.*`
  counterpart, so its forward routes through ComfyUI's cast path with 1:1 parity. Injects the
  `comfy_cast_weights`/`weight_function`/`bias_function` attrs ModelPatcher reads. A custom
  `*RMSNorm` is only re-classed when the fused op **provably** reproduces its forward
  (`_rmsnorm_matches_fused` probes it structurally: real random weight in, module forward vs the
  exact `F.rms_norm` call the re-class would make — which also guards the eps/`normalized_shape`
  `_prep_rmsnorm` inferred). comfy's `RMSNorm` applies the weight verbatim, so a norm with its own
  weight convention (diffusers' `Krea2RMSNorm` normalizes by `1 + weight`, scale stored
  zero-centered) keeps its own forward — as ComfyUI runs it: comfy hands a model `operations` for
  its `Linear`/`Conv` leaves, it never swaps out the model's norm. `keep_uncastable_resident` then
  places it (a norm's weight is tiny). The re-class can only ever be a no-op speedup.
- **`build_patcher` / `build_dynamic_patcher`** — wrap in `ModelPatcher` / `ModelPatcherDynamic`.
  `make_patchable` shadows diffusers/HF's read-only `.device` property (which ComfyUI assigns to)
  while keeping `str(cls)` identical so transformers' output-capture registry still fires.
- **`keep_uncastable_resident`** — move every non-castable param/buffer (custom norms, pad tokens,
  rotary caches) to the compute device so nothing is stranded on the offload device mid-forward.
  Under manual_cast (below) it also casts float stragglers to the compute dtype, matching ComfyUI's
  params being built in that dtype.
- **`add_lora`** — on-cast LoRA: build the `{lora_key → model_weight_key}` map the way ComfyUI's
  `model_lora_keys_unet` does (same generic aliases: bare / `transformer.` / `diffusion_model.` /
  `lora_unet_`), hand it whole to `comfy.lora.load_lora` (vendored `weight_adapter`), register via
  `ModelPatcher.add_patches`. `load_lora` picks only the bases present in the file, so every
  published naming and format loads — lora_up/down, lora_A/B, `diff`, **LoKr/LoHa** — from one
  path. The engine may supply extra namings via the `lora_keys` spec (below); Krea2 uses it to map
  ComfyUI-native keys onto the diffusers module tree. An entry may target a slice as ComfyUI's do,
  `(weight_key, (dim, offset, size))`: FLUX.3 maps q/k/v LoRAs onto thirds of its fused qkv.
  Deltas apply while each weight streams in: no fuse, no resident copy.
- **`install_unpadded_encode`** — drop padded text tokens from the conditioning before the
  transformer sees them. ComfyUI never pads the modern LLM text encoders
  (`pad_to_max_length=False`: qwen3vl/qwen_image/flux-t5; only CLIP keeps `SDTokenizer`'s default
  `True`, since its 77-token pad is required) and drops an all-ones mask outright
  (`text_encoders/krea2.py:65-66`); diffusers' own qwen pipeline agrees (`if
  prompt_embeds_mask.all(): prompt_embeds_mask = None`). Its **krea2** pipeline is the outlier --
  `padding="max_length"` pads every prompt to a fixed 512 -- so the DiT dragged ~500 dead tokens
  through all 28 blocks: **1536 tokens at 512² where ComfyUI runs ~1054**, which was the entire
  per-step gap (**5.17 → 3.50 s/it**). Model-agnostic: keyed only off the mask the pipeline itself
  returns. Sound because a key-padding mask *means* those tokens are excluded as attention keys,
  so removing them is a no-op for any model that honours it. Gathers (krea2 pads mid-template),
  re-pads ragged batches, and falls through untouched for a pipeline returning a second embeds
  tensor instead of a mask (z-image's `prompt_embeds, negative_prompt_embeds`).
- **`load_comfy_single_file` / `mixed_precision_operations`** — a model of ComfyUI's own from its
  single file (or shards), as sd.py's `load_diffusion_model_state_dict` loads it: mmap the file,
  `convert_old_quants` (the fp8 markers from either the classic `scaled_fp8` + `scale_weight` keys
  or the file's `_quantization_metadata`), `detect_layer_quantization`, ComfyUI's ops for it
  (`comfy.ops.mixed_precision_ops` for a quantized file, cast ops otherwise), build, then
  `load_state_dict`, so each quantized Linear builds a comfy `QuantizedTensor`. The weights stay
  fp8 or int8 (mmap, never materialized) and dequantize per forward, as ComfyUI does. As sd.py
  does too, the model is built in the storage dtype `_prepare_offload` picks (ComfyUI's
  `unet_dtype`: fp32 on a CPU unless streamed), computes in its manual_cast dtype where that
  differs, and takes the file's tensors as they are (`assign`) only under a dynamic patcher
  (`model_patcher.is_dynamic()`); otherwise they are copied in. Assigning them everywhere once
  left bf16 weights under fp32 inputs on CPU, an error on the first matmul.
  **Device-agnostic like the rest:** the `disabled` set is computed from comfy's
  `supports_fp8_compute(load_device)` (& nvfp4/mxfp8), so a device without fp8 tensor cores — CPU,
  MPS, Ampere sm_86 — emulates the dequant to the compute dtype, as ComfyUI does. MPS
  untested. (Note: the practical blocker to a *full* CPU run is unrelated to the quant path. On CPU
  `load_torch_file` takes its non-aimdo branch — a full-file `safetensors.safe_open` mmap — and
  mmapping the 39GB transformer can exceed Windows' commit limit on a 32GB box, OS error 1455
  "paging file too small", intermittently. On CUDA `aimdo_enabled` routes to comfy-aimdo's
  file-slice `load_safetensors` instead, so there is no full host mmap and no 1455 — the same
  reason ComfyUI-on-CUDA is fine. A larger page file fixes the CPU case; a 39GB model on a 32GB box
  is edge either way, since materializing it wouldn't fit RAM.)
- **`install_encode_cache`** — wrap the pipeline's `encode_prompt` so a repeated prompt returns
  cached text embeddings and never re-runs (nor re-streams) the encoder — the diffusers analogue of
  ComfyUI's default node cache. Copies `comfy_execution/caching.py::RAMPressureCache`:
  full-input-signature key, embeddings held on CPU, eviction under host-RAM pressure via
  `comfy.model_management.get_free_memory` (below `min(10GB, max(2GB, 10% RAM))`, worst
  `1.3**age × bytes` first). Validated against ComfyUI's own `generate.sh`: identical cold time,
  encode skipped on a repeat prompt in both. An image or tensor argument is keyed by its content
  hash, as ComfyUI's `LoadImage` keys its output by the file's, and the VAE encode of an edit's
  references goes through the same store (vae.encode takes no generator), so a repeated edit with
  a new seed skips both encodes as ComfyUI's cached edit node does.

### Compute parity with ComfyUI

Diffusers runs some ops on slower kernels than ComfyUI. Copied from ComfyUI, not reimplemented:

- **`manual_cast_dtype` / `install_manual_cast`** — ComfyUI's storage-vs-compute dtype split
  (`comfy.model_management.unet_manual_cast`). On a GPU without bf16 tensor cores (e.g. Turing
  sm_75) a bf16 checkpoint's matmuls fall off the tensor cores and run ~5× slower than fp16;
  ComfyUI computes such a model in **fp16** while the weights stay bf16. We do the same: storage
  stays bf16 (mmap — no load-time cast, so no RAM blow-up), compute runs fp16. ComfyUI enforces
  this *inside* its native model; the diffusers model doesn't, so we reproduce the same dtype
  discipline from outside via forward hooks, each mirroring a specific comfy cast:
    - input cast ← `model_base.py:211` (`xc = xc.to(dtype)`);
    - **per-leaf input cast** ← `lumina/model.py:859` (`t_embedder(..., dtype=x.dtype)`) — the
      crux: diffusers computes the time-embed/adaLN in fp32, so `norm(x) * scale` promotes to fp32
      and `cast_bias_weight` then casts weights to fp32, dropping every matmul onto the fp32
      `volta_sgemm` path; casting each comfy-ized leaf's input back to the compute dtype forces
      fp16 tensor cores (`s1688gemm`);
    - straggler cast (in `keep_uncastable_resident`) ← lumina params built `dtype=x.dtype`;
    - `clamp_fp16` ← `lumina/model.py:71-74` — guards fp16 overflow (→ NaN → black image) after
      each block; without it the output is all-zero;
    - output cast back to the storage dtype (diffusers pipeline latent-dtype contract).
  The per-layer weight cast itself is comfy's own `cast_bias_weight`, unchanged. Which leaves take
  it is comfy's too: under manual_cast `comfy_ize` re-classes each leaf to `comfy.ops.manual_cast`
  (the namespace ComfyUI's `pick_operations` selects), so **every** leaf casts — resident ones
  included — exactly as ComfyUI builds a manual_cast model; on Ampere `pick_operations` returns
  `disable_weight_init` (plain forward, no cast). Applied to both the transformer and the text
  encoder (ComfyUI runs the TE fp16 too — `text_encoder_dtype`). Gated by `unet_manual_cast`, so
  **Ampere+ is unchanged** (returns None → compute stays bf16). z-image 1024×768 warm: ~10.4 → ~2.3
  s/step, matching ComfyUI; output seed-valid.
- **`install_prefetch`** — drives ComfyUI's `comfy.model_prefetch` (`prefetch_queue_pop`) via
  forward hooks on the transformer's block lists, so block N+1's weights stream while block N
  computes (overlap; helps when streaming-bound). This is comfy's own mechanism applied to models
  ComfyUI doesn't enable it for: in their codebase only Lightricks/LTXV calls
  `prefetch_queue_pop`; their krea2 forward stalls each block on its weights. It is why we beat
  ComfyUI per-step under memory pressure — parity at 512² (3.50 vs 3.62 s/it, streaming mostly
  hidden either way) growing to ~25% ahead at 1024×768 (6.97 vs ~8.5–9.8 s/it, same-seed
  same-image runs, both cool-started): bigger activations leave less VRAM for weight residency,
  more re-streaming per step, and we hide it behind compute while ComfyUI pays it serially.
- **`use_comfy_attention`** — a verbatim copy of `comfy.ops.scaled_dot_product_attention`
  (`comfy/ops.py:58-109`): on CUDA it forces ComfyUI's SDPA priority (`ops.SDPA_BACKEND_PRIORITY`,
  `[FLASH, CUDNN, EFFICIENT, MATH]` since v0.39, read from the vendored ops so a re-sync carries
  it) **per call**, but only for large inputs (`q.nelement() >= 1024*128`); small attentions use
  torch's default backend, and a masked GQA call repeats K/V when no backend runs GQA natively
  (`ops.repeat_kv_for_gqa`, as comfy does). We reproduce it
  (rather than import it) because it calls `F.sdpa` internally, so pointing torch's `F.sdpa` at it
  self-recurses; instead we patch `torch.nn.functional.scaled_dot_product_attention` once
  (diffusers calls it by attribute) and delegate to the saved original. **The per-call size gate
  matters:** forcing cuDNN on *every* attention (incl. small inputs, which comfy skips) makes the
  first z-image forward after a flux2→z-image switch pick a nondeterministic cuDNN plan and
  diverge; copying comfy's gate verbatim is deterministic. **Dtype-mismatch coercion:** under
  manual_cast (fp16 compute on a bf16 checkpoint) some native HF models leave the attention inputs
  mismatched (transformers' qwen3 RoPE, in flux2's text encoder, promotes q,k to fp32 while v
  stays fp16), and torch's SDPA requires one dtype. We coerce
  mismatched q/k/v to the lowest-precision float present (the compute dtype); ComfyUI never hits
  this because it runs its OWN qwen3, but its attention does the same class of thing (upcasts
  q/k/v, `comfy/ldm/modules/attention.py:259-302`). Model/GPU- agnostic: fires only on a real
  mismatch, no-op on Ampere+ / non-manual-cast.
- **`use_kitchen_rope`** — routes the diffusers transformer's RoPE through comfy-kitchen's fused
  `apply_rope1` (the kernel ComfyUI's `comfy/ldm/flux/math.py` uses), via a `(cos,sin)→freqs_cis`
  shim + a module-scoped patch of `diffusers.models.embeddings.apply_rotary_emb`. Lazy:
  comfy-kitchen is imported only on a matching call, so engines with their own rope (z-image) never
  touch it. It engages for comfy-krea2-turbo but buys no measurable per-step there (rope is a small
  slice of a 12B DiT; `OFFLOADER_KITCHEN_ROPE=0` leaves the per-step unchanged, 4.96 vs 4.93 s/it).
  Its init is ~0.3 s marginal per process (torch is already imported; an earlier ~16 s reading was
  a cold-cache/thermal artifact -- order-reversed A/B totals are equal, 55 s vs 55 s). Worth
  keeping on regardless of speed: comfy's `apply_rope1` is NOT bit-equivalent to diffusers' native
  rope (measured: different image md5, each path individually deterministic), so it is what keeps
  the output on ComfyUI's exact numerics. The packed freqs_cis is cached by
  (cos, sin) tensor identity — ComfyUI builds `freqs` once per forward and hands it to every block
  (`ldm/krea2/model.py:364`), and diffusers passes one (cos, sin) tuple to all blocks, so the
  pack is loop-invariant: 56 rebuilds/step on krea2 (~9 ms, ~90 MB fp32 churn) collapse to one
  per generation. Verified bit-identical.
- **fused RMSNorm** — routing custom norms through `disable_weight_init.RMSNorm` gives ComfyUI's
  fused `F.rms_norm` (≈ 3.5× the eager `mul`/`rsqrt` diffusers/transformers use).

## Engines

The offloader is **model-agnostic**: it places whatever diffusers pipeline the runner hands it,
keying its mechanics off module structure (leaf type, detected rope/attention symbols, file-sliced
streaming), never a model name. The engines below are the ones **turboCLI** currently wires -- each
declaring its `PIPELINE`/`TRANSFORMER` classes -- so adding one is a turboCLI-side edit with no
change here.

- **flux2** (`Flux2KleinPipeline`) — text-to-image and image-input (edit); the pipeline accepts
  `image=`, so the seam is unchanged.
- **z-image** (`ZImagePipeline`) — text-to-image, Turbo (few-step).
- **qwen-image-edit** (`QwenImageEditPlusPipeline`) — image-edit; on-cast Lightning 4-step LoRA;
  ~55GB of weights (transformer + Qwen2.5-VL TE). Both components are mmap file-sliced, so their
  weights fault straight from the file — page cache when the working set fits RAM, disk when it
  doesn't — and run where they wouldn't fit VRAM. (The transformer is meta-loaded so it never sits
  in RAM; the text encoder is loaded by `from_pretrained` then swapped for slices, a transient ~1×
  peak.) The ComfyUI-reuse engines (`comfy-*`) go through `load_pipe_comfy`
  instead: their transformer and text encoder are ComfyUI's own models, loaded from a ComfyUI
  install's files as they are. All model-specifics live engine-side or in each model's adapter.

## Benchmarks

Reference wall-clock (load + generate, seed-fixed, `offloader` mode) on the boxes below -- yours
will differ with disk/RAM/VRAM and thermal state. Both models exceed VRAM (and, at fp32, RAM) on
both, so CUDA runs VBAR partial residency and CPU auto-selects `stream` (bf16 mmap + fp32
`manual_cast`); neither fits a plain full load.

**Machine:** Intel i7-4770K (4c/8t, 3.5 GHz) · 16 GB RAM · RTX 2070 SUPER (8 GB, Turing sm_75) ·
Windows 10 · torch 2.12 + cu130.

| Engine (on-disk bf16) | Device | Res | Steps | Wall-clock | Path |
|---|---|---|---|---|---|
| flux2-4b (15 GB) | CUDA | 1024×768 | 4 | ~140 s warm (~65 s gen) | VBAR stream |
| z-image-turbo (20 GB) | CUDA | 1024×768 | 8 | ~197 s warm (~104 s gen) | VBAR stream |
| flux2-4b | CPU | 512×512 | 4 | ~576 s | stream |
| flux2-4b | CPU | 256×256 | 4 | ~400 s | stream |
| z-image-turbo | CPU | 256×256 | 8 | ~669 s | stream |

Notes: **CUDA** is streaming-bound on an 8 GB card (the GPU idles between steps waiting on the
VBAR disk/RAM→VRAM fault); the first (cold) generation is ~30–50 % slower than warm as the DiT
streams cold from disk. **CPU** is fp32 `manual_cast` compute (hence minutes/image); it runs at all
only because `stream` keeps weights mmap-backed — a plain fp32 full load OOMs both models on 16 GB.

**Machine:** Intel i7-12800H (14c/20t) · 32 GB RAM · RTX A1000 Laptop (4 GB, Ampere sm_86) ·
Windows 11 · torch 2.12 + cu130.

| Engine (on-disk bf16) | Device | Res | Steps | Wall-clock | Path |
|---|---|---|---|---|---|
| flux2-4b (15 GB) | CUDA | 512×512 | 4 | ~30 s warm (~7 s gen) | VBAR stream |
| flux2-4b | CUDA | 1024×768 | 4 | ~60 s warm (~22 s gen) | VBAR stream |
| comfy-z-image-turbo (20 GB) | CUDA | 1024×768 | 8 | ~4.5 s/it (≈ z-image-turbo) | VBAR stream |
| comfy-qwen-image-edit-2511-lightning (41 GB DiT + 9 GB fp8 TE) | CUDA | 512×512 | 4 | ~22 s/it (vs stock ~31) | VBAR stream |
| comfy-krea2-turbo (13 GB fp8 DiT + 9 GB fp8 TE) | CUDA | 512×512 | 8 | ~3.5 s/it (ComfyUI ~3.6) | VBAR stream, both fp8 |
| comfy-krea2-turbo | CUDA | 1024×768 | 8 | ~7.0 s/it (ComfyUI ~8.5–9.8) | VBAR stream, both fp8 |
| flux2-4b | CPU | 512×512 | 4 | ~220 s | stream |
| flux2-4b | CPU | 1024×768 | 4 | ~325 s | stream |
| z-image-turbo (20 GB) | CPU | 512×512 | 8 | ~315 s | stream |

These rows predate the v0.39 re-sync (ComfyUI v0.27, torch 2.12). The current head-to-head against
ComfyUI v0.39.1 on torch 2.14.1 (same files and graphs, interleaved) is in the v0.39 re-sync note
under Notes: turbo on par or faster end to end, comfy-flux2-4b sampling faster than ComfyUI.

Notes: the A1000 is **Ampere** (sm_86, bf16 tensor cores), so CUDA compute stays bf16 -- no
`manual_cast` (unlike the Turing box above, which runs fp16); with 32 GB RAM the ≤20 GB models fit
the page cache, so VBAR streaming is RAM-fed. **The ComfyUI-reuse engines match or beat their stock
counterparts per-step** (same weights + offloader path): `comfy-z-image-turbo` ≈ `z-image-turbo`
(~4.5 s/it), and `comfy-qwen-image-edit-2511(-lightning)` is actually **~30% faster** than stock
(rigorous cool back-to-back, 2 reps each: ~21.6 vs ~31.1 s/it @512²). Both only **after fixing the
fp8 text-encoder offload** — `load_quant_single_file` must `comfy_ize` the non-Linear leaves too,
else the ~1 GB input `Embedding` stays pinned resident and starves the 41 GB DiT's VBAR budget
(free VRAM 3.2 → 2.1 GB), ~2.5× slower per step. The qwen speed-up over stock is because its DiT
**comfy-krea2-turbo has no stock counterpart** (krea2 is comfy-reuse only), so it is quoted against
**ComfyUI itself**: ~3.5 vs ~3.6 s/it @512², i.e. per-step parity, and marginally ahead. That was
5.17 s/it until `install_unpadded_encode` stopped feeding the DiT 500 padding tokens. It is quoted
from two independent thermal states rather than one reading — cool 3.50 vs 3.62, mid-session
throttled 13.33 vs 13.46 — because on this laptop a single number says more about temperature than
about code (z-image measured 13.4 against its ~4.5 row in that same window). At 1024×768 (both
cool-started, 58/61 °C, same-seed same-image) the parity becomes a ~25% lead — the prefetch
advantage above grows with memory pressure. The qwen speed-up over
stock is because its DiT **exceeds the 32 GB RAM** (disk-stream-bound), where comfy has two edges:
the fp8 TE is 9 GB vs
stock's 16.6 GB bf16, leaving ~7 GB more page cache for the DiT; and it streams from one contiguous
40 GB file vs stock's scattered shards (better readahead). z-image (12 GB DiT, fits RAM) shows
parity because neither edge matters when it isn't disk-bound. **These rows quote per-step, not
total wall-clock, on purpose:** the A1000 throttles across a long qwen run (a hot back-to-back gen
measured 62 s/it vs ~22 s/it cool), so per-step at a cool start is the
stable number; the qwen 41 GB DiT also exceeds the 32 GB page cache, so it disk-streams (unlike the
≤20 GB models). **CPU** per-step *climbs within a run* as the laptop throttles the i7-12800H from
~3.6 → ~1.05 GHz over ~5 min (flux2 1024×768: 61 → 90 s/step). Forcing `comfy` CPU mode (fp32
full-load) is slower at high res and OOM-thrashes z-image (fp32 working set ~48 GB > 32 GB RAM), so
the auto-picker correctly stays on `stream`.

**Machine:** Apple M1 (Mac mini, 8-core) · 8 GB unified RAM · MPS · macOS 15.5 · torch 2.12 ·
Python 3.14.

| Engine (on-disk bf16) | Device | Res | Steps | Wall-clock | Path |
|---|---|---|---|---|---|
| flux2-4b (15 GB) | MPS | 512×512 | 4 | ~185 s (~106 s gen) | direct-load, TE-CPU |
| flux2-4b | MPS | 1024×768 | 4 | ~249 s | direct-load, TE-CPU |
| comfy-flux2-4b (single files) | MPS | 512×512 | 4 | ~233 s (`sequential_cpu` ~458 s) | direct-load, TE-CPU |

Notes: MPS runs **fp16-resident** (no `manual_cast`) — the transformer is direct-loaded onto the
device (file mapped on the CPU, each tensor copied once onto MPS) and stays resident; there is no
VBAR/stream path (MPS has no separate VRAM pool). The text encoder runs on **CPU** (ComfyUI's
`text_encoder_device()` under `vram_state` SHARED), so residency is the transformer alone (~7 GB,
down from ~15 GB with both on-device) and only the conditioning crosses to MPS. On this 8 GB box
the run still spills to swap and **that** is the scaling wall, not compute: peak swap ~5.3 GB at
512×512, ~9.7 GB at 1024×768 (both produce clean images — the 1024 corruption seen in
image-to-image *edit* is that mode's larger footprint, not text-to-image). The faithful fp16-on-CPU
encode (~50 s) dominates the non-gen time and is cached (`install_encode_cache`), so a repeated
prompt skips it. Output is bit-deterministic across runs at a fixed seed (verified: two seed-42
runs, identical SHA-256). z-image-turbo (20 GB) is not benchmarked here — its working set exceeds
this 8 GB box's usable RAM+swap headroom, so it belongs on a larger-RAM Mac.

Re-measured on torch 2.14.1 (with the MPS load fix): comfy-flux2-4b 512×512 ~264 s
(`sequential_cpu` ~451 s), 1024×768 ~292 s (`sequential_cpu` ~475 s).

Known limit on 8 GB: the VAE decode takes ~15–40 s at 512×512 (vs ~1 s possible). MPS keeps a
memory heap alive while any tensor in it lives, and the VAE shares heaps with the resident
transformer, so its swapped memory is never returned. Left as is: releasing it would mean deviating
from ComfyUI.

**Machine:** NVIDIA L40S (48 GB, Ada sm_89) on Modal · 64 GB RAM · Debian · torch
2.14.1 + cu130 · models on a network Volume. Measured on 2026-11-17 with turboCLI's
`test/modal.sh run --record --repeat 2 --comfy /opt/ComfyUI`: one container per engine, turbo and
ComfyUI v0.45.4 alternating on the same GPU, each side's fastest of two runs, 1024², seed 42.
Sampling only (the sampler's first to last step, stamped as printed), turbo against ComfyUI:

| Engine | Mode | turbo | ComfyUI | turbo vs ComfyUI | images |
|---|---|---|---|---|---|
| comfy-flux2-4b | text to image | 4.69 s | 5.64 s | −16% | 47.8 dB |
| comfy-flux2-9b-distilled | text to image | 3.04 s | 3.31 s | −8% | 38.3 dB |
| comfy-flux3-6b | text to image | 3.75 s | 3.80 s | −1% | 42.0 dB |
| comfy-krea2-turbo | text to image | 8.50 s | 8.54 s | 0% | 28.0 dB |
| comfy-krea2-turbo-edit-identity | image to image | 13.48 s | 13.93 s | −3% | 27.7 dB |
| comfy-krea2-turbo-edit-general | image to image | 15.59 s | 14.40 s | +8% | 34.9 dB |
| comfy-qwen-image-2-1-turbo | text to image | 3.72 s | 3.76 s | −1% | 43.7 dB |
| comfy-qwen-image-2-1-turbo | image to image | 3.84 s | 3.93 s | −2% | (sizes differed) |
| comfy-z-image-turbo | text to image | 5.92 s | 6.26 s | −5% | 47.6 dB |
| comfy-moss-image | text to image | 9.17 s | 9.28 s | −1% | 40.4 dB |
| comfy-moss-image | image to image | 11.17 s | 11.13 s | 0% | 30.8 dB |
| comfy-minimax-h3-image | text to image | 5.22 s | 5.19 s | +1% | identical |
| comfy-minimax-h3-image | image to image | 8.81 s | 8.76 s | +1% | identical |

MiniMax-H3-Image, added after on the 224 GiB container (`modal.sh run --big`), is made for 4 to
20 MP: at 2048² sampling took 8.07 s against 8.11 s (edit 12.29 against 12.32), at 4096² (16 MP)
19.30 s against 19.32 s (edit 25.94 against 26.53), the images identical to ComfyUI's at every
size. From launch to image turbo is ahead: 22.3 s against 23.1 s at 1024², 26.5 against 27.2 at
2048², 42.1 against 43.5 at 4096² (edits 28.3 / 31.9 / 52.1 against 29.3 / 32.6 / 54.6). At
2048² both sides' text to image move by one level on a few pixels between a cold and a warm run
(ComfyUI's VAE sizes its tiles to the free VRAM), the same on both.

Turbo-only on the same run (no ComfyUI graph, or a stock engine): comfy-flux2-9b 5.99 s (edit
6.53), comfy-flux2-4b edit 5.73, comfy-flux2-9b-distilled edit 3.57, comfy-flux3-6b edit 5.18,
comfy-qwen-image-2-1 7.90 (edit 8.92), flux2-4b 4.52 (edit 4.57), flux2-9b 4.90 (edit 5.60),
flux3-6b 7.76 (edit 11.94), z-image-turbo 20.05. MOSS-Image, added after on another L40S:
moss-image 33.36 (50 steps, CFG 3). MiniMax-H3-Image on the 224 GiB container, its 66 GB bf16
transformer streamed through the card every step: minimax-h3-image 28.12 (edit 34.31) at 1024²,
rendered on a 4 MP canvas as diffusers does, 23.25 (33.56) at 2048², 86.37 (119.66) at 4096², a
single cold run.

Read with care:

- turbo is at parity or faster everywhere. The Krea 2 general edit's +8% above was noise: three
  runs a side on a fresh host put it at −4% (14.53 s against 15.19 s; to the saved image 16.04
  against 16.60), the identity edit at −2%, the same images. A container's runs spread ±0.6 s
  (about 4%), and turbo's first run there is 5 to 7 s slower, reading the weights off the Volume
  that ComfyUI then finds cached: compare with `--repeat 2` or more, which now swaps which side
  goes first. With `--repeat 1` turbo always goes first, and the cold read grows with the files:
  comfy-minimax-h3-image at 4096² (49 GB) sampled 23.27 s against 18.90 s (+23%) that way, the
  first step streaming its weights off the Volume, and 19.30 s against 19.32 s with two repeats.
  Cold against cold, each side alone in a fresh container, turbo starts faster:
  comfy-minimax-h3-image at 1024² reached its first step in 31.5 s against 33.9 s, its image in
  45.8 s against 48.4 s.
- From launch to image, timed alike (ComfyUI from its server's launch, turbo to its `Saved:`
  line), turbo is ahead: comfy-moss-image 20.3 s against 22.6 s, edit 25.0 to 25.2 s against 26.3
  to 26.7 s. Only the first run, turbo's, reading the files cold, is behind (29.6 s against 28.4
  s). The earlier harness timed ComfyUI from its server's readiness and turbo to its process's
  exit, which put turbo 6 to 10 s behind.
- The same engine on two L40S hosts at once sampled 2× apart (comfy-flux2-4b: 4.69 / 5.64 s and
  2.25 / 2.57 s), the ratio holding (−16%, −12%): on Modal, compare turbo with ComfyUI in one
  container, not a time with another host's.
- CUDA runs ahead of the step lines, so the end of sampling can land in the decode that follows
  (on the A1000, 1.8 s of comfy-flux2-4b's sampling was still queued at its `4/4` line); sampling
  to the saved image would be the sturdier measure.
- Images sit further from ComfyUI's here than on the A1000 for Krea 2 (27.7 to 34.9 dB, 37.4 to
  42.1 there) and FLUX.2 9B distilled (38.3, 48.3 there); the others match. Not yet explained.
- The stock z-image-turbo (diffusers' model on `load_pipe`) samples 3.4× slower than
  comfy-z-image-turbo on the same weights. Not yet explained.
- The Qwen-Image-Edit 2511 family is not part of it.

## The node boundary (`node_teardown`)

- **Each node loads its own model.** ComfyUI's teardown also marks every model's pins inactive
  (`reset_cast_buffers`, `model_management.py:1484`), and only the next node's `load_models_gpu`
  marks its model active again; CLIPTextEncode loads the encoder, KSampler the sampling models
  (`prepare_sampling`). Loading all of them once in `prepare` left the transformer as inactive as
  the encoder while sampling, so on the 4 GB card (Qwen-Image 2.1, 1024×768) it held 4.1-6.0 of its
  7.26 GB pinned and swapped pins between its own blocks every step, where ComfyUI pins all of it
  and trims the idle encoder to 5.3 GB. Now `prepare` loads the sampling models, the encode
  boundary loads the encoder first and the sampling models again after its teardown: the same pins
  as ComfyUI, same md5 on flux2-4b and comfy-krea2-turbo.

- **z-image was non-deterministic at a fixed seed — fixed by honouring ComfyUI's node boundary.**
  Two runs, same seed, differed (meandiff ~6–14/255: chaotic amplification of a small drift over 8
  steps, not different noise). Both `z-image-turbo` and `comfy-z-image-turbo`, never krea2 (fp8
  quant route), only at 512²+ (needs VBAR eviction pressure), diverging at **forward 3** — the
  first eviction. Pre-dates the krea2 work (present at `82508b2`).

  Root cause: **ComfyUI tears down the offload state after every node**
  (`comfy/execution.py:559-563`: `cleanup_prefetch_queues` + `reset_cast_buffers` +
  `vbars_reset_watermark_limits`, in that order), so CLIPTextEncode's per-stream cast buffers,
  prefetch queues and VBAR watermarks are gone before KSampler starts. A diffusers pipeline has
  **no node boundary**: `encode_prompt` flows straight into the denoise loop, so the text encoder's
  stream/buffer state leaked into the transformer's forwards: with 2 async offload streams that
  raced (non-determinism) **and** starved the DiT of VRAM the encoder still held.

  Fix: `node_teardown()` -- the same three comfy calls, verbatim -- runs after `encode_prompt`
  (installed in `_finalize_pipe`, under the encode cache so a cache hit, like a cached comfy node,
  skips it) and in `reclaim()`. Result at 512²: both z-image engines **bit-identical across runs**
  (3/3, same md5) and **faster** -- 1.83 → ~1.4 s/it -- with the default 2 streams kept. Clamping
  `--async-offload` to 1/0 also cured it (+37/46% cost), which is how the race was pinned before
  the real boundary was found. Ruled out on the way, each by test: `install_prefetch`,
  `use_comfy_attention`, cuDNN attention, `_assign_sd` vs `load_state_dict(assign=True)`, VBAR
  itself (comfy is deterministic on it), seeding, the text encoder (conditioning hash identical
  across runs), `_rmsnorm_matches_fused`, stochastic rounding, `cudnn.benchmark`, and the
  cast/uncast pairing (measured: 2460/2460 Linear, 1785/1785 RMSNorm).

- **The sampler→VAE boundary (`install_tiled_vae_fallback`) — krea2 1600×1200 OOM'd + aborted.**
  Same bug class at the pipeline's other seam: ComfyUI's KSampler node ENDS before VAEDecode, so
  the teardown has run and `VAE.decode` then hands its working-memory estimate to
  `load_models_gpu(memory_required=memory_used_decode(...))` (comfy/sd.py:1369-1371), whose guts
  are `free_memory(max(inference_memory, memory_required + extra_reserved_memory()))`
  (model_management.py:944,1001) — the DiT is evicted through free_memory's CONTROLLED path before
  the VAE's first kernel. A diffusers pipeline calls `vae.decode` inside the same `__call__`: the
  DiT stayed pinned (measured: 2.66 GB held, 0 free) and a 1600×1200 decode died after all 8 steps
  — and both halfway fixes crash: without the pre-free, the OOM poisons the context and even the
  tiled retry / tensor deallocs abort (C++ throw in a destructor → terminate); with only the
  watermark reset, the first cudnn workspace request evicts on demand INSIDE the allocator and
  aborts on the VAE's first conv3d. Fix, comfy verbatim: at `vae.decode`/`encode`,
  `node_teardown()` + the `memory_used` estimate (comfy's own Wan-2.1 / AutoencoderKL constants,
  sd.py:886-887 / :505-506, keyed only off tensor rank) + `free_memory`, then comfy's OOM fallback
  (sd.py:1403-1410: `raise_non_oom`, warn, flag-retry OUTSIDE the except so the exception's tensor
  refs can gc, tiled via diffusers' `enable_tiling`: the same 256px/64px tile comfy starts from,
  sd.py:1426-1427, though since v0.39 comfy then resizes a 3D tile to its VRAM budget). Result:
  krea2 1600×1200 decodes clean — untiled, the fallback never even fires — and z-image stays
  bit-identical. A VAE may carry its own estimates, as comfy's VAE class does
  (`memory_used_decode` / `memory_used_encode(shape, dtype)`). When even that cannot be freed, the
  call tiles up front: the one deviation from comfy, which only tiles after an OOM, because
  running a diffusers VAE into a sure OOM can end in the uncatchable abort above (measured on a
  4 GB card). An engine can also opt in to ComfyUI's own VAE, `comfy_vae`: comfy's model
  (`comfy/ldm/*`, vendored verbatim) built and run as sd.py does, mirrored in `adapter.py` (see
  `resync.md`), under ComfyUI's model management, so it decodes what ComfyUI decodes, as
  fast. It loads through `load_models_gpu` as a dynamic patcher, staged in RAM rather than kept
  on the GPU, so on a small card the transformer keeps that room; a resident VAE decoded 2.5x
  slower there (Krea 2, 2.37 s against ComfyUI's 0.91 s). comfy-qwen-image-2-1's Wan 2.2 decoder
  runs a single image in row strips, exact and in a quarter of diffusers' memory (1.4 against
  5.5 GB at 1024×768), where diffusers' tiles had bent the alpha along their seams. On the 4 GB
  card at 1024×768, warm: Krea 2 0.95 s (ComfyUI 0.91 s, diffusers tiled 3.4 s), the 2511 edit
  0.72 s (diffusers 2.9 s), Qwen-Image 2.1 1.55-1.75 s (ComfyUI 1.61 s); on an L4 at 1024²,
  Qwen-Image 2.1 0.73 s (ComfyUI 0.70 s).

## Notes

- **MPS direct-to-device load.** On MPS (unified memory) `load_pipe` copies the transformer onto
  the device at load: ComfyUI's `load_torch_file` maps the file on the CPU, each tensor is copied
  once onto MPS, then `load_state_dict(assign=True)` — how ComfyUI places a model it builds on
  Apple Silicon — instead of diffusers' CPU load followed by `load_models_gpu`'s per-leaf
  CPU→device copy; `load_models_gpu` then no-ops on the resident weights. Halves placement
  (flux2-4B ~103s → ~51s), flipping offloader one-shot from ~70s slower than a plain `.to("mps")`
  to ~1min faster (also ~15% faster diffusion from the comfy operator swaps → ~21% total).
  MPS-only: CUDA's separate VRAM pool needs `load_models_gpu`'s partial-residency / VBAR decisions
  (a full direct load would OOM), CPU already loads there. Model- agnostic (globs component
  shards); LoRA-safe (patches still merge on top via `patch_weight_to_ device`, verified
  bit-identical); gated off `manual_cast`; any mismatch falls back to normal placement.
  `assign=True` breaks tied weights (Qwen3 `lm_head`), so it re-ties after — on MPS the Qwen3
  text encoder now lands on CPU (next note), so only the transformer takes this direct-load path.
  ComfyUI's own models (`load_pipe_comfy`) load as sd.py loads them; on MPS they are untested.
- **Text encoder on CPU on MPS (ComfyUI's `text_encoder_device()`).** ComfyUI runs the text encoder
  on `text_encoder_device()` — CPU under `vram_state` SHARED (Apple Silicon), the compute device
  otherwise — keeping only the transformer resident on the compute device and moving just the
  conditioning across. `load_pipe` honors that selector for the resident (non-streamed) path:
  `te_dev = text_encoder_device()`. When it lands the TE off the compute device (MPS → CPU) the TE
  patcher is built on `te_dev`, `encode_prompt` runs there (native forward) and only the small
  embeddings move to the compute device — cast to the compute dtype (`manual_cast`, else the
  model dtype, never the storage one fp8 weights keep), as ComfyUI's `_apply_model` casts the
  conditioning (a streamed TE keeps its file's bf16, and flux2 takes its
  latent dtype from `prompt_embeds`, which then mismatched the fp16 VAE) — and the pipeline's
  `_execution_device` is pinned to the compute device (per-instance subclass) so
  timesteps/latents are built to match the transformer.
  Gated on `te_dev != load_device`, so the streamed paths (VBAR / CPU-stream, TE placed their own
  way) and plain CPU (`text_encoder_device()` → CPU == `load_device`) are unchanged; the effect is
  MPS. ComfyUI's own text encoders (`load_pipe_comfy`) sit where their CLIPLoader puts them.
  Was materialising both transformer + TE on MPS (~15GB); now the TE is CPU-resident.
  Measured on an 8GB M1 (512×512): MPS residency ~15 → 7GB, peak swap 12.8 → 5.3GB, diffusion
  2:26 → 1:46, total 247 → 188s.
- **Streaming is mmap file-sliced + pinned (comfy-aimdo VBAR).** Every big model has its weights
  swapped for mmap file-slices (`load_streamed` / `assign_streamed_weights`), then wrapped in
  ComfyUI's `ModelPatcherDynamic`. The transformer is meta-loaded first (`load_streamed`), so it
  never materialises in RAM; the text encoder is built by `from_pretrained` and then swapped, so
  it briefly peaks at ~1× its size before the slices replace it. comfy-aimdo pins the slices in its
  host buffers, so per-forward streaming is **pinned** (fast async HtoD); the slices come from the
  page cache when the model fits
  RAM, or straight from disk when it doesn't (qwen-image-edit ~55GB). VBAR also sizes partial GPU
  residency from live free VRAM, so a model larger than VRAM runs on any card. A residual gap vs
  ComfyUI on some engines is diffusers' unfused-qkv compute, not the offloader -- but **do not
  reach for that explanation before counting tokens.** comfy-krea2-turbo was 32% off ComfyUI and
  "diffusers compute" was the assumed cause; it was wrong. diffusers pipelines may pad the text
  sequence to a fixed width where ComfyUI never pads, and the DiT then chews the padding in every
  block (krea2: 1536 tokens vs ~1054, the whole gap -- see `install_unpadded_encode`). Check
  `prompt_embeds.shape[1]` against the mask's `.sum()` first; it costs a print, and per-step time
  is linear in the sequence length. The attention mask such padding forces is a symptom, not a
  cause.
- **Single files (ComfyUI-reuse engines).** `load_pipe_comfy` loads a ComfyUI install's split
  single files rather than a diffusers component dir, each big model through its adapter
  (`adapter.load_comfy_single_file` for the transformer, `adapter.CLIP` for the text encoder),
  the file mmapped and its weights file slices, loaded as they are. The VAE, scheduler and tokenizer come from the scaffold, or the
  VAE is ComfyUI's own (`comfy_vae`), handed in as a prebuilt `component`. It shares both ends with
  `load_pipe`: the `_prepare_offload` setup (device / CPU comfy-vs-stream mode / dtype / VBAR /
  builder) and the `_finalize_pipe` tail (VAE placement, execution device, encode bridge/cache),
  the CPU full-load fit sized from the single files.
- **Pinning matches ComfyUI (measured).** comfy-aimdo pins the streamed working set lazily per
  module up to `MAX_PINNED_MEMORY` (40% RAM on Windows), degrading via `_steal_pin` past that; the
  weight VBAR survives `reset_cast_buffers` between gens, so there is **no per-gen re-fault**.
  Probed on the A1000: full budget pins 12.33/13.6GB; capped to a 16GB-Turing-like 6GB budget it
  pins 5.49GB (flux2) / 4.94GB (z-image) with gen2/gen3 first-step == steady step (no re-fault).
  turbo drives the same vendored path ComfyUI does — see
  [`doc/COMFYUI_OFFLOAD_MAP.md`](doc/COMFYUI_OFFLOAD_MAP.md). (An older note claimed turbo
  under-pinned ~4.7/6.4GB with a ~30s re-fault; stale — that was the materialized era.) These
  figures predate the v0.39 re-sync: since v0.39, off a fast disk (`fast_disk` False, as on the
  A1000) comfy also pins VBAR-backed weights, so re-measure before quoting them.
- **Why not the static ModelPatcher.** The static lowvram path pins the full model up front
  (`model_management`: `total_pins_required += model_memory()`, gated on `not is_dynamic()`) and is
  a touch faster per step, but its per-layer peak OOMs a small card on a big model — e.g. z-image's
  12GB DiT crashes on 4GB VRAM where VBAR streams it fine. VBAR exists precisely for that, so it
  stays the default; static is not agnostic.
- **Regression this fixed (materialized).** commit `25dbb6c` materialized fits-RAM components via
  `from_pretrained` instead of mmap; materialized host tensors stream **pageable** (~2x slower per
  step). Reverting to always-mmap restores pinned streaming. Measured on an RTX A1000 (4GB VRAM /
  33GB RAM): flux2 1024×768 ~4 → ~2.3 s/step; z-image ~5.6 → ~5 s/step, both fit.
- **comfy-kitchen** is ComfyUI's kernel library (fused RoPE, fp8/fp4 quant), unrelated to
  offloading itself. It is pip-installed, not vendored, at the version the ComfyUI tag pins: an
  older one makes v0.39's `quant_ops.py` import block fail as a whole, which silently drops fp8
  and the kitchen RoPE.
- **Re-syncing** a newer ComfyUI: re-copy the files and re-apply the short edit set in
  `resync.md`; bump the commit pins there, in `comfy/__init__.py`, and in `README.md`.
- **v0.39 re-sync (v0.27.0 → v0.39.1, comfy-aimdo 0.4.10 → 0.5.5, comfy-kitchen 0.2.16 →
  0.2.37).** Besides the re-copy, upstream behavior that reaches the adapter was mirrored:
  `pinned_memory`'s per-subset pin state (`install_pin_rollback_guard`); the quant formats a
  device can't run now come from `ops.get_disabled_quant_formats`; the SDPA wrapper reads
  `ops.SDPA_BACKEND_PRIORITY` (flash first, any OS) and does comfy's GQA repeat; `node_teardown`
  runs `cleanup_prefetch_queues` first, as `execution.py` now does; aimdo inits with
  `nvml_pressure=True` (ComfyUI's default); `ModelPatcherDynamic` gets comfy's per-model
  `fast_disk` (`comfy.storage.model_fast_disk` over the files each loader streams; False on the
  A1000 laptop, as ComfyUI decides there); and `prepare`/`reclaim` drive comfy's
  `PromptModelTracker`, so pin eviction spares the running generation's models. Measured on the
  A1000 at 512² (seed 42, interleaved old/new runs, cool-ish starts): flux2-4b CUDA and CPU and
  comfy-krea2-turbo CUDA give **bit-identical images** across the two stacks, and per-step time is
  at parity (krea2 ~2.2 s/it both, flux2 ~12 s generation both, flux2 CPU ~25 s/step both).
  z-image stays bit-identical across runs. The runtime then moved to ComfyUI's own torch
  (2.12.1 → **2.14.1**+cu130, torchvision 0.29.1; new kernels, so new reference images) so the
  comparison below runs on the same torch. Against ComfyUI v0.39.1 (same files and graph from
  its templates, fresh process each, interleaved, ms phase logs) turbo is on par or faster end to
  end, images matching visually. comfy-flux2-4b's sampling first trailed by ~2 s, all in step 1:
  profiling ComfyUI's and turbo's first forward side by side showed 60 extra weight copies on
  turbo's side, the q/k/v weights diffusers' converter cuts out of ComfyUI's fused qkv with
  `torch.chunk`. Those views are not whole file slices, so comfy's cast rejected its direct
  file→device read (`read_tensor_file_slice_into`) and paged them through the mmap instead.
  `_own_file_slice` rebuilds each such view the way comfy's `load_safetensors` builds every
  tensor (own buffer over the mmap + own `TensorFileSlice`), so they take comfy's own fast path:
  same bytes (image md5 unchanged), and turbo now samples faster than ComfyUI (7.0/10.1 vs
  8.8/11.3 s; end to end 33.9/38.2 vs 35.7/39.8 s). comfy-krea2-turbo's convert only renames, so
  it was unaffected and stays at parity (57.8 vs 58.6 s). (`_own_file_slice` went with the
  converters, once the comfy engines ran ComfyUI's own models on their files as they are.) flux2-4b from the diffusers repo trails
  by a further ~2-4 s for the same reason: a plain sequential read of its text-encoder file is ~2 s
  slower than ComfyUI's byte-identical copy here (disk/page cache, not code). Profiling it found
  a real leak, fixed: `assign_streamed_weights` now re-ties tied weights (Qwen3's `lm_head`),
  which otherwise kept `from_pretrained`'s ~8 GB load buffer alive (flux2-4b peak private memory
  29.9 → 22.4 GB, image unchanged). Upstream's malloc graph for prefetch was tried and dropped:
  ComfyUI only enables it for forwards it owns (never flux2/krea2), and inside a diffusers block
  it aborts on a CUDA sync during recording.
- **v0.41 re-sync (v0.39.1 → v0.41.2, comfy-aimdo 0.5.5 → 0.5.6, comfy-kitchen stays 0.2.37,
  torch unchanged), for FLUX.3 Image Klein.** Five vendored files changed upstream (`ops.py`,
  `model_patcher.py`, `lora.py`, `cli_args.py`, `weight_adapter/bypass.py`) and one joins the
  copy, `configurable.py`, which `model_patcher.py` now imports. Mirrored: a file's attention
  config may be a list (the first method that runs wins) and may name comfy-kitchen's `sol`
  attention besides int8 (then `use_comfy_attention_config`, since gone with the files that
  used it running as ComfyUI's own models), and the SDPA wrapper copies a strided K/V that would
  spill the L2 cache first (AMD gfx1151 only). The other mirrored lines only moved; their
  citations follow. On the A1000 at 512² (seed 42), flux2-4b CUDA and CPU and
  comfy-krea2-turbo CUDA give the same md5 on both stacks. One caution from that check:
  comfy-krea2-turbo's image changed when it ran right after another engine in the same sequence,
  on either stack, so compare its runs under the same conditions.
- **v0.45 re-sync (v0.41.2 → v0.45.4, comfy-aimdo 0.5.6 → 0.6.0, comfy-kitchen 0.2.37 →
  0.2.41, torch unchanged).** Ten vendored files changed upstream (`ops.py`, `model_patcher.py`,
  `model_management.py`, `model_prefetch.py`, `memory_management.py`, `pinned_memory.py`,
  `utils.py`, `lora.py`, `cli_args.py`, `ldm/modules/diffusionmodules/model.py`) and one joins
  the copy, `anonymous_file.py`, which `utils.py` now imports. Mirrored: `ComfyVAE` hands its
  model its patcher (`current_patcher`), as sd.py's VAE now does. The writable models sampling
  now runs under, which only ComfyUI's Qwen-Image 2.1 uses, came with its transformer (below).
  The other mirrored lines only moved; their citations follow. On the A1000 at 512² (seed
  42, old and new stacks interleaved twice), flux2-4b CUDA and CPU give the same md5 on both.
  comfy-krea2-turbo gave two images on the old stack alone, the new stack's one among them, the
  run-order caution above: under the same conditions both stacks match. Those runs were on v0.45.0;
  v0.45.1 to v0.45.4 change nothing the offloader copies or mirrors (partner nodes only).
- **ComfyUI's own models: Qwen-Image 2.1, FLUX.3, Krea 2, FLUX.2, Z-Image, Qwen-Image and
  MOSS-Image (v0.45.4).** An engine names ComfyUI's model instead of building a diffusers one
  (`transformer["model"]`), and its adapter runs ComfyUI's transformer (vendored with what it
  imports, resync.md) behind the call the diffusers pipeline makes, on `adapter.BaseModel`,
  ComfyUI's BaseModel for what every model shares. The engines lost their hand-copied forwards
  and key remaps, and `comfy_api()` and `use_comfy_attention_config` went with them. For
  Qwen-Image 2.1 (`qwen_image21.py`) the packed tokens become ComfyUI's latents, the reference's
  vision tokens leave the text as `image_slots`, and the prefix K/V cache, its v0.45 writable slots
  and the per-block prefetch are ComfyUI's own; the run ends before the VAE decodes, as KSampler
  ends before VAEDecode, so the cache is gone by then. For FLUX.3 (`flux3.py`) the grids come off
  the pipeline's ids and a flow map's step target goes in as ComfyUI's `sample_sigmas`. Krea 2's file loads as
  it is (430/430), and its edits run the comfyui-krea2edit node pack's forward, pixel fit and
  grounded encode. FLUX.2 (`flux2.py`) is ComfyUI's Flux with model_base.Flux2's conditions (the
  text left-padded to 512, references by index); Z-Image (`lumina2.py`) is ComfyUI's NextDiT, the
  diffusers call's 1 - sigma and negated output mapped onto ComfyUI's. Qwen-Image (`qwen_image.py`)
  runs Qwen-Image-Edit 2511 as TextEncodeQwenImageEditPlus does, on the stock text-to-image
  pipeline: its references go to the transformer through `edit`; it has not run yet (its files
  are not on the test machine). MOSS-Image (`moss_image.py`) runs behind the official
  MOSS-Image pipeline's call, its uncond's skipped block as SkipLayerGuidanceDiTSimple replaces
  it, an edit's reference keys and values computed once as extra_conds does, and samples
  through ComfyUI's own KSampler (`comfy_sampler`): its template's er_sde draws noise every
  step, which no diffusers scheduler reproduces. Its VAE is ComfyUI's own too (`comfy_vae`):
  an edit turned a 1% difference in its reference (diffusers' VAE in bf16 against ComfyUI's
  in fp16) into another picture. MiniMax-H3-Image (`minimax_h3_image.py`) has no diffusers call
  to stand behind (diffusers' H3 transformer takes packed rows and position ids, ComfyUI's packs
  them itself), so its engine runs a short loop on ComfyUI's call: `extra_conds` encodes an edit's
  reference (after the text encoder: the H3 VAE sizes its tiles to the free VRAM, and only that
  order gives ComfyUI's images exactly), refines the text and builds the payload once, as
  model_base does, and each step passes the sampler's sigmas, from
  which the PDD heads pick theirs. Its sampler runs on the grid in the file's metadata
  (`sampling`, `FlowSampler(sigma=)`), and its VAE, ComfyUI's own, tiles itself
  (`handles_tiling`). A stock repo may keep its diffusers transformer elsewhere than
  `transformer/` (MiniMax-H3-Image's `transformer_diffusers`): the pipeline names it
  (`transformer_folder`) and `load_pipe` streams it from there. `load_comfy_single_file`
  archives the model's dtypes as it builds it, as model_base does (model_base.py:197): the patcher
  loads a file's tensor in its layer's archived dtype, so a float32 bias in a bf16 layer is bf16,
  as in ComfyUI. Without it MiniMax-H3-Image's float32 biases stayed float32, a 2e-3 difference
  its 6 steps turned into another picture (20 dB against ComfyUI, bit-identical with it).
  `load_streamed` keeps a diffusers model's `_keep_in_fp32_modules` in float32, as
  `from_pretrained` does. `install_prefetch` takes a top-level `ModuleList` of blocks only (each
  with layers of its own): MiniMax-H3-Image's six PDD heads, bare Linears of which a step runs
  one, were prefetched as a sequence, so the head a step ran raced its copy and the image changed
  between runs. Every text encoder is ComfyUI's own too (`text_encoder["model"]`,
  `adapter.CLIP` mirroring sd.py's): the conditioning is ComfyUI's exactly, where transformers'
  Qwen3-VL and its image processing put a grounded prompt 25-125% off and dropped the face from an
  identity edit. A diffusers VAE gets its input in its own dtype, as ComfyUI's VAE casts it, since
  ComfyUI's models sample in float32. `load_pipe_comfy` takes nothing else any more:
  `stream_single_file`, `load_quant_single_file`, `wire_streamed`, `keep_declared_fp32`, the
  engines' `convert` and `_own_file_slice` went; FLUX's QK norm fix runs as its
  `process_unet_state_dict` does (`process`).
  **`load_pipe` and its diffusers-side helpers stay** (`load_streamed`, `comfy_ize`,
  `install_prefetch`, `use_comfy_attention`, `use_kitchen_rope`, ...): turboCLI's stock engines
  run diffusers' own models through them: the published models at full precision for powerful
  hardware, and no dependency on ComfyUI. ComfyUI's models are for its `comfy-*` engines, fast on
  a reasonable computer; never move a stock engine onto them to drop that path.
- **Against ComfyUI, on the A1000 (seed 42, interleaved).** ComfyUI's progress bar moves its
  start back by one step after the second (`model_trange`, utils.py:1299-1319), so its rate
  understates the later steps: real sampling time is its first step plus `elapsed_N -
  elapsed_2 / 2`. Sampling times, turbo against ComfyUI: Qwen-Image-2.1-Turbo at 1024² edits in
  38-40 s (42.5-43.5 s; the engine's own forward took 57-60 s) and generates in 29-33 s
  (28-31.5 s); FLUX.3 Image Klein SD at 1024² in 32-33 s (31-32 s); Krea 2 Turbo at 512² in
  31-34 s (30-32.5 s, 31.75 against 31.5 on average), an identity edit in 84 s (89.5 s), a
  general edit in 75 s (75.5 s); FLUX.2 Klein 4B at 1024² in 13 s (21 s), Klein 9B distilled in
  25-32 s (25.5-30.5 s); Z-Image Turbo at 1024² in 54-58 s (52.5-58 s). Images from ComfyUI's,
  with ComfyUI's text encoders: Qwen-Image 2.1 43.5 dB and edit 48.6, FLUX.3 44.5, Krea 2 42.1
  and identity edit 37.4, FLUX.2 4B 48.5 and 9B 48.3, Z-Image 49.7 (on euler, the sampler of
  diffusers' scheduler, in place of the template's res_multistep).
- **License.** ComfyUI is GPLv3; the vendored copies live in this already-GPLv3 package.
