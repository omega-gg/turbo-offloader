# offloader (v2)

GPL offload backend for the turboCLI diffusion runner. Instead of cherry-picking lines from ComfyUI
(v1, see `doc/aimdo_v1.md`), v2 **vendors ComfyUI's offloading subsystem verbatim** and adds a
**thin adapter** so turboCLI's diffusers pipelines reuse ComfyUI's own memory manager with 1:1
parity. The same code runs a model larger than VRAM by streaming its weights to VRAM per forward
with partial GPU residency — from host RAM when the model fits (ComfyUI's UNet offload), else
disk→VRAM via comfy-aimdo file-slices for a model larger than RAM.

**Style:** code and comments wrap at 99 columns.

## Layout

```
offloader/
  __init__.py   backend seam the runner drives (pre_torch_init/available/supports/load_pipe/...)
  adapter.py    the thin middleman -- the only real logic in the package
  comfy/        byte-for-byte ComfyUI offloading snapshot (see comfy/resync.md)
```

`offloader/comfy/` is a pristine mirror of ComfyUI's `model_management.py`, `model_patcher.py`,
`ops.py`, `memory_management.py`, `model_prefetch.py`, `lora.py`, `utils.py`, etc. Aliased to the
top-level `comfy` package via a `sys.modules` entry in `comfy/__init__.py`, so the vendored
`import comfy.X` lines resolve unchanged. The **only** edits over upstream (all documented in
`comfy/resync.md`): three commented-out off-path imports, `try/except` shims for the optional
`comfy_aimdo` / `comfy_kitchen`, and a stubless `cli_args`+`options` (vendored, parses no argv).

## The seam (`__init__.py`)

The runner discovers `backend/<mode>/` and drives it through this interface only:

| fn | v2 behavior |
|---|---|
| `pre_torch_init()` | init comfy-aimdo's global CUDA hooks **before torch import** (its allocator can't hook afterwards); imports nothing that pulls torch. Optional -- absent/failed → native path. |
| `available()` | True once the vendored offloader imports (any device) |
| `supports(engine)` | `True` -- model-agnostic; offload eligibility is a turboCLI-side call |
| `load_pipe(model, dtype, pipeline_cls, transformer_cls, device, lora_files)` | build a fully-placed diffusers pipeline (below); runner supplies the classes |
| `load_pipe_single_file(scaffold, files, dtype, pipeline_cls, transformer_cls, device, lora_files)` | same, but the big models stream from ComfyUI's split single files (ComfyUI-reuse engines) instead of a diffusers component dir; scaffold holds the tiny configs/tokenizer/scheduler |
| `prepare(pipe)` | `load_models_gpu(patchers)` -- place managed models on the compute device |
| `reclaim(pipe)` | `free_memory` + `soft_empty_cache` between generations |
| `release(pipe)` | `detach` each patcher |

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
  `comfy_cast_weights`/`weight_function`/`bias_function` attrs ModelPatcher reads.
- **`build_patcher` / `build_dynamic_patcher`** — wrap in `ModelPatcher` / `ModelPatcherDynamic`.
  `make_patchable` shadows diffusers/HF's read-only `.device` property (which ComfyUI assigns to)
  while keeping `str(cls)` identical so transformers' output-capture registry still fires.
- **`keep_uncastable_resident`** — move every non-castable param/buffer (custom norms, pad tokens,
  rotary caches) to the compute device so nothing is stranded on the offload device mid-forward.
  Under manual_cast (below) it also casts float stragglers to the compute dtype, matching ComfyUI's
  params being built in that dtype.
- **`add_lora`** — on-cast LoRA: build the `{lora_key → model_weight_key}` map, hand it to
  `comfy.lora.load_lora` (vendored `weight_adapter`), register via `ModelPatcher.add_patches`. The
  deltas apply while each weight streams in — no fuse, no resident copy.
- **`install_encode_cache`** — wrap the pipeline's `encode_prompt` so a repeated prompt returns
  cached text embeddings and never re-runs (nor re-streams) the encoder — the diffusers analogue of
  ComfyUI's default node cache. Copies `comfy_execution/caching.py::RAMPressureCache`:
  full-input-signature key, embeddings held on CPU, eviction under host-RAM pressure via
  `comfy.model_management.get_free_memory` (below `min(10GB, max(2GB, 10% RAM))`, worst
  `1.3**age × bytes` first). Validated against ComfyUI's own `generate.sh`: identical cold time,
  encode skipped on a repeat prompt in both.

### Compute parity with ComfyUI

Diffusers runs some ops on slower kernels than ComfyUI. Copied from ComfyUI, not reimplemented:

- **`manual_cast_dtype` / `install_manual_cast`** — ComfyUI's storage-vs-compute dtype split
  (`comfy.model_management.unet_manual_cast`). On a GPU without bf16 tensor cores (e.g. Turing
  sm_75) a bf16 checkpoint's matmuls fall off the tensor cores and run ~5× slower than fp16;
  ComfyUI computes such a model in **fp16** while the weights stay bf16. We do the same: storage
  stays bf16 (mmap — no load-time cast, so no RAM blow-up), compute runs fp16. ComfyUI enforces
  this *inside* its native model; the diffusers model doesn't, so we reproduce the same dtype
  discipline from outside via forward hooks, each mirroring a specific comfy cast:
    - input cast ← `model_base.py:207` (`xc = xc.to(dtype)`);
    - **per-leaf input cast** ← `lumina/model.py:826` (`t_embedder(..., dtype=x.dtype)`) — the
      crux: diffusers computes the time-embed/adaLN in fp32, so `norm(x) * scale` promotes to fp32
      and `cast_bias_weight` then casts weights to fp32, dropping every matmul onto the fp32
      `volta_sgemm` path; casting each comfy-ized leaf's input back to the compute dtype forces
      fp16 tensor cores (`s1688gemm`);
    - straggler cast (in `keep_uncastable_resident`) ← lumina params built `dtype=x.dtype`;
    - `clamp_fp16` ← `lumina/model.py:68-71` — guards fp16 overflow (→ NaN → black image) after
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
  computes (overlap; helps when streaming-bound).
- **`use_comfy_attention`** — a verbatim copy of `comfy.ops.scaled_dot_product_attention`
  (`comfy/ops.py:39-64`): on Windows+CUDA it forces the SDPA priority
  `[CUDNN, FLASH, EFFICIENT, MATH]` **per call**, but only for large inputs
  (`q.nelement() >= 1024*128`); small attentions use torch's default backend. We reproduce it
  (rather than import it) because it calls `F.sdpa` internally, so pointing torch's `F.sdpa` at it
  self-recurses; instead we patch `torch.nn.functional.scaled_dot_product_attention` once
  (diffusers calls it by attribute) and delegate to the saved original. **The per-call size gate
  matters:** forcing cuDNN on *every* attention (incl. small inputs, which comfy skips) makes the
  first z-image forward after a flux2→z-image switch pick a nondeterministic cuDNN plan and
  diverge; copying comfy's gate verbatim is deterministic. **Dtype-mismatch coercion
  (`adapter.py:471-490`):** under manual_cast (fp16 compute on a bf16 checkpoint) some native HF
  models leave the attention inputs mismatched -- transformers' qwen3 RoPE (flux2's text encoder)
  promotes q,k to fp32 while v stays fp16 -- and torch's SDPA requires one dtype. We coerce
  mismatched q/k/v to the lowest-precision float present (the compute dtype); ComfyUI never hits
  this because it runs its OWN qwen3, but its attention does the same class of thing (upcasts
  q/k/v, `comfy/ldm/modules/attention.py:244-287`). Model/GPU- agnostic: fires only on a real
  mismatch, no-op on Ampere+ / non-manual-cast.
- **`use_kitchen_rope`** — routes the diffusers transformer's RoPE through comfy-kitchen's fused
  `apply_rope1` (the kernel ComfyUI's `comfy/ldm/flux/math.py` uses), via a `(cos,sin)→freqs_cis`
  shim + a module-scoped patch of `diffusers.models.embeddings.apply_rotary_emb`. Lazy:
  comfy-kitchen is imported only on a matching call, so engines with their own rope (z-image) never
  touch it.
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
- **z-image** (`ZImagePipeline`) — text-to-image, Turbo (few-step). A `comfy-z-image-turbo` variant
  reuses a ComfyUI install's split single files (transformer + Qwen3 TE + VAE) via
  `load_pipe_single_file` — same `TYPE`/pipeline, no re-download.
- **qwen-image-edit** (`QwenImageEditPlusPipeline`) — image-edit; on-cast Lightning 4-step LoRA;
  ~55GB of weights (transformer + Qwen2.5-VL TE). Both components are mmap file-sliced, so their
  weights fault straight from the file — page cache when the working set fits RAM, disk when it
  doesn't — and run where they wouldn't fit VRAM. (The transformer is meta-loaded so it never sits
  in RAM; the text encoder is loaded by `from_pretrained` then swapped for slices, a transient ~1×
  peak.)

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
| flux2-4b | CPU | 512×512 | 4 | ~220 s | stream |
| flux2-4b | CPU | 1024×768 | 4 | ~325 s | stream |
| z-image-turbo (20 GB) | CPU | 512×512 | 8 | ~315 s | stream |

Notes: the A1000 is **Ampere** (sm_86, bf16 tensor cores), so CUDA compute stays bf16 -- no
`manual_cast` (unlike the Turing box above, which runs fp16); with 32 GB RAM both models fit the
page cache, so VBAR streaming is RAM-fed rather than disk-bound. **CPU** per-step *climbs within a
run* as the laptop thermal-throttles the i7-12800H from ~3.6 → ~1.05 GHz over ~5 min (flux2
1024×768: 61 → 90 s/step). Forcing `comfy` CPU mode (fp32 full-load) is slower at high res and
OOM-thrashes z-image (fp32 working set ~48 GB > 32 GB RAM), so the auto-picker correctly stays on
`stream`.

**Machine:** Apple M1 (Mac mini, 8-core) · 8 GB unified RAM · MPS · macOS 15.5 · torch 2.12 ·
Python 3.14.

| Engine (on-disk bf16) | Device | Res | Steps | Wall-clock | Path |
|---|---|---|---|---|---|
| flux2-4b (15 GB) | MPS | 512×512 | 4 | ~185 s (~106 s gen) | direct-load, TE-CPU |
| flux2-4b | MPS | 1024×768 | 4 | ~249 s | direct-load, TE-CPU |

Notes: MPS runs **fp16-resident** (no `manual_cast`) — the transformer is direct-loaded straight
onto the device (`safe_open(device="mps")`) and stays resident; there is no VBAR/stream path (MPS
has no separate VRAM pool). The text encoder runs on **CPU** (ComfyUI's `text_encoder_device()`
under `vram_state` SHARED), so residency is the transformer alone (~7 GB, down from ~15 GB with
both on-device) and only the conditioning crosses to MPS. On this 8 GB box the run still spills to
swap and **that** is the scaling wall, not compute: peak swap ~5.3 GB at 512×512, ~9.7 GB at
1024×768 (both produce clean images — the 1024 corruption seen in image-to-image *edit* is that
mode's larger footprint, not text-to-image). The faithful fp16-on-CPU encode (~50 s) dominates the
non-gen time and is cached (`install_encode_cache`), so a repeated prompt skips it. Output is
bit-deterministic across runs at a fixed seed (verified: two seed-42 runs, identical SHA-256).
z-image-turbo (20 GB) is not benchmarked here — its working set exceeds this 8 GB box's usable
RAM+swap headroom, so it belongs on a larger-RAM Mac.

## Notes

- **MPS direct-to-device load.** On MPS (unified memory) `load_pipe` reads the transformer straight
  onto the device via ComfyUI's `load_torch_file`
  (`safetensors.safe_open(device= "mps")`) + `load_state_dict(assign=True)` — exactly how ComfyUI
  loads on Apple Silicon — instead of diffusers' CPU load followed by `load_models_gpu`'s per-leaf
  CPU→device copy; `load_models_gpu` then no-ops on the resident weights. Halves placement
  (flux2-4B ~103s → ~51s), flipping offloader one-shot from ~70s slower than a plain `.to("mps")`
  to ~1min faster (also ~15% faster diffusion from the comfy operator swaps → ~21% total).
  MPS-only: CUDA's separate VRAM pool needs `load_models_gpu`'s partial-residency / VBAR decisions
  (a full direct load would OOM), CPU already loads there. Model- agnostic (globs component
  shards); LoRA-safe (patches still merge on top via `patch_weight_to_ device`, verified
  bit-identical); gated off `manual_cast`; any mismatch falls back to normal placement.
  `assign=True` breaks tied weights (Qwen3 `lm_head`), so it re-ties after — on MPS the Qwen3
  text encoder now lands on CPU (next note), so only the transformer takes this direct-load path.
- **Text encoder on CPU on MPS (ComfyUI's `text_encoder_device()`).** ComfyUI runs the text encoder
  on `text_encoder_device()` — CPU under `vram_state` SHARED (Apple Silicon), the compute device
  otherwise — keeping only the transformer resident on the compute device and moving just the
  conditioning across. `load_pipe` honors that selector for the resident (non-streamed) path:
  `te_dev = text_encoder_device()`. When it lands the TE off the compute device (MPS → CPU) the TE
  patcher is built on `te_dev`, `encode_prompt` runs there (native forward) and only the small
  embeddings move to the compute device, and the pipeline's `_execution_device` is pinned to the
  compute device (per-instance subclass) so timesteps/latents are built to match the transformer.
  Gated on `te_dev != load_device`, so the streamed paths (VBAR / CPU-stream, TE placed their own
  way) and plain CPU (`text_encoder_device()` → CPU == `load_device`) are unchanged; the effect is
  MPS-only. Was materialising both transformer + TE on MPS (~15GB); now the TE is CPU-resident.
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
  residency from live free VRAM, so a model larger than VRAM runs on any card. The residual gap vs
  ComfyUI on some engines is diffusers' unfused-qkv compute, not the offloader.
- **Single-file streaming (ComfyUI-reuse engines).** `load_pipe_single_file` streams the same way
  from a ComfyUI install's split single files rather than a diffusers component dir:
  `adapter.stream_single_file` meta-loads each big model, mmaps the one safetensors via
  `load_torch_file`, applies an optional key `convert` (the diffusers single-file remap for the
  transformer -- renames + a fused-qkv `torch.chunk` that returns views, so the mmap slices survive;
  a `model.`-prefix strip for the Qwen3 text encoder), then rebinds by name (`_assign_sd`). The VAE
  + scheduler + tokenizer come straight from the scaffold. It shares both ends with `load_pipe`: the
  `_prepare_offload` setup (device / CPU comfy-vs-stream mode / dtype / manual_cast / operations /
  VBAR / builder) and the `_finalize_pipe` tail (VAE placement, execution device, encode
  bridge/cache) — only the weight source differs, plus the CPU full-load fit is sized from the
  single files rather than `cpu_fits_full_load`'s component-dir shards. Verified on the A1000
  (z-image-turbo 1024×768, VBAR): 0 unmatched weights, seed-valid image.
- **Pinning matches ComfyUI (measured).** comfy-aimdo pins the streamed working set lazily per
  module up to `MAX_PINNED_MEMORY` (40% RAM on Windows), degrading via `_steal_pin` past that; the
  weight VBAR survives `reset_cast_buffers` between gens, so there is **no per-gen re-fault**.
  Probed on the A1000: full budget pins 12.33/13.6GB; capped to a 16GB-Turing-like 6GB budget it
  pins 5.49GB (flux2) / 4.94GB (z-image) with gen2/gen3 first-step == steady step (no re-fault).
  turbo drives the same vendored path ComfyUI does — see
  [`doc/COMFYUI_OFFLOAD_MAP.md`](doc/COMFYUI_OFFLOAD_MAP.md). (An older note claimed turbo
  under-pinned ~4.7/6.4GB with a ~30s re-fault; stale — that was the materialized era.)
- **Why not the static ModelPatcher.** The static lowvram path pins the full model up front
  (`model_management`: `total_pins_required += model_memory()`, gated on `not is_dynamic()`) and is
  a touch faster per step, but its per-layer peak OOMs a small card on a big model — e.g. z-image's
  12GB DiT crashes on 4GB VRAM where VBAR streams it fine. VBAR exists precisely for that, so it
  stays the default; static is not agnostic.
- **Regression this fixed (materialized).** commit `25dbb6c` materialized fits-RAM components via
  `from_pretrained` instead of mmap; materialized host tensors stream **pageable** (~2x slower per
  step). Reverting to always-mmap restores pinned streaming. Measured on an RTX A1000 (4GB VRAM /
  33GB RAM): flux2 1024×768 ~4 → ~2.3 s/step; z-image ~5.6 → ~5 s/step, both fit.
- **comfy-kitchen** is a quantization-kernel library, unrelated to offloading — not a dependency.
- **Re-syncing** a newer ComfyUI: re-copy the files and re-apply the short edit set in
  `comfy/resync.md`; bump the commit pins there, in `comfy/__init__.py`, and in `README.md`.
- **License.** ComfyUI is GPLv3; the vendored copies live in this already-GPLv3 package.
