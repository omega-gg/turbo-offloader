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
| `load_pipe_comfy(pipeline_cls, transformer, text_encoder, components, dtype, device, lora_files)` | same, but the big models stream from ComfyUI's split single files (ComfyUI-reuse engines) instead of a diffusers component dir. **Model-agnostic**: the engine passes each big model as a data spec (`{meta, file, convert, quant}`) plus prebuilt small `components` (vae/tokenizer/scheduler); no model classes or names appear here. `quant` routes a scaled-fp8 text encoder through the comfy quant path (below). |
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
- **`add_lora`** — on-cast LoRA: build the `{lora_key → model_weight_key}` map, hand it to
  `comfy.lora.load_lora` (vendored `weight_adapter`), register via `ModelPatcher.add_patches`. The
  deltas apply while each weight streams in — no fuse, no resident copy.
- **`install_unpadded_encode`** — drop padded text tokens from the conditioning before the
  transformer sees them. ComfyUI never pads the modern LLM text encoders
  (`pad_to_max_length=False`: qwen3vl/qwen_image/flux-t5; only CLIP keeps `SDTokenizer`'s default
  `True`, since its 77-token pad is required) and drops an all-ones mask outright
  (`text_encoders/krea2.py:69-70`); diffusers' own qwen pipeline agrees (`if
  prompt_embeds_mask.all(): prompt_embeds_mask = None`). Its **krea2** pipeline is the outlier --
  `padding="max_length"` pads every prompt to a fixed 512 -- so the DiT dragged ~500 dead tokens
  through all 28 blocks: **1536 tokens at 512² where ComfyUI runs ~1054**, which was the entire
  per-step gap (**5.17 → 3.50 s/it**). Model-agnostic: keyed only off the mask the pipeline itself
  returns. Sound because a key-padding mask *means* those tokens are excluded as attention keys,
  so removing them is a no-op for any model that honours it. Gathers (krea2 pads mid-template),
  re-pads ragged batches, and falls through untouched for a pipeline returning a second embeds
  tensor instead of a mask (z-image's `prompt_embeds, negative_prompt_embeds`).
- **`keep_declared_fp32`** — honour a model's own `_keep_in_fp32_modules` after a meta-build +
  `assign=True` load (diffusers only applies it via `from_pretrained`). Without it diffusers'
  `Krea2RMSNorm` runs `F.rms_norm(x.float(), ..., weight=self.weight + 1.0)` with a **bf16**
  weight, so torch refuses the fused kernel ("Cannot dispatch to fused implementation") -- measured
  **2.4× slower per call**. ComfyUI casts its scale to fp32 for the same reason
  (`ldm/krea2/model.py:33`); this also restores exact numeric parity with it (a bf16 `+ 1.0` rounds
  a zero-centered scale), taking the cross-framework cosine 0.99997 → **1.0**.
- **`load_quant_single_file` / `mixed_precision_operations`** — `stream_single_file`'s fp8
  sibling, for any ComfyUI **scaled-fp8** single file — a text encoder (qwen-image-edit's
  `qwen_2.5_vl_7b_fp8_scaled`, ~8.7GB) **or a transformer** (comfy-krea2-turbo's
  `krea2_turbo_fp8_scaled`). `convert_old_quants` reads the fp8 markers from either the classic
  `scaled_fp8` + `scale_weight` keys or the file's `_quantization_metadata`. Meta-build →
  `convert_old_quants` →
  `detect_layer_quantization` → re-class each `Linear` to `comfy.ops.mixed_precision_ops`
  (injecting the `__init__` attrs `_load_quantized_module` reads) → `comfy_ize` the **non-Linear**
  leaves (Embedding/Conv/norms) into the same namespace so they stream too → `load_state_dict`
  (`assign=True`), so each Linear builds a comfy `QuantizedTensor`. The weights stay **fp8** (mmap,
  never materialized) and dequantize per forward — "what comfy does", not a load-time dequant.
  **The `comfy_ize` step is load-bearing:** `_quant_ize` only touches Linears, so without it the
  non-quant leaves — notably the ~1 GB input `Embedding` — would be pinned resident by
  `keep_uncastable_resident`, stealing the transformer's VBAR VRAM budget (~2.5× slower per step on
  the 4 GB A1000; see Benchmarks). With it, the fp8 TE offloads to **0 GB resident**, matching the
  stock bf16 TE.
  **Device-agnostic like the rest:** the `disabled` set is computed from comfy's
  `supports_fp8_compute(load_device)` (& nvfp4/mxfp8), so a device without fp8 tensor cores — CPU,
  MPS, Ampere sm_86 — emulates the dequant to the compute dtype, as ComfyUI does. The fp8 TE loads
  and its forward runs on CPU too (probed in isolation: 0 unexpected keys, `QuantizedTensor`
  weights, forward executing), just slowly. Verified end-to-end on CUDA/Ampere (real image); MPS
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
  computes (overlap; helps when streaming-bound). This is comfy's own mechanism applied to models
  ComfyUI doesn't enable it for: in their codebase only Lightricks/LTXV calls
  `prefetch_queue_pop`; their krea2 forward stalls each block on its weights. It is why we beat
  ComfyUI per-step under memory pressure — parity at 512² (3.50 vs 3.62 s/it, streaming mostly
  hidden either way) growing to ~25% ahead at 1024×768 (6.97 vs ~8.5–9.8 s/it, same-seed
  same-image runs, both cool-started): bigger activations leave less VRAM for weight residency,
  more re-streaming per step, and we hide it behind compute while ComfyUI pays it serially.
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
  touch it. It engages for comfy-krea2-turbo but buys no measurable per-step there (rope is a small
  slice of a 12B DiT; `OFFLOADER_KITCHEN_ROPE=0` leaves the per-step unchanged, 4.96 vs 4.93 s/it).
  Its init is ~0.3 s marginal per process (torch is already imported; an earlier ~16 s reading was
  a cold-cache/thermal artifact -- order-reversed A/B totals are equal, 55 s vs 55 s). Worth
  keeping on regardless of speed: comfy's `apply_rope1` is NOT bit-equivalent to diffusers' native
  rope (measured: different image md5, each path individually deterministic), so it is what keeps
  the output on ComfyUI's exact numerics. The packed freqs_cis is cached by
  (cos, sin) tensor identity — ComfyUI builds `freqs` once per forward and hands it to every block
  (`ldm/krea2/model.py:267`), and diffusers passes one (cos, sin) tuple to all blocks, so the
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
- **z-image** (`ZImagePipeline`) — text-to-image, Turbo (few-step). A `comfy-z-image-turbo` variant
  reuses a ComfyUI install's split single files (transformer + Qwen3 TE + VAE) via
  `load_pipe_single_file` — same `TYPE`/pipeline, no re-download.
- **qwen-image-edit** (`QwenImageEditPlusPipeline`) — image-edit; on-cast Lightning 4-step LoRA;
  ~55GB of weights (transformer + Qwen2.5-VL TE). Both components are mmap file-sliced, so their
  weights fault straight from the file — page cache when the working set fits RAM, disk when it
  doesn't — and run where they wouldn't fit VRAM. (The transformer is meta-loaded so it never sits
  in RAM; the text encoder is loaded by `from_pretrained` then swapped for slices, a transient ~1×
  peak.) A `comfy-qwen-image-edit-2511` variant reuses a ComfyUI install via `load_pipe_comfy`:
  bf16 transformer streamed, **scaled-fp8** TE through the quant path (`load_quant_single_file`),
  VAE reused (WAN→diffusers convert). It is offloader-only (the fp8 quant path needs comfy ops) and
  has a `-lightning` sibling that also reuses ComfyUI's Lightning LoRA. `comfy-krea2-turbo` reuses
  the same seam with **both** big models scaled-fp8 (transformer + Qwen3-VL TE), each spec carrying
  `{"quant": True}`. All model-specifics live engine-side.

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

## The node boundary (`node_teardown`)

- **z-image was non-deterministic at a fixed seed — fixed by honouring ComfyUI's node boundary.**
  Two runs, same seed, differed (meandiff ~6–14/255: chaotic amplification of a small drift over 8
  steps, not different noise). Both `z-image-turbo` and `comfy-z-image-turbo`, never krea2 (fp8
  quant route), only at 512²+ (needs VBAR eviction pressure), diverging at **forward 3** — the
  first eviction. Pre-dates the krea2 work (present at `82508b2`).

  Root cause: **ComfyUI tears down the offload state after every node**
  (`comfy/execution.py:543-549`: `reset_cast_buffers` + `cleanup_prefetch_queues` +
  `vbars_reset_watermark_limits`), so CLIPTextEncode's per-stream cast buffers, prefetch queues and
  VBAR watermarks are gone before KSampler starts. A diffusers pipeline has **no node boundary** --
  `encode_prompt` flows straight into the denoise loop -- so the text encoder's stream/buffer state
  leaked into the transformer's forwards: with 2 async offload streams that raced (non-determinism)
  **and** starved the DiT of VRAM the encoder still held.

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
  `load_models_gpu(memory_required=memory_used_decode(...))` (comfy/sd.py:1057-1058), whose guts
  are `free_memory(max(inference_memory, memory_required + extra_reserved_memory()))`
  (model_management.py:854,911) — the DiT is evicted through free_memory's CONTROLLED path before
  the VAE's first kernel. A diffusers pipeline calls `vae.decode` inside the same `__call__`: the
  DiT stayed pinned (measured: 2.66 GB held, 0 free) and a 1600×1200 decode died after all 8 steps
  — and both halfway fixes crash: without the pre-free, the OOM poisons the context and even the
  tiled retry / tensor deallocs abort (C++ throw in a destructor → terminate); with only the
  watermark reset, the first cudnn workspace request evicts on demand INSIDE the allocator and
  aborts on the VAE's first conv3d. Fix, comfy verbatim: at `vae.decode`/`encode`,
  `node_teardown()` + the `memory_used` estimate (comfy's own Wan-2.1 / AutoencoderKL constants,
  sd.py:757-758 / :481-482, keyed only off tensor rank) + `free_memory`, then comfy's OOM fallback
  (sd.py:1080-1087: `raise_non_oom`, warn, flag-retry OUTSIDE the except so the exception's tensor
  refs can gc, tiled via diffusers' `enable_tiling` — same 256px/64px tiles as comfy's
  `decode_tiled_3d` defaults). Result: krea2 1600×1200 decodes clean — untiled, the fallback never
  even fires — and z-image stays bit-identical.

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
  residency from live free VRAM, so a model larger than VRAM runs on any card. A residual gap vs
  ComfyUI on some engines is diffusers' unfused-qkv compute, not the offloader -- but **do not
  reach for that explanation before counting tokens.** comfy-krea2-turbo was 32% off ComfyUI and
  "diffusers compute" was the assumed cause; it was wrong. diffusers pipelines may pad the text
  sequence to a fixed width where ComfyUI never pads, and the DiT then chews the padding in every
  block (krea2: 1536 tokens vs ~1054, the whole gap -- see `install_unpadded_encode`). Check
  `prompt_embeds.shape[1]` against the mask's `.sum()` first; it costs a print, and per-step time
  is linear in the sequence length. The attention mask such padding forces is a symptom, not a
  cause.
- **Single-file streaming (ComfyUI-reuse engines).** `load_pipe_comfy` streams the same way from a
  ComfyUI install's split single files rather than a diffusers component dir:
  `adapter.stream_single_file` meta-loads each big model, mmaps the one safetensors via
  `load_torch_file`, applies an optional key `convert` (the diffusers single-file remap for the
  transformer -- renames + a fused-qkv `torch.chunk` that returns views, so the mmap slices
  survive; a `model.`-prefix strip for the Qwen3 text encoder, a flat→nested rename for the
  Qwen2.5-VL one), then rebinds by name (`_assign_sd`). A scaled-fp8 model -- transformer or text
  encoder -- takes the quant sibling `load_quant_single_file` (spec `{"quant": True}`) instead. Its
  `convert` runs on the fp8 state dict too (comfy-krea2-turbo remaps the whole ComfyUI-native DiT
  to the diffusers `Krea2Transformer2DModel` layout). The VAE + scheduler +
  tokenizer come straight from the scaffold, or a reused ComfyUI VAE is rebuilt engine-side (qwen's
  WAN-keyed VAE → `convert_wan_vae_to_diffusers`) and handed in as a prebuilt `component`. It
  shares both ends with `load_pipe`: the `_prepare_offload` setup (device / CPU comfy-vs-stream
  mode / dtype / manual_cast / operations / VBAR / builder) and the `_finalize_pipe` tail (VAE
  placement, execution device, encode bridge/cache) — only the weight source differs, plus the CPU
  full-load fit is sized from the single files rather than `cpu_fits_full_load`'s component-dir
  shards. Model-agnostic: every model-specific piece (meta-builders, key remaps, quant flag, the
  reused-VAE rebuild) is data the engine passes in; no names here. Verified on the A1000:
  z-image-turbo 1024×768 (VBAR, 0 unmatched) and comfy-qwen-image-edit-2511 image-to-image (fp8 TE
  emulated), both seed-valid.
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
