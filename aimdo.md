# aimdo (v2)

GPL offload backend for the turboCLI diffusion runner. Instead of cherry-picking lines from
ComfyUI (v1, see `doc/aimdo_v1.md`), v2 **vendors ComfyUI's offloading subsystem verbatim** and adds a
**thin adapter** so turboCLI's diffusers pipelines reuse ComfyUI's own memory manager with 1:1
parity. The same code runs a model larger than VRAM by streaming its weights to VRAM per forward with
partial GPU residency — from host RAM when the model fits (ComfyUI's UNet offload), else disk→VRAM via
comfy-aimdo file-slices for a model larger than RAM.

**Style:** code and comments wrap at 99 columns.

## Layout

```
aimdo/
  __init__.py   backend seam the runner drives (unchanged signatures)
  adapter.py    the thin middleman -- the only real logic in the package
  comfy/        byte-for-byte ComfyUI offloading snapshot (see comfy/resync.md)
```

`aimdo/comfy/` is a pristine mirror of ComfyUI's `model_management.py`, `model_patcher.py`,
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
| `supports(engine)` | flux2 / z-image / qwen-image-edit |
| `load_pipe(model, dtype, engine, device, lora_files)` | build a fully-placed diffusers pipeline (below) |
| `prepare(pipe)` | `load_models_gpu(patchers)` -- place managed models on the compute device |
| `reclaim(pipe)` | `free_memory` + `soft_empty_cache` between generations |
| `release(pipe)` | `detach` each patcher |

All GPL-derived code lives in this package; the calling runner stays GPL-free.

## Two paths, one code

- **Native (device-agnostic: CPU / CUDA / MPS).** Weights load onto the offload device (CPU,
  mmap); a `ModelPatcher` streams them to the compute device per forward via ComfyUI's cast path
  (`cast_bias_weight` → `cast_to`). Needs the model to fit RAM. `comfy_aimdo` not required.
- **VBAR (optional, CUDA).** When comfy-aimdo is present, `load_pipe` flips
  `comfy.memory_management.aimdo_enabled` and uses `ModelPatcherDynamic`, which keeps as much of the
  model GPU-resident as fits (partial residency, sized from live free VRAM) and streams the rest per
  forward. Per component (`adapter.fits_in_ram`, on-disk size vs `get_total_memory`): weights stream
  from **host RAM** when the model fits (fast path — `from_pretrained` tensors, same source as native)
  else **disk→VRAM** via comfy-aimdo file-slices (`load_streamed` / `assign_streamed_weights`) for a
  model larger than RAM. This is ComfyUI's "dynamic VRAM loading" path.

The seam picks VBAR automatically on CUDA-with-comfy-aimdo, native otherwise. `set_device()` maps
the seam's `device=` arg onto `comfy.model_management.cpu_state`, the single knob that switches the
whole device stack.

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
- **`install_encode_cache`** — wrap the pipeline's `encode_prompt` so a repeated prompt returns cached
  text embeddings and never re-runs (nor re-streams) the encoder — the diffusers analogue of ComfyUI's
  default node cache. Copies `comfy_execution/caching.py::RAMPressureCache`: full-input-signature key,
  embeddings held on CPU, eviction under host-RAM pressure via `comfy.model_management.get_free_memory`
  (below `min(10GB, max(2GB, 10% RAM))`, worst `1.3**age × bytes` first). Validated against ComfyUI's
  own `generate.sh`: identical cold time, encode skipped on a repeat prompt in both.

### Compute parity with ComfyUI

Diffusers runs some ops on slower kernels than ComfyUI. Copied from ComfyUI, not reimplemented:

- **`manual_cast_dtype` / `install_manual_cast`** — ComfyUI's storage-vs-compute dtype split
  (`comfy.model_management.unet_manual_cast`). On a GPU without bf16 tensor cores (e.g. Turing sm_75)
  a bf16 checkpoint's matmuls fall off the tensor cores and run ~5× slower than fp16; ComfyUI computes
  such a model in **fp16** while the weights stay bf16. We do the same: storage stays bf16 (mmap — no
  load-time cast, so no RAM blow-up), compute runs fp16. ComfyUI enforces this *inside* its native
  model; the diffusers model doesn't, so we reproduce the same dtype discipline from outside via
  forward hooks, each mirroring a specific comfy cast:
    - input cast ← `model_base.py:207` (`xc = xc.to(dtype)`);
    - **per-leaf input cast** ← `lumina/model.py:826` (`t_embedder(..., dtype=x.dtype)`) — the crux:
      diffusers computes the time-embed/adaLN in fp32, so `norm(x) * scale` promotes to fp32 and
      `cast_bias_weight` then casts weights to fp32, dropping every matmul onto the fp32 `volta_sgemm`
      path; casting each comfy-ized leaf's input back to the compute dtype forces fp16 tensor cores
      (`s1688gemm`);
    - straggler cast (in `keep_uncastable_resident`) ← lumina params built `dtype=x.dtype`;
    - `clamp_fp16` ← `lumina/model.py:68-71` — guards fp16 overflow (→ NaN → black image) after each
      block; without it the output is all-zero;
    - output cast back to the storage dtype (diffusers pipeline latent-dtype contract).
  The per-layer weight cast itself is comfy's own `cast_bias_weight`, unchanged. Applied to both the
  transformer and the text encoder (ComfyUI runs the TE fp16 too — `text_encoder_dtype`). Gated by
  `unet_manual_cast`, so **Ampere+ is unchanged** (returns None → compute stays bf16). z-image 1024×768
  warm: ~10.4 → ~2.3 s/step, matching ComfyUI; output seed-valid.
- **`install_prefetch`** — drives ComfyUI's `comfy.model_prefetch` (`prefetch_queue_pop`) via
  forward hooks on the transformer's block lists, so block N+1's weights stream while block N
  computes (overlap; helps when streaming-bound).
- **`use_comfy_attention`** — a verbatim copy of `comfy.ops.scaled_dot_product_attention`
  (`comfy/ops.py:39-64`): on Windows+CUDA it forces the SDPA priority `[CUDNN, FLASH, EFFICIENT,
  MATH]` **per call**, but only for large inputs (`q.nelement() >= 1024*128`); small attentions use
  torch's default backend. We reproduce it (rather than import it) because it calls `F.sdpa`
  internally, so pointing torch's `F.sdpa` at it self-recurses; instead we patch
  `torch.nn.functional.scaled_dot_product_attention` once (diffusers calls it by attribute) and
  delegate to the saved original. **The per-call size gate matters:** forcing cuDNN on *every*
  attention (incl. small inputs, which comfy skips) makes the first z-image forward after a
  flux2→z-image switch pick a nondeterministic cuDNN plan and diverge; copying comfy's gate verbatim
  is deterministic. **Dtype-mismatch coercion (`adapter.py:471-490`):** under manual_cast (fp16
  compute on a bf16 checkpoint) some native HF models leave the attention inputs mismatched --
  transformers' qwen3 RoPE (flux2's text encoder) promotes q,k to fp32 while v stays fp16 -- and
  torch's SDPA requires one dtype. We coerce mismatched q/k/v to the lowest-precision float present
  (the compute dtype); ComfyUI never hits this because it runs its OWN qwen3, but its attention does
  the same class of thing (upcasts q/k/v, `comfy/ldm/modules/attention.py:244-287`). Model/GPU-
  agnostic: fires only on a real mismatch, no-op on Ampere+ / non-manual-cast.
- **`use_kitchen_rope`** — routes the diffusers transformer's RoPE through comfy-kitchen's fused
  `apply_rope1` (the kernel ComfyUI's `comfy/ldm/flux/math.py` uses), via a `(cos,sin)→freqs_cis`
  shim + a module-scoped patch of `diffusers.models.embeddings.apply_rotary_emb`. Lazy: comfy-kitchen
  is imported only on a matching call, so engines with their own rope (z-image) never touch it.
- **fused RMSNorm** — routing custom norms through `disable_weight_init.RMSNorm` gives ComfyUI's
  fused `F.rms_norm` (≈ 3.5× the eager `mul`/`rsqrt` diffusers/transformers use).

## Engines

- **flux2** (`Flux2KleinPipeline`) — text-to-image and image-input (edit); the pipeline accepts
  `image=`, so the seam is unchanged.
- **z-image** (`ZImagePipeline`) — text-to-image, Turbo (few-step).
- **qwen-image-edit** (`QwenImageEditPlusPipeline`) — image-edit; on-cast Lightning 4-step LoRA;
  ~55GB of weights (transformer + Qwen2.5-VL TE). With enough RAM they stream from RAM; on a small box
  each oversized component streams disk→VRAM via file-slices (the fit gate decides per component), so
  it runs where it wouldn't fit in RAM.

## Notes

- **MPS direct-to-device load.** On MPS (unified memory) `load_pipe` reads the transformer + text
  encoder straight onto the device via ComfyUI's `load_torch_file` (`safetensors.safe_open(device=
  "mps")`) + `load_state_dict(assign=True)` — exactly how ComfyUI loads on Apple Silicon — instead of
  diffusers' CPU load followed by `load_models_gpu`'s per-leaf CPU→device copy; `load_models_gpu` then
  no-ops on the resident weights. Halves placement (flux2-4B ~103s → ~51s), flipping aimdo one-shot
  from ~70s slower than a plain `.to("mps")` to ~1min faster (also ~15% faster diffusion from the
  comfy operator swaps → ~21% total). MPS-only: CUDA's separate VRAM pool needs `load_models_gpu`'s
  partial-residency / VBAR decisions (a full direct load would OOM), CPU already loads there. Model-
  agnostic (globs component shards); LoRA-safe (patches still merge on top via `patch_weight_to_
  device`, verified bit-identical); gated off `manual_cast`; any mismatch falls back to normal
  placement. `assign=True` breaks tied weights (Qwen3 `lm_head`), so it re-ties after.
- **Offloading matches ComfyUI.** Same `ModelPatcherDynamic`, same "N MB Staged / M force-preloaded"
  log, same ~13GB pinned host working set (shown as Windows "shared GPU memory", not a VRAM spill).
  Streaming is ~2s/step from pinned RAM; the residual gap vs ComfyUI on some engines is diffusers'
  unfused-qkv model compute, not the offloader.
- **Per-step streaming residual on a RAM/VRAM-tight box (investigated, z-image on Turing 16GB/8GB).**
  After the fp16 fix, compute is at parity with ComfyUI (identical `s1688gemm` / `cutlassF_f16`
  kernels); the only per-step difference is the weight streaming: ComfyUI streams **pinned** (`Memcpy
  HtoD Pinned`, whole offloaded set in the pinned hostbuf → ~0.5s), turbo streams **pageable** (~1.5s)
  because it materializes weights (`from_pretrained`) rather than mmap. Loading mmap-backed
  (`assign_streamed_weights` for both models) frees the RAM and *does* flip streaming to pinned and
  the steady step to ~2s — but adds a ~30s **per-gen first-step disk re-fault**: `reset_cast_buffers`
  bounces the mmap between gens, and turbo only pins ~4.7GB of its 7GB offloaded set while ComfyUI
  pins all 6.3GB, so turbo's unpinned remainder re-reads from disk. That residency/pin split is set by
  `comfy_aimdo`'s compiled VBAR heuristics — which give the diffusers transformer a smaller
  resident+pinned set than ComfyUI's native model — and is **not reachable from the Python bridge**
  (mmap-load, keep-loaded, `--high-ram`, and comfy's `prepare_sampling` `memory_required` sizing were
  all tried; none close it). Net: the **materialized path is kept as default** — it streams from fast
  anonymous RAM (no re-fault), giving ~2.3s/step and ~20s/image, which already beats ComfyUI's ~31s
  end-to-end (ComfyUI pays a per-request CPU→GPU reload turbo avoids by keeping the model resident).
- **comfy-kitchen** is a quantization-kernel library, unrelated to offloading — not a dependency.
- **Re-syncing** a newer ComfyUI: re-copy the files and re-apply the short edit set in
  `comfy/resync.md`; bump the commit pins there, in `comfy/__init__.py`, and in `README.md`.
- **License.** ComfyUI is GPLv3; the vendored copies live in this already-GPLv3 package.
