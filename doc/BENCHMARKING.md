# Benchmarking turboCLI vs ComfyUI (same box, same model)

How to compare turbo-offloader's diffusion path against ComfyUI head-to-head — both at wall-clock
(per-step / end-to-end) and at the CUDA-op level — so a per-step gap can be *attributed* (which
kernels/copies differ) instead of guessed. Written from the z-image / Turing (RTX 2070S) exercise
where turbo went from ~5× slower to per-step parity.

## Ground rules

- **Same everything:** same GPU, same model files, same resolution, prompt, seed, step count. z-image
  used 1024×768, 8 steps, `res_multistep`/`simple`, shift 3, cfg 1.0.
- **One at a time:** the two servers can't share an 8 GB card / 16 GB RAM. Stop one before starting the
  other (`taskkill //F //IM python.exe`, confirm `nvidia-smi` and both ports are down).
- **Warm vs warm is the real number.** The first generation after a server start pays a cold model
  load (disk → RAM/VRAM); its first sampling step is also slow (weights staging into VRAM). Always run
  a warm-up generation, then measure the *next* one. Compare warm-to-warm; report cold separately.
- **Page cache matters.** Running ComfyUI then turbo (or vice-versa) evicts the other's model from the
  OS page cache, so the next cold load is disk-bound and slow. That's a cache artifact, not the engine.
- **Same prompt twice within an engine** exercises turbo's encode cache (2nd gen skips the text
  encoder). Use a *fresh* prompt when you specifically want to measure a cold text-encode.

## Level 1 — wall-clock (per-step + end-to-end)

**turbo** (server on 8080; `SKY_PATH_BIN` must be set):

```sh
cd turboCLI/bash/turbo && SKY_PATH_BIN=D:/omega/sky/bin sh server.sh start   # backgrounded
cd ../z-image
sh run.sh "PROMPT" out.png 1024 768 cuda SEED 8 offloader none none 8080
```

Per-step comes from the runner's tqdm (`… N/8 (mm:ss, X.XXs/it)`); the `s/it` at step 8 is the
running average (it includes the slow warm-first step) — for the true steady rate use the cumulative
timestamps of steps 2→8, not the average.

**ComfyUI** (server on 8188). Its `generate.sh` reports only end-to-end; the per-step tqdm goes to the
**server** stdout, so launch the server with stdout captured and scrape it:

```sh
cd ComfyUI_windows_portable
nohup ./python_embeded/python.exe -s ComfyUI/main.py --port 8188 > comfy_srv.log 2>&1 &
PROMPT="…" WIDTH=1024 HEIGHT=768 SEED=… bash generate.sh          # prints End-to-end
# per-step: scrape the KSampler tqdm that appeared in comfy_srv.log since the mark
```

**Caveat — end-to-end isn't apples-to-apples.** ComfyUI's `generate.sh` measures queue→finished, which
includes a per-request UNet CPU→GPU reload + VAE decode (~16 s on this box); turbo keeps the model
resident, so turbo's e2e is *lower* even when its per-step is slightly higher. Compare **per-step** for
compute, and note the e2e difference is request overhead, not sampling.

**Validate output, not just speed.** A fast step on a broken (fp16-overflow) computation yields a black
image. Check `PIL`+`numpy` `std()` of the saved PNG (`std < 3` ⇒ black/flat), and confirm both engines
produced the intended resolution (a silently-downscaled image invalidates the comparison).

## Level 2 — CUDA-op profiling (attribute the gap)

Wall-clock says *how much* slower; `torch.profiler` says *which ops*. Instrument **both** at the same
granularity — one warm diffusion-model forward — and diff the self-CUDA tables.

**turbo** — `adapter.install_profiler(model)` (env-gated, temporary; wired in `load_pipe` under
`OFFLOADER_PROFILE`). Wraps `transformer.forward`, and on `OFFLOADER_PROFILE_SKIP+1`-th call runs
`torch.profiler` (CPU+CUDA) and prints `key_averages().table(sort_by="self_cuda_time_total")` plus the
forward's wall time. Run **two** generations and profile a forward in the *second* one so it's fully
warm — with 8 steps/gen, `OFFLOADER_PROFILE_SKIP=11` profiles gen-2 step-4:

```sh
OFFLOADER_PROFILE=1 OFFLOADER_PROFILE_SKIP=11 sh server.sh start
# two gens same prompt (2nd is warm), then read the table from srv.log
```

**ComfyUI** — drop a throwaway custom node `ComfyUI/custom_nodes/zzz_profiler.py` that monkeypatches
`comfy.model_base.BaseModel._apply_model` with the identical profiler (skip via `COMFY_PROFILE_SKIP`,
default 11). It patches on startup; delete the file to remove. Run two gens, read the table from
`comfy_srv.log`.

Both print the same shape of table, so they diff directly. (Note: the profiled forward's *wall* is
inflated by profiler overhead — use it only to reason about `wall − ΣCUDA` = host/Python overhead, not
as the real step time.)

### Reading the diff — three buckets

1. **Shared compute kernels** — `aten::mm`→`*s1688gemm*` (fp16 matmul), `_efficient_attention`→
   `cutlassF_f16`, `_fused_rms_norm`, `nan_to_num` (clamp). If these match in time, compute is at
   parity. (Counts may differ — diffusers does unfused q/k/v so *more, smaller* matmuls than ComfyUI's
   fused ones — but total time is what matters.)
2. **`Memcpy HtoD` streaming** — the offloader shipping weights CPU→VRAM per step. **Pinned vs
   Pageable is the thing to watch:** pinned copies are ~3× faster per byte *and* overlap with compute;
   pageable copies are slow and block. Also compare copy *count* (fewer ⇒ more stays VRAM-resident).
3. **Engine-only ops** — anything one side has and the other doesn't (turbo's manual_cast hooks, comfy's
   `comfy_kitchen::apply_rope`, etc.). Usually small; confirm before blaming.

### Reference finding (this exercise)

After the fp16 `manual_cast` fix, the two op tables were nearly identical **except the streaming
bucket**: ComfyUI streamed **pinned** (`Memcpy HtoD Pinned`, ~0.5 s, ~136 copies) while turbo streamed
**pageable** (`Memcpy HtoD Pageable`, ~1.46 s, ~221 copies) — the whole per-step residual. Compute
(fp16 matmul/attention/norm/clamp) matched to within noise. So the remaining lever is getting turbo's
streamed weights **pinned** (ComfyUI's `comfy.pinned_memory` staging buffer), which is RAM-budget-bound:
turbo holds the transformer + text encoder both resident (~21 GB on 16 GB), starving the pinned buffer,
whereas ComfyUI has headroom from releasing the CLIP at sampling.

## Cleanup

The profilers are diagnostic, not shipped: turbo's `install_profiler` stays uncommitted / env-gated
(remove before committing), and `zzz_profiler.py` is deleted from ComfyUI's `custom_nodes/`.
