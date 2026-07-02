# Plan: Integrate comfy-kitchen fused kernels into turbo-aimdo (flux2 bf16 path)

## Context

turbo-aimdo copies ComfyUI's compute path onto turboCLI's **diffusers** pipelines through the thin
bridge `aimdo/adapter.py` (already: fused RMSNorm via `comfy_ize`, cuDNN-flash SDPA via
`prefer_cudnn_attention`, weight prefetch). ComfyUI additionally uses **comfy-kitchen** — a native
CUDA kernel library — for fused **RoPE** (`apply_rope`) on the normal bf16 forward path (in its own
`comfy/ldm/flux/math.py`) and for fp8/fp4 **quantization**. comfy-kitchen is installed in ComfyUI's
embedded Python but **absent from the turboCLI runtime venv** (hence the recurring
`Failed to import comfy_kitchen … fp8/fp4 not available` warning during our runs).

Goal (user-confirmed): **compute-path parity** — install comfy-kitchen and route the diffusers
transformer's RoPE through its fused kernels, mirroring ComfyUI. **Target engine: flux2 first.**

Key facts established by exploration:
- comfy-kitchen is a **native wheel** (`_C.abi3.pyd`, CMake+nanobind+CUDA 12.8), **Apache-2.0**,
  v0.2.16, commit `43b413e`. It has a pure-Python **eager** fallback + optional **triton**. Not
  vendorable byte-for-byte like `comfy/`; it is **pip-installed** (like comfy-aimdo).
- Distribution: **CUDA wheels** (Linux x86_64 / Windows x64) + a **pure-Python `py3-none-any` wheel**
  (eager+triton) for other platforms. Stable-ABI `cp312-abi3` → loads on the runtime's **Python 3.14**.
- The vendored `aimdo/comfy/float.py` + `quant_ops.py` **already** wrap `import comfy_kitchen` in
  try/except, so quant paths + `comfy.quant_ops.ck` auto-activate once the wheel is present — **no
  vendored-file edits needed**. Only the RoPE routing is new adapter work (diffusers ≠ ComfyUI ldm).
- Runtime torch is **2.12.1+cu130** → satisfies ck's `cuda`-backend gate (`quant_ops.py` disables ck
  cuda unless torch is cu130+). So the CUDA kernels should actually engage.

## Out of scope (this phase)
- **adaLN fusion** — Flux2's per-block modulation is inlined (`x = norm(x)*(1+scale)+shift` across two
  statements), no single patchable method; fusing needs rewriting block forwards (fragile, tiny gain).
  Only `AdaLayerNormContinuous.forward` (the final `norm_out`, 1 call/step) is cleanly patchable —
  negligible. Defer.
- **Quantization (fp8/fp4)** — activates for free once the wheel is installed (vendored try/except);
  no bf16 effect. Not the focus.
- **z-image / qwen routing** — the patch targets the shared diffusers symbol, so qwen (which uses
  `apply_rotary_emb`) inherits it for free; z-image uses its own rotary impl — verify per engine
  before claiming a win. Wire flux2 first.

## Approach

**Design philosophy (same as the whole turbo-aimdo v2): copy comfy, import its files, minimum adapter.**
We do NOT reimplement any kernel. We `import comfy_kitchen` (reusing the vendored, already-backend-
configured `comfy.quant_ops.ck`) and call its `apply_rope1` — the exact function ComfyUI calls. The
ONLY new code is (a) a reversible one-function monkeypatch that points diffusers' rope symbol at ck,
and (b) a ~3-line format shim that reproduces ComfyUI's own `rope()` packaging (`[cos,-sin,sin,cos]`,
`comfy/ldm/flux/math.py:27-28`) from the `(cos,sin)` diffusers hands us. No kernel logic is authored.

### 1. RoPE convention shim (the one real correctness risk)
Both diffusers Flux2 and comfy-kitchen use **interleaved** (adjacent-pair) RoPE → use ck's
`apply_rope1` (NOT the `*_split_half` variants). They only package the rotation differently:
- **diffusers** `apply_rotary_emb(x,(cos,sin), use_real=True, use_real_unbind_dim=-1)` with
  `cos,sin` at `[S, D]` (each freq duplicated per pair via `repeat_interleave(2)`).
- **ck** `apply_rope1(x, freqs_cis)` wants `freqs_cis` shaped `[..., D/2, 2, 2]` = `[[cos,-sin],[sin,cos]]`
  (identical to ComfyUI's `rope()` stack in `comfy/ldm/flux/math.py`).

Exact map (build once per positions, cache; broadcast against `x`):
```python
c = cos[..., 0::2]; s = sin[..., 0::2]                       # [S, D/2]
freqs_cis = torch.stack([c, -s, s, c], -1).reshape(*c.shape, 2, 2)   # [S, D/2, 2, 2]
# insert broadcast dim per sequence_dim (1 -> [:,None,:,:,:] for [B,S,H,D]; 2 -> [:,:,None,...] )
```
This is algebraically identical to diffusers' path up to bf16 accumulation order — **provable via
allclose** (see Verification 4a) before any image run.

### 2. Adapter function — `aimdo/adapter.py`  (mirror ComfyUI exactly, NOT device-gated)
ComfyUI (`comfy/ldm/flux/math.py:47-58`) at inference calls `comfy.quant_ops.ck.apply_rope1(x,freqs_cis)`
**unconditionally — no device gate** — and lets comfy_kitchen's own registry choose the backend. The
registry self-configures at import (`quant_ops.py:19-35`): disable `cuda` if `torch.version.cuda < 13`,
disable `triton` unless `--enable-triton-backend`; otherwise cuda→(triton)→eager. The only fallback is
the pure-Python path, used solely when `comfy.model_management.in_training`. (comfy_kitchen is a HARD dep
in ComfyUI — `math.py` has no ck-absent branch.)

Mirror this. Add ONE function `use_kitchen_rope(model)`, analogous to `prefer_cudnn_attention`: a
reversible, idempotent **module-level monkeypatch** of `diffusers.models.embeddings.apply_rotary_emb`
(the single chokepoint all Flux2 processors call) + the `_build_freqs_cis` shim (§1). Reference the
VENDORED, already-backend-configured `comfy.quant_ops.ck` (NOT a fresh `import comfy_kitchen`) so the
cuda/triton disable logic is inherited **identically** to ComfyUI.

Gating (ComfyUI-faithful, **no `x.is_cuda` check**):
- install the patch only if `comfy.quant_ops._CK_AVAILABLE` (comfy_kitchen imported) and env
  `AIMDO_KITCHEN_ROPE != "0"` (escape hatch)
- per call: route to `ck.apply_rope1` when `not comfy.model_management.in_training` AND inputs match the
  flux convention (`use_real`, `use_real_unbind_dim==-1`, tuple `freqs_cis`, `x.dtype in (bf16,fp16)`);
  else fall through to diffusers' original. **comfy_kitchen's registry dispatches per device** — CUDA
  kernel on cu130+, eager on cpu/mps — exactly as ComfyUI does everywhere.
Emit a one-time log of `ck.list_backends()` so the run shows which backend served.

### 3. Wiring — `aimdo/__init__.py:load_pipe()`
Add `adapter.use_kitchen_rope(p.transformer)` next to each existing
`adapter.prefer_cudnn_attention(p.transformer)` (VBAR branch ~L127 and native branch ~L137). Not on
the text encoder. Idempotent, so per-`load_pipe` is safe.

### 4. Build install — `turboCLI/bash/turbo/build.sh`  (user-confirmed location)
comfy-kitchen is **NOT CUDA-only** (unlike comfy-aimdo). Install it **unconditionally** (all build
types), OUTSIDE the `if [ "$1" = "cuda" ]` block — near the general transformers/diffusers install:
- add `comfy_kitchen_version="0.2.16"` next to `comfy_aimdo_version` (~L59)
- `uv pip install --only-binary=:all: "comfy-kitchen==$comfy_kitchen_version"`
  - pip auto-resolves: CUDA wheel on win/linux-x64, pure-Python wheel on mac/mps
  - `--only-binary` avoids an sdist build (which would need CUDA 12.8 + CMake + nanobind → CI break)
  - on a cpu build on win/linux, pip still fetches the CUDA wheel; harmless (runtime → eager fallback)

### 5. Provenance / license
- `turbo-aimdo/README.md` table + `aimdo/comfy/resync.md` section (2): add
  `comfy-kitchen | 43b413e | 0.2.16 (Apache-2.0, pip-installed)`, with a note that it is
  **pip-installed, not vendored**, so its Apache-2.0 does not mix with the GPL sources; absent →
  graceful fallback; `adapter.use_kitchen_rope` routes diffusers RoPE through it when present.

## Critical files
- `C:\dev\workspace\msvc\turbo-aimdo\aimdo\adapter.py` — new `use_kitchen_rope` + `_build_freqs_cis` + import guard
- `C:\dev\workspace\msvc\turbo-aimdo\aimdo\__init__.py` — wire into `load_pipe()` (both branches)
- `C:\dev\workspace\msvc\turboCLI\bash\turbo\build.sh` — unconditional comfy-kitchen install + version pin
- `C:\dev\workspace\msvc\turbo-aimdo\README.md`, `aimdo\comfy\resync.md` — provenance/license
- Reference (patch target, do not edit): `…\.venv\Lib\site-packages\diffusers\models\embeddings.py::apply_rotary_emb`

## Verification (via `turboCLI/bash/flux2/run.sh`)
Prereq: confirm the flux2 diffusers model is installed for the runtime (else fetch via its build/model
script first). Deploy the updated `backend/aimdo` to the runtime and install the wheel.
- **4a Correctness (gate first):** offline `allclose` between diffusers `apply_rotary_emb(x,(cos,sin))`
  and `ck.apply_rope1(x,_build_freqs_cis(...))` on random `[2,4096,24,128]` bf16 CUDA tensors, for
  **both** `sequence_dim` values (bf16 tol ~1e-2). Then end-to-end A/B same seed:
  `AIMDO_KITCHEN_ROPE=1` vs `=0` → near-identical image (PSNR > ~45 dB). Garbage ⇒ convention bug.
- **4b Engaged:** confirm boot log `Found comfy_kitchen backend cuda: available` (not disabled) and
  the one-time `use_kitchen_rope` log shows the CUDA backend serving.
- **4c Timing:** flux2 on the 4GB GPU is streaming-bound, which masks the rope win — isolate compute:
  smallest res that fits VRAM with `cuda_offload=none` (weights resident, math-bound), A/B the flag,
  log per-step transformer time. Three-way: ck-cuda vs `ck.disable_backend("cuda")` (eager) vs native.
  Honest expectation: measurable only when compute-bound.

## Risks (ranked)
1. **RoPE mismatch → wrong images.** Mitigated by the 4a allclose gate before any image run; the
   `[cos,-sin,sin,cos]` sign/transpose and `sequence_dim` broadcast are where bugs hide — test both.
2. **cu130 gate.** ck's registry self-disables its CUDA backend unless torch is cu130+ (→ eager),
   exactly as in ComfyUI (`quant_ops.py:19-25`). Runtime is `2.12.1+cu130` → CUDA kernel engages;
   a future torch downgrade would silently fall to eager (still correct, just no CUDA win). Log
   `list_backends()` to make the active backend visible.
3. **abi3 / py3.14 wheel.** `cp312-abi3` loads on 3.14, but only if uv finds the matching platform
   wheel; `--only-binary` makes a miss fail loud instead of compiling from sdist.
4. **ck-eager slower than diffusers native rope on cpu/mps.** We deliberately do NOT device-gate —
   we mirror ComfyUI, which routes through ck on every device. On cpu/mps that is ck-eager, which may
   be slower than diffusers' native rope; accepted as the parity choice. Escape hatch:
   `AIMDO_KITCHEN_ROPE=0` disables the routing entirely.
5. **Global patch scope.** Patching the shared `apply_rotary_emb` affects the whole process — fine for
   the per-process CLI; per-call gating keeps unsupported callers on the original.
