# Plan: Re-sync the vendored ComfyUI offloader to v0.39 (comfy-aimdo 0.5.5, comfy-kitchen 0.2.37)

## Context

`offloader/comfy/` is a byte-for-byte snapshot of ComfyUI's offloading subsystem, pinned to
ComfyUI v0.27.0 (`bb131be9`), with comfy-aimdo 0.4.10 and comfy-kitchen 0.2.16 installed into the
turbo runtime. ComfyUI is now at v0.39, which requires comfy-aimdo 0.5.5 and comfy-kitchen 0.2.37.
We move the snapshot and both wheels forward, following `offloader/comfy/resync.md`, with the
smallest adapter change that keeps the offloader working, fast and faithful to ComfyUI. Then we
check flux2 and krea2 on small canvases for regressions.

The vendored tree was verified to match upstream v0.27.0 exactly (line endings aside) apart from
the 3 documented comment-outs, so the re-copy is mechanical. The work is in the adapter, where
upstream API and behavior changes reach our code.

User decisions: keep torch at 2.12.1+cu130 (a bump would be a separate measured step); the malloc
graph prefetch is a measured follow-up, not part of the re-sync; the turbo runtime may be modified
after a backup.

## Target pins: tagged releases only (resolved 2026-10-07 via `git ls-remote`)

| repo | tag | commit | note |
|---|---|---|---|
| ComfyUI | `v0.39.1` | `de0125d9` | latest tag (see below) |
| comfy-aimdo | `v0.5.5` | `3b8e8c16` | annotated tag, also what the PyPI 0.5.5 wheel reports |
| comfy-kitchen | `v0.2.37` | `be003b7c` | latest tag |
| torch | 2.12.1+cu130 (unchanged) | | ComfyUI recommends 2.12+ |

ComfyUI v0.39.1's 3 commits over `v0.39.0` (`b0b74356`) touch only partner nodes, version files
and requirements.txt, so the vendored `comfy/` files are exactly v0.39.0's. It requires exactly
comfy-aimdo 0.5.5 and comfy-kitchen 0.2.37, so the three pins are consistent. The 19 untagged
master commits past v0.39.0 (`c9d8a6e6`, an ops.py refactor and an AMD SDPA tweak) are
deliberately not taken.

This also fixes a stale pin. README, `comfy/__init__.py` and resync.md cite comfy-aimdo `afa70d91`,
an untagged commit, while the installed 0.4.10 wheel is the `v0.4.10` tag (`ace72abe`).

## Step 0: Pull and check out the tags

- `git fetch --tags` + `git pull` the local ComfyUI, comfy-aimdo and comfy-kitchen checkouts,
  then check out `v0.39.1`, `v0.5.5`, `v0.2.37` respectively (the copy source is the tag, never a
  branch head).
- The ComfyUI portable install's `ComfyUI/` is on untagged master `c9d8a6e6`: fetch and check out
  `v0.39.1` there too, so the parity comparison runs exactly the vendored code.
- Re-run `git ls-remote --tags`. If a newer tag than these exists by then, re-check its
  requirements and the vendored-set diff and take it instead.
- Confirm the portable `python_embeded` still has comfy_aimdo 0.5.5 and comfy_kitchen 0.2.37.

## Step 1: Baseline on the current offloader (before touching anything)

All runs go through turboCLI `bash/turbo/text-to-image.sh`, in `offloader` mode, at 512x512 with
seed 42. The same matrix is re-run after the re-sync (Step 6), which gives the before/after.

| engine | device | steps | runs | path exercised |
|---|---|---|---|---|
| `flux2-4b` | CUDA | 4 | 2 | VBAR streaming, prefetch, kitchen RoPE |
| `flux2-4b` | CPU | 4 | 1 (about 220 s) | native cast path, `stream` CPU mode, aimdo off |
| `comfy-krea2-turbo` | CUDA | 8 | 2 | fp8 quant path (`mixed_precision_ops`), VBAR |

Record:
- the md5 of each image (two runs should match),
- per-step s/it from tqdm and total wall clock,
- `tests/check_img.py` std,
- GPU temperature and clocks at the start (`nvidia-smi`).

## Step 2: Re-copy the vendored files

1. Copy from `ComfyUI/comfy/` at tag `v0.39.1` over `offloader/comfy/`:
   - the 14 flat modules listed in resync.md,
   - the `comfy_types/` and `weight_adapter/` packages,
   - plus the 4 modules v0.39 now imports: `system_memory.py`, `internal_logging.py`,
     `rmsnorm.py` (now imported by `ops.py`) and `storage.py`.
   None of them pulls anything off-path. Check `git diff --stat` afterwards: it must show real
   changes only, not whole-file line-ending rewrites. Match the repo's existing line endings, and
   don't use `sed -i`, which drops CRLF.
2. Re-apply the 3 `# [turbo-offloader] disabled for turboCLI:` comment-outs:
   - `utils.py`: `from einops import rearrange`, now at L34;
   - `lora.py` L22: `import comfy.model_base`;
   - `hooks.py` L17: `node_helpers`.
   Their off-path reasoning still holds (verified).
3. `offloader/comfy/__init__.py`:
   - add `"storage"` and `"malloc_graph"` to the comfy_aimdo stand-in list (`utils.py` →
     `storage.py` imports `comfy_aimdo.storage`; `model_prefetch.py` imports
     `comfy_aimdo.malloc_graph`), otherwise CPU/MPS imports fail;
   - bump the commit pins in the header.

Note for resync.md: `model_prefetch.py` now imports `comfy_kitchen` unconditionally. This is safe
because only the adapter imports it, lazily, on the VBAR path, and comfy-kitchen is installed on
every build.

## Step 3: Bump the dependency pins

In turboCLI `bash/turbo/build.sh`:
- `comfy_aimdo_version="0.5.5"` (~L69)
- `comfy_kitchen_version="0.2.37"` (~L71)

The kitchen bump is mandatory. v0.39's `quant_ops.py` imports layouts that 0.2.16 lacks, which
would silently set `_CK_AVAILABLE=False`, and that disables both the kitchen RoPE and the fp8 quant
path (krea2). `commit_offloader` (~L37) gets bumped only after the offloader commit, and that
commit waits for the user's go-ahead.

## Step 4: Adapter changes

Each change reuses a vendored function instead of copying code wherever possible.

**Required (would break or silently disable a feature):**

a. **`adapter.install_pin_rollback_guard`.** `pinned_memory.py` now keeps per-module pin state in
   `module._pins[subset]` (keys `balancer_priority`, `pin`, ...), and `_steal_pin` takes a 6th
   argument, `subset`. Mirror the new tail:
   - priority from `module.__dict__.setdefault("_pins", {}).setdefault(subset, {})`;
   - call `pm._steal_pin(module, stack, buckets, size, priority, subset)`.
   Today's 5-argument call would raise TypeError at the exact moment the guard fires.
b. **`adapter.mixed_precision_operations`.** Replace the hand-built `disabled` set with
   `ops.get_disabled_quant_formats(load_device)` (`ops.py:1762`). That is upstream's own helper,
   and it adds the int8 formats, which would otherwise crash on MPS. Net code removal.
c. **`adapter.use_comfy_attention`.** Re-sync to `ops.py:58-101`. Upstream changed three things:
   - the priority is now `[FLASH, CUDNN, EFFICIENT, MATH]`;
   - the Windows-only gate is gone;
   - GQA/mask handling was added.

   Keep our wrapper, since a patched `F.sdpa` can't call the vendored function, which calls
   `F.sdpa` itself. Instead of hard-coding values, read upstream's own state:
   - gate on `hasattr(comfy.ops, "SDPA_BACKEND_PRIORITY")`, which is exactly comfy's guard;
   - use `ops.SDPA_BACKEND_PRIORITY`;
   - use `ops.repeat_kv_for_gqa` for the GQA branch, `mm.is_nvidia()`.

   Future priority changes then come with the re-copy. Keep our q/k/v dtype coercion. Expect the
   images to change, because attention now prefers flash over cuDNN. That is ComfyUI's numerics
   now, so the baseline md5 is no longer the reference.
d. **`node_teardown` (`__init__.py`).** Reorder to upstream's `execution.py:552-554`:
   1. `cleanup_prefetch_queues`
   2. `reset_cast_buffers`
   3. `vbars_reset_watermark_limits`

   `cleanup_prefetch_queues` now also aborts the malloc graph and must run first.

**Faithfulness (ComfyUI does these by default; we currently don't):**

e. **`pre_torch_init`.** Call `ctl.init(nvml_pressure=True)`. That is ComfyUI's default
   (`main.py:77`, `disable-nvml-pressure` flag off); we currently get the new `False` default.
f. **`fast_disk`.** ModelPatcher/ModelPatcherDynamic now take `fast_disk`. ComfyUI passes
   `comfy.storage.state_dict_fast_disk(sd)` per model (`sd.py:2277`); we pass nothing, so we get
   `False`, which pins every streamed weight. Give `build_patcher` / `build_dynamic_patcher` a
   `fast_disk=False` kwarg. Each loader computes it with `comfy.storage.model_fast_disk(paths)`
   from the safetensors files it streams. That is the same function `state_dict_fast_disk` ends
   in, and on Windows it calls `comfy_aimdo.storage.fast_disk`. Watch its
   `Model storage policy: fast_disk=...` log line.
g. **Prompt tracking.** Pin eviction is now tiered on `current_prompt`. Without it, the running
   model's pins can be evicted under RAM pressure, and v0.27's `evict_active=False` no longer
   protects them. Reuse `comfy.model_patcher.PromptModelTracker` as `execution.py` does:
   - module-level tracker;
   - `start()` + `add(patchers)` in `prepare()`;
   - `end()` in `reclaim()` and `release()`.

**Comment and doc refresh.** Update the upstream line references in adapter, `__init__`,
implementation.md and COMFYUI_OFFLOAD_MAP.md:
- `ops.py:39-64` → `58-101` (and the attention paragraph text: flash first, any OS)
- `execution.py:543-549` → `550-554`
- `sd.py:481-482` → `505-506`; `757-758` → `877-878`; `1057-1058` → `1268-1269`;
  `1080-1087` → `1301-1308`
- `av_model.py:913` → `935`
- `krea2/model.py:33` → `32`; `:267` → `364`
- `model_management.py:156` → `158`

## Step 5: Docs

- `offloader/comfy/resync.md`: pins table; file list (+4 modules); comment-out line numbers; stub
  list; the `model_prefetch` hard kitchen import; the smoke test also covering the stub path.
- `README.md` version table and `dummy.md` ("pinned to ComfyUI v0.27.0").
- implementation.md: refresh the line refs (Step 4) and add a short re-sync note covering
  fast_disk, prompt tracking, nvml pressure and the new SDPA priority.
- Historical plans in `doc/` stay as written.
- Code and comments ≤99 columns, American spelling, no double-dash joins.

## Step 6: Verification

**Throttling protocol.** This laptop thermal-throttles, so:
- compare per-step s/it, never single-run wall clock;
- start each timed run cool (GPU below about 60 °C, logged via `nvidia-smi`);
- alternate A/B where both sides are available (turbo vs ComfyUI);
- treat a delta over about 10% as real only after a second cool re-run reproduces it.

Steps:
1. **Import smoke (CUDA venv):** the resync.md one-liner prints `import OK`.
2. **Import smoke (stub path):** the same one-liner with `sys.modules["comfy_aimdo"] = None`
   injected first, so the CPU/MPS stub path is exercised; it prints `aimdo_enabled= False`.
3. **CPU bridge test:** `python tests/bridge_parity.py`.
4. **Deploy to the turbo runtime (`<sky>/gg.omega/turbo`):**
   - back up `backend/offloader` outside `backend/` (the runner treats every `backend/*` as a
     mode) and record the old comfy wheel versions;
   - `uv pip install comfy-aimdo==0.5.5 comfy-kitchen==0.2.37` into its `.venv`;
   - copy the re-synced `offloader/` in.
5. **Boot log checks:**
   - comfy_kitchen backends show cuda available, with no "comfy_kitchen ... not available"
     warning;
   - aimdo init succeeds;
   - the `use_kitchen_rope` and `Model storage policy` lines appear.
6. **Before/after regression:** re-run the Step 1 matrix from turboCLI: flux2-4b at 512x512 on
   CUDA and on CPU, and comfy-krea2-turbo on CUDA. Pass criteria:
   - CUDA: the two runs are bit-identical (determinism kept);
   - `check_img` std is sane and each image is visually equivalent to its before image. Use PSNR
     rather than md5 on CUDA, because of the SDPA change. On CPU the SDPA patch doesn't engage,
     so a changed CPU image needs explaining (e.g. the kitchen 0.2.37 eager RoPE trim);
   - per-step s/it within noise of the before run under the throttling protocol. On CPU the
     throttle ramp makes per-step climb within a run, so compare the first steps of a cool start.
   Put the before/after table (s/it, wall clock, md5, std) in the final report.
7. **ComfyUI parity (krea2):** run the portable install, checked out at `v0.39.1`, at 512x512,
   seed 42, 8 steps, via a scratch variant of its `generate.sh` with the krea2 graph. Alternate it
   with the turbo run. Compare per-step s/it (historically 3.50 vs 3.62) and check the images agree
   visually.
8. **Optional cheap canary:** z-image-turbo 512x512, two runs, same md5. It was the determinism
   canary for the node teardown that Step 4d reorders.

**Rollback:** restore the backed-up `backend/offloader` and reinstall the recorded wheel versions.

Commits wait for the user's explicit go-ahead, with no Claude attribution. After the offloader
commit, bump turboCLI `commit_offloader`.

## Step 7 (follow-up, measured): malloc graph in `install_prefetch`

Wire upstream's prefetch caller pattern (`av_model.py:935-1024`):
- `mp.malloc_graph_begin(dev)` in `_start`;
- `malloc_scope="block"` on every pop;
- `mp.malloc_graph_end()` after the final `None` pop in `_end`.

Caveats:
- one begin/end (or a distinct scope) per hooked block sequence;
- the kitchen-RoPE `freqs_cis` cache is built inside block 0, so wrap that build in
  `mp.pause_malloc_graph()`.

Keep it only if flux2 and krea2 per-step improve in an interleaved, cool-start A/B with the images
unchanged.

## Critical files

- `offloader/comfy/*`: re-copied, plus 4 new modules
- `offloader/comfy/__init__.py`: stubs, pins
- `offloader/comfy/resync.md`
- `offloader/adapter.py`: pin guard, quant formats, SDPA, fast_disk on the builders and loaders
- `offloader/__init__.py`: aimdo init, node_teardown order, prompt tracker in
  prepare/reclaim/release
- `README.md`, `dummy.md`, `implementation.md`, `doc/COMFYUI_OFFLOAD_MAP.md`
- turboCLI `bash/turbo/build.sh`: version pins
- This plan is copied to `doc/COMFYUI_039_RESYNC_PLAN.md`
