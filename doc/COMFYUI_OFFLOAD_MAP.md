# ComfyUI offload / pin / VBAR cartography

Reference map of how ComfyUI streams model weights host↔VRAM, and how turbo-offloader
reproduces it. Line refs are against a ComfyUI checkout at `C:\dev\test\ComfyUI` (ahead of the
vendored pin — see `offloader/comfy/resync.md`); the mechanisms match the vendored snapshot.

The one-line summary: **turbo drives the exact same vendored code the same way ComfyUI does.**
Static `ModelPatcher` vs `ModelPatcherDynamic`, the two pin systems, the budget, the lazy cast-path
pinning, and the per-generation teardown are all ComfyUI's; turbo's adapter just wraps diffusers
modules so they enter that machinery.

---

## 1. Patcher selection — a global launch-time swap, not a per-model test

`ModelPatcherDynamic` is never chosen by a fit-in-VRAM condition. There is one module-level alias
rebound once at startup when comfy-aimdo initialises:

- `model_patcher.py:2052` — `CoreModelPatcher = ModelPatcher` (default: static).
- `main.py:261-263` — on successful `comfy_aimdo.control.init_devices(...)`:
  `CoreModelPatcher = ModelPatcherDynamic`; `comfy.memory_management.aimdo_enabled = True`.

Every loader builds `ModelPatcher if <disable flag> else CoreModelPatcher`: UNet `sd.py:1902,2043`;
text encoder `sd.py:254-256`; VAE `sd.py:931-933`. So with aimdo on, the UNet **and** text encoder
are `ModelPatcherDynamic` — that is what runs a model that fits RAM but not VRAM on a small card.

- `is_dynamic()` is class identity: `ModelPatcher.is_dynamic()==False` (`model_patcher.py:354`),
  `ModelPatcherDynamic.is_dynamic()==True` (`:1732`).
- `ModelPatcherDynamic.__new__` reroutes to static for a **CPU** load device (`:1698-1702`) — dynamic
  is CUDA-only. Off CUDA / no comfy-aimdo → static `ModelPatcher` + classic lowvram partial load.
- Per-model opt-out: `disable_dynamic` / `disable_offload` force static for specific models.

**turbo:** `adapter.build_dynamic_patcher` → `ModelPatcherDynamic` when comfy-aimdo is present
(`use_vbar`), else `adapter.build_patcher` → `ModelPatcher`. Same split. (turbo flips
`aimdo_enabled` itself in `adapter.enable_vbar`, since it has no `main.py`.)

## 2. Sampling → `load_models_gpu`

`samplers.py:1330 sample` → `CFGGuider.outer_sample:1232` →
`sampler_helpers.prepare_sampling:181` → `_prepare_sampling:188` →
`load_models_gpu([model]+models, memory_required=…, minimum_memory_required=…, force_full_load=False)`
(`sampler_helpers.py:201`). `estimate_memory` (`:165-179`) sizes `memory_required` from
`model.memory_required(shape)` (double-batch for CFG) + `inference_memory`.

**Crucial:** for a *pure-dynamic* model `memory_required` is essentially ignored —
`ModelPatcherDynamic.memory_required` (`model_patcher.py:1762-1766`) notes the estimate only matters
when mixing dynamic-after-static; pure dynamic "does everything dynamically." Residency is decided by
comfy-aimdo's VBAR watermarks, not by the lowvram sizing. (This is why passing `memory_required` from
turbo's `prepare()` does nothing on the dynamic path.)

**turbo:** `prepare()` = `mm.load_models_gpu(patchers)`. Correct for the dynamic path.

## 3. Weight loading — mmap only, no eager copy

`utils.py:load_safetensors:85-119` mmaps the file (`comfy_aimdo.model_mmap.ModelMMAP`) and builds
tensors with `torch.frombuffer` over the read-only mmap (`:111`), tagging each storage
`_comfy_tensor_file_slice = TensorFileSlice(f, lock, offset, size)` (`:113-115`). No bulk copy into a
comfy_aimdo buffer at load — slices are read on demand later.

**turbo:** `adapter.load_streamed` (meta-load + `assign_streamed_weights`) and
`assign_streamed_weights` call the vendored `comfy.utils.load_safetensors` / `set_attr_param`, so
turbo's weights are the same mmap file-slices. (This is why materialising via `from_pretrained` —
commit `25dbb6c`, since reverted — was slower: materialised tensors stream pageable, not pinned.)

## 4. Two pin systems + the budget

- **Static, raw-tensor pin** — `model_management.py:1515 pin_memory(tensor)`: `cudaHostRegister`s an
  already-materialised CPU tensor in place, recorded in `PINNED_MEMORY[ptr]`. Called from
  `ModelPatcher.pin_weight_to_device` (`model_patcher.py:877`) on the **static** partial-load path
  only; on dynamic patchers `pin_weight_to_device` raises (`:1753`).
- **Dynamic, managed host-buffer pin** — `pinned_memory.py:66 pin_memory(module, subset, size)`: grows
  a shared `comfy_aimdo.host_buffer.HostBuffer` (`hostbuf.extend`, `:94`), views it as a tensor
  (`hostbuf_to_tensor(...)[off:off+size]`, `:96`), `cudaHostRegister`s that view (`:98`), and records
  it on `module._pin`. This is the path dynamic models use.
- **Budget** — `MAX_PINNED_MEMORY` (`model_management.py:1488`): `-1` unless nvidia/amd, then
  **RAM × 0.40 on Windows / × 0.90 else** (`:1492-1495`). `TOTAL_PINNED_MEMORY` is the running sum.
  `ensure_pin_budget` (`:645`, vs free system RAM) and `ensure_pin_registerable` (`:680`, vs
  `MAX_PINNED_MEMORY`) gate each pin; over budget, `_steal_pin` (`pinned_memory.py:19`) reuses a
  lower-priority peer's slot rather than allocating. **No per-module or registration-count cap** — a
  model pins as much of itself as fits the global budget, degrading gracefully.

## 5. Dynamic pinning is lazy, per-module, at cast time

`ModelPatcherDynamic.load` (`model_patcher.py:1779-1893`) attaches `_pin_state` to each
`comfy_cast_weights` module (`:1865`) and reserves a VBAR slot `m._v = vbar.alloc(size)` (`:1888`),
but pins **nothing** up front for streamed weights. Tiny modules (`module_mem <= 16KiB`, `:1870`) or
LoRA-reshaped ones are force-loaded resident instead (`:1877-1885`) and never pinned.

Pinning happens during the forward cast (`ops.py:91 cast_modules_with_vbar`):
- `signature = vbar_fault(s._v)` (`:129`); `vbar_signature_compare(signature, s._v_signature)` (`:130`)
  — if the weight is already device-resident with the same signature, the module is **skipped**
  (no transfer, no pin) (`:136-138`).
- else `handle_pin` (`:179-188`) pins only when `signature is None or args.high_ram` (`:183`) via
  `pinned_memory.pin_memory` + `get_pin` (`:184-185`), i.e. weights that must be streamed each step
  get pinned to speed the recurring H2D copy.
- `_v_signature` is written after a successful cast (`:234`) and invalidated to `None` by `set_dirty`
  when the patch set changes (`model_patcher.py:1817`).

Because pinning is lazy and budget-bounded, the pinned set converges to "as much of the streamed
working set as fits `MAX_PINNED_MEMORY`."

## 6. comfy_aimdo surface (compiled — Python call sites only)

- `model_vbar` — the virtual-BAR allocator: `ModelVBAR(model_size*10, dev)` (`model_patcher.py:1743`,
  10× is virtual address space), `vbar.alloc` (`:1888`), `vbar_fault` / `vbar_signature_compare`
  (`ops.py:129-130`), `vbar_unpin` (`:392`, `model_prefetch.py:18`), `vbars_analyze` (feeds
  `get_free_memory`, `model_patcher.py:376`), `vbars_reset_watermark_limits` (`execution.py:549`),
  `vbar.free_memory` (sheds resident pages, `:1938`).
- `host_buffer` — `HostBuffer(...)` staging (`model_patcher.py:1803-1804` weights 64MB / patches 8MB
  grow chunks), `read_file_slice` (file→pinned host, `memory_management.py:69`), `read_file_to_device`
  (file→GPU direct when no hostbuf, `:59`).
- `vram_buffer` — `VRAMBuffer(DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE=16GiB, dev)` reused on-device
  cast/scratch buffer (`model_management.py:1309,1346`); virtual, dropped each node.
- `torch` — `aimdo_to_tensor` (view a VBAR / cast-buffer region as a tensor, `ops.py:124,141`),
  `hostbuf_to_tensor` (`pinned_memory.py:96`).
- `model_mmap` — `ModelMMAP(ckpt)` (`utils.py:86`).
- `control` — `init` / `init_devices` (`main.py`), `analyze` (debug, `execution.py:546`).

## 7. Per-generation teardown — weights persist, only scratch/patches reset

After each node exec, when aimdo is on (`execution.py:544-549`): `reset_cast_buffers()` +
`cleanup_prefetch_queues()` + `vbars_reset_watermark_limits()`.

`reset_cast_buffers` (`model_management.py:1350-1383`): syncs offload streams; bounces + clears
`DIRTY_MMAPS`; for each active dynamic model flips `active=False` and `partially_unload_ram(1e30,
subsets=["patches"])` — unregisters+frees the **patches** host buffer only; clears the cast/scratch
`STREAM_*_CAST_BUFFERS` (drops the 16GiB VRAMBuffer). **The weights host buffer and `_v` VBAR
allocations survive** — base weights are *not* force-re-faulted every generation. `cleanup_prefetch_
queues` unpins prefetch-queued modules (`model_prefetch.py:18`). `vbars_reset_watermark_limits` resets
comfy-aimdo's internal residency watermarks.

**turbo:** `reclaim()` runs these exact three calls (guarded on `aimdo_enabled`). Same teardown, so
turbo's weight residency persists between gens just like ComfyUI's — no per-gen weight re-fault.

## 8. Config knobs that size the VBAR / pinned set

`--high-ram` (pin even when resident; `pinned_hostbuf_size = size*2`), `--reserve-vram` /
`--vram-headroom` (VRAM kept free → caps resident VBAR set), `--disable-pinned-memory` (off),
`--fast-disk` (prefer disk-direct over pinned RAM). Hardcoded: `ModelVBAR = model_size*10`, cast
`VRAMBuffer = 16GiB`, hostbuf grow chunks 64MB/8MB. turbo parses no argv, so all take upstream
defaults.

## 9. Empirical parity (measured, RTX A1000 4GB VRAM / 33GB RAM, flux2 & z-image 1024×768)

Probe: load via the offloader, run 3 generations, read `TOTAL_PINNED_MEMORY` and per-gen step times;
`MAX_PINNED_MEMORY` optionally capped to simulate a tight (16GB-Turing-like) budget.

| scenario | budget | pinned | gen1 first step | gen2/3 first step |
|---|---|---|---|---|
| flux2, full budget | 13.6 GB | **12.33 GB** | (cold) | fast |
| flux2, capped 6 GB (Turing sim) | 6.0 GB | **5.49 GB** (91%) | 9.7s (cold cuDNN) | **3.4 / 3.3s — no re-fault** |
| z-image, capped 6 GB (Turing sim) | 6.0 GB | **4.94 GB** (82%) | 19.8s (cold TE encode) | **3.6 / 4.0s — no re-fault** |

Conclusions:
- turbo pins **to the budget** (91% / 82% of a tight cap) — comfy-parity, not the stale "4.7 of 6.4
  GB" under-pinning an older note described.
- **No per-generation re-fault** at either budget: gen2/gen3 first-step ≈ steady step. The weight VBAR
  persists (§7). gen1's cost is the one-time cuDNN plan search + first fault, not a recurring re-fault.
- Caveat the A1000 can't test: the sim caps the *pin* budget but not RAM/page-cache. On a real 16GB
  box a 14–20GB mmap model plus pinned copies can exceed RAM and thrash the *unpinned* mmap remainder
  from the page cache. That is a RAM-capacity limit ComfyUI shares (same mmap + same budget), not a
  turbo divergence — verify on the box, but there is no turbo-specific gap to close.
