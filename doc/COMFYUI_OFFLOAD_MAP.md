# ComfyUI offload / pin / VBAR cartography

Reference map of how ComfyUI streams model weights host↔VRAM, and how turbo-offloader
reproduces it. The map is for ComfyUI v0.39.1 with comfy-aimdo 0.5.5: line refs are against that
tag, the snapshot vendored in `offloader/comfy/` (see `offloader/comfy/resync.md`).

The one-line summary: **turbo drives the exact same vendored code the same way ComfyUI does.**
Static `ModelPatcher` vs `ModelPatcherDynamic`, the two pin systems, the budget, the lazy cast-path
pinning, and the per-generation teardown are all ComfyUI's; turbo's adapter just wraps diffusers
modules so they enter that machinery.

---

## 1. Patcher selection — a global launch-time swap, not a per-model test

`ModelPatcherDynamic` is never chosen by a fit-in-VRAM condition. There is one module-level alias
rebound once at startup when comfy-aimdo initializes:

- `model_patcher.py:2186` — `CoreModelPatcher = ModelPatcher` (default: static).
- `main.py:303-304` — on successful `comfy_aimdo.control.init_devices(...)`:
  `CoreModelPatcher = ModelPatcherDynamic`; `comfy.memory_management.aimdo_enabled = True`.

Every loader builds `ModelPatcher if <disable flag> else CoreModelPatcher`: UNet `sd.py:2275,2416`;
text encoder `sd.py:271-273`; VAE `sd.py:1115-1118`. Each also passes a per-model
`fast_disk=comfy.storage.state_dict_fast_disk(sd)` (§5). So with aimdo on, the UNet **and** text
encoder are `ModelPatcherDynamic`: that is what runs a model that fits RAM but not VRAM on a small
card.

- `is_dynamic()` is class identity: `ModelPatcher.is_dynamic()==False` (`model_patcher.py:403`),
  `ModelPatcherDynamic.is_dynamic()==True` (`:1797`).
- `ModelPatcherDynamic.__new__` reroutes to static for a **CPU** load device (`:1754-1758`), so
  dynamic is CUDA-only. Off CUDA / no comfy-aimdo → static `ModelPatcher` + classic lowvram
  partial load.
- Per-model opt-out: `disable_dynamic` / `disable_offload` force static for specific models.

**turbo:** `adapter.build_dynamic_patcher` → `ModelPatcherDynamic` when comfy-aimdo is present
(`use_vbar`), else `adapter.build_patcher` → `ModelPatcher`. Same split. (turbo flips
`aimdo_enabled` itself in `adapter.enable_vbar`, since it has no `main.py`.)

## 2. Sampling → `load_models_gpu`

`samplers.py:1349 sample` → `CFGGuider.outer_sample:1240` →
`sampler_helpers.prepare_sampling:181` → `_prepare_sampling:188` →
`load_models_gpu([model]+models, memory_required=…,
minimum_memory_required=…, force_full_load=False)` (`sampler_helpers.py:201`). `estimate_memory`
(`:165-179`) sizes `memory_required` from `model.memory_required(shape)` (double-batch for CFG) +
`inference_memory`.

**Crucial:** for a *pure-dynamic* model `memory_required` is essentially ignored:
`ModelPatcherDynamic.memory_required` (`model_patcher.py:1842-1846`) notes the estimate only
matters when mixing dynamic-after-static; pure dynamic "does everything dynamically." Residency is
decided by comfy-aimdo's VBAR watermarks, not by the lowvram sizing. (This is why passing
`memory_required` from turbo's `prepare()` does nothing on the dynamic path.)

**turbo:** `prepare()` = `mm.load_models_gpu(patchers)`. Correct for the dynamic path.

## 3. Weight loading — mmap only, no eager copy

`utils.py:load_safetensors:97-156` mmaps the file (`comfy_aimdo.model_mmap.ModelMMAP`, `:105`) and
builds tensors with `torch.frombuffer` over the read-only mmap (`:148`), tagging each storage
`_comfy_tensor_file_slice = TensorFileSlice(f, lock, offset, size)` (`:150-152`). No bulk copy into
a comfy_aimdo buffer at load: slices are read on demand later. `load_torch_file` also tags each
storage with its source path (`comfy.storage.annotate_state_dict`, `utils.py:204`), which the
per-model `fast_disk` probe reads.

**turbo:** `adapter.load_streamed` (meta-load + `assign_streamed_weights`) and
`assign_streamed_weights` call the vendored `comfy.utils.load_safetensors` / `set_attr_param`, so
turbo's weights are the same mmap file-slices. (This is why materializing via `from_pretrained` —
commit `25dbb6c`, since reverted — was slower: materialized tensors stream pageable, not pinned.)

## 4. Two pin systems + the budget

- **Static, raw-tensor pin**: `model_management.py:1663 pin_memory(tensor)` `cudaHostRegister`s
  an already-materialized CPU tensor in place, recorded in `PINNED_MEMORY[ptr]`. Called from
  `ModelPatcher.pin_weight_to_device` (`model_patcher.py:933`) on the **static** partial-load
  path only; on dynamic patchers `pin_weight_to_device` raises (`:1821-1822`).
- **Dynamic, managed host-buffer pin**: `pinned_memory.py:69 pin_memory(module, subset, size)`
  grows the subset's shared `comfy_aimdo.host_buffer.HostBuffer` (`hostbuf.extend`, `:101`), views
  it as a tensor (`hostbuf_to_tensor(...)[off:off+size]`, `:103`), `cudaHostRegister`s that view
  (`:105`), and records it in `module._pins[subset]` (`:118-125`). This is the path dynamic models
  use. Each patcher keeps six subsets (`model_patcher.py:1783-1789`): `weights` / `patches`,
  `weights-loaded` / `patches-loaded` (pins backing VBAR-resident weights) and `weights-fast` /
  `patches-fast` (fast-disk models).
- **Budget**: `MAX_PINNED_MEMORY` (`model_management.py:1608`) is `-1` unless nvidia/amd and
  pinning is enabled (integrated GPUs disable it, `:1631-1634`), then **RAM × 0.40 on Windows**,
  else `max(RAM × 0.40, min(RAM × 0.90, RAM − 4GiB, RAM + swap − 16GiB))` (`:1636-1643`).
  `TOTAL_PINNED_MEMORY` is the running sum. `ensure_pin_budget` (`:747`, vs available system RAM)
  and `ensure_pin_registerable` (`:768`, vs `MAX_PINNED_MEMORY`) gate each pin. When short they
  evict pins in tiers (`pin_eviction_tiers` `:693`, `registration_eviction_tiers` `:711`) keyed on
  each model's `active` and `current_prompt` flags: models outside the current prompt lose pins
  first (`-fast`, then plain, then `-loaded`), then the prompt's own `-fast` (idle models),
  `-loaded` and idle plain pins; the executing model's plain and `-fast` pins go only with
  `evict_active`. `current_prompt` is set by `PromptModelTracker` (`model_patcher.py:50-91`), fed
  every node output by `execution.py` (`:669,777`); turbo's `prepare()` marks its models the same
  way. Still short, `_steal_pin(module, stack, buckets, size, priority, subset)`
  (`pinned_memory.py:17`) reuses a lower-priority peer's slot in that subset rather than
  allocating. **No per-module or registration-count cap**: a model pins as much of itself as fits
  the global budget, degrading gracefully.

## 5. Dynamic pinning is lazy, per-module, at cast time

`ModelPatcherDynamic.load` (`model_patcher.py:1859-2043`) attaches `_pin_state` (which carries the
patcher's `fast_disk` and `current_prompt` flags, `:1790,1794`) to each `comfy_cast_weights`
module (`:1970`) and reserves a VBAR slot `m._v = vbar.alloc(size)` (`:1993`), but pins
**nothing** up front. Tiny modules (`module_mem <= 16KiB`, `:1975`) or LoRA-reshaped ones are
force-loaded resident instead (`:1982-1990`) and never pinned.

Pinning happens during the forward cast (`ops.py:128 cast_modules_with_vbar`):
- `signature = vbar_fault(s._v)` (`:168`); `vbar_signature_compare(signature, s._v_signature)`
  (`:169`): if the weight is already device-resident with the same signature, the module is
  **skipped** (no transfer, no pin) (`:177-179`).
- else the subset follows the patcher's `fast_disk` (`:188-195`): `weights-fast` on a fast disk,
  otherwise `weights`, or `weights-loaded` when the weight lands in its VBAR slot (`signature`
  set). `handle_pin` (`:228-237`) pins when `signature is None or not fast_disk or args.high_ram`
  (`:232`) via `pinned_memory.pin_memory` + `get_pin` (`:233-234`). Off a fast disk every
  non-resident weight gets pinned; on a fast disk only weights with no VBAR slot (streamed through
  the cast buffer each step) are, and the rest read file→GPU directly. LoRA patches do the same
  with the `patches*` subsets (`:247-256`).
- `fast_disk` is `comfy.storage.model_fast_disk` over the model's source files
  (`storage.py:91-111`): `--fast-disk` / `--disable-fast-disk` force it, else it is True only when
  every file sits on fast storage (a Linux NVMe link probe, `comfy_aimdo.storage.fast_disk` on
  Windows).
- `_v_signature` is written after a successful cast (`:293`) and invalidated to `None` by
  `set_dirty` when the patch set changes (`model_patcher.py:1922-1924`).

Because pinning is lazy and budget-bounded, the pinned set converges to "as much of the cast
working set as fits `MAX_PINNED_MEMORY`" (on a fast disk, only the part with no VBAR slot).

## 6. comfy_aimdo surface (compiled — Python call sites only)

- `model_vbar` — the virtual-BAR allocator: `ModelVBAR(model_size*10, dev)`
  (`model_patcher.py:1811`, 10× is virtual address space), `vbar.alloc` (`:1993`), `vbar_fault` /
  `vbar_signature_compare` (`ops.py:168-169`), `vbar_unpin` (`:451`, `model_prefetch.py:133,136`),
  `vbars_analyze` (feeds `get_free_memory`, `model_patcher.py:425`), `vbars_reset_watermark_limits`
  (`execution.py:554`), `vbar.free_memory` (sheds resident pages, `:2050`).
- `host_buffer`: `HostBuffer(...)` staging, one per pin subset (`model_patcher.py:1883-1888`,
  weights 64MB / patches 8MB grow chunks), `read_file_slice` (file→pinned host,
  `memory_management.py:69`), `read_file_to_device` (file→GPU direct when no hostbuf, `:59`).
- `vram_buffer` — `VRAMBuffer(DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE=16GiB, dev)` reused
  on-device cast/scratch buffer (`model_management.py:1407,1448`); virtual, dropped each node.
- `torch` — `aimdo_to_tensor` (view a VBAR / cast-buffer region as a tensor, `ops.py:163,182`),
  `hostbuf_to_tensor` (`pinned_memory.py:103`).
- `model_mmap` — `ModelMMAP(ckpt)` (`utils.py:105`).
- `storage`: `fast_disk(path)`, the Windows fast-storage probe behind `comfy.storage.fast_storage`
  (`storage.py:79`).
- `malloc_graph`: `record(stream, ...)` (`model_prefetch.py:72`), the Comfy model compiler's
  per-thread allocation graph (`--disable-comfy-compiler` turns it off); `cleanup_prefetch_queues`
  drops it.
- `control`: `init(simple_vram_headroom=, nvml_pressure=)` (`main.py:77`, NVML pressure on unless
  `--disable-nvml-pressure`), `init_devices` (`main.py:281`), `analyze` (debug,
  `execution.py:551`).

## 7. Per-generation teardown — weights persist, only scratch/patches reset

After each node exec, when aimdo is on (`execution.py:549-554`): `cleanup_prefetch_queues()`, then
`reset_cast_buffers()`, then `vbars_reset_watermark_limits()`.

`reset_cast_buffers` (`model_management.py:1452-1491`): syncs offload streams; bounces + clears
`DIRTY_MMAPS`; for each dynamic model prunes the `weights*` steal buckets (if active), flips
`active=False` and calls `partially_unload_ram(1e30, subsets=["patches", "patches-loaded",
"patches-fast"])`, which unregisters and frees the **patches** host buffers only (re-created
empty); clears the cast/scratch `STREAM_*_CAST_BUFFERS` (drops the 16GiB VRAMBuffer). **The
`weights*` host buffers and `_v` VBAR allocations survive**: base weights are *not*
force-re-faulted every generation. `cleanup_prefetch_queues` (`model_prefetch.py:151-172`) drops
the thread's malloc graph and unpins prefetch-queued modules (`:123-137`).
`vbars_reset_watermark_limits` resets comfy-aimdo's internal residency watermarks.

**turbo:** `reclaim()` runs these exact three calls in the same order (via `node_teardown()`, also
run at the encode→denoise boundary; guarded on `aimdo_enabled`). Same teardown, so turbo's weight
residency persists between gens just like ComfyUI's: no per-gen weight re-fault.

## 8. Config knobs that size the VBAR / pinned set

`--high-ram` (pin VBAR-backed weights even on a fast disk; skip the RAM check;
`pinned_hostbuf_size = size*2`), `--reserve-vram` / `--vram-headroom` (VRAM kept free → caps
resident VBAR set), `--disable-pinned-memory` (off), `--fast-disk` / `--disable-fast-disk` (force
the per-model `fast_disk` policy instead of probing storage), `--disable-nvml-pressure` (CUDA
instead of NVML for VRAM pressure). Hardcoded: `ModelVBAR = model_size*10`, cast
`VRAMBuffer = 16GiB`, hostbuf grow chunks 64MB/8MB. turbo parses no argv, so all take upstream
defaults (turbo probes `fast_disk` per model like ComfyUI).

## 9. Empirical parity (measured, RTX A1000 4GB VRAM / 33GB RAM, flux2 & z-image 1024×768)

Probe: load via the offloader, run 3 generations, read `TOTAL_PINNED_MEMORY` and per-gen step
times; `MAX_PINNED_MEMORY` optionally capped to simulate a tight (16GB-Turing-like) budget.

| scenario | budget | pinned | gen1 first step | gen2/3 first step |
|---|---|---|---|---|
| flux2, full budget | 13.6 GB | **12.33 GB** | (cold) | fast |
| flux2, capped 6 GB (Turing sim) | 6.0 GB | **5.49 GB** (91%) | 9.7s (cold cuDNN) | **3.4 / 3.3s — no re-fault** |
| z-image, capped 6 GB (Turing sim) | 6.0 GB | **4.94 GB** (82%) | 19.8s (cold TE encode) | **3.6 / 4.0s — no re-fault** |

Conclusions:
- turbo pins **to the budget** (91% / 82% of a tight cap) — comfy-parity, not the stale "4.7 of 6.4
  GB" under-pinning an older note described.
- **No per-generation re-fault** at either budget: gen2/gen3 first-step ≈ steady step. The weight
  VBAR persists (§7). gen1's cost is the one-time cuDNN plan search + first fault, not a recurring
  re-fault.
- Caveat the A1000 can't test: the sim caps the *pin* budget but not RAM/page-cache. On a real 16GB
  box a 14–20GB mmap model plus pinned copies can exceed RAM and thrash the *unpinned* mmap
  remainder from the page cache. That is a RAM-capacity limit ComfyUI shares (same mmap + same
  budget), not a turbo divergence — verify on the box, but there is no turbo-specific gap to close.
