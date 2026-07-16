# turbo-offloader, explained simply

This is the plain-English companion to `offloader.md`. It explains what turbo-offloader does and
why, step by step, starting from zero — no image-diffusion background required. Each section is a
little more technical than the one before. For the full design, mechanisms and benchmarks, read
`offloader.md`; this document is kept up to date with it at every turbo-offloader iteration.

## What is this?

Modern image-generation models are huge: 4 to 55 GB of neural-network weights. Most consumer GPUs
have 4–16 GB of video memory (VRAM), so the model simply does not fit. turbo-offloader is the
component that lets turboCLI run these models anyway — on small GPUs, on plain CPUs, and even when
the model is bigger than the machine's entire RAM — at per-step speeds matching ComfyUI, the
reference open-source tool for this job.

It is a GPL-licensed Python package driven by the LGPL turboCLI runner through a small fixed
interface (**the seam**), so the two codebases stay legally and structurally separate.

## 30-second primer: how an image gets generated

If you have never looked at image diffusion, this is the whole pipeline:

```
 "a cat in the snow"
         │
         ▼
 [ text encoder ]              words → numbers the model can work with ("embeddings")
         │
         ▼
 [ diffusion transformer ]     starts from pure random noise and removes a bit of it at
         │      × N steps      each step, steered by the embeddings — THE heavy part
         ▼
 [ VAE decoder ]               expands the model's compact internal image ("latents")
         │                     into actual pixels
         ▼
     image.png
```

Three neural networks run in sequence. The text encoder and the VAE are comparatively small; the
**diffusion transformer** (also called the DiT) holds almost all of the gigabytes and runs N times
(typically 4–50 steps). That is the model that does not fit.

## The core problem, and the core trick

**Problem:** the transformer's weights (say 41 GB) must be on the GPU to compute, but the GPU only
has (say) 8 GB.

**Trick:** you never need the whole model at once. A transformer runs *layer by layer*, and each
layer only needs its own weights while it computes. So the weights can live somewhere big and slow
— system RAM, or even the model file on disk — and be copied to the GPU just in time, one layer at
a time, then dropped to make room for the next:

```
   RAM or disk (big, slow)                                GPU (small, fast)
  ┌─────────────────────────────────────┐               ┌─────────────────────────┐
  │ layer 1 │ layer 2 │  ...  │ layer L │ ──copy just──▶│ layer being computed    │
  └─────────────────────────────────────┘   in time     │ + the next one arriving │
                                                        └─────────────────────────┘
```

Copying is slow, so a **prefetcher** overlaps the transfer of layer N+1 with the computation of
layer N — when it hides well, streaming costs almost nothing per step. This "keep weights
elsewhere, stream them per forward pass" idea is called **offloading**, hence the name.

## Step by step: what happens on one generation

When you run `text-to-image.sh`, the turboCLI runner drives the offloader through these stages:

1. **Init before torch.** `pre_torch_init()` runs before PyTorch is even imported, because the
   CUDA memory-allocator hooks it installs cannot be added afterwards.
2. **Load — map, don't read.** Model weights are *memory-mapped* (mmap): the OS pretends the file
   is in memory and only pulls in the pages actually touched. Nothing is copied up front, so a
   41 GB model "loads" in seconds and never has to fit in RAM all at once.
3. **Prepare.** ComfyUI's memory manager (`load_models_gpu`) measures free VRAM and decides what
   can live on the GPU permanently and what must stream.
4. **Encode the prompt.** The text encoder turns the prompt into embeddings. Results are cached,
   so repeating a prompt is free; padding tokens are dropped first, which is a large speedup.
5. **Denoise loop.** The transformer runs its N steps. Each forward pass streams weights in layer
   by layer, with the prefetcher hiding the copies behind compute.
6. **Decode.** The VAE (small, kept resident) turns latents into pixels. If VRAM runs out, it
   automatically retries in tiles.
7. **Save and reclaim.** The PNG is written, caches are flushed, and memory is released so the
   next generation (or the rest of your desktop) gets it back.

## The architecture: three pieces

```
  ┌───────────────────────────┐
  │  turboCLI runner (LGPL)   │   CLI / server, engine definitions, diffusers pipelines
  └─────────────┬─────────────┘
                │  the seam: 8 functions in offloader/__init__.py
  ┌─────────────▼─────────────┐
  │  adapter.py               │   the thin bridge — the only real logic in this package
  └─────────────┬─────────────┘
                │
  ┌─────────────▼─────────────┐
  │  offloader/comfy/  (GPL)  │   ComfyUI's memory-management subsystem, vendored
  └───────────────────────────┘   byte-for-byte (pinned to ComfyUI v0.27.0)
```

The design philosophy (v2): **don't reimplement offloading — borrow it whole.** ComfyUI's memory
manager is mature and battle-tested, so `offloader/comfy/` is an unmodified snapshot of it, and
`adapter.py` is a thin bridge that makes turboCLI's diffusers pipelines drive *exactly* the code
ComfyUI drives. That gives 1:1 behaviour parity with ComfyUI and makes syncing to a newer ComfyUI
release a mechanical copy (`offloader/comfy/resync.md`). The runner only ever calls the seam —
`load_pipe`, `prepare`, `reclaim`, `release`, and friends — and never reaches inside.

## The two offload paths

- **Native** (works on CPU, CUDA and MPS/Apple). Weights sit in system RAM (mmap-backed); a
  `ModelPatcher` streams every layer to the compute device on each forward pass. Requires the
  model to fit in RAM.
- **VBAR** (CUDA, when the optional `comfy-aimdo` package is present). A `ModelPatcherDynamic`
  keeps as much of the model *permanently on the GPU* as free VRAM allows and streams only the
  remainder — directly from the model file, through the OS page cache. Because nothing is ever
  fully materialised in RAM, models bigger than RAM work (e.g. qwen's 41 GB transformer on a
  32 GB machine). Picked automatically when available.

And on pure CPU, where there is no GPU boundary at all, two placement modes exist:

- **`comfy`** — ComfyUI's default: convert everything to fp32 in RAM once, then compute plainly.
  Fastest, but needs roughly 2× the model's on-disk size in free RAM.
- **`stream`** — keep the weights mmap'd in their compact bf16 form and cast each layer to fp32
  on the fly, every forward. Slower per step, but RAM-bounded: big models run instead of OOMing.

The right one is auto-selected (does the fp32 model fit in ~85% of RAM?); `OFFLOADER_CPU_MODE`
forces a choice.

## A few technical details worth knowing

Each of these is one sentence here and a full section in `offloader.md`:

- **Precision juggling** — weights are stored compact (bf16/fp16) and computed in whatever the
  hardware does best (`manual_cast`), with an fp16 overflow clamp that prevents black images.
- **Scaled-fp8 quantization** — some engines ship 8-bit weights that stay fp8 on disk and are
  dequantized per forward, halving memory again at a small quality cost.
- **LoRA on the cast path** — LoRA weight patches are applied while a layer streams in, so they
  cost no extra resident memory.
- **Prompt-encode cache** — embeddings are cached per prompt (with RAM-pressure eviction), so only
  the first use of a prompt pays for the text encoder.
- **Node boundaries** — ComfyUI resets its streaming state between graph nodes; a diffusers
  pipeline has no nodes, so the adapter replays that teardown between encode, denoise and decode —
  skipping it broke fixed-seed determinism and starved the transformer of VRAM.
- **Pinned memory** — host RAM staging buffers are page-locked so GPU copies run asynchronously at
  full PCIe speed, which is what lets prefetch actually hide the transfers.
- **Determinism** — a fixed seed produces bit-identical images across runs; several of the
  mechanisms above exist precisely to preserve that.

## Where to go next

- `offloader.md` — the full design document: every mechanism, rationale, and benchmark tables.
- `doc/COMFYUI_OFFLOAD_MAP.md` — a map of ComfyUI's offloading internals with line references.
- `doc/BENCHMARKING.md` — how turbo and ComfyUI are compared head-to-head.
- `offloader/comfy/resync.md` — how the vendored ComfyUI snapshot is kept in sync.

## Keeping this document up to date

This file describes the current architecture and is revised with each turbo-offloader iteration:
whenever behaviour documented in `offloader.md` changes, the matching plain-English section here
changes with it. If the two ever disagree, `offloader.md` is right and this file has a bug.
