# Plan: `dummy.md` — plain-English turbo-offloader documentation

## Context

turbo-offloader has one main doc, `implementation.md` — dense and aimed at people who already know
ComfyUI internals and diffusion. There is no gentle entry point. `dummy.md` is that entry point:
a brief-but-complete, plain-English explanation of what turbo-offloader does, walking step by
step through the architecture and gradually getting more technical, readable by technical people
with no diffusion background. It is referenced from `implementation.md` with a note that it is
kept up to date as the offloader evolves.

## Files

1. `dummy.md` (new, top level beside `implementation.md`, since the two cross-reference each
   other).
2. `implementation.md` — one added pointer in the intro: `dummy.md` is the plain-English
   introduction, maintained alongside offloader iterations.

## Content outline for `dummy.md`

Progressive disclosure — each section slightly more technical than the last. All text and
diagrams wrap at 99 columns. ASCII diagrams (not mermaid) so they render in any viewer and
respect the column limit.

1. **What is this?** — models are 4–55 GB, most GPUs can't hold them; turbo-offloader lets
   turboCLI run them anyway (small GPUs, CPUs, models bigger than RAM) at ComfyUI-matching
   speeds. One sentence on the GPL/LGPL split and the seam.
2. **30-second primer** — the diffusion pipeline in plain English: prompt → text encoder →
   diffusion transformer × N steps (the heavy part) → VAE decode → PNG. ASCII diagram.
3. **The core problem and the core trick** — weights don't fit VRAM; transformers run layer by
   layer, so stream weights just in time and let prefetch hide the copies. Conveyor diagram.
4. **Step by step: one generation** — init before torch, mmap load (map don't read),
   `load_models_gpu` placement, prompt encode (cached, unpadded), denoise loop with streaming,
   tiled-fallback VAE decode, save + reclaim.
5. **The architecture: three pieces** — runner (LGPL) → seam (8 functions) → `adapter.py` →
   vendored ComfyUI (`offloader/comfy/`, byte-for-byte v0.27.0). The v2 philosophy: vendor
   verbatim + thin adapter = 1:1 parity, mechanical resync.
6. **The two offload paths** — native (`ModelPatcher`, RAM-resident mmap, all devices) vs VBAR
   (`ModelPatcherDynamic`, partial GPU residency, disk→VRAM, bigger-than-RAM); plus the CPU
   `comfy`/`stream` placement modes and their auto-selection.
7. **Technical details worth knowing** — one sentence each: precision juggling + fp16 clamp,
   scaled-fp8, on-cast LoRA, prompt-encode cache, node boundaries/determinism, pinned memory.
   Each points to `implementation.md` for depth.
8. **Where to go next** — links to `implementation.md`, `doc/COMFYUI_OFFLOAD_MAP.md`,
   `doc/BENCHMARKING.md`, `offloader/comfy/resync.md`.
9. **Keeping this document up to date** — revised with each iteration; if it disagrees with
   `implementation.md`, `implementation.md` is right.

## Style rules

- 99-column wrap everywhere, including diagrams.
- Plain English first; every technical term (VRAM, mmap, latents, VAE, DiT) defined in one
  clause at first use. No unexplained jargon.
- Match existing doc conventions: `#`/`##` headings, backticked identifiers, **bold** key terms,
  relative links, no GPL banner in markdown.
- Brief but complete — no benchmark tables, no line-number archaeology (that is
  `implementation.md`'s job); `dummy.md` explains *what and why*, `implementation.md` explains
  *exactly how and how fast*.

## Verification

- Markdown render check — diagrams aligned, links resolve.
- No line exceeds 99 columns in `dummy.md` or the touched part of `implementation.md`.
- Every factual claim checked against the code (`offloader/__init__.py`, `adapter.py`, the
  runner's `core.py`) — no invented behaviour.
