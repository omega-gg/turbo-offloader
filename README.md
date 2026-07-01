# turbo-aimdo

turbo-aimdo is a [turboCLI](https://omega.gg/turboCLI) high performance offloader based on [comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo).
Paired with a compliant CUDA GPU it improves generate times substantially through optimal memory
allocations when the model does not fit into VRAM, even on a very low VRAM GPU.

Credits: [comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo)

## Vendored ComfyUI snapshot

turbo-aimdo's offloader is a **byte-for-byte port** of ComfyUI's memory-management / model-patcher /
ops subsystem (under `aimdo/comfy/`), driven through the thin adapter in `aimdo/adapter.py`. The
port is pinned to exact upstream commits — bump these together with the vendored files and re-apply
the short edit list in [`aimdo/comfy/resync.md`](aimdo/comfy/resync.md):

| dependency  | commit                                     | version                 |
|-------------|--------------------------------------------|-------------------------|
| ComfyUI     | `dd17debce517f8818ae9910b437cb1ebaa673176` | `v0.0.2-3160-gdd17debc` |
| comfy-aimdo | `afa70d91ec9f6e1ab6758089d1b551f0269b6457` | `0.4.10`                |

Validated against the turboCLI diffusion runtime: `torch 2.12.1+cu130`, `diffusers 0.39.0.dev0`,
`transformers 5.12.1`, `accelerate 1.14.0`. The same commits are recorded in
[`aimdo/comfy/__init__.py`](aimdo/comfy/__init__.py) and `resync.md` — keep all three in sync when
re-vendoring.

## Contribute

PR(s) are welcomed

## Platforms

- Windows 10 and later
- Linux 64 bit

## Requirements

- [turboCLI](https://omega.gg/turboCLI) and a CUDA compliant GPU

## License

Copyright (C) 2026-2026 turbo-aimdo authors | https://omega.gg/turbo-aimdo

### Authors

- Benjamin Arnaud aka [bunjee](https://bunjee.me) | <bunjee@omega.gg>

### GNU General Public License Usage

turbo-aimdo may be used under the terms of the GNU General Public License version 3 as published
by the Free Software Foundation and appearing in the LICENSE.md file included in the packaging
of this file. Please review the following information to ensure the GNU General Public License
requirements will be met: https://www.gnu.org/licenses/gpl.html.
