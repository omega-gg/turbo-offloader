# turbo-offloader

turbo-offloader is a [turboCLI](https://omega.gg/turboCLI) high performance offloader based on [ComfyUI](https://github.com/comfy-org/comfyui).
It improves generate times substantially through optimal memory allocations when the model does not
fit into RAM / VRAM, even on a low end GPU. It's particularly efficient for CUDA GPU(s) but also
works on CPU and Apple MPS.

- [Dummy](dummy.md) - plain-English introduction, no diffusion background required
- [Implementation](implementation.md) - architecture and implementation choices, kept up to date

## ComfyUI

turbo-offloader ports ComfyUI's memory-management / model-patcher / ops subsystem
(under `offloader/comfy`), driven through the thin adapter in `offloader/adapter.py`. The port is
pinned to exact upstream commits. Bump these together with the vendored files and follow the
re-sync steps in [`offloader/comfy/resync.md`](offloader/comfy/resync.md):

| dependency    | commit                                     | version                 |
|---------------|--------------------------------------------|-------------------------|
| ComfyUI       | `de0125d9aefc225ec1a6f787a3a806730b772678` | `v0.39.1`               |
| comfy-aimdo   | `3b8e8c162efeb9470d912609a7a6e7a2b1c693ec` | `0.5.5`                 |
| comfy-kitchen | `be003b7c23c5b01328657955b8bc5d3f073d868e` | `0.2.37`                |

## Credits

- [ComfyUI](https://github.com/comfy-org/comfyui)
- [comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo)
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)

## Contribute

PR(s) are welcomed

## Platforms

- Windows 10 and later
- macOS 64 bit
- Linux 64 bit

## Requirements

- [turboCLI](https://omega.gg/turboCLI) and a CUDA compliant GPU

## License

Copyright (C) 2026-2026 turbo-offloader authors | https://omega.gg/turbo-offloader

### Authors

- Benjamin Arnaud aka [bunjee](https://bunjee.me) | <bunjee@omega.gg>

### GNU General Public License Usage

turbo-offloader may be used under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation and appearing in the LICENSE.md file included in the
packaging of this file. Please review the following information to ensure the GNU General Public
License requirements will be met: https://www.gnu.org/licenses/gpl.html.
