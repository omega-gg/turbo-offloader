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
pinned to exact upstream commits. Bump these together with the vendored files and re-apply the
short edit list in [`offloader/comfy/resync.md`](offloader/comfy/resync.md):

| dependency    | commit                                     | version                 |
|---------------|--------------------------------------------|-------------------------|
| ComfyUI       | `bb131be9e83d2f773c90f1d6f1e4b248a498c8c5` | `v0.27.0`               |
| comfy-aimdo   | `afa70d91ec9f6e1ab6758089d1b551f0269b6457` | `0.4.10`                |
| comfy-kitchen | `43b413e402c93b21b14f437758bcac0cd5130bd4` | `0.2.16`                |

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
