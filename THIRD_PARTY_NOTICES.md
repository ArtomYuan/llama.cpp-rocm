# Third-Party Notices

This repository is based on `llama.cpp` and preserves the upstream MIT license
(`LICENSE`, Copyright (c) 2023-2026 The ggml authors) and the third-party
license files that ship with the tree.

## ROCmFP4 / ROCmFPx formats

- Source: https://github.com/charlie12345/ROCmFPX
- License: MIT (identical text and copyright line to this repository's `LICENSE`)
- Taken over as-is: `ggml/rocmfp4/` and `ggml/rocmfpx/` (format definitions,
  CPU reference quantizers/dequantizers, HIP helper headers, reference tests),
  the CPU vec-dot kernels in `ggml/src/ggml-cpu/ggml-cpu.c`, the HIP
  dequant/copy/FlashAttention integration across `ggml/src/ggml-cuda/`, the
  quantization recipes in `src/llama-quant.cpp`, and the TurboQuant KV-cache
  type definitions that travel with them.
- Re-expressed here rather than copied: the FP4 MMQ load-tile kernels
  (`mmq-load-tiles.cuh`, `mmq-config-*.cuh` entries), which the source tree
  implements against the pre-refactor MMQ architecture.

The upstream `llama.cpp` bundled components (cpp-httplib, nlohmann/json, stb,
etc.) are unchanged from upstream; see their respective headers.
