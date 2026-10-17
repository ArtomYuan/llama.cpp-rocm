# strix.cpp

**A llama.cpp engine deeply tuned for AI MAX+ 395, ROCmFPX-native.**

strix.cpp is an independently maintained llama.cpp branch. It does three things well: it continuously syncs upstream; it natively supports ROCmFPX quantization (formats upstream cannot load); and it folds in multimodal vision. But its reason to exist is kernel-level specialization for this one chip — Strix Halo (gfx1151), up to 128 GiB of unified memory: the same model and the same quant, with an attention kernel that runs over twenty percent faster.

Formerly `llama.cpp-rocm` (renamed 2026-10-17; old links redirect automatically).

[中文](README.md)

---

## Highlights

- **Quantization that just works**: the full ROCmFP4 / ROCmFPX family (2-bit through 8-bit, plus agent-routing variants) with `llama-quantize` included. See the [quantization guide](docs/quantization-guide.en.md).
- **Multimodal**: vision models and LocateAnything grounding models work out of the box (`--special` emits `<ref>/<box>`).
- **Tuned for this chip**: per-shape attention-kernel configs, plus the FP4-specific K tile on the prefill GEMM — 40~66% faster prefill end to end (method in the [kernel optimization record](docs/development/kernel-optimization.en.md)).
- **One binary**: text, vision and grounding models without switching engines.
- **Pure ROCm**: HIP-only, no Vulkan, device locked to ROCm0 — no backend ambiguity.

## Measured performance

Same machine, same model, same quant, same flags; the control arm is the stock config of the same source tree (i.e. the previous production build):

| Scenario (Qwen3.8-27B, Q4_0_ROCMFP4_FAST) | stock | strix.cpp | Delta |
|:--|--:|--:|--:|
| Prefill pp2048 (short context) | 437 t/s | **721 t/s** | **+66%** |
| Prefill pp16384 (deep context) | 389 t/s | **614 t/s** | **+55%** |
| Prefill pp65536 (very deep) | 295 t/s | **414 t/s** | **+40%** |
| Decode tg256 | 14.0 t/s | 14.0 t/s | flat (decode takes a different path) |
| Decode @ depth 16k | 12.69 t/s | 12.69 t/s | flat |
| Decode @ depth 64k | 9.714 t/s | 9.710 t/s | flat |

(The other quant tiers: `Q8_0_ROCMFPX` and `Q6_0_ROCMFPX` are outside this change and match their previous records; `Q4_0_ROCMFP4_STRIX` belongs to the same ROCmFP4 family but was not re-measured on its own.)

**How to read this**: the prefill wins come from two cuts — per-shape attention-kernel configs (**+22.6%** in the kernel itself) and the FP4-specific K tile on the prefill GEMM (**+90%** in the kernel, still +40~66% after dilution by the whole network). Decode is unchanged in both versions: it takes the MMVQ path, which this change does not touch. Item-by-item evidence is in the [kernel optimization record](docs/development/kernel-optimization.en.md).

## Quickstart

### Build

```bash
export ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build -j $(nproc) --target llama-server llama-quantize llama-bench
```

Artifacts land in `build/bin/`.

### Run

```bash
build/bin/llama-server -m <model.gguf> --port 8080
```

OpenAI-compatible API (default `/v1`). Vision/grounding models need `--special`; plain text models do not.

### Keep it running as a service

```ini
[Service]
Type=simple
Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1
Environment=HIP_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/path/to/engine/bin
ExecStart=/path/to/engine/bin/llama-server --special -m /path/to/model.gguf --port <port>
```

## Documentation

- [Quantization guide](docs/quantization-guide.en.md): family tables, how to choose, quantize commands, KV types
- [Kernel optimization record](docs/development/kernel-optimization.en.md): evidence and reproduction for every kernel specialization
- [Contributing](CONTRIBUTING.md) (Chinese)

## Known limitations

- **MUL_MAT_ID (MoE routing) with duplicate expert ids**: the `dup=1` cases (the same expert selected more than once) produce wrong values, and extreme shapes can fault the GPU; the baseline behaves the same (not introduced here, and real model routing normally avoids it).
- **Deep-context precision edge**: beyond 32k tokens, f16-KV attention slowly drifts from the f32 reference (about 3e-3 at 131k). This is pre-existing llama.cpp behaviour (the same on upstream) and amounts to roughly a few tenths of a percent of model output — keep it in mind for very-long-context heavy workloads.
