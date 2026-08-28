# llama.cpp-rocm

A fusion engine for AMD Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151): upstream llama.cpp master + **ROCmFPX quantization** + **multimodal vision tower** in a single binary.

## What is this

An independently maintained branch of llama.cpp that merges three capabilities into one engine:

| Source | Capability |
|---|---|
| Upstream llama.cpp (continuously synced) | All baseline inference capabilities |
| charlie12345/ROCmFPX | ROCmFPX quantization formats (ROCmFP2/3/4/6/8 + TurboQuant) |
| yuuko-eth mtmd-grounders | Multimodal vision tower (incl. LocateAnything grounding projector) |

**One binary**: runs both ROCmFPX-quantized models (text) and vision/grounding models, no engine switching.

## Differences from upstream

- **ROCmFPX quantization**: upstream llama.cpp cannot load ROCmFP4/FP8 formats - this engine supports them natively (GGML types 100-107) and ships `llama-quantize` for conversion
- **Multimodal vision**: merged mtmd vision stack, supports vision models and grounding models (`--special` mode emits `<ref>/<box>`)
- **Pure ROCm HIP backend**: HIP-only build (no Vulkan), device locked to ROCm0 - no backend ambiguity
- **MMQ decision**: gfx1151 MMQ kernels are not ready yet; MMQ is disabled for MoE (measured 69.17 t/s vs 67.26 t/s default)
- **No upstream CI**: upstream CI matrix removed (validated by local builds)

## Features

- ROCmFPX quantization: ROCmFP2/3/4/6/8 + TurboQuant (FAST variants), requantize from F32 or quantized sources
- Multimodal vision: moonvit vision tower, Gemma4 image processing aligned with HF, LocateAnything grounding (`<ref>/<box>`)
- Performance: gfx1151 HIP kernel adaptation, pure ROCm0 backend, 69+ t/s on 35B.A3B MoE
- OpenAI-compatible API (llama-server)

## Measured performance

| Item | Result |
|---|---|
| Ornith 35B.A3B (Q4_ROCmFPX_FAST) | tg32 **69.17 t/s** (mmq off) |
| Same model, mmq on | 67.26 t/s (MMQ not ready - disabled is optimal) |
| Qwen 27B (Q4_ROCmFPX_FAST) | tg32 12.17 t/s |
| Pure ROCm0 vs old Vulkan path | combined engine+backend improvement (see CHANGELOG) |

## Deployment

### Build (ROCm 10.0.0 / gfx1151)

```bash
export ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build -j $(nproc) --target llama-server llama-quantize llama-bench
```

Artifacts land in `build/bin/` (llama-server / llama-quantize / llama-bench + shared libraries).

### Quantize

```bash
build/bin/llama-quantize --allow-requantize <source.gguf> <output.gguf> Q4_0_ROCMFP4_FAST 32
```

### Inference service (systemd user service example)

```ini
[Service]
Type=simple
Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1
Environment=HIP_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/path/to/engine/bin
ExecStart=/path/to/engine/bin/llama-server --special -m /path/to/model.gguf --port 8030
```

- Text models: `llama-server -m <model.gguf> --port 8010`
- Vision/grounding models: `llama-server --special -m <model.gguf> --port 8030` (`--special` enables `<ref>/<box>` special tokens)

### Sync upstream

```bash
git fetch official && git merge official/master
```

## Known limitations

- ROCmFPX MMQ kernels for gfx1151 not adapted yet - MMQ disabled for MoE (measured 69.17 t/s vs 67.26 t/s with MMQ on; re-enable once kernels are ready)
