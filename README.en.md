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

All ROCmFPX formats supported (llama-quantize output types):

**Q4_0_ROCMFP4 family (4-bit):**

| Type | Description | bpw |
|:--|:--|:--|
| Q4_0_ROCMFP4 | ROCmFP4 UE4M3-scale base | 4.50 |
| Q4_0_ROCMFP4_EVEN | even tensor conversion (implies --pure) | 4.50 |
| Q4_0_ROCMFP4_LEAN | + Q5_K token embeddings | 4.60 |
| Q4_0_ROCMFP4_COHERENT | + Q6_K token embeddings | 4.70 |
| Q4_0_ROCMFP4_FAST | single-scale speed layout | 4.25 |
| Q4_0_ROCMFP4_FAST_EVEN | fast even tensor conversion (implies --pure) | 4.25 |
| Q4_0_ROCMFP4_FAST_COHERENT | fast + Q6_K token embeddings | ~4.45 |
| Q4_0_ROCMFP4_STRIX | Strix Halo attn-K/V quality recipe | ~4.49 |
| Q4_0_ROCMFP4_STRIX_LEAN | Strix K/V + Q5_K token embeddings | ~4.38 |

**Qx_ROCMFPX family:**

| Type | Description | bpw |
|:--|:--|:--|
| Q2_0_ROCMFPX | 2-bit S40 codebook + dual UE4M3 scales | 2.50 |
| Q3_0_ROCMFPX | 3-bit reference layout | 3.50 |
| Q6_0_ROCMFPX | 6-bit reference layout | 6.50 |
| Q8_0_ROCMFPX | 8-bit reference layout | 8.25 |
| Q3_0_ROCMFPX_AGENT | agent/tool-call coherent Q3 routing | 3.50 |
| Q6_0_ROCMFPX_AGENT | agent/tool-call coherent Q6 routing | 6.50 |
| Q8_0_ROCMFPX_AGENT | agent/tool-call coherent Q8 routing | 8.25 |
| Q6_0_ROCMFPX_LEAN | size/speed-biased Q6 routing | 6.50 |
| Q6_0_ROCMFPX_AGENT_LEAN | agent Q6 routing (no Q8-heavy boosts) | 6.50 |

**KV-cache types** (runtime parameters, not quantize outputs): TURBO3_0 (3.50 bpw) / TURBO4_0 (4.50 bpw)

```bash
# Basic quantize
build/bin/llama-quantize --allow-requantize <source.gguf> <output.gguf> Q4_0_ROCMFP4_FAST 32

# Agent routing variant (tool-call/coding scenarios)
build/bin/llama-quantize --allow-requantize <source.gguf> <output.gguf> Q8_0_ROCMFPX_AGENT 32
```

### Inference service (systemd user service example)

```ini
[Service]
Type=simple
Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1
Environment=HIP_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/path/to/engine/bin
ExecStart=/path/to/engine/bin/llama-server --special -m /path/to/model.gguf --port <port>
```

- llama-server serves an OpenAI-compatible API (default `/v1`), pick your own `--port`
- Vision/grounding models need `--special` (enables `<ref>/<box>` special tokens); plain text models do not

## Known limitations

- ROCmFPX MMQ kernels for gfx1151 not adapted yet - MMQ disabled for MoE (measured 69.17 t/s vs 67.26 t/s with MMQ on; re-enable once kernels are ready)
