# strix.cpp

> Formerly `llama.cpp-rocm` (renamed 2026-10-17); old links redirect automatically, history and release line continue.

**A llama.cpp engine deeply tuned for Strix Halo (gfx1151) — ROCmFPX-native.**

[中文](README.md)

---

## What it is

strix.cpp is an independently maintained llama.cpp branch. On top of a fused base (upstream sync + ROCmFPX quantization + multimodal vision), it applies **kernel-level specialization for gfx1151** — the reason this engine exists as its own project.

| Pillar | Content |
|---|---|
| Upstream llama.cpp (continuously synced) | All baseline inference capabilities; current sync point `091b6a7fc` (2026-10-14, window 552 commits, merge commit `b5a7b6118`) |
| ROCmFPX quantization | ROCmFP2/3/4/6/8 + TurboQuant (GGML types 100-107), not loadable by upstream |
| Multimodal vision | mtmd vision stack + LocateAnything grounding projector (`--special` emits `<ref>/<box>`) |
| **gfx1151 specialization** | Attention kernel config split, full MMQ quant paths, branch-free hot-path decode (see below) |

**One binary**: ROCmFPX-quantized text models + vision/grounding models run without switching engines.

## Specializations (gfx1151)

Every item is validated with same-moment back-to-back A/B, byte-equivalence gates and numeric gates; profiling details and reproduction commands live in the [kernel optimization record](docs/development/kernel-optimization.en.md).

| Item | Content | Measured gain |
|---|---|---|
| **Attention kernel config split** | FA configs are keyed by `(ncols1,ncols2)`: only the `(8,8)` and `(16,4)` instances use the 512-thread config (-60% register spills); all other shapes stay on stock 256 threads | kv=16k deep contexts **+22.6% / +20.2%**; P1 `(32,2)` shape unaffected |
| Branch-free hot-path decode (fp8 family) | branch-free ue4m3 scale decode | prefill **+11.4%** (Q8_0_ROCMFPX pp2048) |
| Full MMQ quant paths | four-layer MMVQ + MMQ wiring on RDNA3.5 (incl. MoE) | prefill ~**2x** (pp512 +105%, pp2048 +109%) |
| KV quantization line | TURBO3_0/TURBO4_0 + q4fast trade-off tier | q6x deep context d4k **11.7 -> 724 t/s** (62x) |
| Pure ROCm HIP backend | HIP-only (no Vulkan), device locked to ROCm0 | no backend ambiguity |

## Features

- ROCmFPX quantization: ROCmFP2/3/4/6/8 + TurboQuant (FAST variants), requantize from F32 or quantized sources
- Multimodal vision: moonvit vision tower, Gemma4 image processing aligned with HF, LocateAnything grounding (`<ref>/<box>`)
- Performance: gfx1151 HIP kernel adaptation, pure ROCm0 backend, 69+ t/s on 35B.A3B MoE
- OpenAI-compatible API (llama-server)

## Measured performance

| Item | Result |
|---|---|
| Ornith 35B.A3B (Q4_ROCmFPX_FAST) | tg32 **≈67 t/s** (decode, via MMVQ - does not go through MMQ; r5 re-run 2026-09-21) |
| Same model, prefill (pp512) | **≈1460 t/s** (MMQ on; r5 re-run 2026-09-21) |
| Qwen3-Embedding-8B (Q8_0_ROCMFPX) | prefill pp2048 **1233 t/s** (v2026.9.22 decode optimization; 1107 before, +11.4%) |
| Qwen 27B (Q4_ROCmFPX_FAST) | tg32 12.17 t/s |
| Attention kernel gqa8 `(8,8)`, kv=16k | **14.08 TFLOPS** (512-thread config; stock 11.48, **+22.6%**; same-moment A/B, v2026.10.17) |
| Attention kernel `(16,4)`, kv=16k | **10.63 TFLOPS** (stock 8.84, **+20.2%**; same-moment A/B, v2026.10.17) |
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

> The fp8 family (Q2_0/Q3_0/Q6_0/Q8_0_ROCMFPX) GPU compute paths (MMVQ + MMQ) are enabled on RDNA3.5 (gfx1151), with a branch-free hot-path decode since v2026.9.22 (Q8 prefill now matches/beats std Q8_0); other architectures fall back to dequant + hipBLAS (functional, slower).

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

- Dual-scale `Q4_0_ROCMFP4` (not `_FAST`): the MMQ numerical defect was fixed in v2026.9.16; older builds can set `GGML_HIP_NO_ROCMFP4_MMQ=1` to work around it.
- **Deep-context precision edge**: f16-KV attention diverges from the f32 reference as context grows (kv>=32k: ~8e-4 at 32k, ~1.5e-3 at 64k, ~3.2e-3 at 131k; suite threshold 5e-4). This is pre-existing behaviour (identical on stock) and amounts to ~0.1-0.3% of model output — keep it in mind for very-long-context heavy workloads.

> Development docs: [kernel optimization record](docs/development/kernel-optimization.en.md) (profiling, fixes and validation for kernel-level changes); contribution workflow in [CONTRIBUTING.md](CONTRIBUTING.md) (Chinese).
