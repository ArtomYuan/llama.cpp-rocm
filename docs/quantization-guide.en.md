# Quantization guide (ROCmFPX)

strix.cpp natively supports the ROCmFPX quantization formats (GGML types 100-107), which upstream llama.cpp cannot load. This guide covers how to choose, how to quantize, and the bit widths of each family.

[中文](quantization-guide.md)

## How to choose

In most cases this table is all you need:

| Your case | Pick this |
|:--|:--|
| Everyday use, speed first | `Q4_0_ROCMFP4_FAST` (4.25 bpw) |
| Quality first (27B-class models) | `Q4_0_ROCMFP4_STRIX` (attention K/V quality recipe) |
| Tool calling / agents / coding | `Q8_0_ROCMFPX_AGENT` (drop to the Q6/Q3 AGENT tiers if memory is tight) |
| Low on memory | `Q2_0_ROCMFPX` / `Q3_0_ROCMFPX` |
| Speed/quality middle ground | `Q6_0_ROCMFPX` |

Quantizing advice: keep token embeddings at high precision (the `LEAN`/`COHERENT` variants do this for you) and follow the table above for the remaining tensors.

## Q4_0_ROCMFP4 family (4-bit)

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

## Qx_ROCMFPX family

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

## KV-cache types (runtime parameters)

| Type | bpw | Description |
|:--|:--|:--|
| TURBO3_0 | 3.50 | deep-context memory saver (noticeable quality loss; gate with numeric tests first) |
| TURBO4_0 | 4.50 | middle ground between memory and quality |

> KV-cache types are selected via `-ctk/-ctv` at runtime; they are not quantize output types. They trade memory, not speed — prefer f16 KV for deep-context workloads.

## Quantize commands

```bash
# Basic quantize
build/bin/llama-quantize --allow-requantize <source.gguf> <output.gguf> Q4_0_ROCMFP4_FAST 32

# Agent routing variant (tool-call/coding scenarios)
build/bin/llama-quantize --allow-requantize <source.gguf> <output.gguf> Q8_0_ROCMFPX_AGENT 32
```

## GPU compute paths

The fp8 family (Q2_0/Q3_0/Q6_0/Q8_0_ROCMFPX) GPU compute paths (MMVQ + MMQ) are enabled on RDNA3.5 (gfx1151), with a branch-free hot-path decode since v2026.9.22 (Q8 prefill now matches/beats std Q8_0); other architectures fall back to dequant + hipBLAS (functional, slower).

## Known issues

- `q3_0_rocmfpx` on the K side is automatically promoted to `q6_0_rocmfpx` (coherency floor); this is a behaviour note, not an unfixed issue.
