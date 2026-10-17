# 量化指南（ROCmFPX）

strix.cpp 原生支持 ROCmFPX 量化格式（GGML 类型 100-107）——这些格式上游 llama.cpp 加载不了。本指南说明怎么选、怎么量化，以及各家族的位宽。

[English](quantization-guide.en.md)

## 怎么选

大多数情况下照这张表选就够了：

| 你的需求 | 选这个 |
|:--|:--|
| 日常通用、速度优先 | `Q4_0_ROCMFP4_FAST`（4.25 bpw） |
| 质量优先（27B 级模型） | `Q4_0_ROCMFP4_STRIX`（注意力 K/V 质量配方） |
| 工具调用 / Agent / 编码 | `Q8_0_ROCMFPX_AGENT`（显存紧就降到 Q6/Q3 的 AGENT 档） |
| 显存告急 | `Q2_0_ROCMFPX` / `Q3_0_ROCMFPX` |
| 速度与质量折中 | `Q6_0_ROCMFPX` |

量化时建议：token embeddings 保留高精度（`LEAN`/`COHERENT` 变体会自动处理），其余张量按上表走。

## Q4_0_ROCMFP4 家族（4-bit）

| 类型 | 说明 | bpw |
|:--|:--|:--|
| Q4_0_ROCMFP4 | ROCmFP4 UE4M3-scale 基础版 | 4.50 |
| Q4_0_ROCMFP4_EVEN | 偶数张量转换（隐含 --pure） | 4.50 |
| Q4_0_ROCMFP4_LEAN | + Q5_K token embeddings | 4.60 |
| Q4_0_ROCMFP4_COHERENT | + Q6_K token embeddings | 4.70 |
| Q4_0_ROCMFP4_FAST | single-scale 速度布局 | 4.25 |
| Q4_0_ROCMFP4_FAST_EVEN | fast 偶数张量转换（隐含 --pure） | 4.25 |
| Q4_0_ROCMFP4_FAST_COHERENT | fast + Q6_K token embeddings | ~4.45 |
| Q4_0_ROCMFP4_STRIX | Strix Halo attn-K/V 质量配方 | ~4.49 |
| Q4_0_ROCMFP4_STRIX_LEAN | Strix K/V + Q5_K token embeddings | ~4.38 |

## Qx_ROCMFPX 家族

| 类型 | 说明 | bpw |
|:--|:--|:--|
| Q2_0_ROCMFPX | 2-bit S40 codebook + dual UE4M3 scales | 2.50 |
| Q3_0_ROCMFPX | 3-bit 参考布局 | 3.50 |
| Q6_0_ROCMFPX | 6-bit 参考布局 | 6.50 |
| Q8_0_ROCMFPX | 8-bit 参考布局 | 8.25 |
| Q3_0_ROCMFPX_AGENT | agent/工具调用连贯性 Q3 路由 | 3.50 |
| Q6_0_ROCMFPX_AGENT | agent/工具调用连贯性 Q6 路由 | 6.50 |
| Q8_0_ROCMFPX_AGENT | agent/工具调用连贯性 Q8 路由 | 8.25 |
| Q6_0_ROCMFPX_LEAN | 尺寸/速度偏向 Q6 路由 | 6.50 |
| Q6_0_ROCMFPX_AGENT_LEAN | agent Q6 路由（无 Q8-heavy 提升） | 6.50 |

## KV-cache 类型（运行时参数）

| 类型 | bpw | 说明 |
|:--|:--|:--|
| TURBO3_0 | 3.50 | 深上下文省内存档（质量折损较明显，先做数值门） |
| TURBO4_0 | 4.50 | 省内存与质量的中间档 |

> KV-cache 类型通过 `-ctk/-ctv` 指定，属运行时参数，不是量化输出类型。换内存不换速度——深上下文场景优先 f16 KV。

## 量化命令

```bash
# 基础量化
build/bin/llama-quantize --allow-requantize <源模型.gguf> <输出.gguf> Q4_0_ROCMFP4_FAST 32

# Agent 路由变体（工具调用/编码场景）
build/bin/llama-quantize --allow-requantize <源模型.gguf> <输出.gguf> Q8_0_ROCMFPX_AGENT 32
```

## GPU 计算路径说明

fp8 家族（Q2_0/Q3_0/Q6_0/Q8_0_ROCMFPX）的 GPU 计算路径（MMVQ + MMQ）已在 RDNA3.5（gfx1151）启用，热路径解码自 v2026.9.22 起完成无分支优化（Q8 档 prefill 追平/反超标准 Q8_0）；其他架构自动回退 dequant + hipBLAS（可用、较慢）。

## 已知问题

- `q3_0_rocmfpx` 作 K 侧会自动升格为 `q6_0_rocmfpx`（coherency floor）；这是行为说明，不是待修问题。
