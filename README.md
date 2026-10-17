# strix.cpp

> 原名 `llama.cpp-rocm`（2026-10-17 更名）；旧链接由 GitHub 自动重定向，历史与发布线延续。

**为 AI MAX+ 395（Strix Halo / gfx1151）深度特调、原生支持 ROCmFPX 的 llama.cpp 引擎。**

[English](README.en.md)

---

## 这是什么

strix.cpp 是 llama.cpp 的独立维护分支。在「官方同步 + ROCmFPX 量化 + 多模态视觉」的融合底座之上，**针对 gfx1151 做内核级深度特调**——这是本引擎区别于其他 llama.cpp 分支的立身之本。

| 支柱 | 内容 |
|---|---|
| 官方 llama.cpp（持续同步 master） | 全部基础推理能力；当前同步点 `091b6a7fc`（2026-10-14，窗口 552 commits，merge commit `b5a7b6118`） |
| ROCmFPX 量化 | ROCmFP2/3/4/6/8 + TurboQuant（GGML 类型 100-107），上游无法加载 |
| 多模态视觉 | mtmd 视觉栈 + LocateAnything 定位投影器（`--special` 输出 `<ref>/<box>`） |
| **gfx1151 深度特调** | 注意力内核配置分裂、MMQ 全量化路径、无分支热路径解码（见下节） |

**一个二进制**：ROCmFPX 量化模型（文本）+ 视觉/定位模型都能跑，无需切换引擎。

## 特调明细（gfx1151）

每项都经「同刻背靠背 A/B + 字节等价 + 数值门」验证，定位过程与复现命令见[内核优化记录](docs/development/kernel-optimization.md)。

| 特调项 | 内容 | 实测收益 |
|---|---|---|
| **注意力内核配置分裂** | FA 配置按 `(ncols1,ncols2)` 分键：仅 `(8,8)` 与 `(16,4)` 实例启用 512 线程配置（寄存器溢出 −60%），其余形状保持 stock 256 线程 | kv=16k 深上下文 **+22.6% / +20.2%**；P1 `(32,2)` 形状零代价 |
| 无分支热路径解码（fp8 家族） | ue4m3 scale 解码去分支 | prefill **+11.4%**（Q8_0_ROCMFPX pp2048） |
| MMQ 全量化路径 | RDNA3.5 的 MMVQ + MMQ 四层接线（含 MoE） | prefill 约 **2×**（pp512 +105%、pp2048 +109%） |
| KV 量化线 | TURBO3_0/TURBO4_0 + q4fast 权衡档 | q6x 深上下文 d4k **11.7 → 724 t/s**（62×） |
| 纯 ROCm HIP 后端 | 仅 HIP（无 Vulkan），设备锁定 ROCm0 | 无后端选择歧义 |

## 功能

- ROCmFPX 量化：ROCmFP2/3/4/6/8 + TurboQuant（FAST 变体），从 F32/量化源 requantize
- 多模态视觉：moonvit 视觉塔、Gemma4 图像处理对齐 HF、LocateAnything 定位（`<ref>/<box>`）
- 高性能：gfx1151 HIP kernel 适配、纯 ROCm0 后端、MoE 场景 69+ t/s（35B.A3B）
- OpenAI 兼容 API（llama-server）

## 优化实测

| 项 | 结果 |
|---|---|
| Ornith 35B.A3B（Q4_ROCmFPX_FAST） | tg32 **≈67 t/s**（decode，走 MMVQ 不经 MMQ；2026-09-21 r5 复测） |
| 同模型 prefill（pp512） | **≈1460 t/s**（MMQ 开；2026-09-21 r5 复测） |
| Qwen3-Embedding-8B（Q8_0_ROCMFPX） | prefill pp2048 **1233 t/s**（v2026.9.22 解码优化；前 1107，+11.4%） |
| Qwen 27B（Q4_ROCmFPX_FAST） | tg32 12.17 t/s |
| 注意力内核 gqa8 `(8,8)`，kv=16k | **14.08 TFLOPS**（512 线程配置；对照 stock 11.48，**+22.6%**；v2026.10.17 同刻对照） |
| 注意力内核 `(16,4)`，kv=16k | **10.63 TFLOPS**（对照 stock 8.84，**+20.2%**；v2026.10.17 同刻对照） |
| 纯 ROCm0 后端 vs 旧 Vulkan 路径 | 提升为引擎+后端综合效应（详见 CHANGELOG） |

## 部署

### 构建（ROCm 10.0.0 / gfx1151）

```bash
export ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build -j $(nproc) --target llama-server llama-quantize llama-bench
```

产物在 `build/bin/`（llama-server / llama-quantize / llama-bench + 共享库）。

### 量化

支持全部 ROCmFPX 格式（llama-quantize 输出类型）：

**Q4_0_ROCMFP4 家族（4-bit）：**

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

**Qx_ROCMFPX 家族：**

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

> fp8 家族（Q2_0/Q3_0/Q6_0/Q8_0_ROCMFPX）的 GPU 计算路径（MMVQ + MMQ）已在 RDNA3.5（gfx1151）启用，热路径解码自 v2026.9.22 起完成无分支优化（Q8 档 prefill 追平/反超标准 Q8_0）；其他架构自动回退 dequant + hipBLAS（可用、较慢）。

**KV-cache 类型**（运行时参数，非量化输出）：TURBO3_0（3.50 bpw）/ TURBO4_0（4.50 bpw）

```bash
# 基础量化
build/bin/llama-quantize --allow-requantize <源模型.gguf> <输出.gguf> Q4_0_ROCMFP4_FAST 32

# Agent 路由变体（工具调用/编码场景）
build/bin/llama-quantize --allow-requantize <源模型.gguf> <输出.gguf> Q8_0_ROCMFPX_AGENT 32
```

### 推理服务（systemd 用户服务示例）

```ini
[Service]
Type=simple
Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1
Environment=HIP_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/path/to/engine/bin
ExecStart=/path/to/engine/bin/llama-server --special -m /path/to/model.gguf --port <port>
```

- llama-server 提供 OpenAI 兼容 API（默认 `/v1`），`--port` 自选
- 视觉/定位模型需 `--special`（启用 `<ref>/<box>` 特殊 token）；纯文本模型不需要

## 已知限制

- 双尺度 `Q4_0_ROCMFP4`（非 `_FAST`）：MMQ 数值缺陷已在 v2026.9.16 修复；旧版本可设 `GGML_HIP_NO_ROCMFP4_MMQ=1` 规避。
- **深水精度边缘**：f16 KV 的注意力在 kv≥32k 时与 f32 参考的误差随深度增长（32k ≈ 8e-4、64k ≈ 1.5e-3、131k ≈ 3.2e-3；套件阈值 5e-4）。属既有行为（stock 同款），对模型输出影响 ~0.1~0.3% 量级——超长上下文重负载场景请留意。

> 开发文档：[内核优化记录](docs/development/kernel-optimization.md)（本库内核级改动的定位、修复与验证）；贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。
