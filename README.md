# llama.cpp-rocm

面向 AMD Strix Halo（Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151）的融合引擎：官方 llama.cpp master + **ROCmFPX 量化格式** + **多模态视觉塔** 三合一。

[English](README.en.md)

---

## 这是什么

llama.cpp 的独立维护分支，把三个能力整合到一个引擎：

| 来源 | 能力 |
|---|---|
| 官方 llama.cpp（持续同步 master） | 全部基础推理能力 |
| charlie12345/ROCmFPX | ROCmFPX 量化格式（ROCmFP2/3/4/6/8 + TurboQuant） |
| yuuko-eth mtmd-grounders | 多模态视觉塔（含 LocateAnything 定位投影器） |

**一个二进制**：ROCmFPX 量化模型（文本）+ 视觉/定位模型都能跑，无需切换引擎。

## 与上游的不同

- **ROCmFPX 量化支持**：上游 llama.cpp 无法加载 ROCmFP4/FP8 格式——本引擎原生支持（GGML 类型 100-107），并带量化工具 `llama-quantize`
- **多模态视觉**：合并 mtmd 视觉栈，支持视觉模型与 Grounding 定位模型（`--special` 模式输出 `<ref>/<box>`）
- **纯 ROCm HIP 后端**：构建仅含 HIP（无 Vulkan），设备锁定 ROCm0——无后端选择歧义
- **MMQ 决策**：gfx1151 的 MMQ kernel 尚未就绪，MoE 场景禁用 MMQ（实测 69.17 t/s vs 默认 67.26 t/s）
- **无上游 CI**：官方 CI 矩阵已移除（本库本地构建验证）

## 功能

- ROCmFPX 量化：ROCmFP2/3/4/6/8 + TurboQuant（FAST 变体），从 F32/量化源 requantize
- 多模态视觉：moonvit 视觉塔、Gemma4 图像处理对齐 HF、LocateAnything 定位（`<ref>/<box>`）
- 高性能：gfx1151 HIP kernel 适配、纯 ROCm0 后端、MoE 场景 69+ t/s（35B.A3B）
- OpenAI 兼容 API（llama-server）

## 优化实测

| 项 | 结果 |
|---|---|
| Ornith 35B.A3B（Q4_ROCmFPX_FAST） | tg32 **69.17 t/s**（mmq off） |
| 同模型 mmq on 对比 | 67.26 t/s（MMQ 未就绪——禁用为最优） |
| Qwen 27B（Q4_ROCmFPX_FAST） | tg32 12.17 t/s |
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

```bash
build/bin/llama-quantize --allow-requantize <源模型.gguf> <输出.gguf> Q4_0_ROCMFP4_FAST 32
```

### 推理服务（systemd 用户服务示例）

```ini
[Service]
Type=simple
Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1
Environment=HIP_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/path/to/engine/bin
ExecStart=/path/to/engine/bin/llama-server --special -m /path/to/model.gguf --port 8030
```

- 文本模型：`llama-server -m <模型.gguf> --port 8010`
- 视觉/定位模型：`llama-server --special -m <模型.gguf> --port 8030`（`--special` 启用 `<ref>/<box>` 特殊 token）

### 同步上游

```bash
git fetch official && git merge official/master
```

## 已知限制

- gfx1151 的 MMQ kernel 未就绪——MoE 场景禁用 MMQ（ggml-cuda/mmq.cu）
- 上游 CI 已移除——请本地构建验证
