# strix.cpp

**为 AI MAX+ 395 深度特调的 llama.cpp 引擎，原生支持 ROCmFPX 量化。**

strix.cpp 是 llama.cpp 的独立维护分支。它把三件事做扎实：持续同步官方上游；原生支持 ROCmFPX 量化（上游加载不了的格式）；合并多模态视觉能力。而它真正的立身之本，是在这颗芯片——Strix Halo（gfx1151）、最多 128 GiB 统一内存——上做内核级的深度特调：同样的模型、同样的量化，注意力内核快两成以上。

原名 `llama.cpp-rocm`（2026-10-17 更名，旧链接由 GitHub 自动重定向）。

[English](README.en.md)

---

## 特点

- **量化格式开箱即用**：ROCmFP4 / ROCmFPX 全家族（含 2-bit 到 8-bit、Agent 路由变体），自带 `llama-quantize`。详见[量化指南](docs/quantization-guide.md)。
- **多模态**：视觉模型与 LocateAnything 定位模型直接可用（`--special` 输出 `<ref>/<box>`）。
- **为这块芯片调过**：注意力内核按形状分裂配置；预填 GEMM 接上 FP4 专用 K 分块，整网预填快 40~66%（方法见[内核优化记录](docs/development/kernel-optimization.md)）。
- **一个二进制**：文本、视觉、定位模型不用换引擎。
- **纯 ROCm**：只编 HIP、无 Vulkan、设备锁定 ROCm0，没有后端选择歧义。

## 实测

同机、同模型、同量化、同参数，对照臂 = 同源码的 stock 配置（也就是上一个生产版本）：

| 场景（Qwen3.8-27B，Q4_0_ROCMFP4_FAST） | stock | strix.cpp | 变化 |
|:--|--:|--:|--:|
| 预填 pp2048（短上下文） | 437 t/s | **721 t/s** | **+66%** |
| 预填 pp16384（深上下文） | 389 t/s | **614 t/s** | **+55%** |
| 预填 pp65536（超深上下文） | 295 t/s | **414 t/s** | **+40%** |
| 生成 tg256 | 14.0 t/s | 14.0 t/s | 持平（解码走另一条路径） |
| 16k 深度生成 | 12.69 t/s | 12.69 t/s | 持平 |
| 64k 深度生成 | 9.714 t/s | 9.710 t/s | 持平 |

（其余量化档：`Q8_0_ROCMFPX`、`Q6_0_ROCMFPX` 不在本次改动范围内，与此前记录一致；`Q4_0_ROCMFP4_STRIX` 属同族 ROCmFP4，未单独复测。）

**怎么读这张表**：预填的收益来自两刀——注意力内核按形状分裂配置（内核本身 **+22.6%**），以及预填 GEMM 接上 FP4 专用的 K 分块（内核 **+90%**；被整网摊薄后仍 +40~66%）。生成侧两版一致：解码走 MMVQ 路径，不在本次改动范围内。逐项验证见[内核优化记录](docs/development/kernel-optimization.md)。

## 快速开始

### 构建

```bash
export ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build -j $(nproc) --target llama-server llama-quantize llama-bench
```

产物在 `build/bin/`。

### 运行

```bash
build/bin/llama-server -m <模型.gguf> --port 8080
```

OpenAI 兼容 API（默认 `/v1`）。视觉/定位模型加 `--special`；纯文本模型不需要。

### 作为服务常驻

```ini
[Service]
Type=simple
Environment=HSA_OVERRIDE_GFX_VERSION=11.5.1
Environment=HIP_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/path/to/engine/bin
ExecStart=/path/to/engine/bin/llama-server --special -m /path/to/model.gguf --port <port>
```

## 文档

- [量化指南](docs/quantization-guide.md)：家族表、怎么选、量化命令、KV 类型
- [内核优化记录](docs/development/kernel-optimization.md)：本库每项内核特调的证据与复现
- [贡献指南](CONTRIBUTING.md)

## 已知限制

- **MUL_MAT_ID（MoE 路由）的重复专家用例**：在 `dup=1`（同一专家被选多次）的用例下存在数值错误，极端形状可触发 GPU 故障；基线同为如此（非本版引入，正式模型路由通常不触发）。
- **超长上下文的精度边缘**：f16 KV 的注意力在 32k 以上时，与 f32 参考的误差会随深度缓慢增长（131k 时约 3e-3）。这是 llama.cpp 的既有行为（上游同样如此），对模型输出影响约百分之零点几——超长上下文的重负载场景请留意。
