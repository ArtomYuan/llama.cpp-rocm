# llama.cpp-rocm

ROCmFPX 量化 + 多模态视觉融合引擎——基于官方 llama.cpp，面向 AMD Strix Halo（gfx1151 / Radeon 8060S）。

## 定位

三合一融合库（官方 master + ROCmFPX + mtmd 视觉）：

- **官方 llama.cpp**：持续同步上游 master
- **ROCmFPX**：ROCmFP2/3/4/6/8 + TurboQuant 量化格式（HIP kernel，gfx1151 适配）
- **mtmd 视觉**：多模态视觉塔（含 LocateAnything 定位投影器）

一个引擎同时支持：ROCmFPX 量化模型（Ornith/Qwen）+ 视觉模型（LocateAnything/多模态）。

## 构建（ROCm 10.0.0 / gfx1151）

```bash
export ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build -j $(nproc) --target llama-server llama-quantize llama-bench
```

产物在 `build/bin/`（llama-server / llama-quantize / llama-bench + 共享库）。

## 使用

```bash
# 量化（支持 ROCmFPX 格式）
build/bin/llama-quantize --allow-requantize <源模型> <输出> Q4_0_ROCMFP4_FAST 32

# 推理（文本模型）
build/bin/llama-server -m <模型.gguf> --port 8010

# 推理（视觉/定位模型，--special 启用特殊 token）
build/bin/llama-server --special -m <模型.gguf> --port 8030
```

## 同步上游

```bash
git fetch official && git merge official/master
```

## 说明

- CI：上游 workflow 已移除（融合库本地构建验证，不跑官方矩阵）
- 已知限制：gfx1151 的 MMQ kernel 未就绪——MoE 场景禁用 MMQ（ggml-cuda/mmq.cu）
