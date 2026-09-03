# 更新日志

## [Unreleased]

### 引擎

- 上游同步：ggml-org/llama.cpp master → e107984bc（117 commits，含 ROCm radix TOP_K for long rows #27466、mtmd const 传播、DeepSeek4V projector、Nemotron-3-Puzzle 模型、Metal sparse FA、Vulkan FA dequant 修复等）。
- 本地 TOP_K chunked-merge fix（f883d66c8）退役：上游 radix 实现覆盖同一 HIP 长行崩溃场景（#27021），实测快 3.5-13.5x。

### 文档

- 上游同步记录（2026-09-03，merge commit e1697de49）。

## [v0.1.0] (2026-08-28)

### 新增

- 三合一融合引擎：官方 llama.cpp master + ROCmFPX 量化 + mtmd 多模态视觉。
- ROCmFPX 量化格式支持（ROCmFP2/3/4/6/8 + TurboQuant，GGML 类型 100-107）。
- 多模态视觉塔支持（含 LocateAnything 定位投影器，`--special` 模式）。
- HIP 后端 gfx1151 适配（ROCm 10.0.0 构建）。

### 变更

- MoE 场景禁用 MMQ（gfx1151 的 MMQ kernel 未就绪，实测 mmq off 69.17 vs on 67.26 t/s）。

### 移除

- 上游 CI workflows（官方矩阵不适用于融合库，本地构建验证）。

### 文档

- 添加 README（定位/构建/使用）。
- README 重写（上游差异/功能/优化实测/部署）。
- 双语 README（中文默认 + English）。
- 量化格式全清单（Q4_0_ROCMFP4 家族 9 变体 + Qx_ROCMFPX 家族 9 类型）。
- 社区文件：CONTRIBUTING / CODE_OF_CONDUCT / SECURITY / Issue 模板。
