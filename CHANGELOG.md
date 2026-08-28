# 更新日志

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
