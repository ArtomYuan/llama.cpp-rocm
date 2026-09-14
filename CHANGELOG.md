# 更新日志

## [Unreleased] (2026-09-14)

### 引擎

- 上游同步：ggml-org/llama.cpp master → 97e4ca735（窗口 e107984bc..97e4ca735 = **172 commits**，merge commit cfeb42ff6，pre/post 留证 tag sync-20260914-pre/post）。
- 修复既有缺陷：`ggml_validate_row_data` 未接入 ROCmFP4/ROCmFPx/TurboQuant 校验分发，llama-quantize 产 FPX 格式时报 "invalid type 101" 导致量化生成失败（commit da935e773；补 100-107 八个 case，TurboQuant 用 FP16 L2-norm 校验）。
- 融合面四项全部保留：①ROCmFPX 量化（GGML 类型 100-107）②mtmd-grounders 多模态视觉（LocateAnything + DeepSeek4V 双 projector）③MoE 场景 MMQ 禁用（`ggml_cuda_should_use_mmq` 内 `return false`）④上游 CI workflows 删除态（本窗口新增的 `.github/workflows/fusion.yml` 一并删除）。
- 冲突裁决：机械冲突 24 个（`.github/workflows/*.yml` modify/delete，裁决 `git rm` 保持本地删除态，另有上游本窗口新增的 `.github/workflows/fusion.yml` 一并删除）+ 真内容冲突 4 个（README.md 裁决为整体取本地中文重写版；`ggml/src/ggml-cuda/CMakeLists.txt`、`ggml/src/ggml-hip/CMakeLists.txt`、`ggml/src/ggml-cuda/fattn.cu` 手工逐块融合。注：`ggml/src/CMakeLists.txt` 为自动合并，非内容冲突）。
- 与预判相反：`mmq.cu` / `mmq.cuh` / `mmq-config-*.cuh` / `ggml/include/ggml.h` / `include/llama.h` / `tools/mtmd/clip.cpp` / `src/models/gemma4.cpp` 全部**零冲突自动合并**，融合面未被动过（八组审查 grep 全 PASS：GGML_TYPE_COUNT=108、mmq-config-rdna3 ROCMFP=24、mmq.cu MoE `return false`、arg.cpp kv_cache FPX、llama-quant.cpp ROCMFP=84、clip.cpp 双 projector、workflows 无残留、无冲突标记）。
- 上游关键演进面：构建系统 PCH + unity build 上马（#28091）后续按上游修正；GGML_FA_QUANTS 拆分（#28079）——FlashAttention 向量 kernel 实例列表 helper 化（`ggml_cuda_fattn_vec_instances`）+ 分发改为 `ggml_cuda_get_fattn_vec_case` 查找，本库 FPX/turbo 实例与 `GGML_CUDA_FA_*_ROCMFP*` 编译定义已重新接入；RDNA3/4 MMQ 的 MoE N-tile 尺寸优化加→撤→重做，并新增 `mmq-config-gcn.cuh`；HIP/CUDA 面 BF16 fallback、gfx1201 FA 调整、mmf/mmid 并发修复、gfx90c HIP 支持；mtmd 面 gemma4 vision 修复、video ID 传播、`mtmd_tokenize_from_parts`；server 面子进程重构、LRU 挂起修复、schema 内部表示与 UI 渲染性能；模型面 Maple 20B-A1B ternary MoE、腾讯 Hy4 preview、Kimi-K3 回滚、Nemotron-3-Puzzle、Spark2_5、Qwen3-Next/Qwen3.5 recurrent_layers。
- 上游版本号同步前进：`LLAMA_VERSION_BASE` 0.3.0 → **0.4.0**（对应上游 v0.4.0 里程碑）。本库自身版本 tag 与上游 tag 命名空间冲突问题，见 S11a 版本建议（待管理员确认）。
- 部署面：8010/8020 两个 systemd 服务的 `--no-mmap` 改为 `--load-mode none`（上游已正式移除 `--no-mmap` 弃用别名，新引擎不再接受旧参数；`--load-mode none` 与 `--no-mmap` 语义等价，已源码+实测双向确证）。

### 文档

- 上游同步记录（2026-09-14，merge commit cfeb42ff6；上游窗口 e107984bc..97e4ca735，172 commits）。
- 逐块融合最复杂面 = `ggml/src/ggml-cuda/fattn.cu`：上游 #28079 把 FA 向量 kernel 分发由内联级联改成 `ggml_cuda_get_fattn_vec_case()` 查找函数，本库把 5 条 ROCmFPX 对角线组合注册进该函数、并把 `ggml_cuda_fattn_kv_type_supported` 扩展到 ROCmFPX/TurboQuant（否则 `is_rocmfp_family` 路由不可达）；两个 CMakeLists 在 helper 调用后追加 FPX/turbo 实例与编译定义。该面为本次唯一高语义风险点，编译正确性由构建阶段兜底。
- README 未随本次同步改动：上游 README 在本窗口仅 1 行变更（维护者导航链接列表），本库 README 为中文重写版，冲突裁决取本地；README 复核结论见 S11a-readme-notes.md。

## [Unreleased] (2026-09-03)

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
