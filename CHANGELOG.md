# 更新日志

[English](CHANGELOG.en.md)

## [Unreleased]

## [v2026.9.24] (2026-09-24)

### 引擎

- **embeddings 模式跳过未消费的全词表 lm_head 输出（`result_output`）**：重排/embeddings 路径（`cparams.embeddings=true`）下 `qwen3.cpp` 原先无条件构建 `[151669, n_tokens]` 的 lm_head 输出（16K 批 ≈ 9.9 GB F32），而消费方只取 `n_cls_out` 个池化值（`llama-context.cpp` 的 RANK/MEAN/CLS/LAST 提取只读 `t_embd_pooled`/`t_embd`）。新增 `llm_graph_context::should_build_logits()`（判据 `!cparams.embeddings`，非 rnk 专用），`qwen3.cpp` 据此在 embeddings 模式跳过 lm_head 构建与 `ggml_build_forward_expand`；`llama-context.cpp` 的 `output_reserve` 同步改 `has_logits = !cparams.embeddings`，不再分配 `n_vocab*n_outputs` 主机缓冲与全词表 D2H。生成模式（`embeddings=false`）gate 不生效、行为不变。验证：重排分数 6 尺寸（含 14209/26534 边界）逐 bit 一致、向量逐元素 max|Δ|=0、生成冒烟逐 token 一致、`test-backend-ops MUL_MAT rocmfpx` 61/61。实测 16K 单任务 24.687s→24.667s（+0.1%）——消除的是真实但小的浪费（lm_head 约占整网前向计算 3.9%），不改变「GPU 前向吞吐 vs 突发到达率」的结构性退化。

- **`llama-embedding` 示例：jina-reranker-v3.5 支持链**（列表式重排服务化配套）：
  - `qwen3`：支持**逐层滑动窗口注意力（SWA）**——按 GGUF `sliding_window` 与逐层 pattern 路由（jina-reranker-v3.5 为 16 SWA 层 + 12 全注意力层）。
  - `--output-token-ids`：仅输出指定 token（如 `<|embed_token|>` / `<|rerank_token|>`）的隐状态，重排打分提取不再消费全序列输出。
  - `--serve-stdin`：**常驻模式**——模型只加载一次，按 stdin 逐请求处理（重排服务化 wrapper 的基础）。

### 修复

- **修复 MMQ 目标偏移 `offset_dst` 的 int32 有符号溢出（大词表 × 大批量）**：lm_head（tied `token_embd.weight[4096,151669]`，`stride_col_dst = vocab = 151669`、`J = 128`）在 `n ≥ 14209`（即 `ntx = ceil(n/128) ≥ 112`、`jt ≥ 111`）时 `jt*J*stride_col_dst = 111*128*151669 = 2,154,913,152 > 2^31-1` 溢出为负 → 写回 `dst − ~8.6 GiB` 野地址 → Memory Fault（`kernel: mul_mat_q<(ggml_type)103,128,true>`）。`ggml/src/ggml-cuda/mmq.cuh` 四处同型位置（`:1094` 常规 dense、`:1182` stream-k 循环、`:1271` stream-k 尾段、`:1414` stream-k fixup）全部把声明改为 `int64_t` 并在乘法显式 cast `(int64_t) jt*J*stride_col_dst`（只改声明不够，RHS 仍按 int 计算）。修前实测阈值精确到 1：`m=151669, k=4096` 时 `n ≤ 14208` 安全、`n = 14209` 必崩；修后最小复现全矩阵（n 至 32768）不崩，且 MMQ 与关阀 hipBLAS 逐元素一致（max abs diff 3.557e2、NMSE 5.46e-5）。该缺陷在 `official/master`（`9a9f939b9`）`mmq.cuh:1005/1093/1182/1325` 为同型 int32 溢出，fork 经 `8cbe3f39a` 为 ROCmFPX 接通 MMQ 后才将其暴露。

- **`llama-embedding`：未指定 attention 类型被误判为非因果，`--ubatch-size` 被静默覆写为整包**：`UNSPECIFIED` 落入 `!= LLAMA_ATTENTION_TYPE_CAUSAL` 判据 → `n_ubatch = n_batch`（`n_batch` 已被抬到 `n_ctx`）→ 超长输入以**单个 ubatch** 进图——O(n²) 注意力掩码（base + SWA 双掩码、主机/设备各一份）+ 全序列激活，131K 上下文 11 万 token 单包实测内存约 **93 GB**（整机 OOM）。判据改为 `== LLAMA_ATTENTION_TYPE_NON_CAUSAL`（`UNSPECIFIED` 交由模型默认，与 `llama-context.cpp` 语义一致）后，同场景峰值约 **15.6 GB**、44K 单包 108 s → 17 s、零 500/OOM；语义无变化（该 GGUF 无 `causal` 键、模型默认因果）。

### 测试

- `tests/test-backend-ops.cpp`：新增 MMQ `offset_dst` int32 溢出回归用例（`Q8_0_ROCMFPX`，`m=151669, n=16384, k=32`，覆盖 `m*n > 2^31` 的大词表 × 大批量形状；n=16384 使 17 个列 tile 溢出、NMSE 0.165 远高于 5e-4 阈值，可被数值判据捕获；最小形状 n=14209 仅 1 列溢出、低于阈值，故不用最小形状）。修前该用例 `ERR = 0.1649 > 5e-4` 必失败、修后通过；`rocmfpx` 定向自测 60/60 → **61/61**，全套 `MUL_MAT` 1381 → **1382/1382**，`MUL_MAT_ID` 935/935。
- 长输入内存回归：沙盒五档阶梯（8K–120K token）测得修复后 GTT 增量恒 **15.6 GB**（与长度无关）、零熔断；生产 110K 单包实测成功（90.9 s / 200）；切换后观察窗零 500/504、`journalctl -k` 无 oom。

## [v2026.9.22] (2026-09-22)

### 引擎

- **热路径解码无分支化（ROCmFPX/ROCmFP4）**：ue4m3 尺度解码（每权重块每线程执行一次）去掉数据相关分支（三元选择 → SEL/CSEL），全部 256 输入位等价。实测（gfx1151，Qwen3-Embedding/Reranker-8B pp2048）：Q8_0_ROCMFPX **1107→1233 t/s（+11.4%，反超 std Q8_0 的 1227）**、Q6_0_ROCMFPX 920→1013（+10.1%）、Q4_0_ROCMFP4_FAST 1246→1294（+3.9%）。定位依据：rocprofv3 计数器（旧 fp8 MMQ 核 SALU +39%、分支 +40%、WAIT_ANY +86%、波周期 +18%，访存计数逐项一致）；修复后与 std 逐项打平，fp4 MMQ 核周期 −8.6%。验证：test-backend-ops rocmfpx 60/60、rocmfp4 24/24。

## [v2026.9.21] (2026-09-21)

### 引擎

- **ROCmFPX fp8 家族（Q2/Q3/Q6/Q8_0_ROCMFPX）GPU 计算路径接通**（按提取法从 charlie12345/ROCmFPX 移植 + 本库配置化 MMQ 架构适配；此前该家族在 GPU 上无法运行，加载预热即 abort）：
  - MMVQ（`mmvq.cu`）：补 fp8 四类型派发接线（vec_dot 表、表选择、内核特化、RDNA3.5 参数表），并移植 fp2 专用多列内核 `vec_dot_rocmfpx_fp2_q8_1_ncols`（多列时权重的 fp2→int8 展开只做一次）；参数宏默认值=旧行为（调优留路，暂不启用）。
  - MMQ（`mmq-load-tiles.cuh` / `mmq.cuh` / `mmq-config-rdna3-5.cuh`）：新增 fp2/fp3/fp6/fp8 四个装载器（fp2/3/6 用 Q3_K SRAM 布局、fp8 用 Q8_0 布局）、ds 布局与 tile 尺寸表、util_funcs 双通道（dp4a/mma）接线、RDNA3.5 配置表 48 条 (I,J) 条目（与 Q8_0/Q3_K 同网格）；实例化文件（`mmq-instance-*.cu`）为既有、本次接入调度。
  - 门控：fp8 家族 MMQ 当前仅 RDNA3.5 启用（其他架构自动回退 dequant+hipBLAS，避免命中「无 J 配置」缺省路径）；`GGML_HIP_NO_ROCMFPX_MMQ=1` 可整族关闭（调试/回退用）。
- 回归：fp4 家族路径行为不变（RDNA3.5 参数表改动默认值=旧行为；Ornith Q4_FAST 实测新旧引擎差异在噪声内）。

### 测试

- `tests/test-backend-ops.cpp`：`all_types[]` 纳入 fp8 四类型，并新增 fp8 大形状用例（m=4096/251、n=128/512、k=1024，覆盖大 J tile 与 fallback 两种网格）。实测 **60/60 全过**（覆盖 MMVQ、MMQ、hipBLAS 三条路径）；端到端 Qwen3-Embedding-8B-Q8_ROCMFPX GPU 冒烟（含 MMQ 批路径）与 CPU 基准偏差 e-4 量级、保真度与 CPU 对照一致（近邻 top-5 96.4%）。

## [v2026.9.20] (2026-09-20)

### 引擎

- 上游同步：ggml-org/llama.cpp master → f072b10371（窗口 97e4ca735..f072b10371 = **98 commits**，merge commit c305d4a9a）。
- qwen4exp 系列：新增 hc ops（#28901，含 metal #29000 / vulkan #28988 同步支持）；rms_norm + mul 融合（#28896）。
- HIP 6 笔：RDNA3.5 MoE ncols_opt tile 启发式放宽（#28935）、ROCm AllReduce（#27825）、CUDA/HIP im2col 访问模式优化（#28013）、fattn-mma fp32 累积（#28576）、ubuntu rocm 发布包新增 gfx1103（#28423）、hip-quality-check 忽略已知 spill（#28909）。
- RPC 3 笔：跳过 ACCEL 设备（#29020）、buffer 释放时失效缓存计算图（#24292）、权重 hash-cache（#28789）。
- GGUF data 段对齐基准变化（#28993，影响写出的 GGUF 布局）。
- 版本号同步前进：llama.cpp → 0.4.1（#28900）、ggml → 0.24.0；本库构建 0.4.1-dev（build 11094，commit c305d4a9a）。
- 融合面保留：①ROCmFPX 量化 ②ROCmFP4（含 MMQ 双尺度修复与 env 开关）③mtmd 多模态视觉（LocateAnything + DeepSeek4V 双 projector）④上游 CI workflows 删除态 ⑤CONTRIBUTING/README 中文改写。枚举复核 GGML_TYPE_COUNT=108、FTYPE 100-119 全家 17 个逐一在、冲突标记零残留。
- 冲突裁决：mechanical=20（.github/workflows/*.yml modify/delete，保持本地删除态）+ content=1（CONTRIBUTING.md 取本地中文改写）。

### 文档

- 文档中英双语：README.en.md 与中文版同步维护；新增 CHANGELOG.en.md（本文件英文版）；发行版说明自本版起中英双语。

## [v2026.9.16] (2026-09-15)

### 修复

- 修复双尺度 `Q4_0_ROCMFP4` 在 MMQ 路径的派发错位（`mmq.cuh` 改用与 Q3_K 同构的 per-16 值向量点积变体）：数值误差 ~3e-2 → ~3e-8，2296/2296 用例全过。此前双尺度模型走 MMQ 会静默算错；生产用 `_FAST`（单尺度）不受影响。

## [v2026.9.15] (2026-09-15)

### 引擎

- 测试覆盖修复：`tests/test-backend-ops.cpp` 的 `all_types[]` 纳入 `GGML_TYPE_Q4_0_ROCMFP4` / `GGML_TYPE_Q4_0_ROCMFP4_FAST`，修正假绿（此前全仓 `tests/` 内 ROCmFP4 用例数为 0）。两臂实测：默认臂 2233/2236（3 败全为双尺度 `q4_0_rocmfp4` 非 FAST，系已知数值缺陷）；`GGML_HIP_NO_ROCMFP4_MMQ=1` 臂 2236/2236 全过。
- `mmq.cu`：清理错位守卫——把 `highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A` 分支恢复为上游 pre-Pascal NVIDIA 写法（旧 `return false` + ROCmFPX 误导注释对 AMD 恒不可达，本库曾误以为「禁用 MMQ」）；新增可达的 `GGML_HIP_NO_ROCMFP4_MMQ` 兜底开关（默认开、行为不变；设置后仅 ROCmFP4 双类型转 hipBLAS，标准格式 MMQ 与 MMVQ 不受影响）。
- **已知问题**：双尺度 `Q4_0_ROCMFP4`（非 `_FAST`）在 MMQ 路径下数值超标（NMSE 0.01–0.04，超 `max_nmse_err=5e-4` 阈值约 21–80 倍）；非本版引入；生产用 `_FAST`（单尺度）不受影响；影响面 = 未来若使用双尺度 `Q4_0_ROCMFP4` 量化的模型（当前生产模型均为 `_FAST`）。

### 文档

- 勘误 README/CHANGELOG 中「MMQ 未就绪/已禁用」及「off 69.17 vs on 67.26 更优」错误表述（旧 69.17/67.26 是 tg32/batch-1 decode 口径，走 MMVQ 不经过 MMQ）；实证 MMQ 一直开启且 prefill 约 2×（pp512 +105%、pp2048 +109%，2026-09-15 实测）。

## [v2026.9.22] (2026-09-22) (2026-09-14)

### 引擎

- 上游同步：ggml-org/llama.cpp master → 97e4ca735（窗口 e107984bc..97e4ca735 = **172 commits**，merge commit cfeb42ff6，pre/post 留证 tag sync-20260914-pre/post）。
- 修复既有缺陷：`ggml_validate_row_data` 未接入 ROCmFP4/ROCmFPx/TurboQuant 校验分发，llama-quantize 产 FPX 格式时报 "invalid type 101" 导致量化生成失败（commit da935e773；补 100-107 八个 case，TurboQuant 用 FP16 L2-norm 校验）。
- 融合面四项全部保留：①ROCmFPX 量化（GGML 类型 100-107）②mtmd-grounders 多模态视觉（LocateAnything + DeepSeek4V 双 projector）③MoE 场景 MMQ「禁用」（`ggml_cuda_should_use_mmq` 内 `return false`；⚠ 历史误记、已勘误——该守卫对 AMD 恒不可达、MMQ 实为一直开启，见 2026-09-15 条目）④上游 CI workflows 删除态（本窗口新增的 `.github/workflows/fusion.yml` 一并删除）。
- 冲突裁决：机械冲突 24 个（`.github/workflows/*.yml` modify/delete，裁决 `git rm` 保持本地删除态，另有上游本窗口新增的 `.github/workflows/fusion.yml` 一并删除）+ 真内容冲突 4 个（README.md 裁决为整体取本地中文重写版；`ggml/src/ggml-cuda/CMakeLists.txt`、`ggml/src/ggml-hip/CMakeLists.txt`、`ggml/src/ggml-cuda/fattn.cu` 手工逐块融合。注：`ggml/src/CMakeLists.txt` 为自动合并，非内容冲突）。
- 与预判相反：`mmq.cu` / `mmq.cuh` / `mmq-config-*.cuh` / `ggml/include/ggml.h` / `include/llama.h` / `tools/mtmd/clip.cpp` / `src/models/gemma4.cpp` 全部**零冲突自动合并**，融合面未被动过（八组审查 grep 全 PASS：GGML_TYPE_COUNT=108、mmq-config-rdna3 ROCMFP=24、mmq.cu MoE `return false`、arg.cpp kv_cache FPX、llama-quant.cpp ROCMFP=84、clip.cpp 双 projector、workflows 无残留、无冲突标记）。
- 上游关键演进面：构建系统 PCH + unity build 上马（#28091）后续按上游修正；GGML_FA_QUANTS 拆分（#28079）——FlashAttention 向量 kernel 实例列表 helper 化（`ggml_cuda_fattn_vec_instances`）+ 分发改为 `ggml_cuda_get_fattn_vec_case` 查找，本库 FPX/turbo 实例与 `GGML_CUDA_FA_*_ROCMFP*` 编译定义已重新接入；RDNA3/4 MMQ 的 MoE N-tile 尺寸优化加→撤→重做，并新增 `mmq-config-gcn.cuh`；HIP/CUDA 面 BF16 fallback、gfx1201 FA 调整、mmf/mmid 并发修复、gfx90c HIP 支持；mtmd 面 gemma4 vision 修复、video ID 传播、`mtmd_tokenize_from_parts`；server 面子进程重构、LRU 挂起修复、schema 内部表示与 UI 渲染性能；模型面 Maple 20B-A1B ternary MoE、腾讯 Hy4 preview、Kimi-K3 回滚、Nemotron-3-Puzzle、Spark2_5、Qwen3-Next/Qwen3.5 recurrent_layers。
- 上游版本号同步前进：`LLAMA_VERSION_BASE` 0.3.0 → **0.4.0**（对应上游 v0.4.0 里程碑）。本库自身版本 tag 与上游 tag 命名空间冲突问题，见 S11a 版本建议（待管理员确认）。
- 部署面：8010/8020 两个 systemd 服务的 `--no-mmap` 改为 `--load-mode none`（上游已正式移除 `--no-mmap` 弃用别名，新引擎不再接受旧参数；`--load-mode none` 与 `--no-mmap` 语义等价，已源码+实测双向确证）。

### 文档

- 上游同步记录（2026-09-14，merge commit cfeb42ff6；上游窗口 e107984bc..97e4ca735，172 commits）。
- 逐块融合最复杂面 = `ggml/src/ggml-cuda/fattn.cu`：上游 #28079 把 FA 向量 kernel 分发由内联级联改成 `ggml_cuda_get_fattn_vec_case()` 查找函数，本库把 5 条 ROCmFPX 对角线组合注册进该函数、并把 `ggml_cuda_fattn_kv_type_supported` 扩展到 ROCmFPX/TurboQuant（否则 `is_rocmfp_family` 路由不可达）；两个 CMakeLists 在 helper 调用后追加 FPX/turbo 实例与编译定义。该面为本次唯一高语义风险点，编译正确性由构建阶段兜底。
- README 未随本次同步改动：上游 README 在本窗口仅 1 行变更（维护者导航链接列表），本库 README 为中文重写版，冲突裁决取本地；README 复核结论见 S11a-readme-notes.md。

## [v2026.9.22] (2026-09-22) (2026-09-03)

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

- MoE 场景禁用 MMQ（gfx1151 的 MMQ kernel 未就绪，实测 mmq off 69.17 vs on 67.26 t/s）。（⚠ 历史误记、已勘误：该「禁用」为不可达死代码、MMQ 一直开启；旧 69.17/67.26 是 tg32 decode 口径，走 MMVQ 不经过 MMQ，见 2026-09-15 条目）

### 移除

- 上游 CI workflows（官方矩阵不适用于融合库，本地构建验证）。

### 文档

- 添加 README（定位/构建/使用）。
- README 重写（上游差异/功能/优化实测/部署）。
- 双语 README（中文默认 + English）。
- 量化格式全清单（Q4_0_ROCMFP4 家族 9 变体 + Qx_ROCMFPX 家族 9 类型）。
- 社区文件：CONTRIBUTING / CODE_OF_CONDUCT / SECURITY / Issue 模板。
