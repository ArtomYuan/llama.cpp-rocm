# 内核优化记录（ROCmFPX / ROCmFP4）

[English](kernel-optimization.en.md)

本库（strix.cpp）对 ROCmFPX / ROCmFP4 量化家族在 HIP（gfx1151）上**内核级优化**的记录：每条含问题、定位证据、根因、修复与验证数据，均可按文内命令复现。

---

## 1. 热路径 ue4m3 尺度解码无分支化（v2026.9.22）

### 1.1 问题

`Q8_0_ROCMFPX` 在 gfx1151 上的 prefill 比标准 `Q8_0` 慢约 9–11%（同条件 pp2048：**1107 vs 1235 t/s**）。此前推测与「33 字节块步长 / SRAM 布局 / 访存对齐」有关。

### 1.2 定位（rocprofv3 两级）

**第一级 · 逐内核采样**（`--kernel-trace`，两侧同条件）：

- 两档的 GPU 时间差 **100% 落在 `mul_mat_q`（MMQ）主核**内（每次调用 +11%）；注意力、归一化、量化、搬运等内核逐项相同。
- 两档内核调用数一致；VGPR（232）/SGPR/工作组与网格配置一致 → 排除「多做步骤」与「占用率」两类原因。

**第二级 · 硬件计数器**（`--pmc`，仅采 `mul_mat_q`；旧 fp8 内核 vs std）：

| 计数器 | fp8/std | 读法 |
|:--|:--|:--|
| `SQ_INSTS_SALU` | **1.389** | 标量指令偏多 |
| `SQ_INSTS_BRANCH` | **1.402** | 分支偏多 |
| `SQ_WAIT_ANY` | **1.860** | 等待周期暴涨 |
| `SQ_WAVE_CYCLES` | **1.177** | 同工作量耗时更长 |
| `SQ_INSTS_FLAT` / `SQC_DCACHE_REQ` / `GL2C_MC_RDREQ` / LDS 冲突 | 1.000 | **访存侧完全一致** |

结论：非访存 / 带宽 / 对齐问题；差距集中在**解码路径的标量指令与分支**造成的流水线停顿。

### 1.3 根因

`ggml/rocmfp4/rocmfp4_hip_scale.cuh` 中两个 ue4m3 尺度解码函数（`rocmfpx_ue4m3_to_fp32_finite` 与 `rocmfp4_ue4m3_to_fp32_half_finite`）带**数据相关 if 分支**，且位于 MMQ 装载器、vec_dot、cpy-utils 的**每权重块每线程**热路径上。

```cpp
// 修复前（rocmfpx_ue4m3_to_fp32_finite 节选）
if (x > 0x7e) { return 0.0f; }
const int exp = (x >> 3) & 0xF;
const int man = x & 0x7;
if (exp == 0) { return (float) man * (1.0f / 1024.0f); }
```

### 1.4 修复（提交 `0347fac53`）

改为无分支选择（三元选择 → SEL/CSEL 类选择指令），逐位等价：

```cpp
// 修复后
const int exp = (x >> 3) & 0xF;
const int man = x & 0x7;
const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
const float    norm = rocmfp4_u32_as_f32(bits);
const float    subn = (float) man * (1.0f / 1024.0f);
const float    val  = (exp == 0) ? subn : norm;
return (x > 0x7e) ? 0.0f : val;
```

一处修改同时覆盖 fp8（Q2/Q3/Q6/Q8）与 fp4 两个家族的全部热路径。

### 1.5 验证

- **位等价**：两个函数改前 / 改后，256 个输入全枚举逐一比对一致（数值零变化）。
- **内核测试**：`test-backend-ops` MUL_MAT `rocmfpx` **60/60**、`rocmfp4` **24/24**。
- **计数器复核**：新 fp8 内核与 std 在所有采集计数器上**逐项打平**（SALU / 分支 / 等待 = 1.00x、波周期 0.991）；fp4 MMQ 核波周期 −8.6%、等待 −20%。
- **部署双锚**（同目录新旧二进制背靠背）：Qwen3-Embedding-8B Q8_0_ROCMFPX pp2048 **1114.9 → 1235.3 t/s（+10.8%）**；Ornith Q4_0_ROCMFP4_FAST tg128 71.99 → 71.95（持平）。

### 1.6 效果（llama-bench pp2048）

| 档位（Emb / Rnk） | 修复前 | 修复后 | 变化 |
|:--|:--|:--|:--|
| Q8_0_ROCMFPX | 1107 / 1099 | 1233 / 1227 | **+11.4% / +11.6%** |
| Q6_0_ROCMFPX | 920 / 916 | 1013 / 1010 | +10.1% / +10.2% |
| Q4_0_ROCMFP4_FAST | 1246 / 1244 | 1294 / 1298 | +3.9% / +4.3% |

### 1.7 复现命令

```bash
# 逐内核采样（两侧同条件各跑一次；采样开销约 5%，相对对比不受影响）
rocprofv3 --kernel-trace -f csv -o /tmp/prof -- \
  llama-bench -m <模型.gguf> -embd 1 -ngl 99 -p 2048 -n 0 -r 1

# 硬件计数器（注意 -- 分隔符；计数器名逐个传参；-f csv 必加）
rocprofv3 --pmc SQ_WAVES SQ_BUSY_CYCLES SQ_INSTS_SALU SQ_INSTS_BRANCH \
  SQ_WAIT_ANY SQ_WAVE_CYCLES --kernel-include-regex "mul_mat_q" \
  -f csv -o /tmp/pmc -- llama-bench -m <模型.gguf> -embd 1 -ngl 99 -p 2048 -n 0 -r 1

# 内核正确性
test-backend-ops test -o MUL_MAT -p "rocmfpx"   # 60/60
test-backend-ops test -o MUL_MAT -p "rocmfp4"   # 24/24
```

### 1.8 交付

- 提交：`0347fac53`（代码）、`f46544c29`（README 实测表刷新）
- 版本：**v2026.9.22 (v0.2.5)**（CHANGELOG 见仓库 `[v2026.9.22]` 条目）

---

## 2. 注意力内核：512 线程配置按 (ncols1,ncols2) 分裂（v2026.10.17）

### 2.1 问题

gfx1151 的 FLASH_ATTN_EXT mma-f16 主内核在 stock 配置（`(256,256,64): 256,2,64` = 256 线程/occ2/combine64）下 VGPR 拉满 256、每线程溢出 **515 处**、scratch 1076B——kv≥8k 的深上下文吞吐被寄存器溢出拖累：kv=16k gqa8 基线 **11.48 TFLOPS**。

### 2.2 定位过程与证据

- 逐内核采样 + 硬件计数器：等待类 82%、L2 命中 6.8% → 延迟受限；空间维度指向寄存器压力。
- 单实例资源报告（`-Rpass-analysis=kernel-resource-usage`）：VGPR 256（上限）、溢出 515 处/线程、scratch 1076B。
- 512 线程配置单试（occ 2→1、combine 64→32）：溢出 **−60%**、kv=16k **+21%**（11.5→13.97）。
- 整行修改版（ncols=64 全实例共享）导致 `(32,2)` 家族 **−16~21%**——同宽度不同头分组共享同一行，必须按 `(ncols1,ncols2)` 分裂。

### 2.3 根因

配置表按 `ncols` 取键：`ncols=64` 一行同时服务 `(8,8)`/`(16,4)`/`(32,2)` 三种实例；512 线程收益只对头分组宽的 `(8,8)`/`(16,4)` 成立，对 `(32,2)` 相反。

### 2.4 修复

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`：新增分裂判定双 helper——设备侧 `ggml_cuda_fattn_mma_xaihi_split_512_dev`（宏判定）、主机侧 `ggml_cuda_fattn_mma_xaihi_split_512_host`（运行时 `amd_wmma_available(cc)`），接 5 处调用点（launch_bounds / kernel body nthreads / process_tile combine / host nthreads / host combine）；仅 `(256,256,8,8)` 与 `(256,256,16,4)` 启用。
- **关键坑（主机/设备宏可见性）**：`RDNA3` 宏源自设备趟专属 `__GFX11__`，主机编译趟不可见——单一 `__host__ __device__` 判定会形成「主机 256 / 设备 512」错位，表现为 **NaN（首元素）+ 约 2× 假快**（半量计算）。必须拆双 helper。

### 2.5 验证链

- 数值门：`test-backend-ops test -o FLASH_ATTN_EXT -p "hsk=256.*type_K=f16"` = **148/152**；4 挂均为 kv≥32k 的既有精度边缘（stock 同款，非本轮引入）。
- 定位矩阵：nb 梯度（2/4/8/16/32/64）与 n_kv_max 梯度（256/512/1024/2048）全过；初版 `(8,8)`-nb≥8 的 NaN 回归在修复后清零。
- 性能（同刻背靠背）：`(8,8)` kv=16k **11.48→14.08 TF（+22.6%）**；`(16,4)` **8.84→10.63（+20.2%）**；P1 `(32,2)` 持平；32k/64k 深水 **+4%**；4k 短上下文 −3~6%（权衡内）。
- 构建确定性：同源同工具链重编两次产物逐字节一致（md5 复核）。

### 2.6 复现命令

```bash
# 数值门
test-backend-ops test -o FLASH_ATTN_EXT -p "hsk=256.*type_K=f16"

# 性能（同刻背靠背；perf 用例由参数化循环生成：nh=8 × ratio∈{1,4,8} × kv∈{4096..65536} × nb∈{1,4096}）
test-backend-ops perf -o FLASH_ATTN_EXT -p "hsk=256,hsv=256,nh=8,nr23=\[8,1\],kv=16384,nb=4096"
test-backend-ops perf -o FLASH_ATTN_EXT -p "hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=16384,nb=4096"
test-backend-ops perf -o FLASH_ATTN_EXT -p "hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=16384,nb=512"
```

### 2.7 交付

- 提交：`43ea02cb3`（配置分裂）、`c57487a5b`（FA 向量闸门）、`6d4fc573a`（用例覆盖）、`0eb0adcf4`（llama-bench 类型）
- 版本：**v2026.10.17 (v0.2.7)**

---

## 3. 预填 GEMM：ROCmFP4 的 MMQ K 分块接上 FP4 专用值（v2026.10.18）

### 3.1 问题

Qwen3.8-27B 的预填 GEMM（`Q4_0_ROCMFP4_FAST`）在 27B 主投影形状上只有 **27.5 TFLOPS**，仅为同形状 f16（33.3）的 82%——4.25 bpw 的权重读得比 f16 少 3.7 倍，吞吐反而更低。对照 gufo 同款模型的公开记录（预填 ≈ llama.cpp 的 2×），差距不在注意力，在 GEMM 本身。

### 3.2 定位过程与证据

1. **立尺**（`tests/test-backend-ops perf -o MUL_MAT`，27B 投影形状）：Q4 = 27.47 / 25.75 / 25.23 TFLOPS（三个主形状），f16 = 33.27 / 24.31 / 27.73。
2. **分派面审查**（`ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`）：类型 101（`Q4_0_ROCMFP4_FAST`）的条目挂着 `MMQ_ITER_K=256`；而 `mmq.cuh:10` 里已有 FP4 专用的 `MMQ_ITER_K_FP4=512`，从未被 ROCmFP4 条目引用。
3. **变体实测**（每次重编 + 同刻测量）：

   | 变体 | 17408×2048×5120 | 判定 |
   |:--|--:|:--|
   | baseline（256） | 27.90 TF | 对照 |
   | **iter512（K=512）** | **53.33 TF（+90.6%）** | 采用 |
   | occ1（occupancy 2→1） | 27.34 TF | 否决（无收益） |
   | both（512 + occ1） | 52.57 TF | 否决（无叠加） |

### 3.3 根因

类型 101 接入 MMQ 时，配置表照抄了通用路径的 K 分块 256，FP4 专用值漏接——**纯配置遗漏，不是算法缺陷**。

### 3.4 修复

`ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`：`Q4_0_ROCMFP4` / `Q4_0_ROCMFP4_FAST` 的 12 条 CASE 条目将 `MMQ_ITER_K` 改为 `MMQ_ITER_K_FP4`。

### 3.5 验证链

- **数值门**：`MUL_MAT -p rocmfp4` **24/24**、`-p rocmfpx` **61/61**。
- **整网 A/B**（Qwen3.8-27B Q4_0_ROCMFP4_FAST，同机同模型同协议背靠背）：

  | 场景 | 改前 | 改后 | 变化 |
  |:--|--:|--:|--:|
  | pp2048 | 433.5 t/s | **720.8 t/s** | **+66.3%** |
  | pp16384 | 396.4 t/s | **613.5 t/s** | **+54.8%** |
  | pp65536 | 295.0 t/s | **413.6 t/s** | **+40.2%** |
  | tg256 | 14.04 t/s | 14.04 t/s | 不变（解码走 MMVQ） |

- **横向对照**：gufo 同款模型的公开记录为预填 656 / 575 / 387（0 / 16k / 64k 深度），本版为 721 / 614 / 414——三档均在其上（量化档不同，口径为同芯片同模型级别）。

### 3.6 边界与否决项

- **fp8 族不适用**：把同一改动套到 `Q2/Q3/Q6/Q8_0_ROCMFPX` → 内核 GEMM 20.2→34.8 TF（+72~86%），但数值门 **49/61**，失败用例全部为 **k=1024**（ERR≈1.0，结果整块错），k=256/k=4096 通过。K 瓦片与 SRAM 布局绑定（K 分位超块粒度 256），改动涉及块索引/加载器，属独立改造，未纳入本版。
- occ1 / both 无收益，不采用。

### 3.7 复现命令

```bash
# 内核用例（新增 27B 投影形状）
LD_LIBRARY_PATH=build/bin:/opt/rocm/core-10.1/lib \
  build/bin/test-backend-ops perf -o MUL_MAT -p "m=17408,n=2048,k=5120"
# 数值门
LD_LIBRARY_PATH=build/bin:/opt/rocm/core-10.1/lib build/bin/test-backend-ops test -o MUL_MAT -p "rocmfp4"
# 整网
build/bin/llama-bench -m <27B Q4_0_ROCMFP4_FAST.gguf> -p 2048 -n 0 -r 3
```

### 3.8 交付

- 代码：`mmq-config-rdna3-5.cuh`（12 条条目）+ `tests/test-backend-ops.cpp`（新增 perf 用例）。
- 数值门、端到端 A/B 数据在 `~/bench/gemm-e2e-1017/`、`~/bench/mmq-scan2-1017.log`。

---

## 附：条目结构（后续内核级改动沿用）

1. 问题（含基线数字与适用条件）
2. 定位过程与证据（逐内核采样 / 硬件计数器 / 对照臂）
3. 根因（具体文件与函数）
4. 修复（提交号 + 代码要点）
5. 验证链（位等价 / 内核测试 / 计数器复核 / 部署双锚）
6. 效果数据表与复现命令
