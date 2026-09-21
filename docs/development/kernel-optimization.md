# 内核优化记录（ROCmFPX / ROCmFP4）

本库（llama.cpp-rocm）对 ROCmFPX / ROCmFP4 量化家族在 HIP（gfx1151）上**内核级优化**的记录：每条含问题、定位证据、根因、修复与验证数据，均可按文内命令复现。

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

## 附：条目结构（后续内核级改动沿用）

1. 问题（含基线数字与适用条件）
2. 定位过程与证据（逐内核采样 / 硬件计数器 / 对照臂）
3. 根因（具体文件与函数）
4. 修复（提交号 + 代码要点）
5. 验证链（位等价 / 内核测试 / 计数器复核 / 部署双锚）
6. 效果数据表与复现命令
