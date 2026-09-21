# Kernel optimization record (ROCmFPX / ROCmFP4)

[简体中文](kernel-optimization.md)

Records of kernel-level optimizations applied to the ROCmFPX / ROCmFP4 quantization families on HIP (gfx1151) in this repo (llama.cpp-rocm). Each entry covers the problem, profiling evidence, root cause, fix and validation data, and can be reproduced with the commands included.

---

## 1. Branch-free hot-path ue4m3 scale decode (v2026.9.22)

### 1.1 Problem

On gfx1151, `Q8_0_ROCMFPX` prefill was ~9–11% slower than standard `Q8_0` (same-condition pp2048: **1107 vs 1235 t/s**). The gap had been tentatively attributed to the 33-byte block stride / SRAM layout / memory alignment.

### 1.2 Profiling (two levels with rocprofv3)

**Level 1 — per-kernel trace** (`--kernel-trace`, both builds under identical conditions):

- 100% of the GPU-time difference falls inside the `mul_mat_q` (MMQ) main kernel (+11% per dispatch); attention, norm, quantize and copy kernels are identical.
- Both builds launch the same number of kernels, with identical VGPR (232) / SGPR / workgroup and grid configurations → rules out the "extra work" and "occupancy" causes.

**Level 2 — hardware counters** (`--pmc`, `mul_mat_q` only; old fp8 kernel vs std):

| Counter | fp8/std | What it says |
|:--|:--|:--|
| `SQ_INSTS_SALU` | **1.389** | excess scalar instructions |
| `SQ_INSTS_BRANCH` | **1.402** | excess branches |
| `SQ_WAIT_ANY` | **1.860** | waiting cycles blow up |
| `SQ_WAVE_CYCLES` | **1.177** | longer per-wave time for equal work |
| `SQ_INSTS_FLAT` / `SQC_DCACHE_REQ` / `GL2C_MC_RDREQ` / LDS conflicts | 1.000 | **memory side identical** |

Conclusion: not a memory / bandwidth / alignment issue; the gap is pipeline stalls caused by **the scalar instructions and branches of the decode path**.

### 1.3 Root cause

Two ue4m3 scale-decode helpers in `ggml/rocmfp4/rocmfp4_hip_scale.cuh` (`rocmfpx_ue4m3_to_fp32_finite` and `rocmfp4_ue4m3_to_fp32_half_finite`) used **data-dependent if branches**, and they sit on the **per-weight-block, per-thread** hot path of the MMQ loaders, vec_dot and cpy-utils.

```cpp
// before (excerpt of rocmfpx_ue4m3_to_fp32_finite)
if (x > 0x7e) { return 0.0f; }
const int exp = (x >> 3) & 0xF;
const int man = x & 0x7;
if (exp == 0) { return (float) man * (1.0f / 1024.0f); }
```

### 1.4 Fix (commit `0347fac53`)

Rewritten branch-free (ternaries → SEL/CSEL-class select instructions), bit-identical:

```cpp
// after
const int exp = (x >> 3) & 0xF;
const int man = x & 0x7;
const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
const float    norm = rocmfp4_u32_as_f32(bits);
const float    subn = (float) man * (1.0f / 1024.0f);
const float    val  = (exp == 0) ? subn : norm;
return (x > 0x7e) ? 0.0f : val;
```

A single change covers the entire hot path of both the fp8 (Q2/Q3/Q6/Q8) and fp4 families.

### 1.5 Verification

- **Bit-exactness**: both helpers, before vs after, compared exhaustively over all 256 inputs (zero numeric change).
- **Kernel tests**: `test-backend-ops` MUL_MAT `rocmfpx` **60/60**, `rocmfp4` **24/24**.
- **Counter re-check**: the new fp8 kernel matches std on every collected counter (SALU / branch / wait = 1.00x, wave cycles 0.991); the fp4 MMQ kernel is 8.6% lower in wave cycles and 20% lower in waits.
- **Deployment double-anchor** (old vs new binaries side by side in the deploy dir): Qwen3-Embedding-8B Q8_0_ROCMFPX pp2048 **1114.9 → 1235.3 t/s (+10.8%)**; Ornith Q4_0_ROCMFP4_FAST tg128 71.99 → 71.95 (flat).

### 1.6 Effect (llama-bench pp2048)

| Variant (Emb / Rnk) | Before | After | Change |
|:--|:--|:--|:--|
| Q8_0_ROCMFPX | 1107 / 1099 | 1233 / 1227 | **+11.4% / +11.6%** |
| Q6_0_ROCMFPX | 920 / 916 | 1013 / 1010 | +10.1% / +10.2% |
| Q4_0_ROCMFP4_FAST | 1246 / 1244 | 1294 / 1298 | +3.9% / +4.3% |

### 1.7 Reproduction

```bash
# Per-kernel trace (run both builds under identical conditions; ~5% sampling overhead, relative comparisons unaffected)
rocprofv3 --kernel-trace -f csv -o /tmp/prof -- \
  llama-bench -m <model.gguf> -embd 1 -ngl 99 -p 2048 -n 0 -r 1

# Hardware counters (note the `--` separator; pass counter names as separate args; `-f csv` is required)
rocprofv3 --pmc SQ_WAVES SQ_BUSY_CYCLES SQ_INSTS_SALU SQ_INSTS_BRANCH \
  SQ_WAIT_ANY SQ_WAVE_CYCLES --kernel-include-regex "mul_mat_q" \
  -f csv -o /tmp/pmc -- llama-bench -m <model.gguf> -embd 1 -ngl 99 -p 2048 -n 0 -r 1

# Kernel correctness
test-backend-ops test -o MUL_MAT -p "rocmfpx"   # 60/60
test-backend-ops test -o MUL_MAT -p "rocmfp4"   # 24/24
```

### 1.8 Delivery

- Commits: `0347fac53` (code), `f46544c29` (README measured-performance refresh)
- Version: **v2026.9.22 (v0.2.5)** (see the `[v2026.9.22]` entry in the changelog)

---

## Appendix: entry structure (follow for future kernel-level changes)

1. Problem (with baseline numbers and applicable conditions)
2. Profiling process and evidence (per-kernel trace / hardware counters / control arms)
3. Root cause (specific files and functions)
4. Fix (commit id + code essentials)
5. Verification chain (bit-exactness / kernel tests / counter re-check / deployment double-anchor)
6. Effect table and reproduction commands
