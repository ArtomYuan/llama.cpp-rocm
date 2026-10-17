# Kernel optimization record (ROCmFPX / ROCmFP4)

[简体中文](kernel-optimization.md)

Records of kernel-level optimizations applied to the ROCmFPX / ROCmFP4 quantization families on HIP (gfx1151) in this repo (strix.cpp). Each entry covers the problem, profiling evidence, root cause, fix and validation data, and can be reproduced with the commands included.

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

## 2. Attention kernel: 512-thread config split by (ncols1,ncols2) (v2026.10.17)

### 2.1 Problem

On gfx1151 the FLASH_ATTN_EXT mma-f16 main kernel under the stock config (`(256,256,64): 256,2,64` = 256 threads/occ2/combine64) saturates VGPR at 256, spills **515 slots/thread**, scratch 1076B - deep-context throughput (kv>=8k) is spill-bound: kv=16k gqa8 baseline **11.48 TFLOPS**.

### 2.2 Profiling evidence

- Per-kernel sampling + hardware counters: wait-class 82%, L2 hit 6.8% -> latency-bound; the spatial dimension points to register pressure.
- Single-instance resource report (`-Rpass-analysis=kernel-resource-usage`): VGPR 256 (cap), 515 spills/thread, scratch 1076B.
- 512-thread config trial (occ 2->1, combine 64->32): spills **-60%**, kv=16k **+21%** (11.5->13.97).
- Whole-row variant (shared by all ncols=64 instances) costs the `(32,2)` family **-16~21%** - the row must be split by `(ncols1,ncols2)`.

### 2.3 Root cause

The config table is keyed by `ncols`: the `ncols=64` row serves `(8,8)`/`(16,4)`/`(32,2)`; the 512-thread win only holds for the wide-head-group instances, and is negative for `(32,2)`.

### 2.4 Fix

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: twin split helpers - device side `ggml_cuda_fattn_mma_xaihi_split_512_dev` (macro), host side `ggml_cuda_fattn_mma_xaihi_split_512_host` (runtime `amd_wmma_available(cc)`), wired at 5 call sites (launch_bounds / kernel body nthreads / process_tile combine / host nthreads / host combine); only `(256,256,8,8)` and `(256,256,16,4)` enabled.
- **Trap (host/device macro visibility)**: the `RDNA3` macro derives from the device-pass-only `__GFX11__` and is invisible in the host pass - a single `__host__ __device__` predicate yields a host-256/device-512 mismatch: **NaN (first elements) + ~2x fake speed** (half work). Twin helpers are mandatory.

### 2.5 Verification

- Numerics: `test-backend-ops test -o FLASH_ATTN_EXT -p "hsk=256.*type_K=f16"` = **148/152**; the 4 failures are the pre-existing kv>=32k precision edge (same on stock).
- Localization matrix: nb gradient (2/4/8/16/32/64) and n_kv_max gradient (256/512/1024/2048) all pass; the initial `(8,8)`-nb>=8 NaN regression is gone after the fix.
- Performance (same-moment A/B): `(8,8)` kv=16k **11.48->14.08 TF (+22.6%)**; `(16,4)` **8.84->10.63 (+20.2%)**; P1 `(32,2)` neutral; 32k/64k deep contexts **+4%**; 4k short contexts -3~6% (inside the trade-off).
- Build determinism: two rebuilds from the same source/toolchain are byte-identical (md5).

### 2.6 Reproduction

```bash
# numerics
test-backend-ops test -o FLASH_ATTN_EXT -p "hsk=256.*type_K=f16"

# performance (same-moment A/B; perf cases are generated by parameterized loops: nh=8 x ratio in {1,4,8} x kv in {4096..65536} x nb in {1,4096})
test-backend-ops perf -o FLASH_ATTN_EXT -p "hsk=256,hsv=256,nh=8,nr23=\[8,1\],kv=16384,nb=4096"
test-backend-ops perf -o FLASH_ATTN_EXT -p "hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=16384,nb=4096"
test-backend-ops perf -o FLASH_ATTN_EXT -p "hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=16384,nb=512"
```

### 2.7 Delivery

- Commits: `43ea02cb3` (config split), `c57487a5b` (FA vector gate), `6d4fc573a` (test coverage), `0eb0adcf4` (llama-bench types)
- Version: **v2026.10.17 (v0.2.7)**

---

## 3. Prefill GEMM: the ROCmFP4 MMQ K tile now uses the FP4-specific value (v2026.10.18)

### 3.1 Problem

The 27B prefill GEMM (`Q4_0_ROCMFP4_FAST`) reached only **27.5 TFLOPS** on the main projection shapes — 82% of f16 (33.3) on the same shapes, despite reading 3.7× fewer bytes at 4.25 bpw. Compared with gufo's published records for the same model class (prefill ≈ 2× llama.cpp), the gap was not in attention but in the GEMM itself.

### 3.2 Investigation

1. **Baseline** (`tests/test-backend-ops perf -o MUL_MAT`, 27B projection shapes): Q4 = 27.47 / 25.75 / 25.23 TFLOPS, f16 = 33.27 / 24.31 / 27.73.
2. **Dispatch review** (`ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`): type 101 (`Q4_0_ROCMFP4_FAST`) entries used `MMQ_ITER_K=256`, while the FP4-specific `MMQ_ITER_K_FP4=512` (`mmq.cuh:10`) existed but was never referenced by the ROCmFP4 entries.
3. **Variant measurements** (rebuild + same-session measurement each time):

   | Variant | 17408×2048×5120 | Verdict |
   |:--|--:|:--|
   | baseline (256) | 27.90 TF | control |
   | **iter512 (K=512)** | **53.33 TF (+90.6%)** | adopted |
   | occ1 (occupancy 2→1) | 27.34 TF | rejected (no gain) |
   | both (512 + occ1) | 52.57 TF | rejected (no stacking) |

### 3.3 Root cause

When type 101 was wired into MMQ, its config entries copied the generic K tile of 256 and missed the FP4-specific value — a pure configuration omission, not an algorithmic defect.

### 3.4 Fix

`ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`: 12 CASE entries for `Q4_0_ROCMFP4` / `Q4_0_ROCMFP4_FAST` switch `MMQ_ITER_K` to `MMQ_ITER_K_FP4`.

### 3.5 Verification

- **Numeric gates**: `MUL_MAT -p rocmfp4` **24/24**, `-p rocmfpx` **61/61**.
- **End-to-end A/B** (Qwen3.8-27B Q4_0_ROCMFP4_FAST, same machine/model/protocol, back-to-back):

  | Case | Before | After | Change |
  |:--|--:|--:|--:|
  | pp2048 | 433.5 t/s | **720.8 t/s** | **+66.3%** |
  | pp16384 | 396.4 t/s | **613.5 t/s** | **+54.8%** |
  | pp65536 | 295.0 t/s | **413.6 t/s** | **+40.2%** |
  | tg256 | 14.04 t/s | 14.04 t/s | unchanged (decode uses MMVQ) |

- **Cross-reference**: gufo's published records for the same model class are 656 / 575 / 387 (depths 0 / 16k / 64k); this version is 721 / 614 / 414 — above all three (different quant, same chip/model class).

### 3.6 Boundaries and rejected items

- **Not applicable to the fp8 family**: applying the same change to `Q2/Q3/Q6/Q8_0_ROCMFPX` lifts the kernel GEMM from 20.2 to 34.8 TF (+72~86%), but the numeric gate drops to **49/61** — every failing case is **k=1024** (ERR≈1.0, blocks completely wrong), while k=256/k=4096 pass. The K tile is bound to the SRAM layout (256-element super-block), so fixing it means changing block indexing/loaders — a separate change, not part of this version.
- occ1 / both give no gain; rejected.

### 3.7 Reproduction

```bash
# kernel case (the added 27B projection shapes)
LD_LIBRARY_PATH=build/bin:/opt/rocm/core-10.1/lib \
  build/bin/test-backend-ops perf -o MUL_MAT -p "m=17408,n=2048,k=5120"
# numeric gate
LD_LIBRARY_PATH=build/bin:/opt/rocm/core-10.1/lib build/bin/test-backend-ops test -o MUL_MAT -p "rocmfp4"
# end to end
build/bin/llama-bench -m <27B Q4_0_ROCMFP4_FAST.gguf> -p 2048 -n 0 -r 3
```

### 3.8 Deliverables

- Code: `mmq-config-rdna3-5.cuh` (12 entries) + `tests/test-backend-ops.cpp` (new perf cases).
- Numeric gate and end-to-end A/B data in `~/bench/gemm-e2e-1017/`, `~/bench/mmq-scan2-1017.log`.

---

## Appendix: entry structure (follow for future kernel-level changes)

1. Problem (with baseline numbers and applicable conditions)
2. Profiling process and evidence (per-kernel trace / hardware counters / control arms)
3. Root cause (specific files and functions)
4. Fix (commit id + code essentials)
5. Verification chain (bit-exactness / kernel tests / counter re-check / deployment double-anchor)
6. Effect table and reproduction commands
