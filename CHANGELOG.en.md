# Change Log

[简体中文](CHANGELOG.md)

## [Unreleased]

### Engine

- **Branch-free hot-path decode (ROCmFPX/ROCmFP4)**: removed the data-dependent branches from the ue4m3 scale decode (executed once per weight block per thread; ternaries now lower to SEL/CSEL), bit-exact for all 256 inputs. Measured on gfx1151 (Qwen3-Embedding/Reranker-8B, pp2048): Q8_0_ROCMFPX **1107→1233 t/s (+11.4%, now above std Q8_0's 1227)**, Q6_0_ROCMFPX 920→1013 (+10.1%), Q4_0_ROCMFP4_FAST 1246→1294 (+3.9%). Root-caused with rocprofv3 counters (the old fp8 MMQ kernel showed +39% SALU, +40% branches, +86% WAIT_ANY, +18% wave cycles vs std; memory counters identical throughout); after the fix every counter matches std, and the fp4 MMQ kernel's wave cycles drop by 8.6%. Verified: test-backend-ops MUL_MAT rocmfpx 60/60, rocmfp4 24/24.

## [v2026.9.21] (2026-09-21)

### Engine

- **ROCmFPX fp8 family (Q2/Q3/Q6/Q8_0_ROCMFPX): GPU compute paths wired up** (ported from charlie12345/ROCmFPX via the extraction method and adapted to this repo's config-based MMQ architecture; previously this family could not run on GPU — abort during load warmup):
  - MMVQ (`mmvq.cu`): dispatch wiring for the four fp8 types (vec-dot table, table selection, kernel specializations, RDNA3.5 parameter table); ported the fp2-specific multi-column kernel `vec_dot_rocmfpx_fp2_q8_1_ncols` (fp2→int8 expansion done once per row for multi-column). Macro defaults keep the old behavior (no-op knobs, tuning not yet enabled).
  - MMQ (`mmq-load-tiles.cuh` / `mmq.cuh` / `mmq-config-rdna3-5.cuh`): four new loaders (fp2/fp3/fp6 use the Q3_K SRAM layout, fp8 uses the Q8_0 layout), ds-layout and tile-size entries, util_funcs wiring in both the dp4a and mma sections, and 48 (I,J) config entries for RDNA3.5 (same grid as Q8_0/Q3_K). The template-instance files (`mmq-instance-*.cu`) already existed and are now reachable.
  - Gating: fp8-family MMQ is currently enabled on RDNA3.5 only (other architectures fall back to dequant + hipBLAS instead of hitting the "no J config" default path); `GGML_HIP_NO_ROCMFPX_MMQ=1` disables the family anywhere (debug/rollback).
- Regression: fp4-family behavior unchanged (the RDNA3.5 table change defaults to old behavior; Ornith Q4_FAST new-vs-old engine differences within noise).

### Tests

- `tests/test-backend-ops.cpp`: `all_types[]` now includes the four fp8 types, plus large-shape fp8 cases (m=4096/251, n=128/512, k=1024, covering the large J tile and both fallback grids). Result: **60/60 pass** (MMVQ, MMQ and hipBLAS paths). End-to-end: Qwen3-Embedding-8B-Q8_ROCMFPX GPU smoke (including the MMQ batch path) within e-4 of the CPU baseline; fidelity matches the CPU reference (top-5 neighbour overlap 96.4%).

## [Unreleased] (2026-09-15)

### Docs

- Bilingual docs: README.en.md is now kept in sync with the Chinese README; added CHANGELOG.en.md (this file). Release notes are bilingual as of this version.

## [v2026.9.16] (2026-09-15)

### Fixed

- Fixed the dispatch mismatch for dual-scale `Q4_0_ROCMFP4` on the MMQ path (`mmq.cuh` now uses the per-16-value vec-dot variant shared with Q3_K): numerical error ~3e-2 -> ~3e-8, 2296/2296 cases pass. Previously dual-scale models silently computed wrong results via MMQ; production `_FAST` (single-scale) is unaffected.

## [v2026.9.15] (2026-09-15)

### Engine

- Test coverage fix: `tests/test-backend-ops.cpp` `all_types[]` now includes `GGML_TYPE_Q4_0_ROCMFP4` / `GGML_TYPE_Q4_0_ROCMFP4_FAST`, fixing the false green (previously the whole `tests/` tree had 0 ROCmFP4 cases). Two-arm measurement: default arm 2233/2236 (all 3 failures are dual-scale `q4_0_rocmfp4` non-FAST, a known numerical defect); `GGML_HIP_NO_ROCMFP4_MMQ=1` arm 2236/2236 all pass.
- `mmq.cu`: cleaned up the misplaced guard - restored the `highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A` branch to the upstream pre-Pascal NVIDIA form (the old `return false` + misleading ROCmFPX comment is unreachable on AMD, and this repo once mistook it for "MMQ disabled"); added a reachable `GGML_HIP_NO_ROCMFP4_MMQ` fallback switch (on by default, behavior unchanged; when set, only the dual ROCmFP4 types fall back to hipBLAS, standard-format MMQ and MMVQ are unaffected).
- **Known issue**: dual-scale `Q4_0_ROCMFP4` (non-`_FAST`) exceeds the numerical threshold on the MMQ path (NMSE 0.01-0.04, about 21-80x the `max_nmse_err=5e-4` threshold); not introduced in this version; production `_FAST` (single-scale) is unaffected; impact = future models quantized with dual-scale `Q4_0_ROCMFP4` (current production models are all `_FAST`).

### Docs

- Corrected the "MMQ not ready / disabled" and "off 69.17 vs on 67.26 is better" wording in README/CHANGELOG (the old 69.17/67.26 is a tg32/batch-1 decode metric that goes through MMVQ, not MMQ); verified MMQ has always been on and prefill is ~2x (pp512 +105%, pp2048 +109%, measured 2026-09-15).

## [Unreleased] (2026-09-14)

### Engine

- Upstream sync: ggml-org/llama.cpp master -> 97e4ca735 (window e107984bc..97e4ca735 = **172 commits**, merge commit cfeb42ff6, pre/post evidence tags sync-20260914-pre/post).
- Fixed an existing defect: `ggml_validate_row_data` was not wired into ROCmFP4/ROCmFPx/TurboQuant validation dispatch, so llama-quantize reported "invalid type 101" when producing FPX formats and quantization failed (commit da935e773; added the eight cases 100-107, TurboQuant uses FP16 L2-norm validation).
- All four fusion-surface items preserved: (1) ROCmFPX quantization (GGML types 100-107) (2) mtmd-grounders multimodal vision (LocateAnything + DeepSeek4V dual projector) (3) MoE-scenario MMQ "disabled" (`return false` inside `ggml_cuda_should_use_mmq`; note: historical misrecord, since corrected - this guard is unreachable on AMD and MMQ has actually always been on, see the 2026-09-15 entry) (4) upstream CI workflows deleted (the `.github/workflows/fusion.yml` added in this window was also deleted).
- Conflict resolution: 24 mechanical conflicts (`.github/workflows/*.yml` modify/delete, resolved with `git rm` to keep the local deleted state, plus the `.github/workflows/fusion.yml` added upstream in this window was also deleted) + 4 real content conflicts (README.md resolved as the whole local Chinese rewrite; `ggml/src/ggml-cuda/CMakeLists.txt`, `ggml/src/ggml-hip/CMakeLists.txt`, `ggml/src/ggml-cuda/fattn.cu` hand-merged block by block. Note: `ggml/src/CMakeLists.txt` was auto-merged, not a content conflict).
- Contrary to prediction: `mmq.cu` / `mmq.cuh` / `mmq-config-*.cuh` / `ggml/include/ggml.h` / `include/llama.h` / `tools/mtmd/clip.cpp` / `src/models/gemma4.cpp` all **auto-merged with zero conflicts**, the fusion surface untouched (eight review greps all PASS: GGML_TYPE_COUNT=108, mmq-config-rdna3 ROCMFP=24, mmq.cu MoE `return false`, arg.cpp kv_cache FPX, llama-quant.cpp ROCMFP=84, clip.cpp dual projector, no workflow residue, no conflict markers).
- Key upstream evolution areas: build system PCH + unity build rolled out (#28091), to be corrected against upstream later; GGML_FA_QUANTS split (#28079) - FlashAttention vector kernel instance list turned into a helper (`ggml_cuda_fattn_vec_instances`) + dispatch changed to `ggml_cuda_get_fattn_vec_case` lookup, this repo's FPX/turbo instances and `GGML_CUDA_FA_*_ROCMFP*` compile definitions re-wired; RDNA3/4 MMQ MoE N-tile size optimization added-then-reverted-then-redone, plus new `mmq-config-gcn.cuh`; HIP/CUDA BF16 fallback, gfx1201 FA adjustments, mmf/mmid concurrency fixes, gfx90c HIP support; mtmd gemma4 vision fixes, video ID propagation, `mtmd_tokenize_from_parts`; server subprocess refactor, LRU hang fix, schema internal representation and UI render performance; models Maple 20B-A1B ternary MoE, Tencent Hy4 preview, Kimi-K3 rollback, Nemotron-3-Puzzle, Spark2_5, Qwen3-Next/Qwen3.5 recurrent_layers.
- Upstream version number advanced: `LLAMA_VERSION_BASE` 0.3.0 -> **0.4.0** (matching the upstream v0.4.0 milestone). The namespace conflict between this repo's own version tags and upstream tags is covered in the S11a version proposal (pending admin confirmation).
- Deployment: the two systemd services 8010/8020 switched `--no-mmap` to `--load-mode none` (upstream formally removed the deprecated `--no-mmap` alias, the new engine no longer accepts the old flag; `--load-mode none` is semantically equivalent to `--no-mmap`, confirmed bidirectionally by source + measurement).

### Docs

- Upstream sync record (2026-09-14, merge commit cfeb42ff6; upstream window e107984bc..97e4ca735, 172 commits).
- The most complex block-merge surface = `ggml/src/ggml-cuda/fattn.cu`: upstream #28079 changed FA vector kernel dispatch from an inline cascade to the `ggml_cuda_get_fattn_vec_case()` lookup function; this repo registered 5 ROCmFPX diagonal combinations into that function and extended `ggml_cuda_fattn_kv_type_supported` to ROCmFPX/TurboQuant (otherwise the `is_rocmfp_family` route is unreachable); two CMakeLists append FPX/turbo instances and compile definitions after the helper call. This is the only high-semantic-risk surface in this sync; compile correctness is backstopped by the build stage.
- README unchanged in this sync: upstream README had only a 1-line change in this window (maintainer navigation link list), this repo's README is a Chinese rewrite, conflict resolution kept local; README review conclusions in S11a-readme-notes.md.

## [Unreleased] (2026-09-03)

### Engine

- Upstream sync: ggml-org/llama.cpp master -> e107984bc (117 commits, incl. ROCm radix TOP_K for long rows #27466, mtmd const propagation, DeepSeek4V projector, Nemotron-3-Puzzle model, Metal sparse FA, Vulkan FA dequant fix, etc.).
- Local TOP_K chunked-merge fix (f883d66c8) retired: the upstream radix implementation covers the same HIP long-row crash scenario (#27021), measured 3.5-13.5x faster.

### Docs

- Upstream sync record (2026-09-03, merge commit e1697de49).

## [v0.1.0] (2026-08-28)

### Added

- Three-in-one fusion engine: upstream llama.cpp master + ROCmFPX quantization + mtmd multimodal vision.
- ROCmFPX quantization format support (ROCmFP2/3/4/6/8 + TurboQuant, GGML types 100-107).
- Multimodal vision tower support (incl. LocateAnything grounding projector, `--special` mode).
- HIP backend gfx1151 adaptation (ROCm 10.0.0 build).

### Changed

- Disabled MMQ for MoE (gfx1151 MMQ kernel not ready, measured mmq off 69.17 vs on 67.26 t/s). (note: historical misrecord, since corrected - this "disabled" is unreachable dead code and MMQ has always been on; the old 69.17/67.26 is a tg32 decode metric that goes through MMVQ, not MMQ; see the 2026-09-15 entry)

### Removed

- Upstream CI workflows (the official matrix does not apply to the fusion repo; validated by local builds).

### Docs

- Added README (positioning/build/usage).
- README rewrite (upstream differences/features/measured performance/deployment).
- Bilingual README (Chinese default + English).
- Full quantization format list (Q4_0_ROCMFP4 family 9 variants + Qx_ROCMFPX family 9 types).
- Community files: CONTRIBUTING / CODE_OF_CONDUCT / SECURITY / Issue templates.
