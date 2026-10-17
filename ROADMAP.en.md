# Roadmap

This document describes **what we plan to do next**. For what has already shipped, see [CHANGELOG.en.md](CHANGELOG.en.md); item-by-item evidence lives in the kernel optimization record and in each release's notes.

Current version: **v2026.10.18 (GEMM K tile, not yet tagged)**

The roadmap lists work in progress, not commitments — scope and order may change. What gets done first, and whether it gets done at all, is the maintainer's call.

---

## Next batch

### MoE duplicate experts: fill the hole

**Area: MoE path (MUL_MAT_ID)**

Some models (Ornith and the like) route through experts. The same expert being selected more than once is a legal routing outcome, but today such input produces wrong numbers, and extreme shapes take the GPU down outright. Regular routing rarely hits it — but it has no guard either.

- Fix the mmid duplicate-id path: the wrong values and the out-of-bounds access together
- Add a `dup` numeric gate to the full test suite
- Add nothing new on that path until it is all green

> A rare error is still an error.

### The fp8 family's K tile: collect the money behind the gate

**Area: ROCmFPX family (Q2/Q3/Q6/Q8)**

Wiring the 512 K tile into ROCmFP4 made prefill 66% faster. The fp8 family still sits on 256, but inserting the same key makes `k=1024` results completely wrong — the tile size and the SRAM layout are bound together.

- Rework the fp8 family's K block indexing/loaders (address by 256-element super-blocks)
- Separate branch, separate numeric gate — never mixed into the already-verified version
- Per-shape A/B; book the gains before talking end-to-end

> Speed goes through the gate before it goes into the books.

### Fused activation quantization: drop the per-layer round trip

**Area: prefill path (RMSNorm → MMQ)**

Every RMSNorm writes its result back to VRAM as F32, and MMQ reads it back to quantize. At batch 2048 that is a couple of hundred MB per layer. What gufo does is have the norm's exit write the quantized tile directly.

- Fuse it at the graph level; validate on a single model and layer first
- Byte-for-byte equivalence before any end-to-end A/B
- Keep the retreat wide: the fused op must be switchable off

> One less trip beats a faster trip.

### Prefill GEMM tiles and routing: push 53 higher

**Area: prefill GEMM (all dense layers)**

The K tile already lifted the main shapes to 53 TFLOPS, but the tile layout is still the generic dispatch. gufo's published 30–34 TF was their old number; our target is 60+.

- Re-tile for the 27B shape family (I=128/256, J=128)
- Evaluate whether stream_k is worth it
- Keep it only if end-to-end A/B does not regress

> Generic is for everyone; tuning is for this one chip.

### Long-context decode: assess before promising

**Area: decode (MMVQ / KV path)**

Decode is currently level with the previous version. gufo's 1.7× comes from KV tiles shared across heads plus sparse splitting, but that is their model's structure — whether it holds for 27B/Ornith needs to be costed first.

- Assess the gain and the VRAM cost; write a one-page conclusion
- Schedule it if it holds; record a rejection if it does not
- Do not experiment on the live decode path

> Measure first, then move.

---

## Decide with us

- **Production switch**: new lib `95f2ed` (K-tile build) → `~/llama-rocm/bin`, via full mirror + smoke test + rollback plan.
- **Release**: whether to tag this change `v2026.10.18` and write a Release.
- **What runs first**: MoE fix protects correctness, the fp8 family buys performance — the priority is the maintainer's call.

---

## Known boundaries (current version)

- Deep-context precision edge: beyond 32k tokens, f16-KV attention slowly drifts from the f32 reference (about 3e-3 at 131k); the same on upstream.
- The fp8 family still runs K tile 256.
- The MoE duplicate-expert defect is unfixed and disclosed in the README's known limitations.
