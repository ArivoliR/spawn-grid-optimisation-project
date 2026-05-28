# Design Document — Monster Spawning Grid
**Name:** Arivoli Ramamoorthy, Sumanyu Aggarwal, Abhijai Chugh  
**Date:** 28 May 2026  
**Final median time (10 runs, public_1 32768):** 116,830 ms  
**Reference median time (10 runs, public_1 32768):** ~124,000,000 ms  
**Speedup:** ~1063×

---

## 1. Cell Representation

We represented each cell as a 2-bit bitplane, with the states being EMPTY=00, EGG=01, JUVENILE=10, ADULT=11. Finding if a cell is ADULT can be done using `low & high` which is one `vandq_u8` operation. The whole grid is represented as 2 arrays, s0 and s1, representing the low and high bits of the cells respectively.

This splitting into 2 separate arrays helped in a simpler and performant implementation for SIMD as we could directly bitwise and the two bitplanes without shifting. Each plane packs 128 cells into one `uint8x16_t`, so every boolean op runs on 128 cells at a time.

The neighbourhood is 5x5 (= 25 cells max), which fits in 5 bits. We keep the count as 5 separate bitplanes `V = (b0..b4)` and update it with bit-level adders, not normal integer add. The horizontal 5-input row sum uses two full-adders and one half-adder; the vertical transition is `V -= H_leaving; V += H_entering`. SHA3's EOR3 collapses each 3-input XOR into one instruction, and BSL does the same for the majority/carry term.

Alternatives considered:

- One byte per cell - Used by the reference. Uses an extra 6 bits for each cell resulting in higher memory traffic and less SIMD parallelism.
- Two bits per cell packed contiguously: SIMD operations needed masking and shifting to separate out the bits, increasing instructions.

---

## 2. Parallelisation Strategy

The work is distributed using an 8-thread thread pool. Each thread is assigned N/8 rows of the grid - rows `[t*N/8, (t+1)*N/8)`, where N is the grid size and t is the thread number. As each thread works on different cells/rows, there is no false sharing and no need for data locks, atomic variabls. The pool is created once at startup and synchronised twice per generation with std::barrier (one releases the workers, one rejoins them).

Contiguous row blocks keep each thread on a linear sweep, which lets the CPU's hardware prefetcher predict the next cache line. Threads only need to read the 2 rows above and below their current block for the 5-row stencil. Since the source grid is read-only and all threads write to disjoint row ranges in the destination, no locks or coordination are needed at boundaries between thread partitions.

The toroidal halo at the boundaries are handled during the H-fill pass.
Horizontal: While computing the horizontal count for the first register, the prev register is explicitly set to the last register in the row. Similarly, for the last register, next is set as the first register.
Vertical: During H-row indexing, we mask the row index with N-1, which correctly wraps the row indices.

We tried `std::execution::par` early on. On this libstdc++ setup, the parallel policy needs the TBB backend, which is outside the allowed libraries. So we went with std::thread.

---

## 3. SIMD Strategy

Yes, we are using SIMD. 
We use NEON with the sha3 extension which provides 3-input boolean instructions. All the bitwise operations are vectorised and use SIMD (except for the case when N < 128, in which case we fallback to scalar). We use SIMD for H adders, V transitions and for finding the next state of a cell.

Why not SVE2 - SVE2 doesn't provide any advantage as on Graviton4, it has 128 bits wide registers, the same as NEON. We microbenchmarked NEON vs SVE2 op-by-op at 128 bits: most ops were tied, NEON was ~24 % faster on BSL (which sits inside our hot-path MAJ, called many times per output cell), SVE2 was ~11 % faster on shifts by a constant (which only appear in the H-fill pass, ~0.7 s at 32K by paper math). We ported the shift sites to SVE2, but didn't see any improvements, so we stayed with NEON.

---

## 4. Memory Layout and Tiling

The core structure is a block-H buffer with a vertical sliding count. One spatial block (BLOCK_ROWS rows at a time per thread), no temporal blocking. Per generation, per thread:
1. Pass 1: compute (BLOCK_ROWS + 4) horizontal-sum rows into block-H scratch.
2. Pass 2: V := sum of the first 5 H rows.
3. Slide: for each output row, `V -= H_leaving; V += H_entering;`, then apply rule for getting the next state, write into destination.

We chose BLOCK_ROWS as 88 after benchmarking with different values. At N=32K, the block-H scratch is (88+4) * 3 planes * 4096 = 1.08 MiB per thread. V scratch is 20 KiB. So ~1.10 MiB total, which fits comfortably withing the L2 cache (2 MiB per core on Graviton4) with margin.

Both scratch structures use a register-local layout: for each column register, all the planes (3 for H, 5 for V) sit in adjacent bytes instead of being separated by 4096-byte full-row strides. Adjacent records means the loads hit one cache line as row-strided would put them on the same L1 sets (due to cache associativity) and cause evictions.

All buffers use 64-byte aligned allocation. We also call `madvise(MADV_HUGEPAGE)` on src, dst, and per-thread scratch to hint the OS to use a 2MB page. When followed, the measured TLB miss rate ended up low enough that TLB pressure was not the bottleneck.

Why no temporal blocking - Advancing multiple generations per tile (tested at K=4) was slower — slab copy overhead and more complex halo management outweighed the reduced DRAM traffic.

Why we picked block-H over a 5-row H ring - The early ring versions rebuilt V from five H rows per output row, so all five had to be hot in L1 at once. At 32K that's `5 * 3 * 4096 = 60 KiB`, 94 % of L1 on its own; with V, source/destination streams, and stack on top, the ring would thrash. Block-H pushes the big scratch to L2 (2 MiB has room) and takes an L2 hit per slide read instead. We measured 4.1 % L1d miss and settled there.

---

## 5. What Didn't Work

1. Temporal Blocking (vb11 and 12): Our idea was to do K=4 generations on a local stripe before writing back, so we touch the global grid 4× less often. Each thread loads 64 output rows plus an 8-row halo (~640 KiB slab, fits in L2), advances it 4 generations, then copies back only the valid inner rows. In practice the slab copy and per-generation halo bookkeeping cost more than the DRAM traffic it saved - vb11 ran in 163,317 ms vs vb10's 161,781 ms, and the best temporal variant (vb12, ~156,000 ms) was still outpaced by the simpler single-generation vb14 at 127,231 ms. To reproduce: Compile and run `bitplane_versions/11_bitplane_temporal_stripe.cpp` and `bitplane_versions/12_bitplane_temporal_ring.cpp` against `bitplane_versions/14_bitplane_byte_h4_compact.cpp`.
2. Fused vertical slide (vb15):  Profiling showed the vertical slide + rule phase accounted for ~68% of worker time. The hot path `v = sub_v5_h3(v, h_out); v = add_v5_h3(v, h_in)` duplicates carry/borrow propagation. vb15 replaced both calls with a single fused carry-save network `slide_v5_h3_h3(v, h_out, h_in)`. It ran in 13,224 ms vs vb14's ~12,700 ms for 1000 generations. The fused network adds enough boolean work and register pressure that Neoverse-V2's out-of-order scheduler cannot compensate for the shorter carry chain. To reproduce: Compile and run `bitplane_versions/15_bitplane_fused_slide.cpp` against `bitplane_versions/14_bitplane_byte_h4_compact.cpp`.

---

## 6. What You Would Do With Another Week

With another week, we'll re-test the ring buffer.
The ring buffer was ruled out early because the original kernel needed all five H rows simultaneously to initialise the vertical accumulator — a working set of 5 × 3 × row_bytes ≈ 60 KiB, which saturates the 64 KiB Neoverse-V2 L1d before source and destination rows are counted. That constraint no longer applies. Under the sliding-window update (`V -= H_leaving; V += H_entering`), only two H rows are touched per output row — the one exiting the window and the one entering it. The other three sit idle in scratch between accesses and are free to fall to L2 without penalty. The revised active footprint is approximately 2 H slots (≈ 24 KiB) + V scratch (≈ 20 KiB) + source/destination row chunks (≈ 16 KiB) ≈ 60 KiB, but now the cold slots are excluded from that count. A ring would also reduce the gap between writing an H row and reading it again from 80+ slide iterations under block-H to roughly 5, directly shortening the backend-stall window. 

---

## 7. Benchmark Methodology

Platform: AWS c8g.2xlarge (Graviton4, Neoverse-V2, 8 vCPUs, 16 GiB RAM). Graviton4 vCPUs run at a fixed frequency — no explicit CPU governor tuning was needed.

Build: `g++-14 -std=c++23 -Ofast -mcpu=neoverse-v2+sha3 -pthread -DSPAWN_BLOCK_ROWS=88 spawn_sim.cpp -o spawn_sim`

Runs used `taskset -c 0-7` to pin to the 8 physical cores. Timing uses `chrono::steady_clock` around the generation loop only; file I/O and bitplane conversion are excluded. For day-to-day A/B we used 1000-gen runs at 32K boundary (~12 s/run); promising candidates got the full 10000-gen confirmation. Correctness was byte-identical `cmp` against the reference on all 5 public patterns at sizes 512 and 2048.

Variance: run-to-run spread ≤ 0.5% for the same binary on the same day (vb16: 120,416/120,469 ms; vb17: 117,368/117,626 ms). ASLR was not explicitly disabled; timing was consistent enough that it was not a factor.

Following benchmarking was done on public_1_random_low grids for 10,000 generations:

| N | Median (ms) | CV | Reference (est) | Speedup |
|---|---:|---:|---:|---:|
| 512   | 43.9    | 2.4 %  | ~30,300       | ~690x |
| 2048  | 443.4   | 0.20 % | ~485,000      | ~1100x |
| 8192  | 7,262   | 0.16 % | ~7,760,000    | ~1070x |
| 32768 | 116,830 | 0.05 % | ~124,000,000  | ~1063x |

Key counters from `perf stat` for 32K grid and 1000 generations:

| Metric | Value | What it says |
|---|---|---|
| IPC | 3.14 (T=8) / 3.57 (T=1) | 78 / 89 % of CPU peak; per-thread drop is shared-L2 contention. |
| Backend-stalled cycles | **31 %** | Binding constraint; memory-load latency. |
| L1d / L2 miss rate | 4.1 % / 1.6 % | Block-H too big for L1; L2-resident as designed. |
| Branch miss rate | 0.46 % | Hot loop is branchless. |

The 31 % backend stall is what bounds us, and it's why section 6's ring-buffer plan aims at exactly that number.

---
