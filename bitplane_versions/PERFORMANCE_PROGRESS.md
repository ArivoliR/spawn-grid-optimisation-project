# Bitplane Performance Progress

This documents the bitplane-only ladder in `bitplane_versions/`. The older
byte-grid ladder remains in `versions/PERFORMANCE_PROGRESS.md`.

## Version Ladder

| Version | File | Added change |
|---|---|---|
| vb01 | `bitplane_versions/01_bitplane_scalar.cpp` | Rewrites the simulator around two state bitplanes, processing 64 cells per `uint64_t` word. |
| vb02 | `bitplane_versions/02_bitplane_ring.cpp` | Adds a 5-slot horizontal row-sum ring buffer to reuse each source row's 5-wide adult sum. |
| vb03 | `bitplane_versions/03_bitplane_ring_split.cpp` | Removes modulo wrap from the hot row-sum word path by splitting interior words from edge words. |
| vb04 | `bitplane_versions/04_bitplane_tree_reduce.cpp` | Replaces serial ripple-add chains in row-sum fill and vertical combine with tree reductions. |
| vb05 | `bitplane_versions/05_bitplane_column_tiled.cpp` | Tests column tiling so the row-sum ring is tile-local instead of full-row width. |
| vb06 | `bitplane_versions/06_bitplane_neon_ripple.cpp` | ARM-only NEON port of the `vb03` ripple kernel, processing two words per vector. |
| vb07 | `bitplane_versions/07_bitplane_neon_vertical_slide.cpp` | ARM-only NEON experiment with a running vertical 5-row count and adjusted thresholds. |
| vb08 | `bitplane_versions/08_bitplane_neon_eor3_scratch.cpp` | ARM-only NEON vertical slide with SHA3 EOR3 and persistent per-worker scratch. |
| vb09 | `bitplane_versions/09_bitplane_neon_block_h.cpp` | ARM-only NEON version with an L2-sized block-H buffer before the vertical sliding pass. |
| vb10 | `bitplane_versions/10_bitplane_neon_predicate_huge.cpp` | `vb09` plus compact rule predicates and Linux transparent huge-page hints. |
| vb11 | `bitplane_versions/11_bitplane_temporal_stripe.cpp` | Experimental full-width row-stripe temporal blocking with `K=4`, `H=64`. |
| vb12 | `bitplane_versions/12_bitplane_temporal_ring.cpp` | Faster temporal stripe using a slab-local 5-row H ring, with the `k_this` writeback fix. |
| vb13 | `bitplane_versions/13_bitplane_eor3_aligned.cpp` | Minimal `vb10` improvement: SHA3 `EOR3` plus 64-byte aligned vector storage, without explicit CPU pinning. |
| vb14 | `bitplane_versions/14_bitplane_byte_h4_compact.cpp` | Adds the best target-side byte-lane NEON experiment: H-row adder tree, H unroll by 4, shared `vext`, compact next-state boolean. |
| vb15 | `bitplane_versions/15_bitplane_fused_slide.cpp` | Tests a fused carry-save vertical slide update for `V = V - H_out + H_in`; correct, but slower than `vb14` in the first target benchmark. |
| vb16 | `bitplane_versions/16_bitplane_v_interleaved.cpp` | Cleaner follow-up to `vb14`: stores the five vertical-count scratch planes as adjacent per-register `V5` records. |
| vb17 | `bitplane_versions/17_bitplane_h_interleaved.cpp` | Applies the same interleaving to H scratch (h0/h1/h2 adjacent per register per row), plus three micro-opts in `apply_rule_byte`: drop `nc1`, EOR3 for `next_high`, BCAX for `next_low`. |
| vb22 | `bitplane_versions/22_ring_buffer.cpp` | Drops vb17's block-H scratch (1.5 MiB / thread) in favor of a streaming 6-slot H ring (~72 KiB) with a Karnaugh-minimised `apply_rule_byte` (17 boolean ops, no `vmvnq`). |
| vb23 | `bitplane_versions/23_ring_k4_temporal.cpp` | vb22 + full-slab K-temporal blocking (each thread processes K generations locally with a 2K-row ghost margin before writeback). K configurable via `-DSPAWN_K`. |
| vb24 | `bitplane_versions/24_diamond_tile.cpp` | vb22 + 2D diamond tiling: column strips (`TILE_W_REGS=8`, ~2.6 MiB per thread at 32K) with trapezoidal writeback, designed to fit L3 across 8 threads. |
| vb25 | `bitplane_versions/25_pairwise_sync.cpp` | vb22 with the global `std::barrier` replaced by per-thread atomic gen counters: each thread only waits for its two row-band neighbours (toroidal wrap). |
| vb26 | `bitplane_versions/26_bsl_maj_fold.cpp` | vb22 with carry-chain MAJ-folding throughout `sum_of_5`, `add_v5_h3`, `sub_v5_h3`. Uses the identity `MAJ(a,b,c) = BSL(a^b, c, a)` to collapse each 3-op ripple stage to 1 BSL. Eliminates `vmvnq` in `sub_v5_h3`. |
| vb27 | `bitplane_versions/27_single_pass.cpp` | vb26 with the H computation fused into the slide loop. The new row's H lives only in registers between produce and consume; the ring still holds the 4 surviving H rows. Ring shrinks from 6 to 5 slots (2-row unroll dropped, replaced by 2-column unroll for ILP). |
| vb28 | `bitplane_versions/28_single_pass_pairwise.cpp` | vb27 + vb25's pairwise neighbour sync. vb27's tighter inner loop made the global barrier relatively more expensive; pairwise atomic counters recover the threading loss. |
| vb29 | `bitplane_versions/29_prefetch.cpp` | vb28 + software prefetch hints in the column loop (entering row, V scratch, ring, source). **Negative result** — the HW L2 prefetcher already handles the streaming pattern, and SW prefetches polluted the pipeline. |
| vb30 | `bitplane_versions/30_peeled_boundary.cpp` | vb28 with the `(r + 2 < R_REGS) ? load(r+2) : load(0)` ternary peeled out into a separate boundary block, so the main loop's `adult_next_1` load is unconditional. **Negative result** — the compiler was already handling the conditional well; the peeled version regressed. |
| vb31 | `bitplane_versions/31_column_strip.cpp` | Column-strip-major iteration: per thread, walk column strips of S=2 col regs, keeping V state in registers across all rows in the strip. Aimed to eliminate the V scratch round-trip (~12% of instructions). **Negative result** — column-major access on row-major src/dst data caused the L1 miss rate to jump from 4.83% to 15.87%. Cache penalty dwarfed the saved V loads. |
| vb32 | `bitplane_versions/32_unroll4.cpp` | vb28 with 4-column unroll instead of 2 (40+ live V vectors per iter). **Negative result** — register spill regressed the wall by ~17%. |
| vb33 | `bitplane_versions/33_fused_slide.cpp` | Combined `sub_v5_h3` + `add_v5_h3` into a single `slide_v5_h3` that interleaves the borrow and carry chains at the bit level, reducing combined dep depth from 10 to 6. **Marginal/neutral** — backend stalls dropped 10% (confirming the dep-chain reduction), but the OOO engine was already overlapping the chains via 2-col unroll, so wall time didn't improve. |
| vb34 | `bitplane_versions/34_pinned.cpp` | vb28 + explicit `pthread_setaffinity_np` per worker thread. **Neutral** — profile showed many cmp/yield instructions in the hot path, but pinning didn't reduce them. Thread migration was not the cause. |
| vb35 | `bitplane_versions/35_no_yield.cpp` | vb28 with the `yield` instructions removed from the spin-wait, pure busy-spin. **Neutral** — the `yield` cost was already negligible. |
| vb36 | `bitplane_versions/36_combined.cpp` | Stack vb33 + vb34 (fused slide + CPU pinning). Hypothesis: neutral parts might compound. **Regression** — the larger inlined slide body increased register pressure when combined with 2-col unroll, eating any threading gain. |
| vb37 | `bitplane_versions/37_byte_v.cpp` | Conversion-cost measurement: vb28 with `sum3_to_byte8` conversion added as dead code in the inner loop (result held alive via inline asm). Quantifies what a full byte-form V would cost. **Catastrophic regression** (167 s vs 94.5 s) — adding only the conversion adds 72 s of wall time, proving the bit→byte expansion on NEON dwarfs any 22→2-op slide savings. |

All versions keep the same CLI and binary I/O format:

```bash
spawn_sim <input.bin> <output.bin> [generations]
```

## Benchmark Environment

These numbers were measured locally, not on the target AWS Graviton4 instance.

- Host architecture: `x86_64`
- Compiler: local `g++`
- Flags: `-std=c++23 -O3 -march=native -pthread`
- Inputs: generated public-style patterns
- Correctness: compared byte-for-byte against the reference implementation on
  generated small cases

Treat these as development measurements only. Final reporting should be rerun on
the c8g.2xlarge target with `g++-14 -mcpu=neoverse-v2`.

## Summary Results

The tables below use the geometric mean of median timings across five
public-style patterns:

- `random_low`
- `random_high`
- `structured`
- `sparse_clusters`
- `boundary_stress`

### 512x512, 100 Generations

| Version | Geomean median time | Incremental speedup |
|---|---:|---:|
| vb01 bitplane scalar | 3.871 ms | - |
| vb02 row-sum ring | 1.763 ms | 2.20x |
| vb03 ring + word split | 1.632 ms | 1.08x |
| vb04 tree reduce | 1.038 ms | 0.91x vs same-run vb03; noisy local result |
| vb05 column tiled | not used | slower on 32768 local short run |

### 1024x1024, 25 Generations

| Version | Geomean median time | Incremental speedup |
|---|---:|---:|
| vb01 bitplane scalar | 2.671 ms | - |
| vb02 row-sum ring | 1.350 ms | 1.98x |
| vb03 ring + word split | 1.083 ms | 1.25x |
| vb04 tree reduce | 0.675 ms | 1.18x vs same-run vb03 |
| vb05 column tiled | not used | intended for 32768 sweep |

### 32768x32768, 10 Generations, Local x86 Boundary Input

| Version / tile words | Time |
|---|---:|
| vb03 full-width ring | 390.182 ms |
| vb05 tile 128 words | 670.544 ms |
| vb05 tile 256 words | 564.619 ms |
| vb05 tile 512 words | 378.135 ms |

Tile 512 is effectively full-width for a 32768-wide grid (`32768 / 64 = 512`
words), so it is comparable to `vb03` and within local noise. The smaller tiles
were slower locally.

### 32768x32768, 10000 Generations, Target Graviton4 Boundary Input

These runs were measured on the AWS Graviton4 target with:

- Compiler: `g++-14`
- Flags: `-std=c++23 -O3 -mcpu=neoverse-v2` or `-mcpu=neoverse-v2+sha3` where
  SHA3 `EOR3` intrinsics are used
- Affinity: `taskset -c 0-7`
- Input: `/tmp/input_32768_boundary.bin`
- Correctness: compared byte-for-byte against the best known correct output
  for that run family

| Version / experiment | Time | Notes |
|---|---:|---|
| vb03 scalar bitplane ring split | 282189.153 ms | Pre-explicit-NEON baseline on target |
| vb06 NEON ripple | 204685.173 ms | First explicit NEON port |
| vb07 NEON vertical slide | 182156.612 ms | Running vertical count |
| vb09 NEON block-H buffer | 170286.970 ms | L2-sized block-H scratch |
| vb10 compact predicate + huge-page hints | 162803.725 ms | First vb10 run |
| vb10 compact predicate + huge-page hints | 161781.177 ms | Later same-input rerun |
| vb11 temporal stripe, block-H slab | 163317.298 ms | Correct but slower than vb10 |
| vb12 temporal stripe, H-ring slab | ~156000 ms | Correct contender run, before cleanup |
| vb13 EOR3 + aligned storage, no pinning | 149701.896 ms | Current minimal cleaned candidate |
| vb14 byte-lane H-tree + H4 + compact state | 127231.528 ms | Best pre-vb16 run; `cmp` clean against vb13 output |
| vb14 byte-lane H-tree + H4 + compact state | 130412.640 ms | Fresh rerun; observed range is ~127-130 s |
| vb15 fused vertical slide | 13223.584 ms for 1000 gens | Correct, but slower than vb14's best 1000-gen runs |
| vb16 V-interleaved scratch, block 128, `-O3` | 12244.251 ms for 1000 gens | Clean architecture; one scratch-layout change over vb14 |
| vb16 V-interleaved scratch, block 96, `-Ofast` | 120843.475 ms | Full 10000-gen run; correct output; clean candidate |
| vb16 V-interleaved scratch, block 96, `-Ofast` | 120416.799 ms | Rerun same day as vb17 baseline; consistent |
| vb17 H+V interleaved scratch + apply_rule micro-opts, block 96, `-Ofast` | 117625.887 ms | First run; `cmp` clean vs vb16 output |
| vb17 H+V interleaved scratch + apply_rule micro-opts, block 96, `-Ofast` | 117367.164 ms | Second run; consistent; ~2.5% faster than vb16 |
| vb17 H+V interleaved scratch + apply_rule micro-opts, block 128, `-Ofast` | 119663.212 ms | Default block size; b96 remains best |
| EOR3-only experiment | 158443.486 ms | Archived experiment; isolated EOR3 effect |
| EOR3 + pinning experiment | 160163.706 ms | Archived experiment; pinning did not help under `taskset` |
| EOR3 + pinning + aligned experiment | 149408.142 ms | Archived experiment; aligned storage supplied most of the win |

The main target-side lesson through `vb13` was that aligned storage was a larger
win than EOR3 alone in the `uint64x2_t` kernel, while explicit thread pinning
was neutral or slightly negative when the process was already launched with
`taskset`. `vb14` is the larger byte-lane kernel rewrite that was deferred from
the minimal `vb13` cleanup, and it is now the best measured candidate.

## x86 AVX2 SIMD Experiment

These numbers came from a separate local AVX2 experiment, not from files in this
folder. They are recorded here because they affect the next implementation
choice.

- Hardware: AMD Ryzen 7 PRO 8840HS, Zen 4, 8 cores
- Affinity: `taskset 0-7`
- Compiler: `g++ 16.1.1`
- Flags: `-O3 -mavx2 -mbmi2 -mtune=native -lpthread`
- Correctness: both SIMD variants produced bit-identical output to v13 on all
  tested grids

| Grid / generations | v13 scalar | v13.5 ripple SIMD | v14.5 tree SIMD |
|---|---:|---:|---:|
| 512x512 / 500 | ~5.0 ms | ~2.5 ms | ~3.2 ms |
| 2048x2048 / 1000 | skipped | 56.5 ms | 56.0 ms |
| 8192x8192 / 30 | 65.6 ms | 61.3 ms | 63.6 ms |

Interpretation: ripple SIMD and tree SIMD were within roughly 5% on larger
cases. The expected tree-reduction advantage did not clearly materialize on
this wide out-of-order x86 core. The major win was SIMD widening itself, not the
adder structure.

Implication for ARM NEON/SVE2: do not choose tree over ripple on performance
theory alone. Start with the cleaner NEON ripple port, benchmark it on
Neoverse-V2, then revisit tree reduction only if the target numbers show a real
benefit.

## Interpretation

`vb01` is already faster than the best byte-grid version because each
`uint64_t` word carries 64 cells through the count and transition logic.

`vb02` is the largest bitplane improvement so far. It avoids rebuilding the same
horizontal 5-cell row sums for neighboring output rows, which cuts repeated
work in the bitplane kernel.

`vb03` is a smaller but useful cleanup: it removes modulo wrap from the row-sum
hot path. Only the first and last words need toroidal word-boundary handling.
The improvement is workload-dependent, but the change is low risk and makes the
kernel shape more friendly for future NEON/SVE2 work.

`vb04` changes the count-expression shape. The horizontal row sum now uses a
5-input popcount tree, and the vertical combine popcounts each bit column of the
five row sums before doing short shifted adds. This targets the carry-chain
dependency visible in both hot phases of `vb03`. Local x86 timings are noisy,
especially on 512x512, where the quick same-run set was slower. The 1024x1024
median set showed about 1.18x over `vb03`. This needs rerunning on the target
ARM machine before treating it as a confirmed win.

`vb05` tests column tiling to reduce the row-sum ring working set at full grid
width. On local x86, the extra tile passes over source rows outweighed the
smaller ring for 128- and 256-word tiles. This should only replace `vb03` if the
target Neoverse-V2 sweep shows a clear win.

`vb06` is the first explicit ARM NEON version. It is based on `vb03`, not
`vb04`, because `vb03` is the best confirmed full-size scalar version. It needs
AWS correctness and performance validation before it can replace `vb03`.

`vb07` removes the repeated 5-row vertical recombine from `vb06`. It keeps a
running vertical count, updates it by subtracting the outgoing H row and adding
the incoming H row, and uses thresholds that include the center cell. This is
expected to be the next high-upside experiment after `vb06`, but it must pass
AWS correctness before being trusted.

`vb08` keeps the `vb07` algorithm and adds EOR3 plus persistent scratch. Build
with `-mcpu=neoverse-v2+sha3` on AWS. It should be compared directly against
the already-correct `vb07` output before benchmarking full-size workloads.

`vb09` keeps the `vb07` vertically sliding count but changes the row-sum cache
shape. Instead of a 5-row ring that is constantly updated, it materializes all
horizontal sums for a 128-row output block plus its 2-row halos into per-worker
scratch, then sweeps vertically over that block. On the target Graviton4 box,
the 32768x32768 boundary input for 10000 generations improved from
`vb07 = 182156.612 ms` to `vb09 = 170286.970 ms`.

`vb10` is a smaller follow-up on top of `vb09`. It replaces the explicit
`eq3..eq10` rule masks with closed-form boolean ranges for `V in {3,4,5}` and
`V in {5..10}`. The compact predicate was truth-table checked for all 32
possible 5-bit count values before benchmarking. It also calls
`madvise(..., MADV_HUGEPAGE)` for the large bitplane and scratch vectors on
Linux. This does not change the algorithm or the output; it is only a
cache/TLB/codegen experiment. On the target Graviton4 box, the 32768x32768
boundary input for 10000 generations improved from `vb09 = 170286.970 ms` to
`vb10 = 162803.725 ms`.

`vb11` is the first temporal-blocking experiment. It loads a full-width stripe
of 64 output rows plus a `2*K` vertical halo on each side, advances that slab for
up to 4 generations using the `vb10` block-H kernel, then writes back only the
valid inner stripe rows. This is still O(generations * cells): every cell in the
valid region is evaluated once per generation. The goal is to reduce global
DRAM traffic by reusing nearby generations while the stripe is in per-worker
scratch. This version is higher risk than `vb10`; validate on small generation
counts before running the 32768x32768 workload.

`vb12` keeps the same temporal stripe idea but uses a slab-local 5-row H ring
inside each generation instead of the larger block-H buffer from `vb11`. The
ring version benchmarked around `156 s` on the 32768x32768 boundary workload,
slightly ahead of `vb10 = 161.781 s`. The checked-in `vb12` also fixes the
writeback offset for trailing generation blocks where `k_this < TIME_K`: final
rows are copied from `2 * k_this`, not always from `2 * TIME_K`. This matters
for arbitrary generation counts even though the 10000-generation benchmark is
divisible by 4.

`vb13` is the cleaned-up form of the direct `vb10` feature experiments.
Measurements on the 32768x32768 boundary workload showed EOR3 helped modestly,
explicit CPU pinning did not help under `taskset`, and aligned vector storage was
the main useful addition. `vb13` therefore keeps EOR3 and aligned storage, and
removes explicit pinning. The large remaining feature from the 130s contender is
the byte-lane NEON H-row adder tree, which is a representation/kernel rewrite
rather than a small patch to the `uint64x2_t` `vb10` kernel.

`vb14` adds the proven byte-lane rewrite. It keeps the same block-H plus
vertical-slide simulation structure, but processes rows as `uint8x16_t` NEON
registers instead of `uint64_t` / `uint64x2_t` words. The horizontal 5-cell
adult count uses a tree-shaped `FA + FA + HA` adder, the H-row kernel is
unrolled by four registers, the shared `vextq_u8` carry vectors are reused for
`x-1/x-2` and `x+1/x+2`, and the state transition directly derives next
low/high bitplanes. On the target 32768x32768 boundary workload, this moved the
best observed full run from `vb13 = 149701.896 ms` to `vb14 = 127231.528 ms`,
with a fresh rerun at `130412.640 ms`.

The `vb14` profile showed the remaining work is dominated by the vertical
slide/rule phase, not H-row fill:

```text
32768x32768, 100 generations, summed worker time
H fill:       3150979623 ns  (~31%)
V init:         56193445 ns  (<1%)
first row:      27673054 ns  (<1%)
slide + rule: 6921339152 ns  (~68%)
```

`vb15` targets the slide phase directly. The current `vb14` hot path does:

```cpp
v = sub_v5_h3(v, h_out);
v = add_v5_h3(v, h_in);
```

The tested experiment is a fused bit-sliced update:

```cpp
v = slide_v5_h3_h3(v, h_out, h_in); // computes V - H_out + H_in
```

The goal was to remove duplicated carry/borrow propagation from the 68% hotspot
without changing the algorithm, memory layout, or row-major access pattern.
Correctness passed, but the first target benchmark regressed to
`13223.584 ms` for 32768x32768 / 1000 generations. The fused carry-save network
appears to add enough boolean work/register pressure to outweigh the shorter
carry chain.

`vb17` applies the same interleaving principle to H scratch that `vb16` applied
to V scratch. In `vb16` the three H bitplanes for a register `r` in H row `i`
are 4096 bytes apart (one per full-width plane), causing cache-set conflicts on
the Neoverse-V2 L1d. `vb17` stores `h0[r], h1[r], h2[r]` as 48 consecutive
bytes at `hp(i) + r*48`, collapsing the slide loop's 6 strided H loads into 2
sequential 48-byte reads. The `apply_rule_byte` function also drops the
`nc1 = vmvnq_u8(v.b1)` intermediate (replaced by `vbicq_u8(v.b2, v.b1)`),
uses `vxor3_u8` (EOR3) for `next_high` (disjointness proof: `adult_r` requires
`low=high=1`, so `high^low=0` there, making XOR and OR equivalent), and uses
`vbcaxq_u8` (BCAX) for `next_low` on SHA3 CPUs (same disjointness argument vs
`(high|born)&~low`). On the 32768x32768 boundary workload, this measured
117367–117626 ms with `block 96, -Ofast`, about 2.5% faster than vb16's
120417–120843 ms range. Output is byte-identical to vb16.

`vb16` keeps the `vb14` kernel shape and changes only the vertical-count scratch
layout. `vb14` stores the running `V` count as five full-row bitplanes:

```text
v0 row, v1 row, v2 row, v3 row, v4 row
```

The slide loop needs all five vectors for the same register at the same time,
so `vb16` stores them as adjacent per-register records:

```text
v0[r], v1[r], v2[r], v3[r], v4[r]
```

This is a cache/locality improvement, not a smarter algorithm. It keeps
row-major traversal, block-H scratch, byte-lane NEON H-tree, compact transition,
and the same generation-by-generation simulation. On the target boundary
workload, `vb16` measured `120843.475 ms` with
`-Ofast -DSPAWN_BLOCK_ROWS=96`, which is about `6.4 s` faster than the best
`vb14` run while staying simple to explain.

### 32768x32768, 10000 Generations, c8g.2xlarge (Graviton4, 8 vCPUs)

Measured on the target instance with `taskset -c 0-7`, 15 GiB RAM,
public_1_random_low_32768 input. Build flags noted per row.

| Version | Time | Flags | Notes |
|---|---:|---|---|
| vb22 ring buffer baseline | 113507 ms | `-Ofast -mcpu=neoverse-v2+sha3 -pthread` | First 8-thread target measurement of the streaming kernel. |
| vb22 + funroll-all-loops + flto | 111142 ms | `-O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto` | Free compiler tuning, kept for all subsequent versions. |
| vb23 K=4 full-slab temporal | 125840 ms | as above, `-DSPAWN_K=4` | Regression: temporal blocking trades DRAM traffic for ghost-row work; here the compute side dominates and the trade is net negative. |
| vb23 K=8 full-slab temporal | 119373 ms | as above, `-DSPAWN_K=8` | Same regression, less severe. |
| vb24 K=4 2D diamond tile | 134704 ms | as above, `-DSPAWN_K=4` | Strip overhead (trapezoidal writeback + ghost-column copies) added more cost than DRAM savings. |
| vb24 K=8 2D diamond tile | 122425 ms | as above, `-DSPAWN_K=8` | Same. |
| vb25 pairwise neighbour sync | 113177 ms | tuned flags | Marginal: ~0.18 of 8 CPUs reclaimed (6.62 → 6.79). Barriers were not the dominant cost. |
| vb26 BSL MAJ-fold | 108974 ms | tuned flags | -8.2% instructions vs vb22 perf-counted; -4% wall. `cmp` clean vs vb22 32K output. |
| vb27 single-pass fused kernel | 95214 ms | tuned flags | -16% wall vs vb22. IPC 3.20 vs vb22's 2.93. Backend memory stalls -68% (the H scratch round-trip is gone). `cmp` clean vs vb22 32K output. 3 stable runs: 95214 / 95261 / 95223 ms. |
| **vb28 single-pass + pairwise sync** | **94538 ms** | tuned flags | -16.7% wall vs vb22. Pairwise atomic gen counters recover threading loss from vb27 (CPU util 5.98→6.61 at 500 gens). `cmp` clean vs vb22 32K output. 3 stable runs: 94538 / 94570 / 94518 ms. |
| vb29 software prefetch | 97809 ms | tuned flags | **Regression** vs vb28. `__builtin_prefetch` 4 column-registers ahead on entering row + V + ring + src; HW L2 prefetcher already handled the streaming pattern, SW prefetch polluted the pipeline. Kept as a documented negative result. |
| vb30 peeled boundary | 97120 ms | tuned flags | **Regression** vs vb28. Peeled the `(r+2 < R_REGS) ? load(r+2) : load(0)` ternary out into a separate boundary block. Compiler was already handling the conditional well — the peeled version added icache pressure without helping. |
| vb31 column-strip | 160164 ms | tuned flags | **Large regression** vs vb28. Column-major iteration of row-major data: L1 miss rate jumped 4.83% → 15.87%, IPC dropped 3.21 → 2.28. The V scratch traffic we saved was much smaller than the cache penalty. |
| vb32 4-col unroll | 110757 ms | tuned flags | **Regression** vs vb28. Register pressure (40+ live vectors per iter) caused spills. |
| vb33 fused slide_v5 | 95912 ms | tuned flags | Neutral/marginal vs vb28. `sub_v5_h3` + `add_v5_h3` combined into single bit-interleaved ripple (dep depth 10 → 6). Backend stalls dropped 27.7% → 24.5% (confirming the dep chain reduction), but the 2-col unroll was already exposing enough ILP for the OOO engine; wall time held at ~95.9 s. |
| vb34 CPU pinning | 94867 ms | tuned flags | Neutral. Pinning each worker to a specific CPU via `pthread_setaffinity_np` did not change wall time, indicating thread migration was not amplifying the spin-wait. |
| vb35 pure spin (no yield) | 94792 ms | tuned flags | Neutral. Removing the `yield` from the spin-wait did not change wall time — the perf-annotated yield% was either mis-attributed or already cheap. |
| vb36 fused slide + pinning | 96570 ms | tuned flags | **Regression** vs vb28. Stacking the neutral optimisations compounded their downsides (register pressure from larger inlined slide, lower OOO freedom). |
| vb37 byte-V conversion cost | 166999 ms | tuned flags | **Decisive regression** (+72 s). Empirical proof that byte-form V is unviable on NEON: the dead-code conversion (bitplane Sum3 → 8 byte-vectors per col reg, no slide replacement) alone added 72 s — 3.6× larger than the maximum possible saving from a full byte-V slide (which would save ~20 s at best). Conversions via `vqtbl1q + vand + vceqq + vmvnq + vshrq` are intrinsically expensive for this access pattern. |

Per-counter snapshot (vb27, 500 gens at 32K x 8 threads):

```text
task-clock       40788 ms   (5.98 CPUs utilized)
cycles           108.4 G    @ 2.66 GHz
instructions     346.4 G    (IPC 3.20)
stall_backend     30.0 G    (27.7% of cycles)
stall_backend_mem  6.4 G    ( 5.9% of cycles)
```

For comparison, vb22 at the same 500-gen workload: IPC 2.93, stall_backend
33.7%, stall_backend_mem 15.5%. vb27 reclaims ~9.6 G cycles of backend memory
stalls — the entire H scratch round-trip.

## Correctness Notes

Current `vb04`/`vb05` checks against the reference:

| Case | Result |
|---|---|
| 64x64, 5 generations | PASS |
| 128x128, 3 generations | PASS |
| 512x512, 2 generations | PASS |
| 512x512, 10 generations | PASS |
| 1024x1024, 2 generations | PASS for vb05 |

Additional `vb04` reducer checks:

- all 32 horizontal 5-bit input combinations passed
- all `6^5` valid vertical row-sum combinations passed

## Reproduction Commands

Compile the bitplane versions locally:

```bash
for f in bitplane_versions/*.cpp; do
  b=/tmp/$(basename "$f" .cpp)
  g++ -std=c++23 -O3 -march=native -pthread "$f" -o "$b"
done
```

Current target build shape:

```bash
g++-14 -std=c++23 -Ofast -mcpu=neoverse-v2+sha3 -pthread \
  -DSPAWN_BLOCK_ROWS=96 \
  bitplane_versions/17_bitplane_h_interleaved.cpp -o spawn_sim
```

Build the column-tiled experiment with a chosen tile width:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -pthread \
  -DDEFAULT_TILE_WORDS=128 \
  bitplane_versions/05_bitplane_column_tiled.cpp -o /tmp/vb05_tw128
```

Build the NEON experiment on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -pthread \
  bitplane_versions/06_bitplane_neon_ripple.cpp -o /tmp/vb06
```

Build the vertical-sliding NEON experiment on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -pthread \
  bitplane_versions/07_bitplane_neon_vertical_slide.cpp -o /tmp/vb07
```

Build the EOR3/scratch experiment on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
  bitplane_versions/08_bitplane_neon_eor3_scratch.cpp -o /tmp/vb08
```

Build the block-H experiment on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -pthread \
  bitplane_versions/09_bitplane_neon_block_h.cpp -o /tmp/vb09
```

Build the compact-predicate/huge-page experiment on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -pthread \
  bitplane_versions/10_bitplane_neon_predicate_huge.cpp -o /tmp/vb10
```

Build the temporal-stripe experiment on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -pthread \
  bitplane_versions/11_bitplane_temporal_stripe.cpp -o /tmp/vb11
```

Build the temporal-ring experiment on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
  bitplane_versions/12_bitplane_temporal_ring.cpp -o /tmp/vb12
```

Build the cleaned direct `vb10` improvement on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
  bitplane_versions/13_bitplane_eor3_aligned.cpp -o /tmp/vb13
```

Build the byte-lane H-tree candidate on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
  bitplane_versions/14_bitplane_byte_h4_compact.cpp -o /tmp/vb14
taskset -c 0-7 /tmp/vb14 /tmp/input_32768_boundary.bin /tmp/vb14_32768.bin 10000
cmp /tmp/output_vb13_32768.bin /tmp/vb14_32768.bin
```

Build the V-interleaved scratch candidate on AWS:

```bash
g++-14 -std=c++23 -Ofast -mcpu=neoverse-v2+sha3 -pthread \
  -DSPAWN_BLOCK_ROWS=96 \
  bitplane_versions/16_bitplane_v_interleaved.cpp -o /tmp/vb16
taskset -c 0-7 /tmp/vb16 /tmp/input_32768_boundary.bin /tmp/vb16_32768.bin 10000
cmp /tmp/output_vb14_32768.bin /tmp/vb16_32768.bin
```

Build the H+V interleaved scratch candidate on AWS:

```bash
g++-14 -std=c++23 -Ofast -mcpu=neoverse-v2+sha3 -pthread \
  -DSPAWN_BLOCK_ROWS=96 \
  bitplane_versions/17_bitplane_h_interleaved.cpp -o /tmp/vb17
taskset -c 0-7 /tmp/vb17 /tmp/input_32768_boundary.bin /tmp/vb17_32768.bin 10000
cmp /tmp/vb16_32768.bin /tmp/vb17_32768.bin
```

Compare against `vb03` on a public grid:

```bash
taskset -c 0-7 /tmp/vb03 test_grids/public_1_random_low_8192.bin /tmp/vb03_8192.bin 10000
taskset -c 0-7 /tmp/vb06 test_grids/public_1_random_low_8192.bin /tmp/vb06_8192.bin 10000
taskset -c 0-7 /tmp/vb07 test_grids/public_1_random_low_8192.bin /tmp/vb07_8192.bin 10000
taskset -c 0-7 /tmp/vb08 test_grids/public_1_random_low_8192.bin /tmp/vb08_8192.bin 10000
cmp /tmp/vb03_8192.bin /tmp/vb06_8192.bin
cmp /tmp/vb03_8192.bin /tmp/vb07_8192.bin
cmp /tmp/vb03_8192.bin /tmp/vb08_8192.bin
```

Build the vb22/23/24 streaming family on AWS:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/22_ring_buffer.cpp -o /tmp/vb22
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  -DSPAWN_K=8 bitplane_versions/23_ring_k4_temporal.cpp -o /tmp/vb23_K8
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  -DSPAWN_K=8 bitplane_versions/24_diamond_tile.cpp -o /tmp/vb24_K8
```

Build the vb25 / vb26 / vb27 ladder on AWS (current best candidates):

```bash
# vb25: pairwise neighbour sync (atomic gen counters)
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/25_pairwise_sync.cpp -o /tmp/vb25

# vb26: BSL MAJ-fold across carry chains (best of the two-pass family)
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/26_bsl_maj_fold.cpp -o /tmp/vb26

# vb27: single-pass fused kernel
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/27_single_pass.cpp -o /tmp/vb27

# vb28: single-pass + pairwise sync (current best at 94.5 s on the target box)
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/28_single_pass_pairwise.cpp -o /tmp/vb28

# vb29 / vb30 / vb31 / vb32 / vb33: documented experiments (mostly regressions or neutral)
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/29_prefetch.cpp -o /tmp/vb29
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/30_peeled_boundary.cpp -o /tmp/vb30
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/31_column_strip.cpp -o /tmp/vb31
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/32_unroll4.cpp -o /tmp/vb32
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/33_fused_slide.cpp -o /tmp/vb33
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/34_pinned.cpp -o /tmp/vb34
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/35_no_yield.cpp -o /tmp/vb35
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/36_combined.cpp -o /tmp/vb36
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread -funroll-all-loops -flto \
  bitplane_versions/37_byte_v.cpp -o /tmp/vb37
```

Cross-check vb27 against vb22 on the 32K boundary input:

```bash
taskset -c 0-7 /tmp/vb22 test_grids/public_1_random_low_32768.bin /tmp/o22.bin
taskset -c 0-7 /tmp/vb27 test_grids/public_1_random_low_32768.bin /tmp/o27.bin
cmp /tmp/o22.bin /tmp/o27.bin && echo MATCH
```

For final reporting, rerun the bitplane ladder on the target Graviton4 machine
with fixed CPU affinity and repeated medians.
