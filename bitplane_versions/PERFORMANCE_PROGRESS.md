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

Final target build shape:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -pthread \
  bitplane_versions/03_bitplane_ring_split.cpp -o spawn_sim
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

For final reporting, rerun the bitplane ladder on the target Graviton4 machine
with fixed CPU affinity and repeated medians.
