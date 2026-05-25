# Spawn Simulator Performance Progress

This documents the cumulative optimization ladder in `versions/` and the local
performance measured for each step.

## Version Ladder

| Version | File | Added change |
|---|---|---|
| 01 | `versions/01_reference.cpp` | Original reference implementation. |
| 02 | `versions/02_unrolled_scalar.cpp` | Inlines/unrolls the 24-neighbor adult count instead of calling `count_adults()` with nested `dx/dy` loops. |
| 03 | `versions/03_bitmask_wrap.cpp` | Replaces toroidal `% N` wrapping with `& (N - 1)`, relying on power-of-two grid sizes. |
| 04 | `versions/04_row_pointers.cpp` | Uses raw pointers and precomputed row pointers `r0..r4`/`rw` in the hot loop. |
| 05 | `versions/05_thread_per_generation.cpp` | Splits rows across up to 8 threads, creating/joining threads each generation. |
| 06 | `versions/06_persistent_pool.cpp` | Uses the current persistent thread pool with barriers between generations. |
| 07 | `versions/07_execution_par_chunks.cpp` | Experiments with `std::for_each(std::execution::par, ...)` over fixed row chunks. |
| 08 | `versions/08_sliding_counts.cpp` | Keeps the persistent pool and replaces 24 explicit neighbor checks with rolling 5-wide adult sums for the five source rows. |
| 09 | `versions/09_hsum_lut.cpp` | Precomputes per-generation horizontal 5-cell adult sums for all rows, then applies a constexpr transition lookup table. |
| 10 | `versions/10_interior_edge_split.cpp` | Builds horizontal sums with direct interior indexing and handles only edge columns with toroidal wrap. |
| 11 | `versions/11_branchless_transition.cpp` | Replaces the scalar transition lookup table with branchless compare/select logic. |

All versions keep the same CLI and binary I/O format:

```bash
spawn_sim <input.bin> <output.bin> [generations]
```

## Benchmark Environment

These numbers were measured locally, not on the target AWS Graviton4 instance.

- Host architecture: `x86_64`
- Compiler: `g++ 16.1.1`
- Flags: `-std=c++23 -O3 -Wall -Wextra -lpthread`
- Inputs: generated with `test_grids/generate.py`
- Correctness: every variant output was compared byte-for-byte against
  `01_reference.cpp`

Because this is not the target ARM machine and CPU affinity/governor were not
controlled through the official harness, treat these as relative development
measurements only.

## Summary Results

The tables below use the geometric mean of median timings across five generated
public-style patterns:

- `random_low`
- `random_high`
- `structured`
- `sparse_clusters`
- `boundary_stress`

### 512x512, 100 Generations, 5 Runs Per Pattern

| Version | Geomean median time | Incremental speedup | Total speedup vs reference |
|---|---:|---:|---:|
| 01 reference | 263.853 ms | - | 1.00x |
| 02 unrolled scalar | 219.645 ms | 1.20x | 1.20x |
| 03 bitmask wrap | 181.785 ms | 1.21x | 1.45x |
| 04 row pointers | 172.437 ms | 1.05x | 1.53x |
| 05 thread per generation | 41.199 ms | 4.19x | 6.40x |
| 06 persistent pool | 41.408 ms | 0.99x | 6.37x |
| 07 execution par chunks | 59.336 ms | 0.70x vs 06 | 4.45x |
| 08 sliding counts | 21.347 ms | 1.94x vs 06 | 12.36x |
| 09 hsum + LUT | 17.459 ms | 1.77x vs 08 | 15.11x |
| 10 interior/edge split | 5.523 ms | 2.00x vs 09 | 47.77x |
| 11 branchless transition | 4.936 ms | 1.79x vs 10 | 53.45x |

### 1024x1024, 25 Generations, 3 Runs Per Pattern

| Version | Geomean median time | Incremental speedup | Total speedup vs reference |
|---|---:|---:|---:|
| 01 reference | 264.046 ms | - | 1.00x |
| 02 unrolled scalar | 217.907 ms | 1.21x | 1.21x |
| 03 bitmask wrap | 180.841 ms | 1.20x | 1.46x |
| 04 row pointers | 169.367 ms | 1.07x | 1.56x |
| 05 thread per generation | 37.880 ms | 4.47x | 6.97x |
| 06 persistent pool | 38.485 ms | 0.98x | 6.86x |
| 08 sliding counts | 16.844 ms | 2.28x vs 06 | 15.68x |
| 09 hsum + LUT | 14.661 ms | 1.86x vs 08 | 18.01x |
| 10 interior/edge split | 7.680 ms | 1.64x vs 09 | 34.38x |
| 11 branchless transition | 4.081 ms | 1.91x vs 10 | 64.70x |

## Interpretation

The biggest measured jump is row-level multithreading in version 05. The
single-threaded scalar cleanups still matter: together, unrolling, bitmask wrap,
and row pointers improve the reference by about 1.5x before adding threads.

The persistent pool in version 06 did not show a clear local win over
thread-per-generation in these short local tests. It remains a useful structure
for the production implementation because it avoids repeated thread creation and
gives a stable place to add future per-thread state, SIMD buffers, or tiled work
queues. On this benchmark run, the measured difference was within noise and
slightly favored version 05.

Version 07 tested `std::execution::par` over fixed row chunks. On this local
GCC/libstdc++ setup it did not link with the normal project-style flags; it
required adding `-ltbb`, because the parallel execution backend resolves to TBB
symbols. That makes it unsuitable for the final submission under the no-external
libraries constraint. Even when linked with TBB for experimentation, it was
slower than the explicit thread pool on the 512x512 benchmark.

Version 08 is the first change that reduces the scalar neighbor-count work per
cell. Instead of loading/checking 24 neighbors directly for every cell, it keeps
five rolling horizontal 5-cell adult counts and combines them:

```text
A = hsum(y-2) + hsum(y-1) + hsum(y) + hsum(y+1) + hsum(y+2) - center_adult
```

That preserves the same O(generations * cells) structure, but reuses overlapping
horizontal neighborhood work between adjacent cells. Locally, this produced a
clear improvement over version 06: about 1.5x on the 512x512 benchmark and
1.8x on the 1024x1024 spot check.

Version 09 moves the horizontal sums into a full per-generation scratch buffer.
This avoids recomputing the same row-local 5-cell sums for neighboring output
rows. It also replaces the transition `switch` with a fixed constexpr table of
the official rules. The scratch buffer is recomputed from the current grid every
generation and is not memoized across generations, so this remains compliant
with the O(generations * cells) constraint. Locally, version 09 improved over
version 08 by about 1.77x on 512x512 and 1.86x on 1024x1024.

Version 10 splits horizontal-sum construction into edge and interior columns.
Only the first and last two columns need toroidal wrap logic. The hot interior
range uses direct `x-2..x+2` indexing. This is still the same per-generation
stencil computation, but avoids wrap/mask arithmetic in almost every horizontal
sum. Locally, version 10 improved over version 09 by about 2.00x on 512x512 and
1.64x on 1024x1024.

Version 11 replaces the scalar transition lookup table with branchless
compare/select logic. Profiling v10 showed that horizontal-sum construction was
only about 10% of runtime, while vertical combine plus transition was about 90%.
The lookup table was branchless but scalar and awkward for vectorization. The
branchless rule logic exposes comparisons and byte arithmetic directly to the
compiler, making the step phase more SIMD-friendly. Locally, version 11 improved
over version 10 by about 1.79x on 512x512 and 1.91x on 1024x1024.

## Per-Pattern 512x512 Timings

Median simulation time in milliseconds, 100 generations, 5 runs per pattern.

| Pattern | 01 ref | 02 unrolled | 03 bitmask | 04 rows | 05 threads/gen | 06 pool | 07 execution | 08 sliding | 09 hsum+LUT | 10 split | 11 branchless |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_low | 279.104 | 233.809 | 199.600 | 192.983 | 39.787 | 33.630 | 66.678 | 31.896 | 15.911 | 9.206 | 5.025 |
| random_high | 282.961 | 234.728 | 203.421 | 194.945 | 41.853 | 48.629 | 61.486 | 36.523 | 17.157 | 9.372 | 5.286 |
| structured | 249.222 | 200.916 | 159.417 | 153.334 | 37.230 | 36.904 | 53.673 | 27.292 | 19.043 | 8.468 | 6.273 |
| sparse_clusters | 251.835 | 214.075 | 172.463 | 159.468 | 33.965 | 40.958 | 58.075 | 28.349 | 18.318 | 7.860 | 3.756 |
| boundary_stress | 257.995 | 216.570 | 177.827 | 165.731 | 56.371 | 49.249 | 57.556 | 31.563 | 17.036 | 9.475 | 4.681 |

## Reproduction Commands

Compile all versions locally:

```bash
for f in versions/*.cpp; do
  b=/tmp/$(basename "$f" .cpp)
  g++ -std=c++23 -O3 -Wall -Wextra "$f" -o "$b" -lpthread
done
```

Generate 512x512 benchmark inputs:

```bash
python3 test_grids/generate.py \
  --output-dir /tmp/spawn_versions_bench \
  --sizes 512 \
  --generations 1
```

For final reporting, rerun the same ladder on the target Graviton4 instance with
the project build flags, preferably through `harness/run.sh` or an equivalent
script using fixed CPU affinity and repeated medians.
