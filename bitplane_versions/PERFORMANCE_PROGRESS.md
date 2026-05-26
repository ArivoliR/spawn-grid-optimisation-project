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

### 1024x1024, 25 Generations

| Version | Geomean median time | Incremental speedup |
|---|---:|---:|
| vb01 bitplane scalar | 2.671 ms | - |
| vb02 row-sum ring | 1.350 ms | 1.98x |
| vb03 ring + word split | 1.083 ms | 1.25x |
| vb04 tree reduce | 0.675 ms | 1.18x vs same-run vb03 |

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

## Correctness Notes

Current `vb04` checks against the reference:

| Case | Result |
|---|---|
| 64x64, 5 generations | PASS |
| 128x128, 3 generations | PASS |
| 512x512, 2 generations | PASS |
| 512x512, 10 generations | PASS |

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
  bitplane_versions/04_bitplane_tree_reduce.cpp -o spawn_sim
```

For final reporting, rerun the bitplane ladder on the target Graviton4 machine
with fixed CPU affinity and repeated medians.
