# Bitplane Design Choices

This file tracks the bitplane-specific implementation decisions. The byte-grid
optimization history stays in `versions/DESIGN_CHOICES.md`.

## Naming

Bitplane versions are named `vbNN` in the notes:

- `vb01` -> `01_bitplane_scalar.cpp`
- `vb02` -> `02_bitplane_ring.cpp`
- `vb03` -> `03_bitplane_ring_split.cpp`
- `vb04` -> `04_bitplane_tree_reduce.cpp`
- `vb05` -> `05_bitplane_column_tiled.cpp`
- `vb06` -> `06_bitplane_neon_ripple.cpp`
- `vb07` -> `07_bitplane_neon_vertical_slide.cpp`
- `vb08` -> `08_bitplane_neon_eor3_scratch.cpp`
- `vb09` -> `09_bitplane_neon_block_h.cpp`
- `vb10` -> `10_bitplane_neon_predicate_huge.cpp`
- `vb11` -> `11_bitplane_temporal_stripe.cpp`
- `vb12` -> `12_bitplane_temporal_ring.cpp`

## vb01: Bitplane Scalar Kernel

`vb01` stores the two state bits in separate `uint64_t` arrays:

```text
s0 = low state bit
s1 = high state bit
ADULT = s0 & s1
```

Each word represents 64 horizontal cells. Neighbor counts are computed with
bitwise adders over whole words.

Chosen over: continuing with byte-per-cell `uint8_t` storage.

Reason: the target grid can be 32768x32768, so memory traffic matters. Bitplanes
reduce active grid-state storage by 4x and expose 64-way scalar word parallelism
before adding ARM NEON/SVE2. This is still the same generation-by-generation
stencil simulation; it does not memoize states or recognize patterns.

## vb02: Horizontal Row-Sum Ring

`vb02` computes each source row's horizontal 5-cell adult sum once and stores it
as three bitplanes in a 5-slot ring buffer. Output rows combine the five row-sum
slots vertically and subtract the center adult bit.

Chosen over: recomputing all shifted adult masks for every output row.

Reason: neighboring output rows reuse the same five source-row horizontal sums.
The ring buffer removes repeated horizontal work while preserving
O(generations * cells) behavior.

## vb03: Interior/Edge Word Split

`vb03` keeps the row-sum ring, but removes modulo wrap from the hot horizontal
word path. Only word `0` and word `rw - 1` need toroidal word-boundary handling.
Interior words use direct neighbor loads:

```text
prev = adult[w - 1]
curr = adult[w]
next = adult[w + 1]
```

Chosen over: `(w - 1 + rw) % rw` and `(w + 1) % rw` for every word.

Reason: `rw = N / 64`, and for valid benchmark sizes it is a power of two, but
even `& (rw - 1)` is unnecessary for almost all words. Splitting edge and
interior word handling removes wrap arithmetic from the hottest row-sum loop.
This is the same kind of constant-factor win as the byte-grid interior/edge
split, adapted to bitplanes.

## vb04: Tree-Reduced Count Chains

`vb04` replaces two serial count chains:

- horizontal 5-input adult row sums
- vertical combine of five 3-bit row sums

Chosen over: repeatedly adding one mask or one row sum into a running count with
ripple-carry logic.

Reason: profiling `vb03` showed the kernel split roughly into row-sum fill and
combine/transition work. Both had the same dependency shape: a series of
carry-propagating bit operations. Tree reduction computes independent partial
sums first, then combines them with much shorter carry chains. The output count
bits are identical; only the expression shape changes.

This version also folds the transition predicates into compact range checks:

```text
birth:   A in [3, 5]
survive: A in [4, 9]
```

The reducer formulas were exhaustively checked for all horizontal 5-bit inputs
and all valid vertical combinations of five row sums.

## vb05: Column-Tiled Row-Sum Ring

`vb05` keeps the `vb03` ripple count structure but processes word-column tiles
inside each worker's row range. The default tile is 128 words, or 8192 cells:

```text
tile ring = 5 slots * 3 row-sum planes * 128 words * 8 bytes = 15 KiB
```

Chosen as an experiment over: full-width row-sum ring.

Reason: at `32768x32768`, the full-width ring is 60 KiB, which is close to the
64 KiB L1d size per Neoverse-V2 core. A tile-local ring should fit more
comfortably in L1. The tradeoff is that each source row is revisited once per
tile, and tile boundary words need explicit handling.

Local x86 testing did not show a win: 128-word and 256-word tiles were slower on
a `32768x32768`, 10-generation run, while 512 words was effectively no tiling
and matched `vb03`. This means tiling is not a new best locally, but the file is
kept as a target-side experiment because the L1/L2 behavior may differ on
Neoverse-V2.

## vb06: NEON Ripple Port

`vb06` ports the confirmed `vb03` ripple kernel to ARM NEON. It keeps:

- two-bitplane state representation
- 5-slot horizontal row-sum ring
- interior/edge word split
- persistent 8-thread row partitioning
- ripple-carry count structure

Chosen over: starting from `vb04` tree reduction.

Reason: the full-size scalar run showed `vb03` beating `vb04`, and the x86 SIMD
experiment showed SIMD widening matters more than the tree-vs-ripple structure.
The NEON version therefore makes the smallest semantic jump: two `uint64_t`
words per `uint64x2_t`, same arithmetic network, scalar edge words for toroidal
carry handling.

This file is ARM-only because it includes `<arm_neon.h>`. Compile and validate
it on the AWS `aarch64` target, not on the local x86 machine.

## vb07: NEON Vertical Sliding Count

`vb07` starts from `vb06` and changes the vertical 5-row combine from repeated
recomputation to a running count:

```text
V[y+1] = V[y] - H[y-2] + H[y+3]
```

The vertical count includes the center cell. Instead of subtracting the center
adult bit from the count, the rule thresholds are adjusted:

- EMPTY birth still uses `V in 3..5`
- ADULT survival uses `V in 5..10`, because `V = A + 1` for adult centers

Chosen over: recombining all five horizontal row sums for every output row.

Reason: the repeated 5-row recombine is still a major hot-path cost in `vb06`.
Sliding the vertical count replaces five 3-bit additions with one 3-bit
subtract and one 3-bit add per row, and removes the center-adult subtract from
the transition path. This keeps the same O(generations * cells) simulation; it
only reduces repeated constant-factor work.

This file is also ARM-only and needs AWS correctness validation against `vb03`
or `vb06`.

## vb08: EOR3 And Persistent Scratch

`vb08` starts from `vb07` and adds two low-risk target-specific optimizations:

- use ARM SHA3 `EOR3` for three-input XORs in vector add/sub chains
- allocate each worker's scratch buffers once before the generation loop

Chosen over: immediately jumping to the full block-H rewrite.

Reason: `vb07` is already correct and fast on the target. `EOR3` directly
targets full-adder sum expressions, while persistent scratch removes repeated
`std::vector` allocation and value-initialization from the generation loop. Both
changes preserve the `vb07` algorithm, so correctness failures should be easier
to diagnose than in a full block-buffer rewrite.

Build with `-mcpu=neoverse-v2+sha3`; GCC may not enable `veor3q_u64` with plain
`-mcpu=neoverse-v2`.

## vb09: Block-H Buffer

`vb09` starts from the winning `vb07` vertical-slide kernel and changes how
horizontal row sums are cached. For each 128-row output block, the worker first
materializes `H` for the block plus its two-row halos into scratch:

```text
H rows stored = block_rows + 4
H storage     = 3 bitplanes * (block_rows + 4) * row_words
```

The vertical pass then sweeps the block with the same running-count update:

```text
V[y+1] = V[y] - H[y-2] + H[y+3]
```

Chosen over: the 5-slot `H` ring from `vb07`.

Reason: the ring minimizes scratch bytes, but it mixes row-sum fill, vertical
count update, and rule application tightly. The block-H buffer gives the CPU a
clean row-sum pass followed by a clean vertical pass, while still fitting in the
2 MiB per-core L2 at the target size. At `N = 32768`, a 128-row block uses about
`132 * 3 * 512 * 8 = 1.55 MiB` for H, leaving enough L2 room for the running
vertical count and current row streams.

Target result: on the Graviton4 dev box, the 32768x32768 boundary input for
10000 generations improved from `vb07 = 182156.612 ms` to
`vb09 = 170286.970 ms`.

## vb10: Compact Predicate And Huge-Page Hints

`vb10` starts from `vb09` and keeps the same algorithm. It changes only two
constant-factor details:

- replace explicit `eq3..eq10` masks with closed-form boolean ranges:
  - birth range: `V in {3,4,5}`
  - survival range: `V in {5,6,7,8,9,10}`
- call `madvise(..., MADV_HUGEPAGE)` for large bitplane and scratch vectors on
  Linux

Chosen over: jumping straight to temporal blocking.

Reason: the compact predicate removes hot-path boolean work with very low
correctness risk. The expression was truth-table checked against the old
`eq3..eq10` form for all 32 possible 5-bit count values. Huge-page hints are a
low-risk TLB experiment at the 32768 grid size, where each state bitplane is
128 MiB.

This should be benchmarked against `vb09`; if it loses, keep `vb09` as the
current best and move on.

## vb11: Temporal Row-Stripe Blocking

`vb11` starts from `vb10` and changes the generation schedule. Instead of
reading and writing the global grid once per generation, each worker loads a
full-width stripe plus vertical halo rows into slab buffers:

```text
K = 4 generations
H = 64 output rows per stripe
halo = 2*K rows above and below
slab rows = H + 4*K
```

The slab is advanced for up to `K` generations with the existing `vb10` block-H
kernel. After those generations, only the valid inner `H` rows are copied back
to the global destination grid.

Chosen over: continuing to tune only the single-generation full-grid pass.

Reason: `vb10` still streams the whole 32768 grid from and to DRAM for every
generation. Temporal striping tries to reuse the loaded rows for multiple
generations before writing back, reducing global memory traffic. This is a
cache-behavior and memory-layout change, not memoization: every cell still runs
the real transition for every generation that affects the output.

Risk: halo indexing is easy to get wrong. The toroidal wrap is applied only when
loading the slab from the global grid. Inside the slab, each generation shrinks
the valid row interval by two rows at the top and bottom:

```text
gen 1 output: rows [2, slab_rows - 2)
gen 2 output: rows [4, slab_rows - 4)
...
gen K output: rows [2K, slab_rows - 2K)
```

This must be correctness-checked against `vb10` on small generation counts
before trusting full-size timings.

## vb12: Temporal Stripe With Slab-Local H Ring

`vb12` keeps the temporal stripe schedule from `vb11` but uses a smaller
slab-local 5-row H ring for each generation step inside the slab. The full
stripe schedule is:

```text
K = 4 generations
H = 64 output rows
slab rows = H + 4*K = 80
```

For each stripe, the worker copies 80 global rows into slab A, runs up to 4
generations with slab A/B ping-pong, then writes the valid 64 inner rows back to
the global destination. Inside a slab generation, the H ring slides down the
slab and reuses the horizontal row sums without materializing a full block-H
buffer.

Chosen over: `vb11`'s block-H buffer inside the slab.

Reason: `vb11` had the right memory-traffic idea but carried too much local
scratch and copy/update overhead. The 5-row H ring has a smaller working set and
benchmarked better on the target. The observed contender result was around
`156 s` on the 32768x32768 boundary workload, compared with `vb10 = 161.781 s`.

Correctness fix applied in checked-in `vb12`: final stripe rows are copied from
offset `2 * k_this`, not always `2 * TIME_K`. This is equivalent for the
10000-generation benchmark because `10000 % 4 == 0`, but it is required for
correct output when the requested generation count is not divisible by 4.

## SIMD Direction: Prefer Widening Before Tree Reduction

Local x86 AVX2 experiments compared two SIMD shapes:

- ripple SIMD: direct SIMD port of the existing ripple-carry bitplane kernel
- tree SIMD: SIMD port of the tree-reduced count chains

The tree form has fewer operations and a shorter dependency chain on paper, but
the measured difference was small on Zen 4. The out-of-order core can overlap
the ripple dependency chain across loop iterations, so throughput was dominated
more by total SIMD work than by the critical path.

Chosen for the next ARM direction: start with the cleaner NEON ripple port.

Reason: the large measured win came from SIMD widening itself, not from the
tree-vs-ripple adder structure. Neoverse-V2 is also an out-of-order core with
wide NEON throughput and register renaming, so the x86 null result may transfer
better than expected. Tree reduction remains correct and useful to keep around,
but it should not block the NEON port.

## Current Next Step

Benchmark `vb12` against `vb10`. If it passes correctness and stays near the
observed `156 s`, it becomes the current best; otherwise the fallback remains
`vb10`.
