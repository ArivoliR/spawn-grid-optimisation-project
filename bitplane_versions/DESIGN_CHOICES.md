# Bitplane Design Choices

This file tracks the bitplane-specific implementation decisions. The byte-grid
optimization history stays in `versions/DESIGN_CHOICES.md`.

## Naming

Bitplane versions are named `vbNN` in the notes:

- `vb01` -> `01_bitplane_scalar.cpp`
- `vb02` -> `02_bitplane_ring.cpp`
- `vb03` -> `03_bitplane_ring_split.cpp`
- `vb04` -> `04_bitplane_tree_reduce.cpp`

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

The next planned bitplane cleanup is persistent per-thread scratch. Right now
`step_rows_bitplane()` allocates `adult_tmp` and `rowsum_store` every generation
per worker. Moving those buffers into a per-thread context should remove malloc
and zero-fill noise from the generation loop and give us a cleaner base for
aligned NEON/SVE2 buffers.
