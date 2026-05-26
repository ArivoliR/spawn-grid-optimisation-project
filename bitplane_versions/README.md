# Bitplane Versions

This folder tracks the bitplane rewrite separately from the byte-grid versions
in `versions/`.

## Files

- `01_bitplane_scalar.cpp`: first bitplane implementation. Stores the two state
  bits in separate `uint64_t` planes and computes the 5x5 adult count with
  bitwise adders.
- `02_bitplane_ring.cpp`: reuses horizontal 5-cell row sums in a 5-row ring
  buffer, reducing repeated work between neighboring output rows.
- `03_bitplane_ring_split.cpp`: keeps the ring-buffer design and splits
  horizontal word handling into interior and edge paths. Interior words use
  direct `w - 1` / `w + 1` loads instead of modulo wrap.
- `04_bitplane_tree_reduce.cpp`: replaces serial ripple-add chains in the
  horizontal row sum and vertical combine phases with tree reductions.
- `05_bitplane_column_tiled.cpp`: tests column tiling for the row-sum ring.
  The default tile is 128 words / 8192 cells and can be changed at compile time
  with `-DDEFAULT_TILE_WORDS=<words>`.
- `06_bitplane_neon_ripple.cpp`: ARM-only NEON port of the `vb03` ripple
  kernel. It keeps the same threading and row-sum ring structure while handling
  two `uint64_t` words per vector.
- `07_bitplane_neon_vertical_slide.cpp`: ARM-only NEON experiment that keeps a
  running vertical 5-row count, replacing the repeated 5-row recombine in
  `vb06`.
- `08_bitplane_neon_eor3_scratch.cpp`: ARM-only NEON vertical-slide experiment
  with SHA3 `EOR3` use and persistent per-worker scratch buffers.
- `09_bitplane_neon_block_h.cpp`: ARM-only NEON version that materializes a
  block of horizontal row sums into per-worker scratch, then vertically slides
  over that block.
- `10_bitplane_neon_predicate_huge.cpp`: `vb09` plus compact rule predicates
  and Linux transparent huge-page hints for the large bitplane and scratch
  buffers.
- `11_bitplane_temporal_stripe.cpp`: experimental temporal row-stripe blocking.
  It advances each full-width stripe for up to 4 generations in per-worker slab
  buffers before writing the valid inner rows back.
- `12_bitplane_temporal_ring.cpp`: temporal row-stripe version using a
  slab-local 5-row H ring instead of the block-H buffer. This is based on the
  faster temporal contender, with the final writeback offset fixed for
  generation counts that are not divisible by 4.
- `13_bitplane_eor3_aligned.cpp`: minimal `vb10` improvement. Keeps SHA3
  `EOR3` in the vertical add/sub chains and 64-byte aligned vector storage;
  leaves out explicit CPU pinning because it showed no benefit under `taskset`.
- `experiments_v13_v15/`: archived one-by-one experiments for EOR3, pinning,
  and aligned storage.

All versions still run every generation over the full grid. The speedup
comes from representation, threading, cache behavior, and removing hot-path
integer work, not from memoization or input-specific precomputation.
