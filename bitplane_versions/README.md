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

All three versions still run every generation over the full grid. The speedup
comes from representation, threading, cache behavior, and removing hot-path
integer work, not from memoization or input-specific precomputation.
