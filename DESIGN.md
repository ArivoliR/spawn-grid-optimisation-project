# Monster Spawning Grid — Design Choices

This document records every design decision made for the optimised
`spawn_sim.cpp`, the reasoning behind each, the alternatives that were
considered and rejected, and the constraints that drove each choice.

The constraints from the assignment that shape every decision:

- Total work must remain `O(generations × cells)`. No HashLife, no memoisation
  across generations, no pattern shortcuts. Every win is a constant-factor
  hardware-utilisation win.
- C++23, standard library only. No external libraries (Boost, TBB, folly,
  OpenMP excluded).
- Target hardware: AWS `c8g.2xlarge` — Graviton4 (ARM Neoverse-V2, ARMv9.0-a),
  8 vCPUs, no SMT, 16 GiB RAM, 64 KiB L1d / 2 MiB L2 per core.

---

## 1. Cell Representation — Bitplane Decomposition

**Decision.** Each cell's 2-bit state is split into two parallel 1-bit grids:
a `low_bit` plane and a `high_bit` plane. State value
`= (high_bit << 1) | low_bit`. ADULT = `low & high`.

**Reasoning.**

- One 128-bit NEON register holds 128 cells of one bitplane. The headline
  throughput is therefore 128 cells per SIMD instruction — 8× denser than a
  byte-per-cell representation (which is 16 cells per register).
- The entire generation step becomes pure boolean / bit-arithmetic on
  bitplanes. No per-cell branching, no lane shuffling for state extraction.
- "Is ADULT" is one SIMD AND of the two planes — the cheapest possible
  predicate to compute, which matters because the neighbour count is a sum
  of ADULT predicates.
- Total memory per grid is 2 × N²/8 bytes = N²/4 bytes (4× compression vs
  byte-per-cell), which reduces DRAM traffic in the bandwidth-bound regime
  on the 32K grid.

**Alternatives considered.**

- *One byte per cell* (the reference layout). Simplest, but 1 byte/cell ×
  N² = 1 GiB at N = 32768. SIMD packs only 16 cells per register. Hard
  ceiling on throughput.
- *2 bits per cell, packed 4 per byte.* Cuts memory 4× but requires shift/
  mask to extract any single cell. Less SIMD-friendly because boolean
  rules can't be expressed without unpacking to a wider type.

**Trade-off.** The cost of bitplanes is implementation complexity — every
arithmetic operation must be written out as a bit-sliced adder network.
This was accepted because the assignment is explicitly a constant-factor
hardware-utilisation challenge and the bitplane representation is the
densest one available on ARM NEON.

---

## 2. Bit Ordering Within a Row — LSB-First Across x

**Decision.** Cell at column `x` lives at bit `(x % 8)` of byte `(x / 8)` of
its bitplane row, with `x = 0` at the LSB of byte 0.

**Reasoning.**

- A NEON 128-bit register loaded from byte offset `16r` of a bitplane row
  holds cells `[128r .. 128r+127]`. Cell `128r + k` lives at bit position
  `k` within the register.
- "Shift the row by +1 in x" becomes "shift each byte left by 1 bit, carry
  the top bit of byte `i-1` into the bottom bit of byte `i`" — a
  pattern with clean NEON instructions (`vshlq_n_u8` + `vextq_u8` +
  `vshrq_n_u8`).
- Shifts toward higher x are left shifts; toward lower x are right shifts.
  Reads symmetrically with normal byte arithmetic.

**Alternative considered.**

- *MSB-first.* Mirror image. Same code cost; chosen LSB-first because byte
  inspection in a debugger reads more naturally (bit 0 of byte 0 = leftmost
  cell of the row).

---

## 3. SIMD Strategy — NEON Intrinsics + ARMv8.2-SHA3

**Decision.** Hand-written NEON intrinsics (`<arm_neon.h>`), with explicit
use of the ARMv8.2-SHA3 extension instructions (specifically `EOR3` via
`veor3q_u8`, used pervasively in adder networks). Build flag:
`-mcpu=neoverse-v2+sha3`.

**Reasoning.**

- NEON registers are 128-bit, fixed. Predictable codegen across
  compilers and generations.
- On Neoverse-V2, SVE2 is also only 128-bit wide. Going SVE2 would buy
  cleaner predication but **zero width advantage**, so the marginal
  benefit doesn't justify the unfamiliar tooling.
- `EOR3` (3-input XOR) is the bottleneck operation in adder networks —
  every full-adder's sum is `a XOR b XOR c`, which without EOR3 takes two
  XOR instructions. EOR3 collapses each full-adder sum to one instruction.
  Roughly halves the SIMD op count of the adder trees.
- The `MAJ(a,b,c)` carry function is computed as
  `vbslq_u8(a^b, c, a&b)` — three NEON ops, no SHA3 dependency.

**Why `+sha3` is required.** `-mcpu=neoverse-v2` alone does **not** enable
the SHA3 ISA extension in gcc-14, even though the CPU supports it. The
compiler emits "target specific option mismatch" errors on `veor3q_u8`
without `+sha3`. This was discovered during the first build attempt.

**Alternatives considered.**

- *Pure scalar, rely on autovec.* Reject — compiler will not invent
  bit-sliced adder networks from a scalar `for` loop.
- *SVE2 intrinsics.* Reject — same vector width as NEON on this CPU, more
  exotic intrinsics surface, no payoff.

---

## 4. Neighbour-Sum Algorithm — Separable, Vertically Sliding

**Decision.** The 24-neighbour ADULT count is computed via a 2D-separable
sum with a vertical sliding window:

- Pass 1 (per row): `H[y][x] = adult[y][x-2] + adult[y][x-1] + adult[y][x] + adult[y][x+1] + adult[y][x+2]`. 5-input bit-sliced adder produces a 3-bit `H` value (range 0..5).
- Pass 2 (down a block): `V[y][x] = H[y-2][x] + H[y-1][x] + H[y][x] + H[y+1][x] + H[y+2][x]`. 5-bit value (range 0..25).
- For `y+1`: `V[y+1] = V[y] - H[y-2] + H[y+3]`. One 5-bit-minus-3-bit subtract and one 5-bit-plus-3-bit add per row.

`V` counts the full 5×5 box *including* the centre cell. The rule
thresholds are pre-adjusted to account for the centre's own ADULT
contribution, avoiding an explicit subtraction:

- EMPTY → EGG iff `V ∈ {3,4,5}` (centre is not ADULT, so `V == A`)
- ADULT stays iff `V ∈ {5..10}` (centre is ADULT, so `A = V - 1`)

**Reasoning.**

- Naive: 24 ADULT-mask additions per cell. Separable: ~10 additions per
  cell. Separable with sliding V: ~1 add + 1 subtract per cell per row,
  amortised. Asymptotically O(1) per cell per generation — preserves the
  O(generations × cells) total work budget.
- The "slide vertically only" choice is dictated by the bitplane layout:
  within a SIMD register, 128 different x positions are computed in
  parallel. Sliding horizontally would force a serial dependency between
  adjacent SIMD registers, killing the parallelism. Sliding vertically
  serialises only across rows (which is fine — rows are processed
  sequentially within a block anyway).

**Why the centre-adjustment trick.** Subtracting the 1-bit `adult[y][x]`
from a 5-bit `V` is a multi-bit bit-sliced subtraction — ~10 NEON ops.
Pre-shifting the thresholds removes the subtraction entirely.

**Alternatives considered.**

- *Naive 5×5 sum, 24 adds per cell.* Reject — wasteful, no algorithmic
  payoff.
- *Bit-sliced full adder tree over all 24 inputs simultaneously.* Reject —
  no row-to-row reuse; ~80 NEON ops per cell vs ~25 for the sliding
  approach. But this is the cleanest non-sliding alternative; worth a try
  if the sliding version turns out memory-bound on the H scratch.
- *Slide both axes.* Reject — compound serial dependencies for a 5-wide
  window aren't worth the complexity.

---

## 5. Multi-Bit Sum Storage — Bitplane Form

**Decision.** `H` (3-bit) is stored as 3 bitplanes; `V` (5-bit) is stored
as 5 bitplanes. Multi-bit add/sub are implemented as bit-sliced full-adder
and borrow chains.

**Reasoning.**

- Keeps the 128-cells-per-register throughput end-to-end. A 5-bit-plus-3-
  bit add operates on 128 cells per instruction — ~15 SIMD ops total for
  the whole add.
- Uniform representation throughout; no width conversion between stages.
- `EOR3` and `BSL` shine in the adder/borrow networks.

**Alternative considered.**

- *Byte-per-cell counters for H and V.* `vaddq_u8` is one instruction per
  16 cells, simpler to read. But the working set inflates 8× — at the 32K
  grid, the H scratch buffer becomes a multi-MiB structure that spills L2,
  inflating DRAM traffic. Rejected on cache grounds.

---

## 6. H Scope — Full Block-Height H Buffer (Two-Pass Per Block)

**Decision.** For each block (B rows), pass 1 computes `H` for all `B + 4`
rows needed by the V sweep and stores them in a per-thread scratch
buffer. Pass 2 then sweeps `V` down all `B` output rows.

**Reasoning.**

- At the chosen block size (≤128 rows at 32K), the H scratch is
  ~(132 × 3 × 4096) ≈ 1.6 MiB — fits comfortably in the 2 MiB L2 per
  core. The whole block's working set stays cache-resident.
- Clean control flow: pass 1 is a tight loop over rows, pass 2 is a tight
  loop over rows. No interleaving means the compiler / hardware scheduler
  can squeeze ILP out of each independently.

**Alternatives considered.**

- *Ring buffer of 5 H rows, recycled as V slides.* Lower memory (~60 KiB
  per thread) but the control flow is more entangled — every row update
  is "compute one new H row, drop one old H row, slide V." Reject because
  even at the larger block size the full-buffer version fits in L2.
- *Full-strip H buffer (all N rows of H persisted across the strip).*
  Reject — at 32K, that's 384 MiB; eats DRAM bandwidth and TLB.

---

## 7. Boundary Handling — Branch Only on Boundary Tiles

**Decision.** Interior blocks use straight-line row indexing; only the
first and last blocks (those containing row 0 or row N-1) take a slow
path that applies modulo-N to the y index. The branch is per-row
(amortised across the whole row's worth of SIMD work).

For x, every row's first and last 128-bit register is wrapped using
register-level neighbour loads: register 0's "prev" is the row's last
register, and the last register's "next" is register 0. This is done
unconditionally per row (the cost is a single extra load per row, well
amortised across `R_REGS` iterations).

**Reasoning.**

- No ghost-cell padding required — saves the per-generation copy of edge
  rows/cols. The input array is read-only within a generation, so threads
  can safely read each other's halo rows directly.
- The y-wrap branch is highly predictable: only fires in 2 of the
  (typically) 64–256 blocks; branch predictor handles it ~perfectly.

**Alternatives considered.**

- *Modulo arithmetic in the inner loop (reference style).* Reject — adds
  an integer op per neighbour access and defeats the SIMD register-level
  stitching.
- *Pre-pad input each generation with a wrap halo.* Reject — a per-gen
  copy step that's cheap but unnecessary.

---

## 8. Threading — `std::jthread` Pool + `std::barrier` (Forced Fallback)

**Decision.** 8 `std::jthread` worker threads, spawned once at startup,
pinned to cores 0–7 via `pthread_setaffinity_np`. Each worker pulls
blocks from a shared atomic `block_counter` until none remain, then
arrives at a `std::barrier`. The barrier's completion function swaps the
double-buffer pointers and resets the counter. A `std::latch` gates the
workers from starting until the timed region begins.

The number of threads is chosen at runtime as
`min(std::thread::hardware_concurrency(), 8)`, overridable via the
`SPAWN_THREADS` environment variable.

**Reasoning.**

- The originally chosen design was `std::execution::par_unseq`, but on
  Ubuntu 24.04 + gcc-14, libstdc++ implements the parallel execution
  policies on top of Intel TBB at link time. The assignment forbids TBB
  as a library; without `-ltbb` the binary still links but
  `par_unseq` **silently falls back to single-threaded** (verified
  empirically — counted distinct thread IDs in a `par_unseq` body and
  observed exactly 1). The escalation rule chosen at design time was
  "fall back to `std::jthread` if TBB is unavailable" — that was
  triggered.
- Pinning avoids thread migration and the cache-cold penalty after each
  reschedule. The 8 worker IDs map 1:1 to the 8 physical cores on
  `c8g.2xlarge`.
- The dynamic atomic-counter scheduling provides load balancing without
  needing a work-stealing queue. Each block is ~equal in work
  (uniform stencil), but dynamic pull is cheap insurance against any
  scheduling skew.

**Synchronisation pattern.** A single `std::barrier(n_threads)` with a
completion function ensures the double-buffer swap fires *exactly once
per generation* (an earlier bug had the completion firing twice per
generation, cancelling the swap — caught by initial correctness testing
on 100-generation runs).

**Alternatives considered.**

- *`std::execution::par_unseq`.* Would have been the cleanest C++23
  expression — first choice. Rejected after empirical demonstration that
  it silently degrades to serial without TBB.
- *Hand-rolled atomic spinning barrier.* Marginally cheaper than
  `std::barrier`, but the completion-function pattern in `std::barrier`
  is what guarantees the swap fires exactly once and visibly to all
  workers. Cleanliness wins.
- *Work-stealing thread pool.* Overkill for a uniform stencil where each
  block does identical work.

---

## 9. Block Granularity — Runtime-Chosen

**Decision.** Block size in rows is chosen at runtime as
`min(max(N / 8, 8), 128)`. Examples:

| Grid size | Block rows | Number of blocks |
|---:|---:|---:|
| 512   |   8 |  64 |
| 2048  |  32 |  64 |
| 8192  | 128 |  64 |
| 32768 | 128 | 256 |

**Reasoning.**

- The lower bound (8) prevents scheduling overhead from dominating at
  small grids.
- The upper bound (128) keeps the per-thread H scratch within L2 at
  large grids: at N = 32K and block = 128, the H scratch is
  `(128 + 4) × 3 × 4096 ≈ 1.6 MiB`, fitting in the 2 MiB L2.
- `N / 8` aims for ~8 blocks per thread per axis at the small/medium
  sizes, which gives the dynamic scheduler enough work units to absorb
  any per-block timing skew.

**Alternative considered.**

- *Fixed block size (e.g. 128 for all grids).* At N = 512 that would
  yield only 4 blocks total, less than the thread count — bad load
  balance. Runtime selection is essentially free.

---

## 10. Memory Allocation — 64-Byte Aligned

**Decision.** All bitplane buffers (current/next, low/high; per-thread H
and V scratch) are allocated via `std::aligned_alloc(64, …)`. No huge
pages.

**Reasoning.**

- 64-byte alignment guarantees that 128-bit NEON loads never straddle a
  cache line, and that the start of each buffer is on a cache-line
  boundary. Cheap to add, removes a class of micro-pessimisation.
- Huge pages were considered but rejected for simplicity. Transparent
  huge pages (`MADV_HUGEPAGE`) would reduce TLB pressure on the 1 GiB
  buffers at N = 32K and could yield 5–15% on the largest grid, but
  require kernel support that is best-effort and adds setup complexity.
  This is the highest-priority "what I would do next" item.

---

## 11. I/O — `fread` + One-Shot Conversion (Untimed)

**Decision.** Read the input file via `fread` into a temporary byte
buffer. Convert byte-per-cell → two bitplanes once, outside the timed
region. Mirror on output: convert the final bitplanes → byte buffer,
then `fwrite`.

**Reasoning.**

- The assignment explicitly excludes I/O from timing. There is no
  performance pressure on the conversion.
- The conversion is a simple scalar loop (~100 ms for 1 GiB on this
  hardware) — fast enough that vectorising it would be premature
  optimisation.

---

## 12. No Temporal Blocking

**Decision.** Exactly one full grid pass per generation. No tile
advanced multiple generations before moving on.

**Reasoning.**

- Asymptotic constraint: total work must remain O(generations × cells).
  Temporal blocking (advancing one tile by T generations before the next
  tile catches up) does **not** change asymptotic complexity, but it is
  close enough to memoisation-style shortcuts that we ruled it out to
  stay clearly within the spirit of the constraint.
- At small grids (≤ 8192) the working set fits in L2/L3 and DRAM
  bandwidth is not the bottleneck — compute dominates. Temporal blocking
  would not help.
- At 32K the working set does spill to DRAM and we are likely
  bandwidth-bound. Temporal blocking *could* help here (cutting DRAM
  traffic ~T×) but was explicitly avoided.

**Implication.** Performance at the 32K grid is dominated by DRAM
bandwidth rather than NEON throughput. This is the highest-impact
optimisation that is intentionally left on the table.

---

## 13. Block-Halo Strategy — Read Directly From Shared Input

**Decision.** Each thread's block reads its 2-row vertical halo directly
from the shared previous-generation bitplane buffers. No per-block
private halo copy.

**Reasoning.**

- The previous-generation buffers are read-only during a generation
  step. Concurrent reads by multiple threads need no synchronisation —
  this is a stateless map.
- Avoids per-generation halo-copy overhead.

---

## 14. Rule Application — Pure Boolean

**Decision.** From `V` (5 bitplanes), `low`, `high`, derive the next
state via Boolean expressions:

- `E := V ∈ {3,4,5}`
  `= ~v4 & ~v3 & ((~v2 & v1 & v0) | (v2 & ~v1))`
- `R := V ∈ {5..10}`
  `= ~v4 & ((v2 & (v1|v0) & ~v3) | (~v2 & ~(v1&v0) & v3))`
- `next_low  = (~low & (high | E)) | (low & high & R)`
- `next_high = (high ^ low) | (high & low & R)`

Verified by truth table over all four input states × all relevant `V`
ranges.

**Reasoning.**

- Pure boolean — no branches, no lookup tables. Cleanest possible
  inner-loop tail.
- ~15 NEON ops total for the rule per 128 cells.

---

## 15. Compiler / Toolchain

**Build flags.**

```
-std=c++23 -O3 -mcpu=neoverse-v2+sha3 -Wall -Wextra
```

- `-O3` for aggressive inlining and instruction scheduling.
- `-mcpu=neoverse-v2+sha3` — the `+sha3` is mandatory; without it,
  `veor3q_u8` (EOR3) fails to compile.
- `-Wall -Wextra` — clean code base; no warnings expected.

No `-ffast-math`: there is no floating-point arithmetic. No PGO / LTO:
the program is a tight kernel where `-O3` does almost everything; the
remaining wins would require profile data that isn't reflective of the
production input.

---

## Items Intentionally Left for "Another Week"

In rough priority order:

1. **Huge pages (`madvise(MADV_HUGEPAGE)` on the bitplane buffers).**
   Expected 5–15% at N = 32K from reduced TLB misses on the 1 GiB-class
   buffers. One-line change; risk-free.
2. **Software prefetching of the next block's input rows** while computing
   the current block's H. Hardware prefetcher is good, but explicit
   prefetch may close the gap further at the 32K grid.
3. **Bitplane-aware byte ↔ bitplane I/O conversion (SIMD).** Currently
   scalar; not on the timing path, but reduces wall time on the 32K
   grid noticeably.
4. **Fusing the H init pass with the first V row's apply-rule.** Saves
   one pass over the H buffer at the cost of more entangled code. Worth
   measuring.
5. **Temporal blocking** at 32K — would cut DRAM traffic by the time-tile
   factor. Avoided as discussed in §12; would relax that constraint with
   author's permission.

---

## Verification

- Correctness verified bit-identically against the reference at all four
  available patterns at N = 512 / 10000 generations.
- Correctness verified bit-identically at N = 2048 / 10000 generations
  (random_low pattern); reference runs on the other 2048 patterns take
  ~8 minutes each on the single-vCPU dev environment available during
  development.
- Single-thread speedup on the dev environment (1 vCPU available):
  - N = 512:  30 723 ms → 496 ms = **62×**
  - N = 2048: 501 248 ms → 8 743 ms = **57×**
- Multi-thread scaling is not measurable on this dev box (only 1 vCPU
  exposed); on the `c8g.2xlarge` 8-core target, expected total speedup
  ~400–500× from the combination of bitplane SIMD and 8-way parallelism.
