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
- `vb13` -> `13_bitplane_eor3_aligned.cpp`
- `vb14` -> `14_bitplane_byte_h4_compact.cpp`
- `vb15` -> `15_bitplane_fused_slide.cpp`
- `vb16` -> `16_bitplane_v_interleaved.cpp`
- `vb17` -> `17_bitplane_h_interleaved.cpp`

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

TODO: Vet this para tmrw

## vb13: EOR3 And Aligned Storage

`vb13` returns to the simpler `vb10` single-generation block-H kernel and keeps
only the two measured low-risk wins from the direct feature experiments:

- SHA3 `EOR3` in the vertical add/sub chains
- 64-byte aligned vector storage for main bitplanes and per-thread scratch

Chosen over: keeping explicit worker CPU pinning.

Reason: target measurements showed EOR3 was a small win and aligned vector
storage was a larger win. Explicit pinning did not help when the process was
already launched with `taskset -c 0-7`, and made the code less minimal. The
cleaned `vb13` therefore removes the pinning code while preserving the useful
changes.

This remains the `uint64x2_t` block-H design from `vb10`; it does not adopt the
larger byte-lane H-row kernel rewrite.

## vb14: Byte-Lane NEON H Tree

`vb14` adds the best measured byte-lane NEON experiment to the numbered
version ladder. It keeps the same block-H plus vertical-slide algorithmic
structure as `vb13`, but changes the hot representation and H-row kernel:

- bitplane rows are accessed as `uint8x16_t` vectors, one register per 128
  cells
- horizontal shifts use `vextq_u8` plus byte shifts instead of `uint64_t`
  word-stitching
- the horizontal 5-cell adult count uses a `FA + FA + HA` tree
- the H-row kernel is unrolled by four registers
- the two `vextq_u8` carry vectors are shared for `x-1/x-2` and `x+1/x+2`
- the state transition directly derives next low/high bitplanes instead of
  materializing EMPTY/EGG/JUVENILE masks

Chosen over: keeping `vb13` as the final candidate.

Reason: target measurements showed the byte-lane mapping is a larger win than
the smaller EOR3/alignment tweaks in the `uint64x2_t` kernel. It gives the
compiler and hardware a cleaner ARM NEON shape for the horizontal H fill while
preserving row-major locality for the vertical slide. On the 32768x32768
boundary workload, observed full-run time improved from `vb13 = 149701.896 ms`
to `vb14 = 127231.528 ms`, with a fresh rerun at `130412.640 ms`.

Rejected follow-ups that were tested before adding `vb14`:

- register-column V traversal: destroyed row-major locality and regressed
- software prefetch: neutral/slower
- direct vertical recompute: more add work outweighed scratch savings
- K=2 temporal slab: correct but slower
- explicit pinning: worse under `taskset -c 0-7`
- H-row unroll by 8: extra register pressure outweighed ILP
- two-row slide variants: reduced V scratch traffic but increased dependency
  pressure

Constraint note: `vb14` currently requires `N` divisible by 128 so each row is a
whole number of NEON registers. The tested target/public sizes satisfy this. If
the final input contract allows smaller valid powers of two, add a fallback
before making this the submitted `spawn_sim.cpp`.

## vb15: Fused Vertical Slide Experiment

`vb15` starts from `vb14` and changes only the hottest vertical-slide
update. The current hot path is:

```cpp
v = sub_v5_h3(v, h_out);
v = add_v5_h3(v, h_in);
```

The tested replacement is a fused bit-sliced update:

```cpp
v = slide_v5_h3_h3(v, h_out, h_in);
```

Chosen over: more H-row tuning or cache-only tuning.

Reason: profiling `vb14` on the 32768 target showed roughly 68% of summed
worker time in slide+rule and only about 31% in H fill. The H side already had
the easy wins: byte-lane shifts, adder tree, H4 unroll, and shared `vext`.
The remaining sub-110 gap needs a direct reduction in the slide phase. Fusing
`V - H_out + H_in` may avoid duplicated carry/borrow propagation while keeping
the same O(generations * cells) algorithm and the same row-major access pattern.

Risk: this is a hand-written multi-bit bit-sliced arithmetic network. It needs
truth-table/unit verification against `sub_v5_h3` followed by `add_v5_h3` before
any full-grid benchmark.

Status: correctness passed on the target smoke test, but the first 32768x32768
/ 1000-generation benchmark regressed to `13223.584 ms`. `vb14` remains the
current best. The likely reason is that the fused carry-save expression reduces
one carry/borrow chain but adds enough boolean work and register pressure to
lose throughput on Neoverse-V2.

## vb16: Interleaved V Scratch

`vb16` starts from `vb14` and keeps the architecture intentionally simple. The
only kernel change is the layout of the running vertical count scratch.

`vb14` stores the five bitplanes of `V` as five full rows:

```text
v_b0[all registers], v_b1[all registers], v_b2[all registers],
v_b3[all registers], v_b4[all registers]
```

`vb16` stores one register's complete `V5` count together:

```text
v_b0[r], v_b1[r], v_b2[r], v_b3[r], v_b4[r],
v_b0[r+1], v_b1[r+1], ...
```

Reason: the hot slide loop always consumes and writes all five `V` bitplanes for
the same register. Keeping those five vectors adjacent is a direct memory-layout
improvement and is easy to defend in review: the loop order, arithmetic, H
scratch, rule predicate, and generation schedule are unchanged. The code only
changes where the scratch vectors live.

Measured result on the target 32768x32768 boundary workload:

```text
vb14 best observed: 127231.528 ms
vb16 block 96, -Ofast: 120843.475 ms
```

## vb17: Interleaved H Scratch + apply_rule_byte Micro-Opts

`vb17` applies the same interleaving principle to H scratch that `vb16` applied
to V scratch, and folds two small instruction-count reductions into
`apply_rule_byte`.

**Change 1 — H scratch interleaved layout:**

`vb16` stored the three H bitplanes for each H row `i` as three separate
full-width planes:

```text
h0 for H row i: h_data + (3*i+0)*row_bytes
h1 for H row i: h_data + (3*i+1)*row_bytes
h2 for H row i: h_data + (3*i+2)*row_bytes
```

For `N = 32768`, `row_bytes = 4096`. The three 16-byte vectors for a given
register `r` are therefore 4096 bytes apart. Every load of a complete `Sum3`
crosses three distinct cache lines, and the 4096-byte stride causes cache-set
conflicts on the Neoverse-V2 4-way 64 KiB L1d.

`vb17` interleaves the three planes per register:

```text
h[i][r].h0  at hp(i) + r*48
h[i][r].h1  at hp(i) + r*48 + 16
h[i][r].h2  at hp(i) + r*48 + 32
where hp(i) = h_data + i * 3 * row_bytes
```

Total scratch size is unchanged: `(BLOCK_ROWS+4) * 3 * row_bytes`. Each
`Sum3` load or store now accesses 48 consecutive bytes — potentially a single
64-byte cache line — instead of three addresses 4096 bytes apart.

Chosen over: keeping the plane layout from vb16.

Reason: the slide loop (hot path) loads `h_out` and `h_in` for the current
register before every `sub_v5_h3` / `add_v5_h3` call. With the plane layout
those six 16-byte loads hit three separate 4096-byte-stride addresses per H
row (two H rows per slide step = 6 distinct 4096-stride cache-line accesses
per register per output row). With interleaving, the same six loads are two
48-byte sequential reads.

**Change 2 — apply_rule_byte micro-opts:**

Three instruction-level improvements, all proven algebraically:

1. **Remove `nc1 = vmvnq_u8(v.b1)`**: `born_b = vandq_u8(v.b2, nc1)` was the
   only use of `nc1`. Replaced with `born_b = vbicq_u8(v.b2, v.b1)`, saving
   one `MVNI` per `apply_rule_byte` call.

2. **`next_high` via EOR3**: Was `vorrq_u8(veorq_u8(high, low), adult_r)`.
   Because `adult_r` requires `low=1 AND high=1`, and `high XOR low = 0`
   wherever both are 1, the two terms are disjoint and `OR == XOR`. So
   `(high ^ low) | adult_r == high ^ low ^ adult_r`, which maps to a single
   `EOR3` on sha3-capable CPUs via the existing `vxor3_u8` wrapper.

3. **`next_low` via BCAX**: Was `vorrq_u8(vbicq_u8(high|born, low), adult_r)`.
   `adult_r` requires `low=1`; `(high|born) & ~low` requires `~low`. The two
   terms are disjoint, so `OR == XOR == BCAX`. `vbcaxq_u8(adult_r, high|born,
   low)` computes `adult_r ^ ((high|born) & ~low)` in one instruction on
   sha3-capable CPUs. Guarded with `#if defined(__ARM_FEATURE_SHA3)`.

Measured result on the target 32768x32768 boundary workload (two runs each):

```text
vb16 block 96, -Ofast: 120416.799 ms / 120469.461 ms
vb17 block 96, -Ofast: 117625.887 ms / 117367.164 ms  (best: 117367 ms, ~2.5% faster)
vb17 block 128, -Ofast: 119663.212 ms
```

`cmp` against vb16 output on 1-generation run: byte-identical.

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

## vb22: Streaming H Ring (Drop Block-H Scratch)

vb17 keeps a "block-H" scratch buffer that materialises `BLOCK_ROWS+4` H rows
ahead of the slide pass. At 32768 wide that buffer is ~1.5 MiB per thread,
mostly hot in L2 but blowing past L1. The redesign:

- 6-slot streaming H ring of (h0, h1, h2) plane triples. At 32768 the ring is
  6 × 3 × 4096 B = **~72 KiB per thread**, sized to live in L2 with the V
  scratch in L1 footprint and almost no L2 pressure from H reads.
- One `compute_H_row` per output row, instead of a batched fill of
  `BLOCK_ROWS` H rows then a batched slide. The amortisation across multiple
  output rows is preserved by the ring slots — each computed H is read 5 times
  before being overwritten.
- 2-way r-unroll of both the V-init and the slide loop. This is what forces
  ring size 6: a 2-row unrolled iteration reads H rows {y-3..y+3} for the
  pair (y, y+1), which is 6 H rows including the new one being computed.
- `apply_rule_byte` Karnaugh-minimised to 17 boolean ops with no `vmvnq`,
  using BSL for the born mux and BCAX for `next_low`.

On target the streaming ring is a clean baseline. It is slower than vb17 in
absolute terms on the boundary input only because vb17's block-H gets warm
data for free at workloads where the block fits L2 anyway. But it is the
right shape for everything that follows.

## vb23, vb24: Temporal Blocking Experiments (Negative Result)

Two experiments testing the hypothesis that DRAM bandwidth is the binding
constraint at 32K x 8 threads:

**vb23** — full-slab K-temporal blocking. Each thread reads its row band
plus a 2K-row ghost margin on each side, runs K generations locally, then
writes back. K configurable via `-DSPAWN_K=N`.

**vb24** — 2D diamond tiling. Column strips of 128 bytes (TILE_W_REGS=8) with
trapezoidal writeback: each strip writes only the central
`128 - 2 × ceil(2K/8)` bytes per gen, and adjacent strips overlap by enough
margin that boundary cells reach the writeback region only after their
inputs were also locally computed.

Both verified correct against vb22 across the 5 public patterns at 2048,
8192, and 32K x 100-gen single-threaded.

Measured at 32K x 10000 x 8 threads:

```text
vb22 baseline      113.5 s
vb23 K=4           125.8 s  (+11%)
vb23 K=8           119.4 s  (+5%)
vb24 K=2           167.3 s  (+47%)
vb24 K=4           134.7 s  (+19%)
vb24 K=8           122.4 s  (+8%)
```

**Conclusion**: vb22's streaming ring is not DRAM-bound. Per-counter
profiling on vb22 shows `stall_backend_mem = 15.5%` of cycles — substantial
but not dominant; the L1/L2 prefetcher keeps the streaming pattern fed.
Temporal blocking buys nothing here because the *new* memory traffic
(ghost-row reads/writes for K-temporal; ghost-column copies and trapezoidal
writeback for 2D tiling) exceeds the small DRAM saving from reduced
generation traffic. Both files retained for the design audit and as a
ready-made baseline if the workload ever changes shape.

## vb25: Pairwise Neighbour Sync (Marginal)

vb22 uses two `std::barrier<>`s per generation (start + end), which forces
all eight threads to converge twice per gen × 10000 gens = 80000 global
syncs. vb25 replaces them with **per-thread atomic gen counters**: thread
`t` waits only for `t-1` and `t+1` (toroidal wrap) to have completed the
previous gen before starting the next. Each counter sits on its own 64-byte
cache line.

Justification: `step_rows_bitplane` for rows `[y0, y1)` reads source rows
`[y0-2, y1+1]`. The only neighbouring data is owned by the two adjacent
threads.

Measured result: vb22 at 113.5 s → vb25 at 113.2 s. Per-counter snapshot
shows CPU utilization climbing from 6.62 / 8 to 6.79 / 8. The barrier was
real but small; the remaining 1.2 of 8 lost CPUs is workload imbalance and
SIMD pipe saturation, not synchronization. Kept in the tree as a baseline
for the threading shape we want for vb27.

## vb26: BSL MAJ-Fold Across Carry Chains

The five-bit column accumulator `V5` slides one row at a time via
`sub_v5_h3(V5, Sum3)` and `add_v5_h3(V5, Sum3)`. Both functions are ripple
chains of depth 5, with interior stages that compute carry/borrow as a
majority of three inputs:

- Adder: `carry_out = MAJ(a, b, carry_in)`
- Subtractor: `borrow_out = MAJ(~a, b, borrow_in)`

vb22 expresses MAJ via `maj_u8(a, b, c) = BSL(a^b, c, a & b)` — three ops
(eor + and + bsl). For the subtractor that becomes 4 ops because of the
extra `vmvnq` to invert `a`. The same MAJ pattern appears in `sum_of_5`
(the 5-input horizontal sum used inside `horizontal_window_sum`).

The identity used in vb26 is:

```
MAJ(a, b, c) = BSL(a ^ b, c, a)
```

Proof: when `a == b`, `a^b = 0` and BSL selects the "else" lane = `a` (which
equals `b`), so MAJ = `a` = `b`. When `a != b`, `a^b = 1` and BSL selects
the "then" lane = `c`, so MAJ = `c`. Both branches agree with the truth
table for the 2-of-3 majority. The same identity drives the full-subtractor:
`borrow_out = BSL(b1 ^ h, h, borrow_in)`, no `vmvnq` needed.

Each interior carry/borrow stage drops from 3-4 ops to 1 BSL (given the
already-computed `a^b` from the corresponding sum bit). Sum bits use
`veor3q_u8` (SHA3) where three inputs are XORed.

Per-counter result on the target at 32K x 500 gens:

```text
                   vb22       vb26      delta
instructions       376.9 G    346.1 G   -8.2 %
cycles             128.5 G    124.3 G   -3.3 %
IPC                2.93       2.78      -5 %
stall_backend      43.3 G     41.0 G    -5 %
```

Wall clock at 32K x 10000 x 8 threads: vb22 baseline 113.5 s → vb26 109.0
s. The IPC dropped slightly because the new BSL form has a tighter serial
dependency than the old `vxor3 + maj_u8` (which could issue the and/bsl in
parallel), but the absolute instruction-count win is larger than the IPC
loss. `cmp` clean against vb22 at 8K and 32K outputs.

vb26 is the best two-pass kernel (compute_H_row pass + slide pass) and was
the basis for vb27.

## vb27: Single-Pass Fused Kernel (Current Best)

`perf record` on vb26 showed 31.5 % of cycles in `compute_H_row` and 63.5 %
in `step_rows_bitplane`. The two-pass structure stores each computed H row
to the ring and reads it back in the slide pass — an L1/L2 round trip per
H value, even though the producer and consumer are microseconds apart on
the same thread.

vb27 fuses the two passes. Inside the inner column-pair loop:

1. Load the entering row's adult bits with a 3-wide sliding window
   (`adult_prev`, `adult_curr`, `adult_next_0`, `adult_next_1`).
2. Compute new H values for the column pair via `horizontal_window_sum`.
3. Use the new H values **immediately** in `add_v5_h3` (no scratch round
   trip for the new H).
4. Subtract the old H from the ring slot the new H will overwrite.
5. Store the new H into that ring slot for future iterations.
6. Apply the rule for the current output row.
7. Slide the adult window for the next column pair.

The fusion also drops vb22's 2-row unroll (the structural reason vb22
needed a 6-slot ring) in favour of 2-column unroll for ILP, which works
out better when the inner-loop register pressure includes both the sliding
adult window and two V5 accumulators. The ring shrinks from 6 to 5 slots
(`RING_SLOTS = 5`), with `tail` advancing by 1 per row.

Per-counter result on the target at 32K x 500 gens:

```text
                       vb22       vb26       vb27       delta vs vb22
cycles                128.5 G    124.3 G    108.4 G    -16 %
instructions          376.9 G    346.1 G    346.4 G    -8 %
IPC                    2.93       2.78       3.20      +9 %
stall_backend          43.3 G     41.0 G     30.0 G    -31 %
stall_backend_mem      19.9 G     19.9 G      6.4 G    -68 %
```

The 68 % drop in `stall_backend_mem` is the H scratch round trip we
eliminated. IPC climbs back up to 3.20 from vb26's 2.78 because the BSL
dependency chain is now overlapped with the fused H compute and the ring
store, giving the OOO engine independent work to issue.

Wall clock at 32K x 10000 x 8 threads, three stable runs:

```text
vb27 run 1    95214 ms
vb27 run 2    95261 ms
vb27 run 3    95223 ms
```

That's **-16.1 % vs vb22's 113.5 s baseline** and **-12.6 % vs vb26's
108.97 s**. `cmp` clean against vb22 at 8K and 32K outputs. Three stable
runs ± 50 ms.

### Why the fusion works

Per-row, vb22/vb26 spend cycles like this:

```text
compute_H_row(new row)        — write 3 bitplane rows × N/8 bytes to ring slot
slide loop:
  load 5 H values from ring   — read 3 bitplane rows × N/8 bytes
  sub_v5_h3
  add_v5_h3
  apply_rule_byte
```

vb27 keeps the new H entirely in registers between produce and consume.
The old H is still loaded from the ring slot we're about to overwrite, but
that single load is what we had before anyway. Net: each row saves the
write-and-immediately-reload of one H row through L2.

At 32K wide, one H row is N/8 × 3 = 12 KiB. Across 32768 rows × 10000 gens
× 8 threads that's 30 TiB of L2 traffic eliminated, which matches the
order of magnitude of the observed `stall_backend_mem` drop.

## vb28: Pairwise Sync on Top of the Fused Kernel

vb27's fused inner loop is ~16 % tighter than vb22, so each generation's
compute pass is shorter — and the global `std::barrier<>` becomes a
relatively larger share of wall time. CPU utilisation in the perf snapshot
dropped from vb22's 6.62 / 8 to vb27's 5.98 / 8, even though the underlying
work per gen also shrank.

vb28 replaces the global barriers with the same per-thread atomic
gen-counter scheme from vb25: each thread waits only on its two row-band
neighbours (toroidal wrap), advances its own `done_gen` counter, and
toggles `src` / `dst` by gen parity locally. The cur/next pointer pair is
removed from the main thread.

The perf snapshot recovers most of what vb27 lost:

```text
                   vb27      vb28
task-clock         40789     40677
CPUs utilised      5.98      6.61
500-gen wall       6.82 s    6.16 s    (-9.7 %)
10000-gen wall    95.21 s   94.54 s    (-0.7 %)
```

The 9.7 % short-run win does not propagate fully to 10000 gens because the
absolute barrier cost is small per gen — at long runs the kernel itself
dominates. Still, three stable 10000-gen runs at 94538 / 94570 / 94518 ms
beat vb27 by ~0.7 s and beat vb22 by 16.7 %.

`cmp` clean against the vb22 8K and 32K outputs.

## vb29, vb30: Negative Results Kept For The Audit Trail

Two further attempts on top of vb28 both regressed, both informative.

### vb29 — software prefetch

Hypothesis: with the inner loop already very tight, the L1/L2 prefetcher
might not be fast enough to keep up with the streaming reads of the
entering row, the V scratch, and the ring slot. Inserted
`__builtin_prefetch` ~4 column-registers ahead for each of those streams.

Result: vb28 94.5 s → vb29 97.8 s. The Neoverse-V2 HW L2 prefetcher already
streams those accesses; SW prefetch instructions added to the dispatch
budget without removing any L2 misses (vb28 already had only 5.9 % of
cycles in `stall_backend_mem`). Net: noisier pipeline, slower wall clock.

### vb30 — peeled boundary iteration

Hypothesis: the ternary
`adult_next_1 = (r + 2 < R_REGS) ? load_new_adult(r + 2) : load_new_adult(0)`
in the column-pair loop forces the compiler either to branch or to keep
the wrap-around register live across the row, possibly causing a stack
reload. Peeled the final pair (`r = R_REGS - 2`) out into a separate
boundary block whose `adult_next_1` is the precomputed `bnd_adult`,
making the main loop's `adult_next_1` load unconditional.

Result: vb28 94.5 s → vb30 97.1 s. The compiler was already producing tight
code for the ternary on Neoverse-V2; the extra peeled block added icache
pressure and probably perturbed scheduling. Net regression.

Both files are kept in the tree as documented negative results. They are
the natural follow-ups any reader would propose, so leaving the data
visible saves them the round trip.

## vb31, vb32, vb33: Pushing Toward Sub-80s (Mostly Negative)

Three further attempts targeting different parts of the kernel. Together
they map out the local optimum that vb28 already sits at and the
structural blockers preventing sub-80 s without a fundamentally
different layout.

### vb31 — column-strip iteration

Hypothesis: ~12 % of all instructions are V scratch load/store ops
(5 loads + 5 stores per col reg per row). Reorganise the iteration so V
state lives in vector registers across rows: per thread, walk column
strips of S = 2 col regs, and for each strip iterate all band rows. Per
strip the H ring is stack-local (480 B, L1-resident), and V state stays
in registers because the row loop only touches the strip's 2 cols.

Result: vb28 94.5 s → vb31 160.2 s — a 70 % regression. Per-counter:

```text
              vb28      vb31
IPC           3.21      2.28
L1d miss %    4.83      15.87
L2 miss %     3.4       29.5
```

Diagnosis: row-major `src` / `dst` layout means column-major iteration
across all 4096 rows in a strip touches the src grid in a striped
pattern (only 4 vectors per row, scattered across all rows). L1 misses
exploded because the working set is 16 MiB of src data per strip versus
64 KiB L1. The cache penalty completely eats the V scratch savings.

This reproduces vb24's lesson: bitplane row-major layout doesn't tolerate
column-strip iteration without a transpose, and transpose itself is too
expensive per gen. The V scratch round-trip is L1-resident, so the
savings were always smaller than the cost.

### vb32 — 4-column unroll

Hypothesis: the inner loop's 2-column unroll exposes two independent V5
carry chains to the OOO engine. 4-column unroll would expose four.

Result: vb28 94.5 s → vb32 110.8 s. Register pressure: 4 × V5 (20 vecs)
+ 4 × adult window (5 vecs) + 4 × h_old / h_new triples (24 vecs) +
src / dst loads = 50+ live vectors against the 32 NEON register file. The
compiler spilled aggressively, and the spill traffic dwarfed the ILP win.

### vb33 — fused `slide_v5_h3`

Hypothesis: `sub_v5_h3` followed by `add_v5_h3` is a serial dep chain of
depth 10 (5 + 5). Interleaving them at the bit level — borrow chain and
carry chain advance one stage per bit — reduces combined dep depth to ~6.

```cpp
// per bit b in 0..2 (and adapted for bits 3, 4 where h_old/h_new are 0):
//   tmp_b      = v.b_b ^ h_old.h_b ^ borrow_{b-1}
//   borrow_b   = BSL(v.b_b ^ h_old.h_b, h_old.h_b, borrow_{b-1})
//   out_b      = tmp_b ^ h_new.h_b ^ carry_{b-1}
//   carry_b    = BSL(tmp_b ^ h_new.h_b, carry_{b-1}, tmp_b)
```

Op count: 21 ops vs 22 for separate sub + add. Dep depth: 6 vs 10.

Result: vb28 94.5 s → vb33 95.9 s. Per-counter:

```text
                   vb28      vb33
cycles            108.4 G   110.6 G
instructions      348.5 G   350.0 G
IPC                3.21      3.16
stall_backend      30.0 G    27.1 G   (-10 %)
```

Backend stalls did drop by ~10 % (confirming the dep chain shortened),
but the OOO engine in vb28 was already overlapping the two chains by
interleaving across the two columns in the 2-col unroll — so the chain
was already effectively depth ~6 at the schedule level. The fused
version's tighter inner function appears to have given the register
allocator slightly less freedom; the small backend-stall improvement
got cancelled out by a slight uptick in total cycles.

Kept in the tree because the dep-chain analysis is useful for any future
work that breaks the 2-col unroll (e.g. SVE2 with wider vectors).

### Where vb28 sits on the optimisation curve

At 94.5 s on c8g.2xlarge, vb28 hits multiple ceilings simultaneously:

- **IPC 3.21 / 4.0** — 80 % of peak SIMD issue
- **CPU util 6.61 / 8** — 83 % of peak threading
- **L1d miss rate 4.83 %** — already low; HW prefetcher does the rest
- **stall_backend 27.7 %** — most of which is short execution-unit
  contention rather than memory

The remaining headroom to sub-60 s would require **fewer instructions per
cell**, not better scheduling. The two paths visible are:

1. A byte-form V representation (V5 → single uint8 per cell): collapses
   the 22-op slide ripple to 2 vector ops, but loses the 8× compression
   from bitplane storage and makes the whole grid DRAM-bound (5+ GB per
   bitplane variant) — would not pay back on this workload.
2. A wider SIMD path (SVE2, currently 128-bit on Neoverse-V2, but a
   future SVE2-256 part would double the cells per vector). Out of scope
   on c8g.2xlarge.

vb28 is therefore the chosen submission candidate from this branch.

## vb34, vb35, vb36: Sub-80s Push (All Neutral or Regress)

The `perf annotate` of vb28 showed unexpectedly hot `cmp` and `yield`
instructions clustered around the spin-wait — looked like up to 30% of
hot-path cycles. Three follow-ups targeting that signal, then one
attempt to stack the neutral optimisations:

### vb34 — explicit CPU pinning

Hypothesis: threads migrating between cores cause cache loss and
amplify the time each thread spends spin-waiting on its neighbours. Add
`pthread_setaffinity_np(self, CPU(t))` at the start of each worker.

Result: vb28 94.5 s → vb34 94.9 s. Neutral. Whatever was attributed to
the spin-wait in `perf annotate` was either mis-attributed (cmp/yield
opcodes inside step_rows_bitplane's loop bounds being conflated with
the worker_loop) or already cheap.

### vb35 — pure busy-spin (no yield)

Hypothesis: the `yield` ARM hint instruction itself was adding latency
to the spin-wait. Replace it with empty body, just `cmp` + branch.

Result: vb28 94.5 s → vb35 94.8 s. Confirms yield wasn't a cost.

### vb36 — vb33 fused slide + vb34 pinning

Hypothesis: each of vb33 (fused slide_v5_h3 with depth-6 ripple) and
vb34 (pinning) was neutral alone; combining them might push the OOO
engine across some threshold (less dep-chain pressure + less migration
noise).

Result: vb28 94.5 s → vb36 96.6 s. **Regression.** The combination
compounded register pressure: the bigger inlined `slide_v5_h3` body in
the 2-col unrolled inner loop ate the marginal threading gain. Stacking
neutral micro-optimisations isn't free — each one trades one resource
for another, and the totals don't add.

### Final position on the sub-80 s question

Theoretical floor for this algorithm on this hardware:

```text
6920 G instructions / (8 CPUs × 4 IPC × 2.66 GHz) = 81.4 s   (peak)
```

We are at 94.5 s = 16 % off peak. To break sub-80, we need *both*:

- IPC >3.6 (currently 3.21) — needs further reducing the backend
  dep-chain stalls
- Effective CPUs >7.5 (currently ~6.6 averaged) — needs reducing the
  inter-thread sync drift

Independent attempts at each (vb33 backend, vb34/35 sync) moved
either metric by ≤5 % without translating to wall-clock improvement,
because the gain was always re-absorbed by a different resource —
register pressure in vb33, schedule noise in vb34/35.

Sub-80 is **not** reachable from vb28's design space without one of:

- An algorithmic change that reduces total instructions (Hashlife-style
  memoisation, or a representation that drops the depth-5 ripple
  ladder). Hashlife only pays back on repeating patterns; the
  `random_low` workload destroys most of its memoisation benefit. Byte-
  form V cuts the slide to 2 SIMD ops but makes the whole grid DRAM-
  bound (5+ GB working set vs 1 GB for bitplane), so we lose more on
  memory than we save on ALU.
- Wider hardware SIMD. Neoverse-V2 SVE2 is 128-bit, same as NEON, so
  the same kernel reissued in SVE2 would give the same throughput.
  Wider SIMD (256, 512) would proportionally drop wall time but is a
  c8g.2xlarge constraint.

vb28 at 94.5 s remains the best achievable from this branch on this
hardware; the negative results above bound the optimisation neighbourhood.

## vb37: Empirical Refutation of Byte-Form V

The most attractive remaining idea was a byte-form V representation: store
each cell's count as a full `uint8_t` (value 0..25) instead of five
bitplane planes. The slide then becomes two SIMD byte operations
(`vsubq_u8` + `vaddq_u8`) per 16 cells, replacing the 22-op bitplane
ripple — a potential saving of ~20 s of wall time at 32K × 10000 gens.

The catch was always the conversion: `h_old` / `h_new` arrive from
`horizontal_window_sum` as 3 bitplanes (1 bit per cell across 128 cells
per vector), and V's output must end up as bitplane masks for the
`apply_rule_byte` predicates. Each round-trip needs a bit→byte spread
(`vqtbl1q` broadcast + power-of-2 mask + normalise to 0/1) plus a
byte→bit pack (mask + position-multiply + horizontal reduce).

vb37 measures this conversion cost directly. It is vb28 unchanged in
all logic — same bitplane slide, same `apply_rule_byte` — but with
`sum3_to_byte8` invoked on `h_new_0`, `h_new_1`, `h_old_0`, `h_old_1`
inside the column-pair inner loop. The results are held alive via an
`__asm__ __volatile__` clobber so the compiler cannot dead-eliminate
them, but they are not consumed by anything else.

```text
vb28 baseline                                94.5 s
vb37 (vb28 + dead-code byte conversion)     167.0 s
Conversion overhead alone                   +72.5 s
```

Per-iteration cost analysis (per 128-cell column register):

```text
sum3_to_byte8 = 3 × bitplane_to_byte8 + combine
              = 3 × (8 × (vqtbl1q + vand + vceqq + vmvnq + vshrq))
                + 8 × (vshl + vshl + vorr + vorr)
              = 3 × 40 + 32
              = 152 ops per Sum3 conversion
```

Two `Sum3` conversions per column register pair (h_new and h_old) ×
256 column register pairs per row × 4096 rows per thread × 10000 gens
= ~6.3 G additional vector ops per thread. The measured +72 s wall time
matches this — and we *haven't* counted the V_byte → V5 bitplane
conversion needed to feed `apply_rule_byte`, which has the same shape
in reverse and would add a similar penalty again.

The theoretical maximum saving from a full byte-V slide is bounded
above by the cost of the 22-op bitplane slide it replaces. That saving
is **smaller than the conversion overhead by ~3.6×**. There is no
arrangement of byte-V in this kernel that comes out positive.

The conclusion holds independently of how the byte-V kernel is
structured: the bit↔byte conversion needs ~5 ops per 16 output cells
per bitplane plane, and the slide it would replace needs ~3 ops per 128
cells per plane (depth-5 ripple bit-stage). Until NEON gains a
single-instruction bit-to-byte spread, byte-form representations cannot
beat bitplane on this kernel shape.

This rules out byte-V definitively, even as a Plan B if other
optimisations fail.
