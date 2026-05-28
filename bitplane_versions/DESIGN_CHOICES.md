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
