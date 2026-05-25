# Design Choices

This file tracks implementation decisions for the optimized spawn simulator and
why we chose each direction over the main alternatives.

## Scalar Baseline Before SIMD
Just makes it easier to work with, benchmark, debug. SIMD should be final. 

## 02 Manual Neighbor Unrolling

Version 02 inlines the 24-neighbor count instead of using `count_adults()` with
nested loops.

**Chosen over** keeping the clean nested `dx/dy` loop.

Reason: the nested loop repeatedly branches, computes offsets, and calls a
helper for every cell. Unrolling gives the compiler straight-line work in the
hot path.

## 03 Bitmask Toroidal Wrapping

Version 03 uses `& (N - 1)` for wrapping.

Chosen over: `% N`.

Reason: grid sizes are guaranteed to be powers of two. Bitmask wrapping is
cheaper than integer modulo and preserves the same toroidal behavior under that
constraint.

## 04 Raw Row Pointers

Version 04 precomputes row pointers like `r0..r4` and writes through `rw`.

Chosen over: repeated `vector[(size_t)y * N + x]` indexing.

Reason: row pointers reduce repeated address arithmetic and make the hot loop
more direct. They also prepare the code for SIMD loads.

## 05 Row-Based Parallelism

Versions 05 and 06 split the output grid by row ranges.

Chosen over: per-cell tasks, dynamic scheduling, or tile queues.

Reason: each output row range can be written independently while reading from
the previous generation. Static row ranges are simple, low-overhead, and avoid
contention. Dynamic scheduling may help later if tiling or load imbalance
becomes important, but the current workload is fairly uniform per cell.

# Parallelism 

## Choice: `std::thread` Instead Of `std::jthread`

We use `std::thread` and explicitly join workers.

Chosen over: `std::jthread`.

Reason: `std::jthread` mainly provides automatic join and stop-token based
cancellation. Our workers have explicit shutdown through `stop = true`, a
barrier wakeup, and manual joins. `std::thread` is minimal and clear for this
performance-focused worker pool.

## 06 Persistent Thread Pool (performance improvement yet to be proven by benchmarks)

Version 06 keeps worker threads alive across generations and synchronizes with
barriers.

Chosen over: creating and joining threads every generation.

Reason: the final workload runs 10,000 generations, so avoiding repeated thread
creation is the better structure. Intuitively I think this will work better with SIMD rather than creating a new thread every time. 

## `std::execution` Experiment (opted out for now? I'm not sure how this can make our lives any better

Version 07 tests `std::for_each(std::execution::par, ...)` over fixed row
chunks.

Chosen over for final path: explicit `std::thread` worker pool.

Reason: on the local GCC/libstdc++ setup, `std::execution::par` did not link
with the normal project-style flags and required `-ltbb`, because the parallel
execution backend resolves to TBB symbols. External libraries are outside the
assignment constraints. Even as a local experiment with `-ltbb`, it was slower
than the explicit thread pool, so we keep manual thread management.

# Computation ? (forgive the topic names pls lmao)

## 08 Sliding 5x5 Adult Counts (pls verify this change once)

Version 08 keeps rolling horizontal 5-cell adult counts for the five source rows
and combines them to get the 5x5 adult count, subtracting the center cell.

**Chosen over** checking all 24 neighbors independently for every cell.

Reason: adjacent cells share most of their 5x5 neighborhood. Sliding counts
reuse overlapping horizontal work and reduce scalar neighbor-count cost while
preserving the required O(generations * cells) simulation structure.  

## 09 Precomputed Horizontal Sums And Transition LUT

Version 09 computes a per-generation scratch buffer:

```text
hsum[y][x] = adult[y][x-2] + adult[y][x-1] + adult[y][x]
           + adult[y][x+1] + adult[y][x+2]
```

Then each cell combines five `hsum` rows and subtracts the center adult value.
It also uses a constexpr transition table for `state x adult_count`.

**Chosen over** recomputing rolling horizontal sums separately for every output
row and using a `switch` for every transition.

Reason: neighboring output rows reuse the same source-row horizontal sums.
Materializing those sums once per generation reduces duplicate work. The
transition lookup table is a fixed encoding of the official rules, independent
of input data. The scratch buffer is rebuilt every generation, so this is a
constant-factor optimization, not memoization across generations.

## 10 Interior/Edge Split

Version 10 changes horizontal-sum construction so only the first and last two
columns use toroidal wrapping. The interior columns use direct indexing:

```text
adult[x-2] + adult[x-1] + adult[x] + adult[x+1] + adult[x+2]
```

**Chosen over** using `& (N - 1)` wrap arithmetic for every column.

Reason: almost every cell is an interior cell. On large grids, only four columns
per row need wrap handling. Splitting edges from the interior keeps the same
stencil and the same O(generations * cells) behavior, but removes unnecessary
wrap arithmetic from the hot path.

An important compiler detail: version 09's rolling horizontal sum has a
loop-carried dependency:

```cpp
h = h + adult[add_x] - adult[remove_x];
out[x] = h;
```

Each iteration needs the previous iteration's `h`, which limits instruction
level parallelism and makes autovectorization harder. Version 10's interior loop
computes each output independently:

```cpp
out[x] = adult[x - 2] + adult[x - 1] + adult[x]
       + adult[x + 1] + adult[x + 2];
```

Even though this performs more direct loads, the access pattern is simple, has no
wrap arithmetic in the hot path, and has no dependency chain between neighboring
outputs. That gives the compiler more room to schedule instructions and
potentially vectorize the loop.

## 11 Branchless Transition Logic

Version 11 replaces the scalar transition lookup table with compare/select
logic:

```text
EMPTY    -> EGG   if 3 <= A <= 5
EGG      -> JUVENILE
JUVENILE -> ADULT
ADULT    -> ADULT if 4 <= A <= 9
```

**Chosen over** `TRANSITION[cell][adult_count]`.

Reason: profiling v10 showed horizontal-sum construction was only about 10% of
runtime, while the vertical combine plus transition step was about 90%. The
lookup table is branchless, but it is still a scalar indexed load per cell. The
branchless compare/select form exposes the rule as arithmetic and comparisons,
which is easier for the compiler to optimize and is a better shape for explicit
SIMD later.
 
## Keep Binary Cell Storage For Now (Need to check this after we implement SIMD. Not sure which one might be better)

The current versions keep one byte per cell with states `0..3`.

Chosen over: bit-packing or separate state planes.

Reason: byte storage matches the input/output format, keeps transitions simple,
and avoids conversion complexity while the algorithm is still evolving. A future
SIMD version may introduce separate adult masks or packed representations if
benchmarks show the conversion cost pays off.
