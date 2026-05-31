// vb24: vb23 ring buffer + K-temporal + COLUMN TILING (2D diamond).
//
// Adds column tiling on top of vb23. Per generation group, each thread iterates
// its slab in narrow column strips of TILE_W_REGS NEON registers (default 8).
// Each strip's local buffer fits L3 (or L2) where vb23's full-width local
// buffer didn't, so the K=N gens inside each strip run from L2/L3 instead of
// DRAM. That's the architectural change that should make temporal blocking
// actually pay at 32K x 8 threads (the regime where vb23 spilled L2/L3).
//
// Strip layout per row in the local buffer:
//   [ghost_L (16 bytes) | data (TILE_W_REGS * 16 bytes) | ghost_R (16 bytes)]
// Ghost is oversized (16 bytes = 128 cells) to cover any K up to ~32. NEON
// alignment maintained. The kernel reads prev/next registers from the ghost
// areas directly, no toroidal wrap inside the strip.
//
// Strip boundary (column wrap):
//   Ghost copy reads from shared src with column wrap, so the strip kernel
//   sees the correct neighbouring cells regardless of position in the row.
//
// Build (target):
//   g++-14 -std=c++23 -Ofast -mcpu=neoverse-v2+sha3 -pthread \
//          [-DSPAWN_K=8] [-DSPAWN_TILE_W_REGS=8] \
//          bitplane_versions/24_diamond_tile.cpp -o /tmp/vb24
//
// === vb23 header below ===
//
// vb23: vb22 ring buffer + K=4 temporal blocking.
//
// Architecture: outer driver runs K=4 generations per "gen group" on a per-thread
// local slab buffer instead of one generation at a time on the global double-buffer.
// Each thread:
//   1. Copies its slab + 2K-row ghost margin from shared src into a local buffer.
//   2. Runs K=4 gens locally with the ring-buffer kernel, shrinking window each gen.
//   3. Copies the central slab back to shared dst.
//
// Why this should work where vb18 failed:
//   vb18 used the block-H kernel (1.55 MiB scratch) with K=4 temporal blocking.
//   The block-H scratch fought the K-local slab buffer (~33 MiB at 32K) for L2,
//   and the hot loop slowed down catastrophically.
//
//   vb23 uses the ring buffer (~72 KiB scratch). The ring fits L1; the local slab
//   sits in L2/L3 but only the active 5 rows (~20 KiB) are touched per output row.
//   No L2 fight between scratch and slab. Across K=4 gens each row of the slab
//   gets touched K times, mostly hitting L2 instead of DRAM.
//
// Validity window inside the local buffer at gen k (1-indexed):
//   write window = [2k, local_h - 2k)
//   read window  = [2k - 2, local_h - 2k + 2)  (5-row stencil)
// At gen K=4, write window collapses to [2K, 2K + slab_rows) = slab_rows correct rows.
//
// Build (target):
//   g++-14 -std=c++23 -Ofast -mcpu=neoverse-v2+sha3 -pthread \
//          bitplane_versions/23_ring_k4_temporal.cpp -o /tmp/vb23
//
// Default K is compile-time tunable: -DSPAWN_K=N (range tested: 2, 4, 8).
//
// === vb22 header below ===
//
// vb22: vb17 + ring-buffer streaming kernel.
//
// Replaces vb17's block-H scratch (BLOCK_ROWS+4 H rows materialised in a
// 1.55 MiB block per thread at 32K) with a streaming 6-slot ring buffer.
// Per generation, each thread:
//
//   1. Pre-fill ring slots 0..4 with H_{y0-2..y0+2} (5 compute_H_row calls).
//   2. V := sum of ring slots 0..4. Apply rule to dst row y0.
//   3. For each output row y = y0+1..y1-1:
//        a. new_slot := (tail + 5) % 6
//        b. compute_H_row(src.row(y+2)) → ring[new_slot]
//        c. Slide loop per r (2-way unrolled): V -= ring[tail][r],
//           V += ring[new_slot][r], apply rule, dst write.
//        d. tail = (tail + 1) % 6.
//
// Slot indexing invariant: at the start of iter y, the 5 valid H rows live
// in slots {tail, tail+1, ..., tail+4} mod 6, holding H_{y-3..y+1}. Slot
// (tail+5) mod 6 is free for the new H_{y+2}. tail and new_slot are exactly
// 5 apart mod 6, so they never alias — read and write are independent.
//
// Hot scratch per thread:
//   - H ring     = 6 slots * 3 planes * row_bytes = 72 KiB at 32K.
//   - V interleav= 5 * row_bytes                 = 20 KiB at 32K.
//   - h_temp     = NONE (compute_H_row writes directly into the ring slot).
//   Total ~92 KiB; spills L1 (64 KiB) slightly, comfortably in L2.
//
// Versus vb17:
//   - Same algorithm: same sliding 5x5 window over carry-save H+V planes.
//   - Same compute_H_row, sum_of_5, add_v5_h3, sub_v5_h3, apply_rule_byte.
//   - Same boolean-minimised apply (vbsl-MUX born, vbcaxq sub_v5_h3 tail,
//     vsri/vsli horizontal shift fusion).
//   - Same 2-way r-unroll, persistent thread pool, huge-page hints,
//     64-byte aligned storage.
//   - NO block-H scratch (deleted). NO Pass-1/Pass-2/Slide block structure.
//   - One compute_H_row per output row (was: amortised over BLOCK_ROWS).
//
// What this trades:
//   - Scratch: -1.48 MiB per thread (1.55 MiB block-H → 72 KiB ring).
//   - compute_H_row calls per gen: same (each src row's H computed once).
//   - L1 hit rate on H scratch: was ~95% (block-H spilled L1 hard), now
//     expected close to 100% (ring fits L1+L2 cleanly).
//   - BLOCK_ROWS define still works (compile-time tunable), but it has no
//     effect on the kernel now since there's no block loop.
//
// Build (target):
//   g++-14 -std=c++23 -Ofast -mcpu=neoverse-v2+sha3 -pthread \
//          bitplane_versions/22_ring_buffer.cpp -o /tmp/vb22
//
// === vb17 header below ===
//
// vb17: t23 + three orthogonal local op-count reductions.
//
// All changes are local to per-cell arithmetic. No scratch-layout, threading,
// or algorithmic change. Block-H, V-interleaved scratch, hsplit wrap, 2-way
// r-unroll, persistent thread pool, huge-page hints — all stay verbatim.
//
//   (1) vsri_n_u8 / vsli_n_u8 in horizontal_window_sum.
//       The four shifted versions of curr (m2, m1, p1, p2) were each built as
//       `vorrq(vshlq(curr,k), vshrq(carry, 8-k))` — 3 ops. Folded into
//       `vsriq_n_u8(vshlq(curr,k), carry, 8-k)` — 2 ops. The shift-and-insert
//       form preserves the high (8-k) bits of the shifted value and inserts
//       carry>>(8-k) into the low k bits without disturbing them. Saves 4
//       ops per row-sum (one per shifted version, c stays as curr).
//
//   (2) Karnaugh-minimised apply_rule_byte.
//       Original: 25 boolean ops + 4 vmvnq (the NOTs add 4 extra issue slots
//       and produce a dep chain of inversions early in the function).
//       New: 17 boolean ops, no vmvnq. Key compressions:
//
//         born     = ~(c4|c3) & (c1 ? (~c2 & c0) : c2)
//                  = vbicq(vbslq(c1, vbicq(c0,c2), c2), vorrq(c4,c3))   [4 ops]
//
//         alive_v  = ~c4 & ((~c3 & c2 & (c1|c0)) | (c3 & ~c2 & ~(c1&c0)))
//                  = vbicq(vorrq(vandq(vbicq(c2,c3), vorrq(c1,c0)),
//                                vbicq(vbicq(c3,c2), vandq(c1,c0))), c4) [7 ops]
//
//         next_high = (high ^ low) | adult_r  ; disjoint, so OR == XOR.
//                   = veor3q_u8(high, low, adult_r)                    [1 op]
//
//         next_low  = ((high|born) & ~low) | adult_r ; also disjoint.
//                   = vbcaxq_u8(adult_r, vorrq(high, born), low)        [2 ops]
//
//       Removes all 4 vmvnq instructions; every ~X is folded into a vbicq at
//       its use site via De Morgan.
//
//   (3) vbcaxq_u8 fusion in sub_v5_h3 final stage.
//       Original final stage:
//           borrow3 = vbicq_u8(borrow2, v.b3);   // borrow2 & ~v.b3
//           diff4   = veorq_u8(v.b4, borrow3);   // v.b4 ^ that
//       Folded:
//           diff4   = vbcaxq_u8(v.b4, borrow2, v.b3);  // v.b4 ^ (borrow2 & ~v.b3)
//       Saves 1 op per slide call.
//
// Build (target):
//   g++-14 -std=c++23 -Ofast -mcpu=neoverse-v2+sha3 -pthread \
//          -DSPAWN_BLOCK_ROWS=88 \
//          bitplane_versions/17_boolean_min.cpp -o /tmp/vb17_b88
//
// === Original t23 header below ===
//
// Temp experiment 23: temp-exp/17 + split H-row wrap block.
// block size + __restrict__ qualifiers.
//
// This keeps temp-exp/17's V-interleaved hot loop. The only kernel change is
// in compute_H_row: the common 4-register unrolled loop stops before the
// wraparound group, then handles the final group separately. That removes the
// per-iteration `(r + 4 == R_REGS) ? 0 : r + 4` select from the H-fill loop.
//
// The vertical count scratch is stored as per-register V5 records:
//   v0[r], v1[r], v2[r], v3[r], v4[r]
// instead of five full-row bitplanes. The hot loop loads/stores all five V
// vectors for a register every output row, so adjacency may reduce L1 set
// pressure and improve load/store pairing.
//
//   1. SOFTWARE-UNROLL the inner r-loop by 2 in the three places that walk
//      registers across a row:
//        - Pass 2 (initial V = sum of H[0..4])
//        - first-row apply
//        - slide loop body (the hottest path)
//      Processing two consecutive registers per iteration keeps the
//      row-major sequential access pattern (cache-friendly, prefetcher-
//      friendly) and gives the compiler two independent slide+apply chains
//      that can interleave across the 4 ASIMD pipes. Register pressure peak
//      is ~25 NEON regs (10 for two V5s, ~12 for two Sum3 pairs, working
//      temps in apply_rule_byte); fits in 32.
//
//      Per-r tail loop handles the case R_REGS is odd. For all valid
//      grading sizes (N >= 128, power of two), R_REGS is a power of two and
//      the tail never triggers.
//
//   2. COMPILE-TIME BLOCK_ROWS override via -DSPAWN_BLOCK_ROWS=N. Lets us
//      sweep block sizes without source edits:
//        default 128 -> 132 H rows * 3 planes * 4096 bytes = 1.62 MiB scratch
//                       (fits 2 MiB L2 with headroom)
//        64          -> ~820 KiB scratch (smaller, more block overhead per gen)
//        256         -> ~3.2 MiB scratch (spills L2; expected to regress)
//
//   3. __restrict__ on compute_H_row's pointer parameters. These pointers
//      (low_row, high_row, h0_out, h1_out, h2_out) provably don't alias each
//      other; the explicit annotation removes any conservative aliasing
//      checks from compiler analysis. Marginal but free.
//
// What stays the same as temp-exp/01:
//   - byte-packed uint8x16_t storage
//   - tree-style sum_of_5 with FA/HA/maj/EOR3
//   - vextq_u8 byte-granular shifts
//   - compact birth/survive predicate
//   - block-H scratch
//   - persistent per-thread scratch + hugepages + 64-byte aligned storage
//   - row-major (k outer, r inner) access pattern
//
// What we deliberately don't add (per docs and the vb13.6 regression):
//   - CPU pinning (docs: neutral under taskset, slightly negative in one run)
//   - Loop-swap to r outer (vb13.6: 92 % regression, 138 s -> 265 s, cache
//     prefetcher couldn't follow the 12 KiB column stride)
//   - Temporal stripe (vb11/vb12 on uint64x2 didn't beat block-H, unlikely
//     to help here either)
//
// Build (target, default block size 128):
//   g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
//          temp-exp/17_byte_tree_v_interleaved.cpp -o /tmp/t17
//
// Sweep block sizes:
//   g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
//          -DSPAWN_BLOCK_ROWS=64 \
//          temp-exp/17_byte_tree_v_interleaved.cpp -o /tmp/t17_b64
//
//   g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
//          -DSPAWN_BLOCK_ROWS=256 \
//          temp-exp/17_byte_tree_v_interleaved.cpp -o /tmp/t17_b256

#include <algorithm>
#include <arm_neon.h>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unistd.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

static constexpr uint8_t EMPTY    = 0;
static constexpr uint8_t EGG      = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT    = 3;

#ifndef SPAWN_K
#define SPAWN_K 8
#endif
static constexpr int K_GENS = SPAWN_K;

#ifndef SPAWN_TILE_W_REGS
#define SPAWN_TILE_W_REGS 8    // 8 NEON regs = 1024 cells = 128 bytes/plane data
#endif
static constexpr int TILE_W_REGS = SPAWN_TILE_W_REGS;
static constexpr int TILE_W_DATA_BYTES = TILE_W_REGS * 16;
static constexpr int TILE_W_GHOST_BYTES = 16;       // 1 NEON reg ghost each side (128 cells)
static constexpr int TILE_W_LOCAL_BYTES = TILE_W_DATA_BYTES + 2 * TILE_W_GHOST_BYTES;
static constexpr int TILE_W_DATA_OFFSET = TILE_W_GHOST_BYTES;   // where data starts in local row

#ifndef SPAWN_BLOCK_ROWS
#define SPAWN_BLOCK_ROWS 128
#endif
static constexpr int BLOCK_ROWS = SPAWN_BLOCK_ROWS;

// =====================================================================
// NEON helpers (identical to vb13.5)
// =====================================================================

static inline uint8x16_t vxor3_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c)
{
#if defined(__ARM_FEATURE_SHA3)
    return veor3q_u8(a, b, c);
#else
    return veorq_u8(veorq_u8(a, b), c);
#endif
}

static inline uint8x16_t shl_row_1(uint8x16_t prev, uint8x16_t curr)
{
    const uint8x16_t carry = vextq_u8(prev, curr, 15);
    return vorrq_u8(vshlq_n_u8(curr, 1), vshrq_n_u8(carry, 7));
}
static inline uint8x16_t shl_row_2(uint8x16_t prev, uint8x16_t curr)
{
    const uint8x16_t carry = vextq_u8(prev, curr, 15);
    return vorrq_u8(vshlq_n_u8(curr, 2), vshrq_n_u8(carry, 6));
}
static inline uint8x16_t shr_row_1(uint8x16_t curr, uint8x16_t next)
{
    const uint8x16_t carry = vextq_u8(curr, next, 1);
    return vorrq_u8(vshrq_n_u8(curr, 1), vshlq_n_u8(carry, 7));
}
static inline uint8x16_t shr_row_2(uint8x16_t curr, uint8x16_t next)
{
    const uint8x16_t carry = vextq_u8(curr, next, 1);
    return vorrq_u8(vshrq_n_u8(curr, 2), vshlq_n_u8(carry, 6));
}

static inline uint8x16_t maj_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c)
{
    return vbslq_u8(veorq_u8(a, b), c, vandq_u8(a, b));
}

struct FA { uint8x16_t s, c; };
static inline FA full_adder_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c)
{
    return { vxor3_u8(a, b, c), maj_u8(a, b, c) };
}
struct HA { uint8x16_t s, c; };
static inline HA half_adder_u8(uint8x16_t a, uint8x16_t b)
{
    return { veorq_u8(a, b), vandq_u8(a, b) };
}

struct Sum3 { uint8x16_t h0, h1, h2; };
static inline Sum3 sum_of_5(uint8x16_t a, uint8x16_t b, uint8x16_t c,
                            uint8x16_t d, uint8x16_t e)
{
    FA fa1 = full_adder_u8(a, b, c);
    FA fa2 = full_adder_u8(d, e, fa1.s);
    HA ha  = half_adder_u8(fa1.c, fa2.c);
    return { fa2.s, ha.s, ha.c };
}

static inline Sum3 horizontal_window_sum(uint8x16_t prev, uint8x16_t curr, uint8x16_t next)
{
    const uint8x16_t left_carry  = vextq_u8(prev, curr, 15);
    const uint8x16_t right_carry = vextq_u8(curr, next, 1);
    // s_m2: high 6 bits = curr<<2, low 2 bits = left_carry>>6.
    //       vsri inserts low 2 bits without touching the high 6.
    const uint8x16_t s_m2 = vsriq_n_u8(vshlq_n_u8(curr, 2), left_carry, 6);
    const uint8x16_t s_m1 = vsriq_n_u8(vshlq_n_u8(curr, 1), left_carry, 7);
    // s_p1: low 7 bits = curr>>1, high 1 bit = right_carry<<7.
    //       vsli inserts high 1 bit without touching the low 7.
    const uint8x16_t s_p1 = vsliq_n_u8(vshrq_n_u8(curr, 1), right_carry, 7);
    const uint8x16_t s_p2 = vsliq_n_u8(vshrq_n_u8(curr, 2), right_carry, 6);
    return sum_of_5(s_m2, s_m1, curr, s_p1, s_p2);
}

struct V5 { uint8x16_t b0, b1, b2, b3, b4; };
static inline V5 add_v5_h3(V5 v, Sum3 h)
{
    HA s0 = half_adder_u8(v.b0, h.h0);
    FA s1 = full_adder_u8(v.b1, h.h1, s0.c);
    FA s2 = full_adder_u8(v.b2, h.h2, s1.c);
    HA s3 = half_adder_u8(v.b3, s2.c);
    const uint8x16_t b4 = veorq_u8(v.b4, s3.c);
    return { s0.s, s1.s, s2.s, s3.s, b4 };
}
static inline V5 sub_v5_h3(V5 v, Sum3 h)
{
    const uint8x16_t diff0   = veorq_u8(v.b0, h.h0);
    const uint8x16_t borrow0 = vbicq_u8(h.h0, v.b0);
    const uint8x16_t diff1   = vxor3_u8(v.b1, h.h1, borrow0);
    const uint8x16_t borrow1 = maj_u8(vmvnq_u8(v.b1), h.h1, borrow0);
    const uint8x16_t diff2   = vxor3_u8(v.b2, h.h2, borrow1);
    const uint8x16_t borrow2 = maj_u8(vmvnq_u8(v.b2), h.h2, borrow1);
    const uint8x16_t diff3   = veorq_u8(v.b3, borrow2);
    // diff4 = v.b4 ^ (borrow2 & ~v.b3) — folds vbicq + veorq into vbcaxq.
    const uint8x16_t diff4   = vbcaxq_u8(v.b4, borrow2, v.b3);
    return { diff0, diff1, diff2, diff3, diff4 };
}

static inline V5 load_v5_interleaved(const uint8_t* v_data, size_t reg_byte_off)
{
    const size_t off = reg_byte_off * 5;
    return {
        vld1q_u8(v_data + off + 0),
        vld1q_u8(v_data + off + 16),
        vld1q_u8(v_data + off + 32),
        vld1q_u8(v_data + off + 48),
        vld1q_u8(v_data + off + 64),
    };
}

static inline void store_v5_interleaved(uint8_t* v_data, size_t reg_byte_off, V5 v)
{
    const size_t off = reg_byte_off * 5;
    vst1q_u8(v_data + off + 0, v.b0);
    vst1q_u8(v_data + off + 16, v.b1);
    vst1q_u8(v_data + off + 32, v.b2);
    vst1q_u8(v_data + off + 48, v.b3);
    vst1q_u8(v_data + off + 64, v.b4);
}

struct LH { uint8x16_t low, high; };
// Karnaugh-minimised apply_rule_byte. 17 boolean ops, no vmvnq.
// V5 = (c4,c3,c2,c1,c0) = bit-planes of A_full (25-cell sum incl centre).
//   born     fires on EMPTY (low=0,high=0) cells with A_full ∈ {3,4,5}.
//   survives fires on ADULT (low=1,high=1) cells with A_full ∈ {5..10}.
// born * survives bit-encodings minimised:
//   born     = ~(c4|c3) & (c1 ? (~c2 & c0) : c2)
//   alive_v  = ~c4 & ((~c3 & c2 & (c1|c0)) | (c3 & ~c2 & ~(c1&c0)))
static inline LH apply_rule_byte(V5 v, uint8x16_t low, uint8x16_t high)
{
    // born: 4 ops.
    const uint8x16_t c0_andn_c2 = vbicq_u8(v.b0, v.b2);                  // ~c2 & c0
    const uint8x16_t born_mux   = vbslq_u8(v.b1, c0_andn_c2, v.b2);      // c1 ? (~c2&c0) : c2
    const uint8x16_t c4_or_c3   = vorrq_u8(v.b4, v.b3);
    const uint8x16_t born       = vbicq_u8(born_mux, c4_or_c3);          // & ~(c4|c3)

    // alive_v: 7 ops.
    const uint8x16_t c2_andn_c3 = vbicq_u8(v.b2, v.b3);                  // ~c3 & c2
    const uint8x16_t c1_or_c0   = vorrq_u8(v.b1, v.b0);
    const uint8x16_t surv_lo    = vandq_u8(c2_andn_c3, c1_or_c0);        // ~c3 & c2 & (c1|c0)
    const uint8x16_t c3_andn_c2 = vbicq_u8(v.b3, v.b2);                  // c3 & ~c2
    const uint8x16_t c1_and_c0  = vandq_u8(v.b1, v.b0);
    const uint8x16_t surv_hi    = vbicq_u8(c3_andn_c2, c1_and_c0);       // c3 & ~c2 & ~(c1&c0)
    const uint8x16_t alive_v    = vbicq_u8(vorrq_u8(surv_lo, surv_hi), v.b4); // & ~c4

    // adult, adult_r: 2 ops.
    const uint8x16_t adult   = vandq_u8(low, high);
    const uint8x16_t adult_r = vandq_u8(adult, alive_v);

    // next_high = (high ^ low) | adult_r. The XOR(high,low) bit is 0 wherever
    // adult_r can be 1 (adult_r ⊆ low & high, where XOR(high,low) = 0), so
    // OR ≡ XOR. Collapse to a single veor3q_u8.
    const uint8x16_t next_high = veor3q_u8(high, low, adult_r);            // 1 op

    // next_low = ((high|born) & ~low) | adult_r. Again disjoint: adult_r ⊆ low,
    // (high|born) & ~low forces ~low=1 so adult_r = 0 in those positions.
    // adult_r ^ ((high|born) & ~low) ≡ bcax(adult_r, high|born, low).
    const uint8x16_t high_or_born = vorrq_u8(high, born);
    const uint8x16_t next_low     = vbcaxq_u8(adult_r, high_or_born, low); // 2 ops

    return { next_low, next_high };
}

// =====================================================================
// Hugepage hint
// =====================================================================

static void advise_huge_pages(void* ptr, size_t bytes)
{
#ifdef __linux__
    if (ptr == nullptr || bytes == 0) {
        return;
    }
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return;
    }
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t base = addr & ~(static_cast<uintptr_t>(page) - 1);
    const uintptr_t end = (addr + bytes + static_cast<uintptr_t>(page) - 1)
                        & ~(static_cast<uintptr_t>(page) - 1);
    (void)madvise(reinterpret_cast<void*>(base), end - base, MADV_HUGEPAGE);
#else
    (void)ptr;
    (void)bytes;
#endif
}

// =====================================================================
// 64-byte aligned allocator (verbatim from vb13.5)
// =====================================================================

template <class T, size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(size_type n)
    {
        if (n > size_type(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }
        const size_t bytes = n * sizeof(T);
        const size_t rounded = (bytes + Alignment - 1) & ~(Alignment - 1);
        void* p = std::aligned_alloc(Alignment, rounded);
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { std::free(p); }
};

template <class T, class U, size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept { return true; }
template <class T, class U, size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept { return false; }

using AlignedBytes = std::vector<uint8_t, AlignedAllocator<uint8_t, 64>>;

// =====================================================================
// BitGrid (byte-packed)
// =====================================================================

struct BitGrid {
    int n = 0;
    int row_bytes = 0;
    AlignedBytes s0;
    AlignedBytes s1;

    void resize(int side)
    {
        n = side;
        row_bytes = n / 8;
        s0.assign((size_t)n * row_bytes, 0);
        s1.assign((size_t)n * row_bytes, 0);
        advise_huge_pages(s0.data(), s0.size());
        advise_huge_pages(s1.data(), s1.size());
    }

    const uint8_t* row0(int y) const { return s0.data() + (size_t)y * row_bytes; }
    const uint8_t* row1(int y) const { return s1.data() + (size_t)y * row_bytes; }
    uint8_t* row0(int y) { return s0.data() + (size_t)y * row_bytes; }
    uint8_t* row1(int y) { return s1.data() + (size_t)y * row_bytes; }
};

// =====================================================================
// Bytes <-> bitgrid conversion (one-time, scalar is fine)
// =====================================================================

static void bytes_to_bitgrid(const std::vector<uint8_t>& cells, BitGrid& out)
{
    const int N = out.n;
    const int row_bytes = out.row_bytes;
    for (int y = 0; y < N; ++y) {
        uint8_t* s0 = out.row0(y);
        uint8_t* s1 = out.row1(y);
        for (int b = 0; b < row_bytes; ++b) {
            uint8_t lo = 0, hi = 0;
            const size_t base = (size_t)y * N + (size_t)b * 8;
            for (int j = 0; j < 8; ++j) {
                const uint8_t cell = cells[base + j];
                lo |= (uint8_t)((cell & 1u) << j);
                hi |= (uint8_t)(((cell >> 1) & 1u) << j);
            }
            s0[b] = lo;
            s1[b] = hi;
        }
    }
}

static void bitgrid_to_bytes(const BitGrid& in, std::vector<uint8_t>& cells)
{
    const int N = in.n;
    const int row_bytes = in.row_bytes;
    cells.assign((size_t)N * N, 0);
    for (int y = 0; y < N; ++y) {
        const uint8_t* s0 = in.row0(y);
        const uint8_t* s1 = in.row1(y);
        for (int b = 0; b < row_bytes; ++b) {
            const uint8_t lo = s0[b];
            const uint8_t hi = s1[b];
            const size_t base = (size_t)y * N + (size_t)b * 8;
            for (int j = 0; j < 8; ++j) {
                cells[base + j] = (uint8_t)(((lo >> j) & 1u) | (((hi >> j) & 1u) << 1));
            }
        }
    }
}

// =====================================================================
// H-row tree adder. __restrict__ on the 5 pointers — they never alias.
// =====================================================================

static inline void compute_H_row(const uint8_t* __restrict__ low_row,
                                 const uint8_t* __restrict__ high_row,
                                 int R_REGS,
                                 uint8_t* __restrict__ h0_out,
                                 uint8_t* __restrict__ h1_out,
                                 uint8_t* __restrict__ h2_out)
{
    auto load_adult = [&](int r) -> uint8x16_t {
        const uint8x16_t lo = vld1q_u8(low_row  + r * 16);
        const uint8x16_t hi = vld1q_u8(high_row + r * 16);
        return vandq_u8(lo, hi);
    };

    uint8x16_t prev = load_adult(R_REGS - 1);
    uint8x16_t curr = load_adult(0);
    int r = 0;
    for (; r + 4 < R_REGS; r += 4) {
        const int r1 = r + 1;
        const int r2 = r + 2;
        const int r3 = r + 3;
        const int r4 = r + 4;
        const uint8x16_t a1 = load_adult(r1);
        const uint8x16_t a2 = load_adult(r2);
        const uint8x16_t a3 = load_adult(r3);
        const uint8x16_t a4 = load_adult(r4);

        const Sum3 h0 = horizontal_window_sum(prev, curr, a1);
        const Sum3 h1 = horizontal_window_sum(curr, a1, a2);
        const Sum3 h2 = horizontal_window_sum(a1, a2, a3);
        const Sum3 h3 = horizontal_window_sum(a2, a3, a4);

        vst1q_u8(h0_out + r * 16, h0.h0);
        vst1q_u8(h1_out + r * 16, h0.h1);
        vst1q_u8(h2_out + r * 16, h0.h2);
        vst1q_u8(h0_out + (r + 1) * 16, h1.h0);
        vst1q_u8(h1_out + (r + 1) * 16, h1.h1);
        vst1q_u8(h2_out + (r + 1) * 16, h1.h2);
        vst1q_u8(h0_out + (r + 2) * 16, h2.h0);
        vst1q_u8(h1_out + (r + 2) * 16, h2.h1);
        vst1q_u8(h2_out + (r + 2) * 16, h2.h2);
        vst1q_u8(h0_out + (r + 3) * 16, h3.h0);
        vst1q_u8(h1_out + (r + 3) * 16, h3.h1);
        vst1q_u8(h2_out + (r + 3) * 16, h3.h2);

        prev = a3;
        curr = a4;
    }
    if (r + 3 < R_REGS) {
        const int r1 = r + 1;
        const int r2 = r + 2;
        const int r3 = r + 3;
        const uint8x16_t a1 = load_adult(r1);
        const uint8x16_t a2 = load_adult(r2);
        const uint8x16_t a3 = load_adult(r3);
        const uint8x16_t a4 = load_adult(0);

        const Sum3 h0 = horizontal_window_sum(prev, curr, a1);
        const Sum3 h1 = horizontal_window_sum(curr, a1, a2);
        const Sum3 h2 = horizontal_window_sum(a1, a2, a3);
        const Sum3 h3 = horizontal_window_sum(a2, a3, a4);

        vst1q_u8(h0_out + r * 16, h0.h0);
        vst1q_u8(h1_out + r * 16, h0.h1);
        vst1q_u8(h2_out + r * 16, h0.h2);
        vst1q_u8(h0_out + (r + 1) * 16, h1.h0);
        vst1q_u8(h1_out + (r + 1) * 16, h1.h1);
        vst1q_u8(h2_out + (r + 1) * 16, h1.h2);
        vst1q_u8(h0_out + (r + 2) * 16, h2.h0);
        vst1q_u8(h1_out + (r + 2) * 16, h2.h1);
        vst1q_u8(h2_out + (r + 2) * 16, h2.h2);
        vst1q_u8(h0_out + (r + 3) * 16, h3.h0);
        vst1q_u8(h1_out + (r + 3) * 16, h3.h1);
        vst1q_u8(h2_out + (r + 3) * 16, h3.h2);

        prev = a3;
        curr = a4;
        r += 4;
    }
    for (; r < R_REGS; ++r) {
        const int r_next = (r + 1 == R_REGS) ? 0 : (r + 1);
        const uint8x16_t next = load_adult(r_next);
        const Sum3 h = horizontal_window_sum(prev, curr, next);
        vst1q_u8(h0_out + r * 16, h.h0);
        vst1q_u8(h1_out + r * 16, h.h1);
        vst1q_u8(h2_out + r * 16, h.h2);
        prev = curr;
        curr = next;
    }
}

// =====================================================================
// Per-thread persistent scratch (same as vb13.5)
// =====================================================================

struct ThreadScratch {
    AlignedBytes h_ring;        // 6 slots × 3 planes × row_bytes
    AlignedBytes vcount_store;  // 5 × row_bytes (V-interleaved)
    uint8_t* v_b0 = nullptr;
    uint8_t* v_b1 = nullptr;
    uint8_t* v_b2 = nullptr;
    uint8_t* v_b3 = nullptr;
    uint8_t* v_b4 = nullptr;

    // K-temporal: local double-buffered slab of (slab_rows + 4*K) rows per plane.
    // Used only inside the K-gen inner loop; toroidal wrap is resolved during the
    // ghost-fill copy and not by the kernel.
    AlignedBytes lbuf_a_low, lbuf_a_high;
    AlignedBytes lbuf_b_low, lbuf_b_high;
    int local_rows = 0;

    void resize(int /*row_bytes_full*/, int slab_rows_inclusive_ghost)
    {
        // Ring + V scratch sized for TILE_W_REGS strip, NOT full row.
        // For 2D diamond tiling, all strip operations work on TILE_W_DATA_BYTES
        // wide data. Same scratch is reused across all strips.
        const size_t strip_data_bytes = (size_t)TILE_W_DATA_BYTES;
        h_ring.resize(6 * 3 * strip_data_bytes);
        vcount_store.resize(5 * strip_data_bytes);
        advise_huge_pages(h_ring.data(), h_ring.size());
        advise_huge_pages(vcount_store.data(), vcount_store.size());
        v_b0 = vcount_store.data() + 0 * strip_data_bytes;
        v_b1 = vcount_store.data() + 1 * strip_data_bytes;
        v_b2 = vcount_store.data() + 2 * strip_data_bytes;
        v_b3 = vcount_store.data() + 3 * strip_data_bytes;
        v_b4 = vcount_store.data() + 4 * strip_data_bytes;

        // Local buffer per strip: (slab_h + 4K) rows * TILE_W_LOCAL_BYTES wide
        // (which includes 16 bytes of column ghost on each side).
        local_rows = slab_rows_inclusive_ghost;
        const size_t bytes = (size_t)local_rows * (size_t)TILE_W_LOCAL_BYTES;
        lbuf_a_low.resize(bytes);
        lbuf_a_high.resize(bytes);
        lbuf_b_low.resize(bytes);
        lbuf_b_high.resize(bytes);
        advise_huge_pages(lbuf_a_low.data(),  bytes);
        advise_huge_pages(lbuf_a_high.data(), bytes);
        advise_huge_pages(lbuf_b_low.data(),  bytes);
        advise_huge_pages(lbuf_b_high.data(), bytes);
    }
};

// =====================================================================
// Block-H step kernel — same control flow as vb13.5, with 2-way r-unrolled
// inner loops in Pass 2, first-row apply, and the slide path.
// =====================================================================

static void step_rows_bitplane(const BitGrid& src, BitGrid& dst,
                               int y0, int y1, ThreadScratch& scratch)
{
    const int N = src.n;
    const int row_bytes = src.row_bytes;
    const int R_REGS = row_bytes / 16;
    const int ymask = N - 1;

    uint8_t* ring   = scratch.h_ring.data();
    uint8_t* v_data = scratch.vcount_store.data();

    // Ring slot pointers: 6 slots, each with 3 planes (h0, h1, h2).
    auto h0p = [&](int slot) { return ring + (size_t)(3 * slot + 0) * row_bytes; };
    auto h1p = [&](int slot) { return ring + (size_t)(3 * slot + 1) * row_bytes; };
    auto h2p = [&](int slot) { return ring + (size_t)(3 * slot + 2) * row_bytes; };

    // -------- Pre-fill ring slots 0..4 with H_{y0-2..y0+2}. --------
    for (int s = 0; s < 5; ++s) {
        const int src_y = (y0 - 2 + s + N) & ymask;
        compute_H_row(src.row0(src_y), src.row1(src_y), R_REGS,
                      h0p(s), h1p(s), h2p(s));
    }

    // -------- Initialise V = sum of ring slots 0..4. 2-way unrolled in r. --------
    {
        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;
            V5 v0 = {
                vld1q_u8(h0p(0) + off0),
                vld1q_u8(h1p(0) + off0),
                vld1q_u8(h2p(0) + off0),
                vdupq_n_u8(0),
                vdupq_n_u8(0),
            };
            V5 v1 = {
                vld1q_u8(h0p(0) + off1),
                vld1q_u8(h1p(0) + off1),
                vld1q_u8(h2p(0) + off1),
                vdupq_n_u8(0),
                vdupq_n_u8(0),
            };
            for (int s = 1; s < 5; ++s) {
                const Sum3 h0 = {
                    vld1q_u8(h0p(s) + off0),
                    vld1q_u8(h1p(s) + off0),
                    vld1q_u8(h2p(s) + off0),
                };
                const Sum3 h1 = {
                    vld1q_u8(h0p(s) + off1),
                    vld1q_u8(h1p(s) + off1),
                    vld1q_u8(h2p(s) + off1),
                };
                v0 = add_v5_h3(v0, h0);
                v1 = add_v5_h3(v1, h1);
            }
            store_v5_interleaved(v_data, off0, v0);
            store_v5_interleaved(v_data, off1, v1);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = {
                vld1q_u8(h0p(0) + off),
                vld1q_u8(h1p(0) + off),
                vld1q_u8(h2p(0) + off),
                vdupq_n_u8(0),
                vdupq_n_u8(0),
            };
            for (int s = 1; s < 5; ++s) {
                const Sum3 h = {
                    vld1q_u8(h0p(s) + off),
                    vld1q_u8(h1p(s) + off),
                    vld1q_u8(h2p(s) + off),
                };
                v = add_v5_h3(v, h);
            }
            store_v5_interleaved(v_data, off, v);
        }
    }

    // -------- Apply rule to first output row (y = y0). 2-way unrolled in r. --------
    {
        const uint8_t* low_row  = src.row0(y0);
        const uint8_t* high_row = src.row1(y0);
        uint8_t* nl_row = dst.row0(y0);
        uint8_t* nh_row = dst.row1(y0);

        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;
            V5 v0 = load_v5_interleaved(v_data, off0);
            V5 v1 = load_v5_interleaved(v_data, off1);
            const uint8x16_t lo0 = vld1q_u8(low_row  + off0);
            const uint8x16_t hi0 = vld1q_u8(high_row + off0);
            const uint8x16_t lo1 = vld1q_u8(low_row  + off1);
            const uint8x16_t hi1 = vld1q_u8(high_row + off1);
            const LH out0 = apply_rule_byte(v0, lo0, hi0);
            const LH out1 = apply_rule_byte(v1, lo1, hi1);
            vst1q_u8(nl_row + off0, out0.low);
            vst1q_u8(nh_row + off0, out0.high);
            vst1q_u8(nl_row + off1, out1.low);
            vst1q_u8(nh_row + off1, out1.high);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = load_v5_interleaved(v_data, off);
            const uint8x16_t lo = vld1q_u8(low_row  + off);
            const uint8x16_t hi = vld1q_u8(high_row + off);
            const LH out = apply_rule_byte(v, lo, hi);
            vst1q_u8(nl_row + off, out.low);
            vst1q_u8(nh_row + off, out.high);
        }
    }

    // -------- Slide loop: stream H_{y+2} through ring, V slides one row at a time. --------
    // Invariant entering iter y:
    //   Ring slots {tail, tail+1, ..., tail+4} mod 6 hold H_{y-3, y-2, y-1, y, y+1}.
    //   V represents the count for window centred on (y-1) = sum H_{y-3..y+1}.
    //   Slot (tail+5) mod 6 is free (overwritable).
    int tail = 0;

    for (int y = y0 + 1; y < y1; ++y) {
        const int new_slot = (tail + 5) % 6;       // free slot — write new H_{y+2} here
        const int src_y_new = (y + 2 + N) & ymask;

        // Compute new H_{y+2} directly into ring[new_slot]. compute_H_row writes
        // through h0p/h1p/h2p which are __restrict__-disjoint from ring[tail].
        compute_H_row(src.row0(src_y_new), src.row1(src_y_new), R_REGS,
                      h0p(new_slot), h1p(new_slot), h2p(new_slot));

        // Slide path: V -= ring[tail], V += ring[new_slot]. Apply rule, dst write.
        const uint8_t* h_out_0 = h0p(tail);
        const uint8_t* h_out_1 = h1p(tail);
        const uint8_t* h_out_2 = h2p(tail);
        const uint8_t* h_in_0  = h0p(new_slot);
        const uint8_t* h_in_1  = h1p(new_slot);
        const uint8_t* h_in_2  = h2p(new_slot);
        const uint8_t* low_row  = src.row0(y);
        const uint8_t* high_row = src.row1(y);
        uint8_t* nl_row = dst.row0(y);
        uint8_t* nh_row = dst.row1(y);

        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;

            V5 v0 = load_v5_interleaved(v_data, off0);
            V5 v1 = load_v5_interleaved(v_data, off1);

            const Sum3 ho0 = {
                vld1q_u8(h_out_0 + off0),
                vld1q_u8(h_out_1 + off0),
                vld1q_u8(h_out_2 + off0),
            };
            const Sum3 ho1 = {
                vld1q_u8(h_out_0 + off1),
                vld1q_u8(h_out_1 + off1),
                vld1q_u8(h_out_2 + off1),
            };
            const Sum3 hi0 = {
                vld1q_u8(h_in_0 + off0),
                vld1q_u8(h_in_1 + off0),
                vld1q_u8(h_in_2 + off0),
            };
            const Sum3 hi1 = {
                vld1q_u8(h_in_0 + off1),
                vld1q_u8(h_in_1 + off1),
                vld1q_u8(h_in_2 + off1),
            };

            v0 = sub_v5_h3(v0, ho0);
            v1 = sub_v5_h3(v1, ho1);
            v0 = add_v5_h3(v0, hi0);
            v1 = add_v5_h3(v1, hi1);

            store_v5_interleaved(v_data, off0, v0);
            store_v5_interleaved(v_data, off1, v1);

            const uint8x16_t lo0 = vld1q_u8(low_row  + off0);
            const uint8x16_t hi0v = vld1q_u8(high_row + off0);
            const uint8x16_t lo1 = vld1q_u8(low_row  + off1);
            const uint8x16_t hi1v = vld1q_u8(high_row + off1);
            const LH outL = apply_rule_byte(v0, lo0, hi0v);
            const LH outR = apply_rule_byte(v1, lo1, hi1v);
            vst1q_u8(nl_row + off0, outL.low);
            vst1q_u8(nh_row + off0, outL.high);
            vst1q_u8(nl_row + off1, outR.low);
            vst1q_u8(nh_row + off1, outR.high);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = load_v5_interleaved(v_data, off);
            const Sum3 h_out = {
                vld1q_u8(h_out_0 + off),
                vld1q_u8(h_out_1 + off),
                vld1q_u8(h_out_2 + off),
            };
            const Sum3 h_in = {
                vld1q_u8(h_in_0 + off),
                vld1q_u8(h_in_1 + off),
                vld1q_u8(h_in_2 + off),
            };
            v = sub_v5_h3(v, h_out);
            v = add_v5_h3(v, h_in);
            store_v5_interleaved(v_data, off, v);

            const uint8x16_t lo = vld1q_u8(low_row  + off);
            const uint8x16_t hi = vld1q_u8(high_row + off);
            const LH out = apply_rule_byte(v, lo, hi);
            vst1q_u8(nl_row + off, out.low);
            vst1q_u8(nh_row + off, out.high);
        }

        tail = (tail + 1) % 6;
    }
}

// =====================================================================
// step_rows_bitplane_local — same ring-buffer kernel as step_rows_bitplane,
// but reads/writes by absolute buffer index with NO toroidal wrap.
// Used inside the K-gen inner loop where wrap is already resolved by the
// ghost-copy phase. Caller MUST ensure y0 - 2 >= 0 and y1 + 1 < local_h.
// =====================================================================

static void step_rows_bitplane_local(
    const uint8_t* __restrict__ src_low,
    const uint8_t* __restrict__ src_high,
    uint8_t* __restrict__ dst_low,
    uint8_t* __restrict__ dst_high,
    int row_bytes,
    int y0, int y1,
    ThreadScratch& scratch)
{
    const int R_REGS = row_bytes / 16;

    uint8_t* ring   = scratch.h_ring.data();
    uint8_t* v_data = scratch.vcount_store.data();

    auto h0p = [&](int slot) { return ring + (size_t)(3 * slot + 0) * row_bytes; };
    auto h1p = [&](int slot) { return ring + (size_t)(3 * slot + 1) * row_bytes; };
    auto h2p = [&](int slot) { return ring + (size_t)(3 * slot + 2) * row_bytes; };

    auto slow  = [&](int y) { return src_low  + (size_t)y * row_bytes; };
    auto shigh = [&](int y) { return src_high + (size_t)y * row_bytes; };
    auto dlow  = [&](int y) { return dst_low  + (size_t)y * row_bytes; };
    auto dhigh = [&](int y) { return dst_high + (size_t)y * row_bytes; };

    // Pre-fill ring slots 0..4 with H_{y0-2..y0+2}. No wrap.
    for (int s = 0; s < 5; ++s) {
        const int src_y = y0 - 2 + s;
        compute_H_row(slow(src_y), shigh(src_y), R_REGS,
                      h0p(s), h1p(s), h2p(s));
    }

    // Initialise V = sum of ring slots 0..4.
    {
        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;
            V5 v0 = {
                vld1q_u8(h0p(0) + off0), vld1q_u8(h1p(0) + off0), vld1q_u8(h2p(0) + off0),
                vdupq_n_u8(0), vdupq_n_u8(0),
            };
            V5 v1 = {
                vld1q_u8(h0p(0) + off1), vld1q_u8(h1p(0) + off1), vld1q_u8(h2p(0) + off1),
                vdupq_n_u8(0), vdupq_n_u8(0),
            };
            for (int s = 1; s < 5; ++s) {
                const Sum3 h0 = {
                    vld1q_u8(h0p(s) + off0), vld1q_u8(h1p(s) + off0), vld1q_u8(h2p(s) + off0),
                };
                const Sum3 h1 = {
                    vld1q_u8(h0p(s) + off1), vld1q_u8(h1p(s) + off1), vld1q_u8(h2p(s) + off1),
                };
                v0 = add_v5_h3(v0, h0);
                v1 = add_v5_h3(v1, h1);
            }
            store_v5_interleaved(v_data, off0, v0);
            store_v5_interleaved(v_data, off1, v1);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = {
                vld1q_u8(h0p(0) + off), vld1q_u8(h1p(0) + off), vld1q_u8(h2p(0) + off),
                vdupq_n_u8(0), vdupq_n_u8(0),
            };
            for (int s = 1; s < 5; ++s) {
                const Sum3 h = {
                    vld1q_u8(h0p(s) + off), vld1q_u8(h1p(s) + off), vld1q_u8(h2p(s) + off),
                };
                v = add_v5_h3(v, h);
            }
            store_v5_interleaved(v_data, off, v);
        }
    }

    // Apply rule to first output row (y = y0).
    {
        const uint8_t* low_row  = slow(y0);
        const uint8_t* high_row = shigh(y0);
        uint8_t* nl_row = dlow(y0);
        uint8_t* nh_row = dhigh(y0);

        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;
            V5 v0 = load_v5_interleaved(v_data, off0);
            V5 v1 = load_v5_interleaved(v_data, off1);
            const uint8x16_t lo0 = vld1q_u8(low_row  + off0);
            const uint8x16_t hi0 = vld1q_u8(high_row + off0);
            const uint8x16_t lo1 = vld1q_u8(low_row  + off1);
            const uint8x16_t hi1 = vld1q_u8(high_row + off1);
            const LH out0 = apply_rule_byte(v0, lo0, hi0);
            const LH out1 = apply_rule_byte(v1, lo1, hi1);
            vst1q_u8(nl_row + off0, out0.low);
            vst1q_u8(nh_row + off0, out0.high);
            vst1q_u8(nl_row + off1, out1.low);
            vst1q_u8(nh_row + off1, out1.high);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = load_v5_interleaved(v_data, off);
            const uint8x16_t lo = vld1q_u8(low_row  + off);
            const uint8x16_t hi = vld1q_u8(high_row + off);
            const LH out = apply_rule_byte(v, lo, hi);
            vst1q_u8(nl_row + off, out.low);
            vst1q_u8(nh_row + off, out.high);
        }
    }

    // Slide loop: stream H_{y+2} through ring (no wrap).
    int tail = 0;

    for (int y = y0 + 1; y < y1; ++y) {
        const int new_slot = (tail + 5) % 6;
        const int src_y_new = y + 2;  // NO wrap; caller guarantees < local_h

        compute_H_row(slow(src_y_new), shigh(src_y_new), R_REGS,
                      h0p(new_slot), h1p(new_slot), h2p(new_slot));

        const uint8_t* h_out_0 = h0p(tail);
        const uint8_t* h_out_1 = h1p(tail);
        const uint8_t* h_out_2 = h2p(tail);
        const uint8_t* h_in_0  = h0p(new_slot);
        const uint8_t* h_in_1  = h1p(new_slot);
        const uint8_t* h_in_2  = h2p(new_slot);
        const uint8_t* low_row  = slow(y);
        const uint8_t* high_row = shigh(y);
        uint8_t* nl_row = dlow(y);
        uint8_t* nh_row = dhigh(y);

        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;

            V5 v0 = load_v5_interleaved(v_data, off0);
            V5 v1 = load_v5_interleaved(v_data, off1);

            const Sum3 ho0 = {
                vld1q_u8(h_out_0 + off0), vld1q_u8(h_out_1 + off0), vld1q_u8(h_out_2 + off0),
            };
            const Sum3 ho1 = {
                vld1q_u8(h_out_0 + off1), vld1q_u8(h_out_1 + off1), vld1q_u8(h_out_2 + off1),
            };
            const Sum3 hi0 = {
                vld1q_u8(h_in_0 + off0), vld1q_u8(h_in_1 + off0), vld1q_u8(h_in_2 + off0),
            };
            const Sum3 hi1 = {
                vld1q_u8(h_in_0 + off1), vld1q_u8(h_in_1 + off1), vld1q_u8(h_in_2 + off1),
            };

            v0 = sub_v5_h3(v0, ho0);
            v1 = sub_v5_h3(v1, ho1);
            v0 = add_v5_h3(v0, hi0);
            v1 = add_v5_h3(v1, hi1);

            store_v5_interleaved(v_data, off0, v0);
            store_v5_interleaved(v_data, off1, v1);

            const uint8x16_t lo0 = vld1q_u8(low_row  + off0);
            const uint8x16_t hi0v = vld1q_u8(high_row + off0);
            const uint8x16_t lo1 = vld1q_u8(low_row  + off1);
            const uint8x16_t hi1v = vld1q_u8(high_row + off1);
            const LH outL = apply_rule_byte(v0, lo0, hi0v);
            const LH outR = apply_rule_byte(v1, lo1, hi1v);
            vst1q_u8(nl_row + off0, outL.low);
            vst1q_u8(nh_row + off0, outL.high);
            vst1q_u8(nl_row + off1, outR.low);
            vst1q_u8(nh_row + off1, outR.high);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = load_v5_interleaved(v_data, off);
            const Sum3 h_out = {
                vld1q_u8(h_out_0 + off), vld1q_u8(h_out_1 + off), vld1q_u8(h_out_2 + off),
            };
            const Sum3 h_in = {
                vld1q_u8(h_in_0 + off), vld1q_u8(h_in_1 + off), vld1q_u8(h_in_2 + off),
            };
            v = sub_v5_h3(v, h_out);
            v = add_v5_h3(v, h_in);
            store_v5_interleaved(v_data, off, v);

            const uint8x16_t lo = vld1q_u8(low_row  + off);
            const uint8x16_t hi = vld1q_u8(high_row + off);
            const LH out = apply_rule_byte(v, lo, hi);
            vst1q_u8(nl_row + off, out.low);
            vst1q_u8(nh_row + off, out.high);
        }

        tail = (tail + 1) % 6;
    }
}

// =====================================================================
// Strip variant of compute_H_row. Reads/writes a TILE_W_REGS-wide strip
// of data. prev/next come from the in-buffer ghost regions immediately
// before and after the data (no toroidal wrap inside the strip).
//
// `low_row` / `high_row` point to the DATA start of the row (offset
// TILE_W_DATA_OFFSET into the local row). So load(-1) = left ghost,
// load(TILE_W_REGS) = right ghost.
//
// `h*_out` are sized for the data only (TILE_W_REGS * 16 bytes).
// =====================================================================

static inline void compute_H_row_strip(const uint8_t* __restrict__ low_row,
                                       const uint8_t* __restrict__ high_row,
                                       uint8_t* __restrict__ h0_out,
                                       uint8_t* __restrict__ h1_out,
                                       uint8_t* __restrict__ h2_out)
{
    auto load_adult = [&](int r) -> uint8x16_t {
        const uint8x16_t lo = vld1q_u8(low_row  + r * 16);
        const uint8x16_t hi = vld1q_u8(high_row + r * 16);
        return vandq_u8(lo, hi);
    };

    uint8x16_t prev = load_adult(-1);              // left ghost
    uint8x16_t curr = load_adult(0);
    int r = 0;
    // Unroll by 4. Right ghost is at index TILE_W_REGS, so loads at r+4 are
    // always valid as long as r+4 <= TILE_W_REGS.
    for (; r + 3 < TILE_W_REGS; r += 4) {
        const uint8x16_t a1 = load_adult(r + 1);
        const uint8x16_t a2 = load_adult(r + 2);
        const uint8x16_t a3 = load_adult(r + 3);
        const uint8x16_t a4 = load_adult(r + 4);   // possibly right ghost

        const Sum3 h0 = horizontal_window_sum(prev, curr, a1);
        const Sum3 h1 = horizontal_window_sum(curr, a1, a2);
        const Sum3 h2 = horizontal_window_sum(a1, a2, a3);
        const Sum3 h3 = horizontal_window_sum(a2, a3, a4);

        vst1q_u8(h0_out + r * 16, h0.h0);
        vst1q_u8(h1_out + r * 16, h0.h1);
        vst1q_u8(h2_out + r * 16, h0.h2);
        vst1q_u8(h0_out + (r + 1) * 16, h1.h0);
        vst1q_u8(h1_out + (r + 1) * 16, h1.h1);
        vst1q_u8(h2_out + (r + 1) * 16, h1.h2);
        vst1q_u8(h0_out + (r + 2) * 16, h2.h0);
        vst1q_u8(h1_out + (r + 2) * 16, h2.h1);
        vst1q_u8(h2_out + (r + 2) * 16, h2.h2);
        vst1q_u8(h0_out + (r + 3) * 16, h3.h0);
        vst1q_u8(h1_out + (r + 3) * 16, h3.h1);
        vst1q_u8(h2_out + (r + 3) * 16, h3.h2);

        prev = a3;
        curr = a4;
    }
    // Tail for non-multiple-of-4 TILE_W_REGS. (At T=8 this never runs.)
    for (; r < TILE_W_REGS; ++r) {
        const uint8x16_t next = load_adult(r + 1);
        const Sum3 h = horizontal_window_sum(prev, curr, next);
        vst1q_u8(h0_out + r * 16, h.h0);
        vst1q_u8(h1_out + r * 16, h.h1);
        vst1q_u8(h2_out + r * 16, h.h2);
        prev = curr;
        curr = next;
    }
}

// =====================================================================
// step_rows_bitplane_strip_local — same ring-buffer kernel as
// step_rows_bitplane_local but for a column strip of TILE_W_REGS data
// registers.
//
// Inputs:
//   src_low / src_high: pointer to DATA start of row 0 in the local
//     strip buffer (i.e. base + TILE_W_DATA_OFFSET).
//   dst_low / dst_high: same for output strip.
//   local_row_bytes: stride between consecutive rows (TILE_W_LOCAL_BYTES).
//   y0 / y1: row range to process (absolute indices in local buffer).
//   scratch: per-thread (h_ring, vcount_store) reused across strips.
//
// H ring and V scratch are sized for TILE_W_REGS, not full row.
// =====================================================================

static void step_rows_bitplane_strip_local(
    const uint8_t* __restrict__ src_low,
    const uint8_t* __restrict__ src_high,
    uint8_t* __restrict__ dst_low,
    uint8_t* __restrict__ dst_high,
    int local_row_bytes,
    int y0, int y1,
    ThreadScratch& scratch)
{
    const int R_REGS = TILE_W_REGS;
    const int strip_data_bytes = TILE_W_DATA_BYTES;

    uint8_t* ring   = scratch.h_ring.data();
    uint8_t* v_data = scratch.vcount_store.data();

    auto h0p = [&](int slot) { return ring + (size_t)(3 * slot + 0) * strip_data_bytes; };
    auto h1p = [&](int slot) { return ring + (size_t)(3 * slot + 1) * strip_data_bytes; };
    auto h2p = [&](int slot) { return ring + (size_t)(3 * slot + 2) * strip_data_bytes; };

    auto slow  = [&](int y) { return src_low  + (size_t)y * local_row_bytes; };
    auto shigh = [&](int y) { return src_high + (size_t)y * local_row_bytes; };
    auto dlow  = [&](int y) { return dst_low  + (size_t)y * local_row_bytes; };
    auto dhigh = [&](int y) { return dst_high + (size_t)y * local_row_bytes; };

    // Pre-fill ring slots 0..4 with H_{y0-2..y0+2}. No row wrap.
    for (int s = 0; s < 5; ++s) {
        const int src_y = y0 - 2 + s;
        compute_H_row_strip(slow(src_y), shigh(src_y),
                            h0p(s), h1p(s), h2p(s));
    }

    // Initialise V = sum of ring slots 0..4. 2-way unrolled in r.
    {
        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;
            V5 v0 = {
                vld1q_u8(h0p(0) + off0), vld1q_u8(h1p(0) + off0), vld1q_u8(h2p(0) + off0),
                vdupq_n_u8(0), vdupq_n_u8(0),
            };
            V5 v1 = {
                vld1q_u8(h0p(0) + off1), vld1q_u8(h1p(0) + off1), vld1q_u8(h2p(0) + off1),
                vdupq_n_u8(0), vdupq_n_u8(0),
            };
            for (int s = 1; s < 5; ++s) {
                const Sum3 h0 = {
                    vld1q_u8(h0p(s) + off0), vld1q_u8(h1p(s) + off0), vld1q_u8(h2p(s) + off0),
                };
                const Sum3 h1 = {
                    vld1q_u8(h0p(s) + off1), vld1q_u8(h1p(s) + off1), vld1q_u8(h2p(s) + off1),
                };
                v0 = add_v5_h3(v0, h0);
                v1 = add_v5_h3(v1, h1);
            }
            store_v5_interleaved(v_data, off0, v0);
            store_v5_interleaved(v_data, off1, v1);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = {
                vld1q_u8(h0p(0) + off), vld1q_u8(h1p(0) + off), vld1q_u8(h2p(0) + off),
                vdupq_n_u8(0), vdupq_n_u8(0),
            };
            for (int s = 1; s < 5; ++s) {
                const Sum3 h = {
                    vld1q_u8(h0p(s) + off), vld1q_u8(h1p(s) + off), vld1q_u8(h2p(s) + off),
                };
                v = add_v5_h3(v, h);
            }
            store_v5_interleaved(v_data, off, v);
        }
    }

    // Apply rule to first output row (y = y0).
    {
        const uint8_t* low_row  = slow(y0);
        const uint8_t* high_row = shigh(y0);
        uint8_t* nl_row = dlow(y0);
        uint8_t* nh_row = dhigh(y0);

        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;
            V5 v0 = load_v5_interleaved(v_data, off0);
            V5 v1 = load_v5_interleaved(v_data, off1);
            const uint8x16_t lo0 = vld1q_u8(low_row  + off0);
            const uint8x16_t hi0 = vld1q_u8(high_row + off0);
            const uint8x16_t lo1 = vld1q_u8(low_row  + off1);
            const uint8x16_t hi1 = vld1q_u8(high_row + off1);
            const LH out0 = apply_rule_byte(v0, lo0, hi0);
            const LH out1 = apply_rule_byte(v1, lo1, hi1);
            vst1q_u8(nl_row + off0, out0.low);
            vst1q_u8(nh_row + off0, out0.high);
            vst1q_u8(nl_row + off1, out1.low);
            vst1q_u8(nh_row + off1, out1.high);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = load_v5_interleaved(v_data, off);
            const uint8x16_t lo = vld1q_u8(low_row  + off);
            const uint8x16_t hi = vld1q_u8(high_row + off);
            const LH out = apply_rule_byte(v, lo, hi);
            vst1q_u8(nl_row + off, out.low);
            vst1q_u8(nh_row + off, out.high);
        }
    }

    // Slide loop: stream H_{y+2} through ring (no wrap).
    int tail = 0;

    for (int y = y0 + 1; y < y1; ++y) {
        const int new_slot = (tail + 5) % 6;
        const int src_y_new = y + 2;

        compute_H_row_strip(slow(src_y_new), shigh(src_y_new),
                            h0p(new_slot), h1p(new_slot), h2p(new_slot));

        const uint8_t* h_out_0 = h0p(tail);
        const uint8_t* h_out_1 = h1p(tail);
        const uint8_t* h_out_2 = h2p(tail);
        const uint8_t* h_in_0  = h0p(new_slot);
        const uint8_t* h_in_1  = h1p(new_slot);
        const uint8_t* h_in_2  = h2p(new_slot);
        const uint8_t* low_row  = slow(y);
        const uint8_t* high_row = shigh(y);
        uint8_t* nl_row = dlow(y);
        uint8_t* nh_row = dhigh(y);

        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;

            V5 v0 = load_v5_interleaved(v_data, off0);
            V5 v1 = load_v5_interleaved(v_data, off1);

            const Sum3 ho0 = {
                vld1q_u8(h_out_0 + off0), vld1q_u8(h_out_1 + off0), vld1q_u8(h_out_2 + off0),
            };
            const Sum3 ho1 = {
                vld1q_u8(h_out_0 + off1), vld1q_u8(h_out_1 + off1), vld1q_u8(h_out_2 + off1),
            };
            const Sum3 hi0 = {
                vld1q_u8(h_in_0 + off0), vld1q_u8(h_in_1 + off0), vld1q_u8(h_in_2 + off0),
            };
            const Sum3 hi1 = {
                vld1q_u8(h_in_0 + off1), vld1q_u8(h_in_1 + off1), vld1q_u8(h_in_2 + off1),
            };

            v0 = sub_v5_h3(v0, ho0);
            v1 = sub_v5_h3(v1, ho1);
            v0 = add_v5_h3(v0, hi0);
            v1 = add_v5_h3(v1, hi1);

            store_v5_interleaved(v_data, off0, v0);
            store_v5_interleaved(v_data, off1, v1);

            const uint8x16_t lo0  = vld1q_u8(low_row  + off0);
            const uint8x16_t hi0v = vld1q_u8(high_row + off0);
            const uint8x16_t lo1  = vld1q_u8(low_row  + off1);
            const uint8x16_t hi1v = vld1q_u8(high_row + off1);
            const LH outL = apply_rule_byte(v0, lo0, hi0v);
            const LH outR = apply_rule_byte(v1, lo1, hi1v);
            vst1q_u8(nl_row + off0, outL.low);
            vst1q_u8(nh_row + off0, outL.high);
            vst1q_u8(nl_row + off1, outR.low);
            vst1q_u8(nh_row + off1, outR.high);
        }
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            V5 v = load_v5_interleaved(v_data, off);
            const Sum3 h_out = {
                vld1q_u8(h_out_0 + off), vld1q_u8(h_out_1 + off), vld1q_u8(h_out_2 + off),
            };
            const Sum3 h_in = {
                vld1q_u8(h_in_0 + off), vld1q_u8(h_in_1 + off), vld1q_u8(h_in_2 + off),
            };
            v = sub_v5_h3(v, h_out);
            v = add_v5_h3(v, h_in);
            store_v5_interleaved(v_data, off, v);

            const uint8x16_t lo = vld1q_u8(low_row  + off);
            const uint8x16_t hi = vld1q_u8(high_row + off);
            const LH out = apply_rule_byte(v, lo, hi);
            vst1q_u8(nl_row + off, out.low);
            vst1q_u8(nh_row + off, out.high);
        }

        tail = (tail + 1) % 6;
    }
}

// =====================================================================
// Temporal driver: copies (slab + 2K ghost) from src to local buffer,
// runs K_this gens locally, writes slab back to dst. Window shrinks by
// 2 rows on each side per gen; after K_this gens, central slab is valid.
// =====================================================================

static void run_k_gens_on_slab(const BitGrid& src, BitGrid& dst,
                               int y_slab_start, int y_slab_end,
                               int K_this,
                               ThreadScratch& scratch)
{
    const int N            = src.n;
    const int ymask        = N - 1;
    const int row_bytes    = src.row_bytes;     // full grid row, in bytes
    const int xmask_bytes  = row_bytes - 1;     // column wrap mask (power of 2)
    const int slab_h       = y_slab_end - y_slab_start;
    const int ghost        = 2 * K_this;
    const int local_h      = slab_h + 2 * ghost;

    // Trapezoidal column tiling: each strip's K-iteration produces correct output
    // only in the central region. The outer 2*K cells on each side are wrong because
    // they used stale column ghost (which isn't updated across the K gens).
    // We round up to byte granularity for the affected region.
    const int affected_cells_per_side = 2 * K_this;
    const int affected_bytes_per_side = (affected_cells_per_side + 7) / 8;
    const int valid_advance           = TILE_W_DATA_BYTES - 2 * affected_bytes_per_side;
    const int n_strips                = (row_bytes + valid_advance - 1) / valid_advance;

    uint8_t* la_low  = scratch.lbuf_a_low.data();
    uint8_t* la_high = scratch.lbuf_a_high.data();
    uint8_t* lb_low  = scratch.lbuf_b_low.data();
    uint8_t* lb_high = scratch.lbuf_b_high.data();

    // Iterate column strips. Each strip processes TILE_W_DATA_BYTES bytes (per plane)
    // but only writes back valid_advance bytes (skipping affected_bytes_per_side on each side).
    // Adjacent strips overlap by 2 * affected_bytes_per_side so every cell is covered.
    for (int strip = 0; strip < n_strips; ++strip) {
        const int strip_data_off = (strip * valid_advance) & xmask_bytes;   // byte offset of data start in global row

        // ---- 1. Ghost copy: (slab + 2*ghost) rows from src into lbuf_a. ----
        // Per row we copy ghost_L | data | ghost_R into the local buffer. Each chunk
        // may wrap across the toroidal column boundary (row_bytes). copy_wrap handles that.
        auto copy_wrap = [&](uint8_t* dst, const uint8_t* src_row, int start_off, int len) {
            int o = start_off & xmask_bytes;
            int first = std::min(len, row_bytes - o);
            std::memcpy(dst, src_row + o, (size_t)first);
            if (first < len) std::memcpy(dst + first, src_row, (size_t)(len - first));
        };

        for (int local_y = 0; local_y < local_h; ++local_y) {
            const int global_y = (y_slab_start - ghost + local_y + N) & ymask;
            const uint8_t* sl = src.row0(global_y);
            const uint8_t* sh = src.row1(global_y);

            uint8_t* dst_l = la_low  + (size_t)local_y * TILE_W_LOCAL_BYTES;
            uint8_t* dst_h = la_high + (size_t)local_y * TILE_W_LOCAL_BYTES;

            // [ghost_L | data | ghost_R] laid out contiguously in the local row.
            const int ghost_l_off = (strip_data_off - TILE_W_GHOST_BYTES + row_bytes);
            copy_wrap(dst_l,                                              sl, ghost_l_off,                  TILE_W_GHOST_BYTES);
            copy_wrap(dst_h,                                              sh, ghost_l_off,                  TILE_W_GHOST_BYTES);
            copy_wrap(dst_l + TILE_W_GHOST_BYTES,                         sl, strip_data_off,               TILE_W_DATA_BYTES);
            copy_wrap(dst_h + TILE_W_GHOST_BYTES,                         sh, strip_data_off,               TILE_W_DATA_BYTES);
            copy_wrap(dst_l + TILE_W_GHOST_BYTES + TILE_W_DATA_BYTES,     sl, strip_data_off + TILE_W_DATA_BYTES, TILE_W_GHOST_BYTES);
            copy_wrap(dst_h + TILE_W_GHOST_BYTES + TILE_W_DATA_BYTES,     sh, strip_data_off + TILE_W_DATA_BYTES, TILE_W_GHOST_BYTES);
        }

        // ---- 2. Run K_this gens on local strip. Window shrinks by 2 rows / side / gen. ----
        uint8_t* cur_low_base  = la_low;
        uint8_t* cur_high_base = la_high;
        uint8_t* nxt_low_base  = lb_low;
        uint8_t* nxt_high_base = lb_high;

        for (int k = 1; k <= K_this; ++k) {
            const int y0 = 2 * k;
            const int y1 = local_h - 2 * k;
            // Pass DATA pointers (offset past the left ghost) into the strip kernel.
            step_rows_bitplane_strip_local(
                cur_low_base  + TILE_W_DATA_OFFSET,
                cur_high_base + TILE_W_DATA_OFFSET,
                nxt_low_base  + TILE_W_DATA_OFFSET,
                nxt_high_base + TILE_W_DATA_OFFSET,
                TILE_W_LOCAL_BYTES,
                y0, y1, scratch);
            std::swap(cur_low_base,  nxt_low_base);
            std::swap(cur_high_base, nxt_high_base);
        }
        // After K_this gens, cur_*_base holds final values; central slab at rows [ghost, ghost+slab_h).

        // ---- 3. Write slab back to global dst — central valid_advance bytes only. ----
        // The leftmost and rightmost affected_bytes_per_side bytes of the strip's
        // data are wrong (used stale column ghost). Skip them. Adjacent strips overlap
        // on those wrong regions with their correct central regions, so every cell is
        // covered. Writeback may wrap across the row boundary on the last strip(s).
        auto copy_back_wrap = [&](uint8_t* dst_row, const uint8_t* src_local, int start_off, int len) {
            int o = start_off & xmask_bytes;
            int first = std::min(len, row_bytes - o);
            std::memcpy(dst_row + o, src_local, (size_t)first);
            if (first < len) std::memcpy(dst_row, src_local + first, (size_t)(len - first));
        };

        const int valid_start_off = strip_data_off + affected_bytes_per_side;
        for (int local_y = ghost; local_y < ghost + slab_h; ++local_y) {
            const int global_y = (y_slab_start + (local_y - ghost)) & ymask;
            const uint8_t* src_l = cur_low_base  + (size_t)local_y * TILE_W_LOCAL_BYTES + TILE_W_DATA_OFFSET + affected_bytes_per_side;
            const uint8_t* src_h = cur_high_base + (size_t)local_y * TILE_W_LOCAL_BYTES + TILE_W_DATA_OFFSET + affected_bytes_per_side;
            copy_back_wrap(dst.row0(global_y), src_l, valid_start_off, valid_advance);
            copy_back_wrap(dst.row1(global_y), src_h, valid_start_off, valid_advance);
        }
    }
}

// =====================================================================
// Main (same as vb13.5: no CPU pinning, static row partition).
// =====================================================================

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    int generations = 10000;
    if (argc == 4) {
        char* end;
        long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = (int)g;
    }

    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width, height;
    if (std::fread(&width, sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: input file too short (cannot read header)\n");
        std::fclose(fin);
        return 3;
    }
    if (width == 0 || width != height || (width % 128) != 0) {
        std::fprintf(stderr,
            "Error: grid must be square, non-empty, divisible by 128, got %" PRIu64 " x %" PRIu64 "\n",
            width, height);
        std::fclose(fin);
        return 3;
    }

    const int N = (int)width;
    const size_t Ncells = (size_t)N * N;

    std::vector<uint8_t> cells(Ncells);
    if (std::fread(cells.data(), 1, Ncells, fin) != Ncells) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    BitGrid grid_a, grid_b;
    grid_a.resize(N);
    grid_b.resize(N);
    bytes_to_bitgrid(cells, grid_a);
    cells.clear();
    cells.shrink_to_fit();

    BitGrid* cur = &grid_a;
    BitGrid* next = &grid_b;

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    unsigned T = std::min<unsigned>(hw, 8u);
    if ((int)T > N) T = (unsigned)N;

    std::vector<int> row_lo(T), row_hi(T);
    int max_slab = 0;
    for (unsigned t = 0; t < T; ++t) {
        row_lo[t] = (int)((uint64_t)t * N / T);
        row_hi[t] = (int)((uint64_t)(t + 1) * N / T);
        max_slab = std::max(max_slab, row_hi[t] - row_lo[t]);
    }
    // Local buffer: slab + 2*ghost rows. ghost = 2 * K_GENS at runtime.
    const int local_rows = max_slab + 4 * K_GENS;
    std::vector<ThreadScratch> scratch(T);
    for (auto& ctx : scratch) {
        ctx.resize(grid_a.row_bytes, local_rows);
    }

    const BitGrid* shared_src = nullptr;
    BitGrid* shared_dst = nullptr;
    int shared_K_this = 0;
    bool stop = false;

    std::barrier bar_start(T);
    std::barrier bar_done(T);

    std::vector<std::thread> pool;
    pool.reserve(T - 1);
    for (unsigned t = 1; t < T; ++t) {
        pool.emplace_back([&, t]() {
            for (;;) {
                bar_start.arrive_and_wait();
                if (stop) return;
                run_k_gens_on_slab(*shared_src, *shared_dst,
                                   row_lo[t], row_hi[t],
                                   shared_K_this, scratch[t]);
                bar_done.arrive_and_wait();
            }
        });
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int gen = 0; gen < generations; gen += K_GENS) {
        const int K_this = std::min(K_GENS, generations - gen);
        shared_src = cur;
        shared_dst = next;
        shared_K_this = K_this;
        bar_start.arrive_and_wait();
        run_k_gens_on_slab(*cur, *next, row_lo[0], row_hi[0], K_this, scratch[0]);
        bar_done.arrive_and_wait();
        std::swap(cur, next);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    stop = true;
    bar_start.arrive_and_wait();
    for (auto& th : pool) th.join();

    bitgrid_to_bytes(*cur, cells);

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }
    if (std::fwrite(&width, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(cells.data(), 1, Ncells, fout) != Ncells) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }
    std::fclose(fout);
    return 0;
}
