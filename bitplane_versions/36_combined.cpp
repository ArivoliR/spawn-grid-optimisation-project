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
#include <atomic>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

static constexpr uint8_t EMPTY    = 0;
static constexpr uint8_t EGG      = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT    = 3;

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

// vb26: BSL MAJ-fold throughout. Same identity oth/phase15A-E exploited.
// MAJ(a,b,c) = (a&b)|(a&c)|(b&c). When ax=a^b is 0 (a==b), MAJ=a; when ax=1, MAJ=c.
// So MAJ(a,b,c) = BSL(a^b, c, a) — 1 op given a^b already in hand.
// Old maj_u8: 3 ops (eor+and+bsl). Old sub-stage with ~b1: 4 ops (vmvnq+eor+and+bsl).
// New: 1 BSL per ripple stage.

struct FA { uint8x16_t s, c; };
struct HA { uint8x16_t s, c; };
static inline HA half_adder_u8(uint8x16_t a, uint8x16_t b)
{
    return { veorq_u8(a, b), vandq_u8(a, b) };
}

struct Sum3 { uint8x16_t h0, h1, h2; };
// 8 ops (vs 10 before): two BSL MAJ-folds + one veor3 for the final sum bit.
static inline Sum3 sum_of_5(uint8x16_t a, uint8x16_t b, uint8x16_t c,
                            uint8x16_t d, uint8x16_t e)
{
    const uint8x16_t axb   = veorq_u8(a, b);
    const uint8x16_t s_abc = veorq_u8(axb, c);
    const uint8x16_t c_abc = vbslq_u8(axb, c, a);            // MAJ(a,b,c)
    const uint8x16_t dxe   = veorq_u8(d, e);
    const uint8x16_t c_des = vbslq_u8(dxe, s_abc, d);        // MAJ(d,e,s_abc)
    const uint8x16_t out0  = veor3q_u8(s_abc, d, e);         // h0
    const uint8x16_t out1  = veorq_u8(c_abc, c_des);         // h1
    const uint8x16_t out2  = vandq_u8(c_abc, c_des);         // h2
    return { out0, out1, out2 };
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
// Adder ripple with BSL MAJ-fold at the two interior carry stages.
// Per FA: ax = b^h (1 eor), sum = veor3(b,h,c_in) (1 op), carry = BSL(ax, c_in, b) (1 op) — 3 ops.
// Total: 2 ops (HA s0) + 3 (FA s1) + 3 (FA s2) + 2 (HA s3) + 1 (b4 eor) = 11 ops.
// Old: 2 + 4 + 4 + 2 + 1 = 13 ops.
static inline V5 add_v5_h3(V5 v, Sum3 h)
{
    // HA at bit 0: sum = b0^h0, carry = b0&h0.
    const uint8x16_t s0  = veorq_u8(v.b0, h.h0);
    const uint8x16_t c0  = vandq_u8(v.b0, h.h0);
    // FA at bit 1: inputs (b1, h1, c0).
    const uint8x16_t ax1 = veorq_u8(v.b1, h.h1);
    const uint8x16_t s1  = veor3q_u8(v.b1, h.h1, c0);
    const uint8x16_t c1  = vbslq_u8(ax1, c0, v.b1);           // MAJ(b1,h1,c0)
    // FA at bit 2: inputs (b2, h2, c1).
    const uint8x16_t ax2 = veorq_u8(v.b2, h.h2);
    const uint8x16_t s2  = veor3q_u8(v.b2, h.h2, c1);
    const uint8x16_t c2  = vbslq_u8(ax2, c1, v.b2);           // MAJ(b2,h2,c1)
    // HA at bit 3: sum = b3^c2, carry = b3&c2.
    const uint8x16_t s3  = veorq_u8(v.b3, c2);
    const uint8x16_t c3  = vandq_u8(v.b3, c2);
    const uint8x16_t b4  = veorq_u8(v.b4, c3);
    return { s0, s1, s2, s3, b4 };
}
// Subtractor ripple with BSL MAJ-fold (no vmvnq).
// borrow_out at each interior stage = BSL(b^h, h, borrow_in): when b==h, borrow passes;
// when b!=h, borrow_out = h. Old version did vmvnq+maj_u8 (4 ops); now 1 BSL.
static inline V5 sub_v5_h3(V5 v, Sum3 h)
{
    const uint8x16_t diff0 = veorq_u8(v.b0, h.h0);
    const uint8x16_t b0    = vbicq_u8(h.h0, v.b0);              // borrow_0 = ~v.b0 & h.h0
    const uint8x16_t ax1   = veorq_u8(v.b1, h.h1);
    const uint8x16_t diff1 = veor3q_u8(v.b1, h.h1, b0);
    const uint8x16_t b1    = vbslq_u8(ax1, h.h1, b0);           // borrow_1
    const uint8x16_t ax2   = veorq_u8(v.b2, h.h2);
    const uint8x16_t diff2 = veor3q_u8(v.b2, h.h2, b1);
    const uint8x16_t b2    = vbslq_u8(ax2, h.h2, b1);           // borrow_2
    const uint8x16_t diff3 = veorq_u8(v.b3, b2);
    const uint8x16_t diff4 = vbcaxq_u8(v.b4, b2, v.b3);          // v.b4 ^ (b2 & ~v.b3)
    return { diff0, diff1, diff2, diff3, diff4 };
}

// vb36: bit-interleaved combined sub+add (from vb33). Dep depth ~6 vs ~10
// for separate sub then add.
static inline V5 slide_v5_h3(V5 v, Sum3 h_old, Sum3 h_new)
{
    const uint8x16_t tmp0  = veorq_u8(v.b0, h_old.h0);
    const uint8x16_t bor0  = vbicq_u8(h_old.h0, v.b0);
    const uint8x16_t out0  = veorq_u8(tmp0, h_new.h0);
    const uint8x16_t car0  = vandq_u8(tmp0, h_new.h0);

    const uint8x16_t bxh1  = veorq_u8(v.b1, h_old.h1);
    const uint8x16_t tmp1  = veor3q_u8(v.b1, h_old.h1, bor0);
    const uint8x16_t bor1  = vbslq_u8(bxh1, h_old.h1, bor0);
    const uint8x16_t tx1   = veorq_u8(tmp1, h_new.h1);
    const uint8x16_t out1  = veor3q_u8(tmp1, h_new.h1, car0);
    const uint8x16_t car1  = vbslq_u8(tx1, car0, tmp1);

    const uint8x16_t bxh2  = veorq_u8(v.b2, h_old.h2);
    const uint8x16_t tmp2  = veor3q_u8(v.b2, h_old.h2, bor1);
    const uint8x16_t bor2  = vbslq_u8(bxh2, h_old.h2, bor1);
    const uint8x16_t tx2   = veorq_u8(tmp2, h_new.h2);
    const uint8x16_t out2  = veor3q_u8(tmp2, h_new.h2, car1);
    const uint8x16_t car2  = vbslq_u8(tx2, car1, tmp2);

    const uint8x16_t tmp3  = veorq_u8(v.b3, bor2);
    const uint8x16_t bor3  = vbicq_u8(bor2, v.b3);
    const uint8x16_t out3  = veorq_u8(tmp3, car2);
    const uint8x16_t car3  = vandq_u8(tmp3, car2);

    const uint8x16_t out4  = veor3q_u8(v.b4, bor3, car3);

    return { out0, out1, out2, out3, out4 };
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

// vb27: 5-slot ring (no 2-row unroll → no need for the spare slot vb22 had).
static constexpr int RING_SLOTS = 5;

struct ThreadScratch {
    AlignedBytes h_ring;        // RING_SLOTS slots × 3 planes × row_bytes
    AlignedBytes vcount_store;  // 5 × row_bytes (V-interleaved, unchanged from vb17)
    uint8_t* v_b0 = nullptr;
    uint8_t* v_b1 = nullptr;
    uint8_t* v_b2 = nullptr;
    uint8_t* v_b3 = nullptr;
    uint8_t* v_b4 = nullptr;

    void resize(int row_bytes)
    {
        h_ring.resize((size_t)RING_SLOTS * 3 * (size_t)row_bytes);
        vcount_store.resize(5 * (size_t)row_bytes);
        advise_huge_pages(h_ring.data(), h_ring.size());
        advise_huge_pages(vcount_store.data(), vcount_store.size());
        v_b0 = vcount_store.data() + 0 * (size_t)row_bytes;
        v_b1 = vcount_store.data() + 1 * (size_t)row_bytes;
        v_b2 = vcount_store.data() + 2 * (size_t)row_bytes;
        v_b3 = vcount_store.data() + 3 * (size_t)row_bytes;
        v_b4 = vcount_store.data() + 4 * (size_t)row_bytes;
    }
};

// =====================================================================
// vb27: single-pass step kernel.
//
// vb22 / vb26 ran two passes per row: compute_H_row stored H bitplanes to a
// 6-slot ring, then a separate slide loop loaded them back to update V. The
// new-row H value made an L2 round trip every time before being consumed.
//
// vb27 fuses the two passes. Inside the inner column-pair loop:
//   - We load the entering row's adult bits with a 3-wide sliding window.
//   - Compute new H values for the current column pair (h_new_0, h_new_1).
//   - Use h_new_0/1 immediately in add_v5_h3 (no round trip for the new H).
//   - Store h_new_0/1 to the ring slot we just sub'd from (the slot rotates).
//
// Result: per row, the new H value lives only in registers between produce
// and consume; the ring still holds the 4 surviving H rows that future
// iterations need. Net: ring is 5 slots (was 6, since we also drop the
// 2-row unroll), and ~half the H bandwidth is eliminated.
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

    // Ring slot pointers: RING_SLOTS slots, each with 3 planes (h0, h1, h2).
    auto h0p = [&](int slot) { return ring + (size_t)(3 * slot + 0) * row_bytes; };
    auto h1p = [&](int slot) { return ring + (size_t)(3 * slot + 1) * row_bytes; };
    auto h2p = [&](int slot) { return ring + (size_t)(3 * slot + 2) * row_bytes; };

    // -------- Pre-fill ring slots 0..4 with H_{y0-2..y0+2}. --------
    for (int s = 0; s < RING_SLOTS; ++s) {
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

    // -------- Single-pass slide loop. --------
    // Entering iter y (with y > y0):
    //   Ring slots {tail, (tail+1)%5, ..., (tail+4)%5} hold H_{y-3, y-2, y-1, y, y+1}.
    //   V represents window centred at y-1.
    //   We compute H_{y+2} inline, fuse it into the V slide (-H_{y-3} +H_{y+2}),
    //   apply rule for row y, then overwrite ring[tail] with H_{y+2}.
    //   tail then advances by 1.
    int tail = 0;

    for (int y = y0 + 1; y < y1; ++y) {
        const int src_y_new = (y + 2 + N) & ymask;
        const uint8_t* new_lo = src.row0(src_y_new);
        const uint8_t* new_hi = src.row1(src_y_new);

        const uint8_t* h_out_0 = h0p(tail);
        const uint8_t* h_out_1 = h1p(tail);
        const uint8_t* h_out_2 = h2p(tail);
        // After slide, write the new H into the slot we just freed.
        uint8_t* h_write_0 = h0p(tail);
        uint8_t* h_write_1 = h1p(tail);
        uint8_t* h_write_2 = h2p(tail);

        const uint8_t* low_row  = src.row0(y);
        const uint8_t* high_row = src.row1(y);
        uint8_t* nl_row = dst.row0(y);
        uint8_t* nh_row = dst.row1(y);

        // Sliding-window adult bits for the entering row (used to compute H_new).
        // Wrap-around: prev starts at R_REGS-1, curr starts at 0.
        auto load_new_adult = [&](int r) -> uint8x16_t {
            const uint8x16_t lo = vld1q_u8(new_lo + (size_t)r * 16);
            const uint8x16_t hi = vld1q_u8(new_hi + (size_t)r * 16);
            return vandq_u8(lo, hi);
        };

        uint8x16_t adult_prev = load_new_adult(R_REGS - 1);
        uint8x16_t adult_curr = load_new_adult(0);

        int r = 0;
        for (; r + 1 < R_REGS; r += 2) {
            const size_t off0 = (size_t)r * 16;
            const size_t off1 = (size_t)(r + 1) * 16;

            // Load adults for next 2 column registers (with wrap for the very last pair).
            const int r1 = r + 1;
            const int r2 = r + 2;
            const uint8x16_t adult_next_0 = load_new_adult(r1);
            const uint8x16_t adult_next_1 =
                (r2 < R_REGS) ? load_new_adult(r2) : load_new_adult(0);

            // Compute new H for both columns from the sliding adult window.
            const Sum3 h_new_0 = horizontal_window_sum(adult_prev, adult_curr, adult_next_0);
            const Sum3 h_new_1 = horizontal_window_sum(adult_curr, adult_next_0, adult_next_1);

            // Load V state for both columns.
            V5 v0 = load_v5_interleaved(v_data, off0);
            V5 v1 = load_v5_interleaved(v_data, off1);

            // Load OLD H from ring[tail] for both columns (the leaving row's H).
            const Sum3 h_old_0 = {
                vld1q_u8(h_out_0 + off0),
                vld1q_u8(h_out_1 + off0),
                vld1q_u8(h_out_2 + off0),
            };
            const Sum3 h_old_1 = {
                vld1q_u8(h_out_0 + off1),
                vld1q_u8(h_out_1 + off1),
                vld1q_u8(h_out_2 + off1),
            };

            // Slide: V -= h_old, V += h_new. Single bit-interleaved ripple (vb33).
            v0 = slide_v5_h3(v0, h_old_0, h_new_0);
            v1 = slide_v5_h3(v1, h_old_1, h_new_1);

            store_v5_interleaved(v_data, off0, v0);
            store_v5_interleaved(v_data, off1, v1);

            // Write new H to the just-freed ring slot for future iterations.
            vst1q_u8(h_write_0 + off0, h_new_0.h0);
            vst1q_u8(h_write_1 + off0, h_new_0.h1);
            vst1q_u8(h_write_2 + off0, h_new_0.h2);
            vst1q_u8(h_write_0 + off1, h_new_1.h0);
            vst1q_u8(h_write_1 + off1, h_new_1.h1);
            vst1q_u8(h_write_2 + off1, h_new_1.h2);

            // Apply rule for row y.
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

            // Slide adult window.
            adult_prev = adult_next_0;
            adult_curr = adult_next_1;
        }
        // Scalar-tail (R_REGS odd). Same shape, single column.
        for (; r < R_REGS; ++r) {
            const size_t off = (size_t)r * 16;
            const int r1 = (r + 1 == R_REGS) ? 0 : (r + 1);
            const uint8x16_t adult_next = load_new_adult(r1);

            const Sum3 h_new = horizontal_window_sum(adult_prev, adult_curr, adult_next);

            V5 v = load_v5_interleaved(v_data, off);
            const Sum3 h_old = {
                vld1q_u8(h_out_0 + off),
                vld1q_u8(h_out_1 + off),
                vld1q_u8(h_out_2 + off),
            };
            v = slide_v5_h3(v, h_old, h_new);
            store_v5_interleaved(v_data, off, v);

            vst1q_u8(h_write_0 + off, h_new.h0);
            vst1q_u8(h_write_1 + off, h_new.h1);
            vst1q_u8(h_write_2 + off, h_new.h2);

            const uint8x16_t lo = vld1q_u8(low_row  + off);
            const uint8x16_t hi = vld1q_u8(high_row + off);
            const LH out = apply_rule_byte(v, lo, hi);
            vst1q_u8(nl_row + off, out.low);
            vst1q_u8(nh_row + off, out.high);

            adult_prev = adult_curr;
            adult_curr = adult_next;
        }

        tail = (tail + 1) % RING_SLOTS;
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

    // (cur/next pointers no longer needed — workers toggle src/dst by gen parity)

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    unsigned T = std::min<unsigned>(hw, 8u);
    if ((int)T > N) T = (unsigned)N;

    std::vector<int> row_lo(T), row_hi(T);
    for (unsigned t = 0; t < T; ++t) {
        row_lo[t] = (int)((uint64_t)t * N / T);
        row_hi[t] = (int)((uint64_t)(t + 1) * N / T);
    }
    std::vector<ThreadScratch> scratch(T);
    for (auto& ctx : scratch) {
        ctx.resize(grid_a.row_bytes);
    }

    // vb28: pairwise neighbour sync (vb27 inner loop is tighter than vb22, so the
    // global barrier became relatively more expensive). Each thread waits only
    // for its two row-band neighbours (toroidal). Cache-line-aligned counters.
    struct alignas(64) WorkerSync {
        std::atomic<int64_t> done_gen{0};
        char _pad[64 - sizeof(std::atomic<int64_t>)];
    };
    std::vector<WorkerSync> wsync(T);

    // vb34: explicit CPU pinning per worker. Profile of vb28 showed ~32 % of
    // hot-loop cycles in the spin-wait (cmp + yield), suggesting threads were
    // drifting between cores and amplifying the wait. Pinning each thread to
    // a unique CPU stops migration, keeps L1/L2 warm for that thread's row
    // band, and shrinks neighbour-sync drift.
    auto pin_thread_to_cpu = [](unsigned cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    };

    auto worker_loop = [&](unsigned t) {
        pin_thread_to_cpu(t);
        const int t_left  = (t == 0) ? (int)T - 1 : (int)t - 1;
        const int t_right = (t + 1 == T) ? 0 : (int)t + 1;
        for (int gen = 0; gen < generations; ++gen) {
            if (gen > 0) {
                while (wsync[t_left].done_gen.load(std::memory_order_acquire) < gen) {
                    __asm__ __volatile__("yield" ::: "memory");
                }
                while (wsync[t_right].done_gen.load(std::memory_order_acquire) < gen) {
                    __asm__ __volatile__("yield" ::: "memory");
                }
            }
            const BitGrid* src = (gen & 1) ? &grid_b : &grid_a;
            BitGrid* dst       = (gen & 1) ? &grid_a : &grid_b;
            step_rows_bitplane(*src, *dst, row_lo[t], row_hi[t], scratch[t]);
            wsync[t].done_gen.store(gen + 1, std::memory_order_release);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(T - 1);

    auto t0 = std::chrono::steady_clock::now();
    for (unsigned t = 1; t < T; ++t) {
        pool.emplace_back([&, t]() { worker_loop(t); });
    }
    worker_loop(0);
    for (unsigned t = 1; t < T; ++t) {
        while (wsync[t].done_gen.load(std::memory_order_acquire) < generations) {
            __asm__ __volatile__("yield" ::: "memory");
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    for (auto& th : pool) th.join();

    BitGrid* result = (generations & 1) ? &grid_b : &grid_a;
    bitgrid_to_bytes(*result, cells);

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
