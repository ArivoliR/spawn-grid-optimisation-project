// spawn_sim.cpp — Optimised Monster Spawning Grid simulator.
//
// Strategy (chosen, not assumed):
//   * Bitplane decomposition: two parallel 1-bit grids (low_bit, high_bit).
//     ADULT = low & high. EMPTY = ~low & ~high. EGG = low & ~high. JUVENILE = ~low & high.
//     LSB-first across x: cell at column x lives at bit (x%8) of byte (x/8) of its row.
//
//   * Range-2 neighbour count via SEPARABLE sums + vertical sliding window:
//       H[y][x] = sum of adult bits at (x-2..x+2, y)        (3-bit, max 5)
//       V[y][x] = sum of H[y-2..y+2][x]                      (5-bit, max 25)
//     V counts the entire 5x5 box INCLUDING the centre. The rule thresholds account
//     for the centre cell's own ADULT contribution:
//        EMPTY → EGG  iff V in {3,4,5}        (centre not adult, so V == A)
//        ADULT stays  iff V in {5..10}        (V = A + 1, since centre is adult)
//
//   * H computed once per row in pass 1 into a per-block scratch buffer.
//     V slid down the block in pass 2: V[y+1] = V[y] - H[y-2] + H[y+3].
//     V lives across rows in a per-block 5-bitplane buffer.
//
//   * Rule applied per-register as pure boolean ops on (V's 5 bitplanes, low, high).
//
//   * NEON intrinsics + ARMv8.2-SHA3 (EOR3 via veor3q_u8) for adder networks.
//     One 128-bit register = 128 cells.
//
//   * Threading: 8 std::jthread workers, dynamic block scheduling via an atomic
//     block counter, std::barrier between generations.
//     (std::execution::par_unseq requires linking against TBB, which is forbidden
//      by the assignment; without TBB libstdc++ falls back to single-threaded.)
//
//   * Boundary: x-wrap handled per row by loading the row's tail register as the
//     "prev" of register 0 and the head as the "next" of the last register.
//     y-wrap handled per H-row by indexing y modulo N. Interior rows hit the
//     straight-line fast path; only the top/bottom blocks pay the modulo cost.
//
//   * Buffers aligned to 64 bytes (std::aligned_alloc). No huge pages.
//
//   * O(generations × cells): exactly one full grid pass per generation; no
//     temporal blocking; no memoisation across generations.

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <latch>
#include <memory>
#include <thread>
#include <vector>

namespace {

constexpr uint8_t  ADULT     = 3;   // (kept for documentation; not used directly)

// Pick a thread count: 8 on the c8g.2xlarge target, but cap at available cores
// so this code is well-behaved on smaller dev boxes. Override with SPAWN_THREADS.
int choose_thread_count() {
    if (const char* env = std::getenv("SPAWN_THREADS")) {
        long n = std::strtol(env, nullptr, 10);
        if (n >= 1 && n <= 64) return int(n);
    }
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    return int(std::min(hc, 8u));
}

// ---------------------------------------------------------------------------
// Aligned allocation
// ---------------------------------------------------------------------------
struct AlignedDeleter { void operator()(void* p) const noexcept { std::free(p); } };
using AlignedBuf = std::unique_ptr<uint8_t[], AlignedDeleter>;

AlignedBuf aligned_alloc_bytes(size_t n) {
    size_t rounded = (n + 63) & ~size_t(63);
    void* p = std::aligned_alloc(64, rounded);
    if (!p) { std::fprintf(stderr, "aligned_alloc failed for %zu bytes\n", rounded); std::exit(2); }
    std::memset(p, 0, rounded);
    return AlignedBuf(reinterpret_cast<uint8_t*>(p));
}

// ---------------------------------------------------------------------------
// SIMD shift helpers (LSB-first bit packing within each byte).
// "shl_row_n(prev, curr)" shifts the 128 cells in 'curr' toward higher x by n,
// pulling carry bits from 'prev' (the register holding the lower-x neighbours).
// "shr_row_n(curr, next)" shifts toward lower x, pulling from 'next'.
// ---------------------------------------------------------------------------
inline uint8x16_t shl_row_1(uint8x16_t prev, uint8x16_t curr) {
    uint8x16_t carry = vextq_u8(prev, curr, 15);    // [prev[15], curr[0..14]]
    return vorrq_u8(vshlq_n_u8(curr, 1), vshrq_n_u8(carry, 7));
}
inline uint8x16_t shl_row_2(uint8x16_t prev, uint8x16_t curr) {
    uint8x16_t carry = vextq_u8(prev, curr, 15);
    return vorrq_u8(vshlq_n_u8(curr, 2), vshrq_n_u8(carry, 6));
}
inline uint8x16_t shr_row_1(uint8x16_t curr, uint8x16_t next) {
    uint8x16_t carry = vextq_u8(curr, next, 1);     // [curr[1..15], next[0]]
    return vorrq_u8(vshrq_n_u8(curr, 1), vshlq_n_u8(carry, 7));
}
inline uint8x16_t shr_row_2(uint8x16_t curr, uint8x16_t next) {
    uint8x16_t carry = vextq_u8(curr, next, 1);
    return vorrq_u8(vshrq_n_u8(curr, 2), vshlq_n_u8(carry, 6));
}

// ---------------------------------------------------------------------------
// Bit-sliced adder primitives.
// MAJ(a,b,c) = at-least-two-of-three; the carry output of a full adder.
// Full adder: sum = a ^ b ^ c (one EOR3); carry = MAJ(a,b,c) (one XOR + AND + BSL).
// ---------------------------------------------------------------------------
inline uint8x16_t maj(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
    return vbslq_u8(veorq_u8(a, b), c, vandq_u8(a, b));
}

struct FA { uint8x16_t s, c; };
inline FA full_adder(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
    return { veor3q_u8(a, b, c), maj(a, b, c) };
}

struct HA { uint8x16_t s, c; };
inline HA half_adder(uint8x16_t a, uint8x16_t b) {
    return { veorq_u8(a, b), vandq_u8(a, b) };
}

// Sum of 5 single-bit inputs into a 3-bit value (h0=LSB, h2=MSB).
struct Sum3 { uint8x16_t h0, h1, h2; };
inline Sum3 sum_of_5(uint8x16_t a, uint8x16_t b, uint8x16_t c,
                     uint8x16_t d, uint8x16_t e) {
    FA fa1 = full_adder(a, b, c);       // s weight 1, c weight 2
    FA fa2 = full_adder(d, e, fa1.s);   // s weight 1, c weight 2
    HA ha  = half_adder(fa1.c, fa2.c);  // combine the two weight-2 carries
    return { fa2.s, ha.s, ha.c };
}

// V (5-bit) += H (3-bit). Used both for initial sum-of-5 and for sliding update.
struct V5 { uint8x16_t b0, b1, b2, b3, b4; };
inline V5 add_v5_h3(V5 v, Sum3 h) {
    HA s0 = half_adder(v.b0, h.h0);
    FA s1 = full_adder(v.b1, h.h1, s0.c);
    FA s2 = full_adder(v.b2, h.h2, s1.c);
    HA s3 = half_adder(v.b3, s2.c);
    uint8x16_t b4 = veorq_u8(v.b4, s3.c);
    return { s0.s, s1.s, s2.s, s3.s, b4 };
}

// V (5-bit) -= H (3-bit). Borrow chain. Borrow-out of bit i = MAJ(~A_i, B_i, Bin).
inline V5 sub_v5_h3(V5 v, Sum3 h) {
    uint8x16_t diff0   = veorq_u8(v.b0, h.h0);
    uint8x16_t borrow0 = vbicq_u8(h.h0, v.b0);                    // ~v0 & h0
    uint8x16_t diff1   = veor3q_u8(v.b1, h.h1, borrow0);
    uint8x16_t borrow1 = maj(vmvnq_u8(v.b1), h.h1, borrow0);
    uint8x16_t diff2   = veor3q_u8(v.b2, h.h2, borrow1);
    uint8x16_t borrow2 = maj(vmvnq_u8(v.b2), h.h2, borrow1);
    uint8x16_t diff3   = veorq_u8(v.b3, borrow2);                 // H[3]=0
    uint8x16_t borrow3 = vbicq_u8(borrow2, v.b3);                 // ~v3 & borrow2
    uint8x16_t diff4   = veorq_u8(v.b4, borrow3);                 // H[4]=0
    return { diff0, diff1, diff2, diff3, diff4 };
}

// ---------------------------------------------------------------------------
// Rule predicates derived from V (5 bitplanes).
//   E = V in {3,4,5}            (EMPTY → EGG condition; centre not adult, V == A)
//   R = V in {5,6,7,8,9,10}     (ADULT stays condition; centre adult, A = V - 1)
// ---------------------------------------------------------------------------
inline uint8x16_t compute_E(V5 v) {
    // E = ~v4 & ~v3 & ( (~v2 & v1 & v0) | (v2 & ~v1) )
    uint8x16_t not_v4 = vmvnq_u8(v.b4);
    uint8x16_t not_v3 = vmvnq_u8(v.b3);
    uint8x16_t gate   = vandq_u8(not_v4, not_v3);
    uint8x16_t left   = vbicq_u8(vandq_u8(v.b1, v.b0), v.b2);     // ~v2 & v1 & v0
    uint8x16_t right  = vbicq_u8(v.b2, v.b1);                     // v2 & ~v1
    return vandq_u8(gate, vorrq_u8(left, right));
}

inline uint8x16_t compute_R(V5 v) {
    // v4=0 for V≤10. v3=0: V in {5..7} <=> v2 & (v1|v0). v3=1: V in {8..10} <=> ~v2 & ~(v1&v0).
    uint8x16_t not_v4   = vmvnq_u8(v.b4);
    uint8x16_t case_a   = vbicq_u8(vandq_u8(v.b2, vorrq_u8(v.b1, v.b0)), v.b3); // v2 & (v1|v0) & ~v3
    uint8x16_t v1v0     = vandq_u8(v.b1, v.b0);
    uint8x16_t case_b   = vandq_u8(vbicq_u8(vmvnq_u8(v1v0), v.b2), v.b3);       // ~v2 & ~(v1&v0) & v3
    return vandq_u8(not_v4, vorrq_u8(case_a, case_b));
}

// Apply the full rule: from (V, low, high) emit (next_low, next_high).
//   next_low  = (~l & (h | E)) | (l & h & R)
//   next_high = (h ^ l)        | (h & l & R)
struct LH { uint8x16_t low, high; };
inline LH apply_rule(V5 v, uint8x16_t low, uint8x16_t high) {
    uint8x16_t E      = compute_E(v);
    uint8x16_t R      = compute_R(v);
    uint8x16_t h_x_l  = veorq_u8(high, low);
    uint8x16_t h_a_l  = vandq_u8(high, low);
    uint8x16_t hl_R   = vandq_u8(h_a_l, R);
    uint8x16_t n_high = vorrq_u8(h_x_l, hl_R);
    uint8x16_t hOrE   = vorrq_u8(high, E);
    uint8x16_t branch = vbicq_u8(hOrE, low);                       // (h | E) & ~l
    uint8x16_t n_low  = vorrq_u8(branch, hl_R);
    return { n_low, n_high };
}

// ---------------------------------------------------------------------------
// Compute H bitplanes for one full row from the adult bitplane row.
// The adult bitplane is derived inline from low_row & high_row.
// h0_out, h1_out, h2_out: pointers to R_BYTES bytes each.
// x-wrap is toroidal: register r=0's "prev" is the row's last register;
// register r=R_REGS-1's "next" is register 0.
// ---------------------------------------------------------------------------
inline void compute_H_row(const uint8_t* low_row, const uint8_t* high_row,
                          size_t R_REGS,
                          uint8_t* h0_out, uint8_t* h1_out, uint8_t* h2_out) {
    auto load_adult = [&](size_t r) -> uint8x16_t {
        uint8x16_t lo = vld1q_u8(low_row  + r * 16);
        uint8x16_t hi = vld1q_u8(high_row + r * 16);
        return vandq_u8(lo, hi);
    };

    uint8x16_t prev = load_adult(R_REGS - 1);
    uint8x16_t curr = load_adult(0);
    for (size_t r = 0; r < R_REGS; ++r) {
        size_t r_next = (r + 1 == R_REGS) ? 0 : (r + 1);
        uint8x16_t next = load_adult(r_next);
        uint8x16_t s_m2 = shl_row_2(prev, curr);
        uint8x16_t s_m1 = shl_row_1(prev, curr);
        uint8x16_t s_0  = curr;
        uint8x16_t s_p1 = shr_row_1(curr, next);
        uint8x16_t s_p2 = shr_row_2(curr, next);
        Sum3 h = sum_of_5(s_m2, s_m1, s_0, s_p1, s_p2);
        vst1q_u8(h0_out + r * 16, h.h0);
        vst1q_u8(h1_out + r * 16, h.h1);
        vst1q_u8(h2_out + r * 16, h.h2);
        prev = curr;
        curr = next;
    }
}

// ---------------------------------------------------------------------------
// Block processing.
//
// One block covers 'block_rows' consecutive rows of output. To compute its H
// scratch we need 'block_rows + 4' rows of H (rows [block_start - 2 ..
// block_start + block_rows + 1]) so V can slide through all output rows.
//
// H scratch layout (per block):
//   For each H bitplane index p in {0,1,2}:
//     h_scratch[p] is a contiguous buffer of (block_rows + 4) * R_BYTES bytes.
//     Row 'i' in [0..block_rows+3] sits at offset i * R_BYTES.
//     Row i corresponds to absolute grid row ((block_start - 2 + i) mod N).
//
// V scratch layout (per block):
//   5 bitplane buffers, R_BYTES bytes each. Holds V for the row currently
//   being processed; updated in place via add_v5_h3 / sub_v5_h3.
// ---------------------------------------------------------------------------
struct ThreadScratch {
    AlignedBuf h0, h1, h2;
    AlignedBuf v0, v1, v2, v3, v4;
    size_t h_rows_cap = 0;
    size_t v_bytes_cap = 0;
};

thread_local ThreadScratch tls;

inline void ensure_scratch(size_t h_rows_needed, size_t R_BYTES) {
    if (tls.h_rows_cap < h_rows_needed) {
        size_t h_size = h_rows_needed * R_BYTES;
        tls.h0 = aligned_alloc_bytes(h_size);
        tls.h1 = aligned_alloc_bytes(h_size);
        tls.h2 = aligned_alloc_bytes(h_size);
        tls.h_rows_cap = h_rows_needed;
    }
    if (tls.v_bytes_cap < R_BYTES) {
        tls.v0 = aligned_alloc_bytes(R_BYTES);
        tls.v1 = aligned_alloc_bytes(R_BYTES);
        tls.v2 = aligned_alloc_bytes(R_BYTES);
        tls.v3 = aligned_alloc_bytes(R_BYTES);
        tls.v4 = aligned_alloc_bytes(R_BYTES);
        tls.v_bytes_cap = R_BYTES;
    }
}

inline void process_block(size_t block_start, size_t block_rows,
                          size_t N, size_t R_BYTES, size_t R_REGS,
                          const uint8_t* cur_low,  const uint8_t* cur_high,
                          uint8_t* next_low, uint8_t* next_high) {
    const size_t H_ROWS = block_rows + 4;
    ensure_scratch(H_ROWS, R_BYTES);
    uint8_t* h0 = tls.h0.get();
    uint8_t* h1 = tls.h1.get();
    uint8_t* h2 = tls.h2.get();

    // ---- Pass 1: compute H for rows [block_start - 2 .. block_start + block_rows + 1].
    // Wraparound only on the very top / very bottom blocks. Inside the loop
    // the branch is predictable: it fires at most twice per block.
    for (size_t i = 0; i < H_ROWS; ++i) {
        size_t y_signed = block_start + i;  // intended absolute row + 2 (since i=0 is block_start - 2)
        // Compute absolute row index = (block_start + i - 2 + N) % N.
        size_t y;
        if (i >= 2 && y_signed - 2 < N) {
            y = y_signed - 2;
        } else {
            // Need wrap: (block_start + i - 2 + N) % N. block_start, i, N all unsigned.
            size_t raw = block_start + i + N - 2;  // safely > 0 since N > 0
            y = raw % N;
        }
        compute_H_row(cur_low + y * R_BYTES, cur_high + y * R_BYTES, R_REGS,
                      h0 + i * R_BYTES, h1 + i * R_BYTES, h2 + i * R_BYTES);
    }

    // ---- Pass 2: initialise V for first output row, then slide.
    uint8_t* v_b0 = tls.v0.get();
    uint8_t* v_b1 = tls.v1.get();
    uint8_t* v_b2 = tls.v2.get();
    uint8_t* v_b3 = tls.v3.get();
    uint8_t* v_b4 = tls.v4.get();

    // Initial V[block_start] = H_scratch[0] + H_scratch[1] + H_scratch[2] + H_scratch[3] + H_scratch[4].
    {
        const uint8_t* H_row[5][3] = {
            { h0 + 0*R_BYTES, h1 + 0*R_BYTES, h2 + 0*R_BYTES },
            { h0 + 1*R_BYTES, h1 + 1*R_BYTES, h2 + 1*R_BYTES },
            { h0 + 2*R_BYTES, h1 + 2*R_BYTES, h2 + 2*R_BYTES },
            { h0 + 3*R_BYTES, h1 + 3*R_BYTES, h2 + 3*R_BYTES },
            { h0 + 4*R_BYTES, h1 + 4*R_BYTES, h2 + 4*R_BYTES },
        };
        for (size_t r = 0; r < R_REGS; ++r) {
            V5 v = {
                vld1q_u8(H_row[0][0] + r * 16),
                vld1q_u8(H_row[0][1] + r * 16),
                vld1q_u8(H_row[0][2] + r * 16),
                vdupq_n_u8(0), vdupq_n_u8(0)
            };
            for (int s = 1; s < 5; ++s) {
                Sum3 h = { vld1q_u8(H_row[s][0] + r * 16),
                           vld1q_u8(H_row[s][1] + r * 16),
                           vld1q_u8(H_row[s][2] + r * 16) };
                v = add_v5_h3(v, h);
            }
            vst1q_u8(v_b0 + r * 16, v.b0);
            vst1q_u8(v_b1 + r * 16, v.b1);
            vst1q_u8(v_b2 + r * 16, v.b2);
            vst1q_u8(v_b3 + r * 16, v.b3);
            vst1q_u8(v_b4 + r * 16, v.b4);
        }
    }

    // Apply rule to first output row.
    {
        const uint8_t* low_row  = cur_low  + block_start * R_BYTES;
        const uint8_t* high_row = cur_high + block_start * R_BYTES;
        uint8_t* nl_row = next_low  + block_start * R_BYTES;
        uint8_t* nh_row = next_high + block_start * R_BYTES;
        for (size_t r = 0; r < R_REGS; ++r) {
            V5 v = { vld1q_u8(v_b0 + r * 16), vld1q_u8(v_b1 + r * 16),
                     vld1q_u8(v_b2 + r * 16), vld1q_u8(v_b3 + r * 16),
                     vld1q_u8(v_b4 + r * 16) };
            uint8x16_t lo = vld1q_u8(low_row  + r * 16);
            uint8x16_t hi = vld1q_u8(high_row + r * 16);
            LH out = apply_rule(v, lo, hi);
            vst1q_u8(nl_row + r * 16, out.low);
            vst1q_u8(nh_row + r * 16, out.high);
        }
    }

    // Slide for remaining output rows.
    for (size_t k = 1; k < block_rows; ++k) {
        const uint8_t* h_out_0 = h0 + (k - 1) * R_BYTES;
        const uint8_t* h_out_1 = h1 + (k - 1) * R_BYTES;
        const uint8_t* h_out_2 = h2 + (k - 1) * R_BYTES;
        const uint8_t* h_in_0  = h0 + (k + 4) * R_BYTES;
        const uint8_t* h_in_1  = h1 + (k + 4) * R_BYTES;
        const uint8_t* h_in_2  = h2 + (k + 4) * R_BYTES;
        size_t y = block_start + k;
        const uint8_t* low_row  = cur_low  + y * R_BYTES;
        const uint8_t* high_row = cur_high + y * R_BYTES;
        uint8_t* nl_row = next_low  + y * R_BYTES;
        uint8_t* nh_row = next_high + y * R_BYTES;

        for (size_t r = 0; r < R_REGS; ++r) {
            V5 v = { vld1q_u8(v_b0 + r * 16), vld1q_u8(v_b1 + r * 16),
                     vld1q_u8(v_b2 + r * 16), vld1q_u8(v_b3 + r * 16),
                     vld1q_u8(v_b4 + r * 16) };
            Sum3 h_out = { vld1q_u8(h_out_0 + r * 16),
                           vld1q_u8(h_out_1 + r * 16),
                           vld1q_u8(h_out_2 + r * 16) };
            Sum3 h_in  = { vld1q_u8(h_in_0  + r * 16),
                           vld1q_u8(h_in_1  + r * 16),
                           vld1q_u8(h_in_2  + r * 16) };
            v = sub_v5_h3(v, h_out);
            v = add_v5_h3(v, h_in);
            vst1q_u8(v_b0 + r * 16, v.b0);
            vst1q_u8(v_b1 + r * 16, v.b1);
            vst1q_u8(v_b2 + r * 16, v.b2);
            vst1q_u8(v_b3 + r * 16, v.b3);
            vst1q_u8(v_b4 + r * 16, v.b4);

            uint8x16_t lo = vld1q_u8(low_row  + r * 16);
            uint8x16_t hi = vld1q_u8(high_row + r * 16);
            LH out = apply_rule(v, lo, hi);
            vst1q_u8(nl_row + r * 16, out.low);
            vst1q_u8(nh_row + r * 16, out.high);
        }
    }
}

// ---------------------------------------------------------------------------
// Byte ↔ bitplane conversion (untimed).
// ---------------------------------------------------------------------------
void bytes_to_bitplanes(const uint8_t* bytes, uint8_t* low, uint8_t* high, size_t N) {
    const size_t total = N * N;
    for (size_t i = 0; i < total; i += 8) {
        uint8_t lo = 0, hi = 0;
        for (int j = 0; j < 8; ++j) {
            uint8_t v = bytes[i + j];
            lo |= (v & 1u) << j;
            hi |= ((v >> 1) & 1u) << j;
        }
        low [i >> 3] = lo;
        high[i >> 3] = hi;
    }
}

void bitplanes_to_bytes(const uint8_t* low, const uint8_t* high, uint8_t* bytes, size_t N) {
    const size_t total_bp = (N * N) / 8;
    for (size_t p = 0; p < total_bp; ++p) {
        uint8_t lo = low[p], hi = high[p];
        for (int j = 0; j < 8; ++j) {
            bytes[p * 8 + j] = uint8_t(((lo >> j) & 1u) | (((hi >> j) & 1u) << 1));
        }
    }
}

// ---------------------------------------------------------------------------
// Pick a block size: aim for many blocks per thread for load balancing, but cap
// so the per-block H scratch stays comfortably in L2 (2 MiB per core).
// At N = 32768 with block_rows = 128, H scratch = 132 * 3 * 4096 = ~1.6 MiB.
// ---------------------------------------------------------------------------
size_t choose_block_rows(size_t N) {
    size_t br = N / 8;                   // ~8 blocks per thread per axis
    if (br > 128) br = 128;
    if (br < 8)   br = 8;
    // Make N divisible by br (always true for power-of-two N and br).
    while (N % br != 0) br /= 2;
    return br;
}

// ---------------------------------------------------------------------------
// CPU pinning. Best-effort.
// ---------------------------------------------------------------------------
void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

} // namespace

int main(int argc, char** argv) {
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
        generations = int(g);
    }

    // ---- Read input ----
    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) { std::fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]); return 2; }
    uint64_t width = 0, height = 0;
    if (std::fread(&width,  sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: input file too short (header)\n");
        std::fclose(fin); return 3;
    }
    if (width == 0 || width != height) {
        std::fprintf(stderr, "Error: grid must be square and non-empty (%" PRIu64 " x %" PRIu64 ")\n", width, height);
        std::fclose(fin); return 3;
    }
    const size_t N = size_t(width);
    if ((N & (N - 1)) != 0) {
        std::fprintf(stderr, "Error: grid size must be a power of two, got %zu\n", N);
        std::fclose(fin); return 3;
    }
    if (N < 8) {
        std::fprintf(stderr, "Error: grid size must be >= 8 (one byte per bitplane row)\n");
        std::fclose(fin); return 3;
    }
    const size_t total_bytes = N * N;
    const size_t R_BYTES     = N / 8;            // bytes per bitplane row
    const size_t R_REGS      = R_BYTES / 16;     // NEON registers per bitplane row
    if (R_REGS == 0) {
        std::fprintf(stderr, "Error: grid size must be >= 128 cells wide for NEON path\n");
        std::fclose(fin); return 3;
    }

    AlignedBuf byte_buf = aligned_alloc_bytes(total_bytes);
    if (std::fread(byte_buf.get(), 1, total_bytes, fin) != total_bytes) {
        std::fprintf(stderr, "Error: input file too short (cell data)\n");
        std::fclose(fin); return 4;
    }
    std::fclose(fin);

    // ---- Convert byte grid → bitplanes (outside timed region) ----
    AlignedBuf low_a  = aligned_alloc_bytes(N * R_BYTES);
    AlignedBuf high_a = aligned_alloc_bytes(N * R_BYTES);
    AlignedBuf low_b  = aligned_alloc_bytes(N * R_BYTES);
    AlignedBuf high_b = aligned_alloc_bytes(N * R_BYTES);
    bytes_to_bitplanes(byte_buf.get(), low_a.get(), high_a.get(), N);

    // ---- Set up workers and shared block counter ----
    const size_t block_rows = choose_block_rows(N);
    const size_t n_blocks   = N / block_rows;

    uint8_t* cur_low   = low_a.get();
    uint8_t* cur_high  = high_a.get();
    uint8_t* next_low  = low_b.get();
    uint8_t* next_high = high_b.get();

    const int n_threads = choose_thread_count();

    std::atomic<size_t> block_counter{0};

    // Barrier with completion: swap buffers and reset the block counter once per generation.
    // Only workers participate. Completion fires exactly once per generation (after all
    // workers finish their block pulls), so the swap happens exactly once per gen.
    std::barrier gen_barrier(n_threads, [&]() noexcept {
        std::swap(cur_low,  next_low);
        std::swap(cur_high, next_high);
        block_counter.store(0, std::memory_order_relaxed);
    });

    // Latch to release workers at the start of timing.
    std::latch start_latch(1);

    std::vector<std::jthread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t]() {
            pin_to_cpu(t);
            // Pre-allocate per-thread scratch (block_rows + 4 H rows + V planes).
            ensure_scratch(block_rows + 4, R_BYTES);
            start_latch.wait();
            for (int gen = 0; gen < generations; ++gen) {
                while (true) {
                    size_t b = block_counter.fetch_add(1, std::memory_order_relaxed);
                    if (b >= n_blocks) break;
                    process_block(b * block_rows, block_rows, N, R_BYTES, R_REGS,
                                  cur_low, cur_high, next_low, next_high);
                }
                gen_barrier.arrive_and_wait();
            }
        });
    }

    // ---- Simulate (timed) ----
    auto t0 = std::chrono::steady_clock::now();
    start_latch.count_down();
    for (auto& w : workers) w.join();
    auto t1 = std::chrono::steady_clock::now();
    workers.clear();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    // ---- Convert bitplanes → byte grid ----
    bitplanes_to_bytes(cur_low, cur_high, byte_buf.get(), N);

    // ---- Write output ----
    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]); return 5; }
    if (std::fwrite(&width,  sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(byte_buf.get(), 1, total_bytes, fout) != total_bytes) {
        std::fprintf(stderr, "Error: write error on '%s'\n", argv[2]);
        std::fclose(fout); return 6;
    }
    std::fclose(fout);
    return 0;
}
