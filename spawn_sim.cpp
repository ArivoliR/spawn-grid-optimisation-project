// spawn_sim.cpp — Minimal bitplane / EOR3 / threaded variant
// with vertical sliding-window for the neighbour sum.
//
// Derived from reference/spawn_sim.cpp. Three additions only, kept as
// simple as possible:
//
//   (1) BITPLANE STORAGE
//       Each cell's 2-bit state is split into two parallel 1-bit grids
//       (low_bit, high_bit). ADULT iff both bits set. LSB-first across x.
//
//   (2) NEON SIMD + ARMv8.2-SHA3 EOR3
//       The step is computed on 128 cells per NEON register. The 5x5
//       neighbour-ADULT count is built as a separable bit-sliced adder
//       network: a per-row horizontal 5-input sum (H), then a vertical
//       sliding window over 5 successive H rows (V). Every full-adder
//       sum uses one EOR3 instruction (veor3q_u8 in <arm_neon.h>;
//       ARMv8.2-SHA3 extension, native on Neoverse-V2).
//
//       Sliding-window detail: V across y is maintained per cell. As
//       the strip sweep advances by one row, V is updated as
//           V_new = V - H[y-2] + H[y+3]
//       (one bit-sliced subtract + one add per output row). The 5 most
//       recent H rows live in a 5-slot ring buffer.
//
//       Centre handling: V is the 25-cell box sum (includes (x,y)).
//       Before applying the rule, A is computed as V minus the centre's
//       adult bit (one 5-bit-minus-1-bit borrow chain).
//
//   (3) THREADING via std::thread + atomic-counter barrier
//       The grid is partitioned into n_threads horizontal strips. Each
//       strip is owned by one worker thread. Between generations the
//       threads synchronise on a hand-rolled atomic barrier; the last
//       thread to arrive swaps the cur/next buffer pointers and advances
//       a generation counter.
//
// Total work: O(generations * cells). Each cell touched a constant
// number of times per generation.

#include <arm_neon.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static const int RANGE = 2;    // range-2 Moore neighbourhood

static const uint8_t EMPTY    = 0;
static const uint8_t EGG      = 1;
static const uint8_t JUVENILE = 2;
static const uint8_t ADULT    = 3;

// ---------------------------------------------------------------------------
// Byte grid  <->  bitplane conversion (outside timed region).
// LSB-first across x: cell at column x  -> bit (x % 8) of byte (x / 8).
// ---------------------------------------------------------------------------
static void bytes_to_bitplanes(const uint8_t* bytes,
                               uint8_t* low_plane, uint8_t* high_plane, size_t N)
{
    const size_t total = N * N;
    for (size_t i = 0; i < total; i += 8) {
        uint8_t lo = 0, hi = 0;
        for (int j = 0; j < 8; ++j) {
            uint8_t v = bytes[i + j];
            lo |= uint8_t((v & 1u)        << j);
            hi |= uint8_t(((v >> 1) & 1u) << j);
        }
        low_plane [i >> 3] = lo;
        high_plane[i >> 3] = hi;
    }
}

static void bitplanes_to_bytes(const uint8_t* low_plane, const uint8_t* high_plane,
                               uint8_t* bytes, size_t N)
{
    const size_t total_bp = (N * N) / 8;
    for (size_t p = 0; p < total_bp; ++p) {
        uint8_t lo = low_plane[p], hi = high_plane[p];
        for (int j = 0; j < 8; ++j) {
            bytes[p * 8 + j] = uint8_t(((lo >> j) & 1u) | (((hi >> j) & 1u) << 1));
        }
    }
}

// ---------------------------------------------------------------------------
// Row-shift SIMD helpers (LSB-first; +n = toward higher x).
// Carry bits cross the 128-bit register boundary via vextq_u8.
// ---------------------------------------------------------------------------
static inline uint8x16_t shl_row_1(uint8x16_t prev, uint8x16_t curr) {
    uint8x16_t carry = vextq_u8(prev, curr, 15);
    return vorrq_u8(vshlq_n_u8(curr, 1), vshrq_n_u8(carry, 7));
}
static inline uint8x16_t shl_row_2(uint8x16_t prev, uint8x16_t curr) {
    uint8x16_t carry = vextq_u8(prev, curr, 15);
    return vorrq_u8(vshlq_n_u8(curr, 2), vshrq_n_u8(carry, 6));
}
static inline uint8x16_t shr_row_1(uint8x16_t curr, uint8x16_t next) {
    uint8x16_t carry = vextq_u8(curr, next, 1);
    return vorrq_u8(vshrq_n_u8(curr, 1), vshlq_n_u8(carry, 7));
}
static inline uint8x16_t shr_row_2(uint8x16_t curr, uint8x16_t next) {
    uint8x16_t carry = vextq_u8(curr, next, 1);
    return vorrq_u8(vshrq_n_u8(curr, 2), vshlq_n_u8(carry, 6));
}

// ---------------------------------------------------------------------------
// Full adder using EOR3 (3-input XOR).
//   sum   = a XOR b XOR c                       -- single EOR3 instruction
//   carry = (a AND b) OR (c AND (a XOR b))      -- MAJ(a,b,c) via BSL
// ---------------------------------------------------------------------------
static inline uint8x16_t fa_sum  (uint8x16_t a, uint8x16_t b, uint8x16_t c) {
    return veor3q_u8(a, b, c);
}
static inline uint8x16_t fa_carry(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
    return vbslq_u8(veorq_u8(a, b), c, vandq_u8(a, b));
}

// ---------------------------------------------------------------------------
// Sum of 5 single-bit inputs -> 3-bit value {h0=LSB, h1, h2=MSB}.
// Two full-adders + one half-adder; two EOR3 uses (one per FA sum).
// ---------------------------------------------------------------------------
struct H3 { uint8x16_t h0, h1, h2; };
static inline H3 sum_of_5(uint8x16_t a, uint8x16_t b, uint8x16_t c,
                          uint8x16_t d, uint8x16_t e)
{
    uint8x16_t s1 = fa_sum  (a, b, c);
    uint8x16_t c1 = fa_carry(a, b, c);
    uint8x16_t s2 = fa_sum  (d, e, s1);
    uint8x16_t c2 = fa_carry(d, e, s1);
    uint8x16_t s3 = veorq_u8(c1, c2);
    uint8x16_t c3 = vandq_u8(c1, c2);
    return { s2, s3, c3 };
}

// ---------------------------------------------------------------------------
// 5-bit accumulator type.
// ---------------------------------------------------------------------------
struct V5 { uint8x16_t b0, b1, b2, b3, b4; };

// V (5-bit) += H (3-bit). Bits 3 and 4 of H are zero, so the upper carry
// chain is a half-adder + plain XOR.
static inline V5 v5_add_h3(V5 v, H3 h)
{
    uint8x16_t s0 = veorq_u8(v.b0, h.h0);
    uint8x16_t c0 = vandq_u8(v.b0, h.h0);
    uint8x16_t s1 = fa_sum  (v.b1, h.h1, c0);
    uint8x16_t c1 = fa_carry(v.b1, h.h1, c0);
    uint8x16_t s2 = fa_sum  (v.b2, h.h2, c1);
    uint8x16_t c2 = fa_carry(v.b2, h.h2, c1);
    uint8x16_t s3 = veorq_u8(v.b3, c2);
    uint8x16_t c3 = vandq_u8(v.b3, c2);
    uint8x16_t s4 = veorq_u8(v.b4, c3);
    return { s0, s1, s2, s3, s4 };
}

// V (5-bit) -= H (3-bit). Bit-i diff = V_i XOR H_i XOR borrow_in (uses EOR3
// at bits 1, 2). Borrow-out formula (subtractor):
//   borrow_out = (~A & (B | Bin)) | (B & Bin)
// Equivalent to MAJ(~A, B, Bin). Implemented with vbicq + vorrq + vandq
// (no explicit vmvn needed).
// Upper bits (H_3 = H_4 = 0) reduce to plain XOR + propagation.
static inline V5 v5_sub_h3(V5 v, H3 h)
{
    uint8x16_t d0 = veorq_u8(v.b0, h.h0);
    uint8x16_t b0 = vbicq_u8(h.h0, v.b0);                        // ~v0 & h0
    uint8x16_t d1 = veor3q_u8(v.b1, h.h1, b0);
    uint8x16_t b1 = vorrq_u8(vbicq_u8(vorrq_u8(h.h1, b0), v.b1),
                             vandq_u8(h.h1, b0));                // (~v1 & (h1|b0)) | (h1 & b0)
    uint8x16_t d2 = veor3q_u8(v.b2, h.h2, b1);
    uint8x16_t b2 = vorrq_u8(vbicq_u8(vorrq_u8(h.h2, b1), v.b2),
                             vandq_u8(h.h2, b1));
    uint8x16_t d3 = veorq_u8(v.b3, b2);
    uint8x16_t b3 = vbicq_u8(b2, v.b3);                          // ~v3 & b2
    uint8x16_t d4 = veorq_u8(v.b4, b3);
    return { d0, d1, d2, d3, d4 };
}

// V (5-bit) -= b (1-bit). Subtracts the centre's adult bit to convert
// the 25-cell box sum into the 24-neighbour count A. Borrow chain only.
static inline V5 v5_sub_b(V5 v, uint8x16_t b)
{
    uint8x16_t bo = b;                          // initial borrow = the 1-bit
    uint8x16_t d0 = veorq_u8(v.b0, bo);
    bo = vbicq_u8(bo, v.b0);                    // ~v0 & bo
    uint8x16_t d1 = veorq_u8(v.b1, bo);
    bo = vbicq_u8(bo, v.b1);
    uint8x16_t d2 = veorq_u8(v.b2, bo);
    bo = vbicq_u8(bo, v.b2);
    uint8x16_t d3 = veorq_u8(v.b3, bo);
    bo = vbicq_u8(bo, v.b3);
    uint8x16_t d4 = veorq_u8(v.b4, bo);
    return { d0, d1, d2, d3, d4 };
}

// ---------------------------------------------------------------------------
// Compute H for one full row (separable horizontal 5-input sum).
//   H[y][x] = adult[y][x-2] + adult[y][x-1] + adult[y][x] + adult[y][x+1] + adult[y][x+2]
// adult bitplane is derived inline as (low & high). x wraps toroidally.
// ---------------------------------------------------------------------------
static void compute_H_row(const uint8_t* low_row, const uint8_t* high_row,
                          size_t R_REGS,
                          uint8_t* h0_out, uint8_t* h1_out, uint8_t* h2_out)
{
    auto load_adult = [&](size_t r) -> uint8x16_t {
        return vandq_u8(vld1q_u8(low_row + r * 16), vld1q_u8(high_row + r * 16));
    };
    uint8x16_t prev = load_adult(R_REGS - 1);    // x-wrap: prev of register 0 = last register
    uint8x16_t curr = load_adult(0);
    for (size_t r = 0; r < R_REGS; ++r) {
        size_t r_next = (r + 1 == R_REGS) ? 0 : (r + 1);
        uint8x16_t next = load_adult(r_next);
        H3 h = sum_of_5(shl_row_2(prev, curr),
                        shl_row_1(prev, curr),
                        curr,
                        shr_row_1(curr, next),
                        shr_row_2(curr, next));
        vst1q_u8(h0_out + r * 16, h.h0);
        vst1q_u8(h1_out + r * 16, h.h1);
        vst1q_u8(h2_out + r * 16, h.h2);
        prev = curr;
        curr = next;
    }
}

// ---------------------------------------------------------------------------
// Per-thread scratch (5-slot H ring + persistent 5-bitplane V buffer).
// ---------------------------------------------------------------------------
struct ThreadScratch {
    std::vector<uint8_t> h_ring[5][3];   // 5 ring slots, 3 bitplanes each
    std::vector<uint8_t> v_planes[5];    // 5 bitplanes of V (running sum)
    size_t bytes_cap = 0;
};
static thread_local ThreadScratch tls;

static void ensure_scratch(size_t R_BYTES)
{
    if (tls.bytes_cap >= R_BYTES) return;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) tls.h_ring[i][j].assign(R_BYTES, 0);
        tls.v_planes[i].assign(R_BYTES, 0);
    }
    tls.bytes_cap = R_BYTES;
}

// ---------------------------------------------------------------------------
// step_strip — process one generation's worth of output for rows
// [y_start, y_end).  V is kept persistent across rows via sliding window.
// ---------------------------------------------------------------------------
static void step_strip(const uint8_t* cur_low,  const uint8_t* cur_high,
                       uint8_t* next_low, uint8_t* next_high,
                       size_t N, size_t R_BYTES, size_t R_REGS,
                       size_t y_start, size_t y_end)
{
    ensure_scratch(R_BYTES);
    uint8_t* hring[5][3];
    uint8_t* vp[5];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) hring[i][j] = tls.h_ring[i][j].data();
        vp[i] = tls.v_planes[i].data();
    }

    auto y_wrap = [&](long y) -> size_t {
        return size_t((y % long(N) + long(N)) % long(N));
    };

    // Initial fill: H for rows y_start-2 .. y_start+2 -> slots 0..4.
    for (int i = 0; i < 5; ++i) {
        size_t y_abs = y_wrap(long(y_start) + (i - 2));
        compute_H_row(cur_low + y_abs * R_BYTES, cur_high + y_abs * R_BYTES, R_REGS,
                      hring[i][0], hring[i][1], hring[i][2]);
    }

    // Initial V = sum of the 5 H rows just computed.
    for (size_t r = 0; r < R_REGS; ++r) {
        V5 v{
            vld1q_u8(hring[0][0] + r * 16),
            vld1q_u8(hring[0][1] + r * 16),
            vld1q_u8(hring[0][2] + r * 16),
            vdupq_n_u8(0),
            vdupq_n_u8(0)
        };
        for (int i = 1; i < 5; ++i) {
            H3 h{ vld1q_u8(hring[i][0] + r * 16),
                  vld1q_u8(hring[i][1] + r * 16),
                  vld1q_u8(hring[i][2] + r * 16) };
            v = v5_add_h3(v, h);
        }
        vst1q_u8(vp[0] + r * 16, v.b0);
        vst1q_u8(vp[1] + r * 16, v.b1);
        vst1q_u8(vp[2] + r * 16, v.b2);
        vst1q_u8(vp[3] + r * 16, v.b3);
        vst1q_u8(vp[4] + r * 16, v.b4);
    }

    const size_t strip_rows = y_end - y_start;
    for (size_t k = 0; k < strip_rows; ++k) {
        const size_t y = y_start + k;
        const size_t row_off = y * R_BYTES;

        // Apply rule for row y using V (25-cell box sum, includes centre).
        // Fused with the "subtract H[y-2]" half of the V slide so V stays in
        // registers between the two operations.
        const int slot = int(k) % 5;     // slot currently holds H[y-2]
        const bool more = (k + 1 < strip_rows);

        for (size_t r = 0; r < R_REGS; ++r) {
            const size_t reg_off = r * 16;

            // Load V from per-row scratch.
            V5 V{
                vld1q_u8(vp[0] + reg_off),
                vld1q_u8(vp[1] + reg_off),
                vld1q_u8(vp[2] + reg_off),
                vld1q_u8(vp[3] + reg_off),
                vld1q_u8(vp[4] + reg_off)
            };

            // Centre adult = low & high at (x, y).
            uint8x16_t low  = vld1q_u8(cur_low  + row_off + reg_off);
            uint8x16_t high = vld1q_u8(cur_high + row_off + reg_off);
            uint8x16_t centre = vandq_u8(low, high);

            // A = V - centre (24-neighbour count).
            V5 A = v5_sub_b(V, centre);

            // E = A in {3, 4, 5}
            //   = ~a4 & ~a3 & ( (~a2 & a1 & a0) | (a2 & ~a1) )
            uint8x16_t hi_zero = vandq_u8(vmvnq_u8(A.b4), vmvnq_u8(A.b3));
            uint8x16_t bA = vbicq_u8(vandq_u8(A.b1, A.b0), A.b2);   // ~a2 & a1 & a0
            uint8x16_t bB = vbicq_u8(A.b2, A.b1);                    // a2 & ~a1
            uint8x16_t E  = vandq_u8(hi_zero, vorrq_u8(bA, bB));

            // R = A in {4..9}
            //   = ~a4 & ( (~a3 & a2) | (a3 & ~a2 & ~a1) )
            uint8x16_t not_a4 = vmvnq_u8(A.b4);
            uint8x16_t cA     = vbicq_u8(A.b2, A.b3);                      // ~a3 & a2
            uint8x16_t cB     = vandq_u8(A.b3, vbicq_u8(vmvnq_u8(A.b1), A.b2));  // a3 & ~a2 & ~a1
            uint8x16_t R      = vandq_u8(not_a4, vorrq_u8(cA, cB));

            // next_low  = (~low & (high | E)) | (low & high & R)
            // next_high = (high ^ low)        | (high & low & R)
            uint8x16_t h_x_l = veorq_u8(high, low);
            uint8x16_t h_a_l = vandq_u8(high, low);
            uint8x16_t hl_R  = vandq_u8(h_a_l, R);
            uint8x16_t nxt_h = vorrq_u8(h_x_l, hl_R);
            uint8x16_t branch_low = vbicq_u8(vorrq_u8(high, E), low);
            uint8x16_t nxt_l = vorrq_u8(branch_low, hl_R);
            vst1q_u8(next_low  + row_off + reg_off, nxt_l);
            vst1q_u8(next_high + row_off + reg_off, nxt_h);

            // Fused slide-subtract: V -= H[y-2] (the slot we're about to overwrite).
            if (more) {
                H3 H_out{
                    vld1q_u8(hring[slot][0] + reg_off),
                    vld1q_u8(hring[slot][1] + reg_off),
                    vld1q_u8(hring[slot][2] + reg_off)
                };
                V = v5_sub_h3(V, H_out);
                vst1q_u8(vp[0] + reg_off, V.b0);
                vst1q_u8(vp[1] + reg_off, V.b1);
                vst1q_u8(vp[2] + reg_off, V.b2);
                vst1q_u8(vp[3] + reg_off, V.b3);
                vst1q_u8(vp[4] + reg_off, V.b4);
            }
        }

        if (more) {
            // Compute H for row y+3 into the slot we just freed.
            size_t y_new = y_wrap(long(y) + 3);
            compute_H_row(cur_low  + y_new * R_BYTES,
                          cur_high + y_new * R_BYTES,
                          R_REGS,
                          hring[slot][0], hring[slot][1], hring[slot][2]);

            // Slide-add: V += H[y+3].
            for (size_t r = 0; r < R_REGS; ++r) {
                const size_t reg_off = r * 16;
                V5 V{
                    vld1q_u8(vp[0] + reg_off),
                    vld1q_u8(vp[1] + reg_off),
                    vld1q_u8(vp[2] + reg_off),
                    vld1q_u8(vp[3] + reg_off),
                    vld1q_u8(vp[4] + reg_off)
                };
                H3 H_in{
                    vld1q_u8(hring[slot][0] + reg_off),
                    vld1q_u8(hring[slot][1] + reg_off),
                    vld1q_u8(hring[slot][2] + reg_off)
                };
                V = v5_add_h3(V, H_in);
                vst1q_u8(vp[0] + reg_off, V.b0);
                vst1q_u8(vp[1] + reg_off, V.b1);
                vst1q_u8(vp[2] + reg_off, V.b2);
                vst1q_u8(vp[3] + reg_off, V.b3);
                vst1q_u8(vp[4] + reg_off, V.b4);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thread count chooser (capped at 8 for c8g.2xlarge target).
// ---------------------------------------------------------------------------
static int choose_thread_count()
{
    if (const char* env = std::getenv("SPAWN_THREADS")) {
        long n = std::strtol(env, nullptr, 10);
        if (n >= 1 && n <= 64) return int(n);
    }
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    return int(std::min(hc, 8u));
}

// ===========================================================================
// MAIN
// ===========================================================================
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
        generations = int(g);
    }

    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) { std::fprintf(stderr, "Error: cannot open '%s'\n", argv[1]); return 2; }
    uint64_t width = 0, height = 0;
    if (std::fread(&width,  sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: input header truncated\n");
        std::fclose(fin); return 3;
    }
    if (width == 0 || width != height) {
        std::fprintf(stderr, "Error: grid must be square, got %" PRIu64 " x %" PRIu64 "\n", width, height);
        std::fclose(fin); return 3;
    }
    const size_t N = size_t(width);
    if ((N & (N - 1)) != 0 || N < 128) {
        std::fprintf(stderr, "Error: grid size must be a power of two >= 128, got %zu\n", N);
        std::fclose(fin); return 3;
    }
    const size_t total_bytes = N * N;
    const size_t R_BYTES     = N / 8;
    const size_t R_REGS      = R_BYTES / 16;

    std::vector<uint8_t> byte_buf(total_bytes);
    if (std::fread(byte_buf.data(), 1, total_bytes, fin) != total_bytes) {
        std::fprintf(stderr, "Error: input cell data truncated\n");
        std::fclose(fin); return 4;
    }
    std::fclose(fin);

    std::vector<uint8_t> low_a (N * R_BYTES);
    std::vector<uint8_t> high_a(N * R_BYTES);
    std::vector<uint8_t> low_b (N * R_BYTES);
    std::vector<uint8_t> high_b(N * R_BYTES);
    bytes_to_bitplanes(byte_buf.data(), low_a.data(), high_a.data(), N);

    uint8_t* cur_low   = low_a.data();
    uint8_t* cur_high  = high_a.data();
    uint8_t* next_low  = low_b.data();
    uint8_t* next_high = high_b.data();

    const int n_threads = choose_thread_count();
    std::atomic<int> arrival_count{0};
    std::atomic<int> gen_done{0};
    std::atomic<bool> started{false};

    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        size_t y_start = (size_t(t)     * N) / size_t(n_threads);
        size_t y_end   = (size_t(t + 1) * N) / size_t(n_threads);
        workers.emplace_back([&, y_start, y_end]() {
            while (!started.load(std::memory_order_acquire))
                std::this_thread::yield();
            for (int gen = 0; gen < generations; ++gen) {
                step_strip(cur_low, cur_high, next_low, next_high,
                           N, R_BYTES, R_REGS, y_start, y_end);

                int arrived = arrival_count.fetch_add(1, std::memory_order_acq_rel) + 1;
                if (arrived == n_threads) {
                    std::swap(cur_low,  next_low);
                    std::swap(cur_high, next_high);
                    arrival_count.store(0, std::memory_order_relaxed);
                    gen_done.store(gen + 1, std::memory_order_release);
                } else {
                    while (gen_done.load(std::memory_order_acquire) <= gen)
                        std::this_thread::yield();
                }
            }
        });
    }

    auto t0 = std::chrono::steady_clock::now();
    started.store(true, std::memory_order_release);
    for (auto& w : workers) w.join();
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    bitplanes_to_bytes(cur_low, cur_high, byte_buf.data(), N);
    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::fprintf(stderr, "Error: cannot open '%s'\n", argv[2]); return 5; }
    if (std::fwrite(&width,           sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height,          sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(byte_buf.data(),  1, total_bytes, fout) != total_bytes) {
        std::fprintf(stderr, "Error: write failed on '%s'\n", argv[2]);
        std::fclose(fout); return 6;
    }
    std::fclose(fout);
    return 0;
}
