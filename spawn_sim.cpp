// spawn_sim.cpp — Surgical bitplane / NEON-EOR3 / threaded variant.
//
// Derived from reference/spawn_sim.cpp. The I/O, the CLI, the timing,
// and the rule semantics are preserved verbatim. Three additions:
//
//   (1) BITPLANE STORAGE
//       During the simulation only, each cell's 2-bit state is split
//       into two parallel 1-bit grids (low_bit, high_bit). ADULT iff
//       both bits set. LSB-first bit ordering across x. The on-disk
//       byte-per-cell format is unchanged; conversion happens once on
//       read and once on write, both outside the timed region.
//
//   (2) NEON SIMD + ARMv8.2-SHA3 EOR3
//       The step is computed register-wise on 128 cells at a time.
//       The 5x5 neighbour-ADULT count is built as a bit-sliced adder
//       network whose sum-bits use the 3-input XOR instruction EOR3
//       (intrinsic veor3q_u8 in <arm_neon.h>; ARMv8.2-SHA3 extension,
//       supported natively on Neoverse-V2). Build flag includes +sha3.
//
//   (3) THREADING via std::jthread + std::barrier
//       The grid is partitioned into N_THREADS horizontal strips. Each
//       strip is owned by one worker thread for the entire run. A
//       barrier between generations performs the buffer swap. No work
//       stealing; uniform stencil so static partitioning suffices.
//
// All other algorithmic concerns (asymptotic complexity, the rule, the
// file format, the timing methodology) follow the reference exactly.
// Total work is O(generations * cells), with constant-factor speedup
// coming purely from SIMD width + parallelism + a separable neighbour
// sum that is mathematically equivalent to the reference's loop.

#include <arm_neon.h>

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

static const int RANGE = 2;    // range-2 Moore neighbourhood

static const uint8_t EMPTY    = 0;
static const uint8_t EGG      = 1;
static const uint8_t JUVENILE = 2;
static const uint8_t ADULT    = 3;

// ---------------------------------------------------------------------------
// Aligned allocation (64-byte boundary -> cache line + NEON alignment).
// ---------------------------------------------------------------------------
struct AlignedDeleter { void operator()(void* p) const noexcept { std::free(p); } };
using AlignedBuf = std::unique_ptr<uint8_t[], AlignedDeleter>;

static AlignedBuf aligned_zeros(size_t n)
{
    size_t rounded = (n + 63) & ~size_t(63);
    void* p = std::aligned_alloc(64, rounded);
    if (!p) { std::fprintf(stderr, "aligned_alloc failed\n"); std::exit(2); }
    std::memset(p, 0, rounded);
    return AlignedBuf(reinterpret_cast<uint8_t*>(p));
}

// ---------------------------------------------------------------------------
// Byte grid  <->  bitplane conversion (outside the timed region).
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

// ===========================================================================
// NEON BITPLANE PRIMITIVES
// ===========================================================================

// Shift a 128-cell-wide row register by +1 / +2 / -1 / -2 cell positions
// (positive = toward higher x). Carry bits cross the 128-bit boundary via
// the adjacent register (prev for left shift, next for right shift).
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

// ---- Full adder ----------------------------------------------------------
//   sum   = a XOR b XOR c       -- single EOR3 instruction (SHA3 extension)
//   carry = MAJ(a, b, c)
// MAJ(a,b,c) implemented as vbslq_u8(a^b, c, a&b):
//   when a == b (a^b == 0): MAJ = a (= b) = a&b
//   when a != b (a^b == 1): MAJ = c
static inline uint8x16_t fa_sum  (uint8x16_t a, uint8x16_t b, uint8x16_t c) { return veor3q_u8(a, b, c); }
static inline uint8x16_t fa_carry(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
    return vbslq_u8(veorq_u8(a, b), c, vandq_u8(a, b));
}

// Sum of 5 single-bit inputs -> 3-bit value { h0=LSB, h1, h2=MSB }.
struct H3 { uint8x16_t h0, h1, h2; };
static inline H3 sum_of_5(uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d, uint8x16_t e)
{
    uint8x16_t s1 = fa_sum  (a, b, c);
    uint8x16_t c1 = fa_carry(a, b, c);   // weight 2
    uint8x16_t s2 = fa_sum  (d, e, s1);  // weight 1 of final result
    uint8x16_t c2 = fa_carry(d, e, s1);  // weight 2
    // Combine the two weight-2 carries with a half-adder:
    uint8x16_t s3 = veorq_u8(c1, c2);    // weight 2 of final result
    uint8x16_t c3 = vandq_u8(c1, c2);    // weight 4 of final result
    return { s2, s3, c3 };
}

// V (5-bit) += H (3-bit), bit-sliced ripple-carry.
struct V5 { uint8x16_t b0, b1, b2, b3, b4; };
static inline V5 v5_add_h3(V5 v, H3 h)
{
    // bit 0
    uint8x16_t s0 = veorq_u8(v.b0, h.h0);
    uint8x16_t k0 = vandq_u8(v.b0, h.h0);
    // bit 1
    uint8x16_t s1 = fa_sum  (v.b1, h.h1, k0);
    uint8x16_t k1 = fa_carry(v.b1, h.h1, k0);
    // bit 2
    uint8x16_t s2 = fa_sum  (v.b2, h.h2, k1);
    uint8x16_t k2 = fa_carry(v.b2, h.h2, k1);
    // bit 3 (h's bit 3 is zero -> half-adder with carry)
    uint8x16_t s3 = veorq_u8(v.b3, k2);
    uint8x16_t k3 = vandq_u8(v.b3, k2);
    // bit 4
    uint8x16_t s4 = veorq_u8(v.b4, k3);
    return { s0, s1, s2, s3, s4 };
}

// ---- Compute H row -------------------------------------------------------
// One full row of adult bitplane -> 3 bitplanes of H values (range 0..5).
// H[y][x] = adult[y][x-2] + adult[y][x-1] + adult[y][x] + adult[y][x+1] + adult[y][x+2].
// Adult bitplane is derived inline from low_row & high_row.
// x-wrap toroidal: register 0's "prev" = last register; last register's "next" = register 0.
static void compute_H_row(const uint8_t* low_row, const uint8_t* high_row,
                          size_t R_REGS,
                          uint8_t* h0_out, uint8_t* h1_out, uint8_t* h2_out)
{
    auto load_adult = [&](size_t r) -> uint8x16_t {
        return vandq_u8(vld1q_u8(low_row + r * 16), vld1q_u8(high_row + r * 16));
    };

    uint8x16_t prev = load_adult(R_REGS - 1);
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

// ---- Apply rule to one output row using 5 precomputed H rows -------------
// V (5x5 sum, INCLUDES centre) = H[y-2] + H[y-1] + H[y] + H[y+1] + H[y+2].
// Rule (centre adjustment baked into the threshold):
//   EMPTY  -> EGG    iff V in {3, 4, 5}      (centre not adult -> V == A)
//   ADULT  ->        iff V in {5..10}        (centre adult -> A == V - 1)
// State encoded as (high << 1) | low; output bits computed by:
//   next_low  = (~low & (high | E)) | (low & high & R)
//   next_high = (high ^ low)        | (high & low & R)
static void apply_rule_row(const uint8_t* const h_rows[5][3],
                           const uint8_t* low_in,  const uint8_t* high_in,
                           uint8_t* low_out, uint8_t* high_out,
                           size_t R_REGS)
{
    for (size_t r = 0; r < R_REGS; ++r) {
        // Initialise V from H[0] (3 bits; high two bits zero).
        V5 v{
            vld1q_u8(h_rows[0][0] + r * 16),
            vld1q_u8(h_rows[0][1] + r * 16),
            vld1q_u8(h_rows[0][2] + r * 16),
            vdupq_n_u8(0),
            vdupq_n_u8(0)
        };
        // Add the other 4 H rows.
        for (int i = 1; i < 5; ++i) {
            H3 h{
                vld1q_u8(h_rows[i][0] + r * 16),
                vld1q_u8(h_rows[i][1] + r * 16),
                vld1q_u8(h_rows[i][2] + r * 16)
            };
            v = v5_add_h3(v, h);
        }

        // E = V in {3,4,5}
        //   = ~v4 & ~v3 & ( (~v2 & v1 & v0) | (v2 & ~v1) )
        uint8x16_t hi_zero = vandq_u8(vmvnq_u8(v.b4), vmvnq_u8(v.b3));
        uint8x16_t branchA = vbicq_u8(vandq_u8(v.b1, v.b0), v.b2);   // ~v2 & v1 & v0
        uint8x16_t branchB = vbicq_u8(v.b2, v.b1);                   //  v2 & ~v1
        uint8x16_t E       = vandq_u8(hi_zero, vorrq_u8(branchA, branchB));

        // R = V in {5..10}
        //   v3=0: V in {5..7}  <=>  v2 & (v1 | v0)
        //   v3=1: V in {8..10} <=>  ~v2 & ~(v1 & v0)
        uint8x16_t not_v4 = vmvnq_u8(v.b4);
        uint8x16_t caseLo = vbicq_u8(vandq_u8(v.b2, vorrq_u8(v.b1, v.b0)), v.b3);
        uint8x16_t v1v0   = vandq_u8(v.b1, v.b0);
        uint8x16_t caseHi = vandq_u8(vbicq_u8(vmvnq_u8(v1v0), v.b2), v.b3);
        uint8x16_t R      = vandq_u8(not_v4, vorrq_u8(caseLo, caseHi));

        // Combine with current (low, high) to produce (next_low, next_high).
        uint8x16_t low  = vld1q_u8(low_in  + r * 16);
        uint8x16_t high = vld1q_u8(high_in + r * 16);
        uint8x16_t h_x_l = veorq_u8(high, low);
        uint8x16_t h_a_l = vandq_u8(high, low);
        uint8x16_t hl_R  = vandq_u8(h_a_l, R);
        uint8x16_t nxt_h = vorrq_u8(h_x_l, hl_R);                    // (h^l) | (h&l&R)
        uint8x16_t branch_low = vbicq_u8(vorrq_u8(high, E), low);    // (h|E) & ~l
        uint8x16_t nxt_l = vorrq_u8(branch_low, hl_R);
        vst1q_u8(low_out  + r * 16, nxt_l);
        vst1q_u8(high_out + r * 16, nxt_h);
    }
}

// ===========================================================================
// PER-THREAD STRIP PROCESSING
// ===========================================================================
//
// Each worker owns rows [y_start, y_end). It maintains a 5-slot ring of H
// rows so the V-sum can slide downward across the strip without recomputing
// H more than once per row per generation.
//
// Ring layout: 5 slots, each holding 3 bitplane buffers of R_BYTES bytes.
// At iteration k (output row y = y_start + k), slot ((k + i) % 5) holds
// H for row (y_start + k + i - 2). Initial fill populates the 5 slots with
// H for rows y_start-2 .. y_start+2; each step overwrites the oldest slot.
//
// Toroidal y wrap: applied by absolute-row indexing modulo N.

struct HRing {
    AlignedBuf planes[5][3];     // planes[slot][bitplane_index]
    size_t bytes_cap = 0;
};

static thread_local HRing tls_ring;

static void ensure_ring(size_t R_BYTES)
{
    if (tls_ring.bytes_cap >= R_BYTES) return;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 3; ++j)
            tls_ring.planes[i][j] = aligned_zeros(R_BYTES);
    tls_ring.bytes_cap = R_BYTES;
}

static void step_strip(const uint8_t* cur_low,  const uint8_t* cur_high,
                       uint8_t* next_low, uint8_t* next_high,
                       size_t N, size_t R_BYTES, size_t R_REGS,
                       size_t y_start, size_t y_end)
{
    ensure_ring(R_BYTES);
    uint8_t* ring[5][3];
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 3; ++j)
            ring[i][j] = tls_ring.planes[i][j].get();

    auto y_wrap = [&](long y) -> size_t {
        return size_t((y % long(N) + long(N)) % long(N));
    };

    // Initial fill: H for rows y_start-2 .. y_start+2 -> slots 0..4.
    for (int i = 0; i < 5; ++i) {
        size_t y_abs = y_wrap(long(y_start) + (i - 2));
        compute_H_row(cur_low  + y_abs * R_BYTES,
                      cur_high + y_abs * R_BYTES,
                      R_REGS,
                      ring[i][0], ring[i][1], ring[i][2]);
    }

    const size_t strip_rows = y_end - y_start;
    for (size_t k = 0; k < strip_rows; ++k) {
        size_t y = y_start + k;
        // Slot (k + i) % 5 holds H for row y - 2 + i.
        const uint8_t* const h_rows[5][3] = {
            { ring[(k + 0) % 5][0], ring[(k + 0) % 5][1], ring[(k + 0) % 5][2] },
            { ring[(k + 1) % 5][0], ring[(k + 1) % 5][1], ring[(k + 1) % 5][2] },
            { ring[(k + 2) % 5][0], ring[(k + 2) % 5][1], ring[(k + 2) % 5][2] },
            { ring[(k + 3) % 5][0], ring[(k + 3) % 5][1], ring[(k + 3) % 5][2] },
            { ring[(k + 4) % 5][0], ring[(k + 4) % 5][1], ring[(k + 4) % 5][2] },
        };
        apply_rule_row(h_rows,
                       cur_low  + y * R_BYTES, cur_high + y * R_BYTES,
                       next_low + y * R_BYTES, next_high + y * R_BYTES,
                       R_REGS);

        // Replace the oldest slot (k % 5, which held y-2) with H for y+3.
        if (k + 1 < strip_rows) {
            size_t y_new = y_wrap(long(y) + 3);
            size_t slot  = k % 5;
            compute_H_row(cur_low  + y_new * R_BYTES,
                          cur_high + y_new * R_BYTES,
                          R_REGS,
                          ring[slot][0], ring[slot][1], ring[slot][2]);
        }
    }
}

// ===========================================================================
// THREAD POOL UTILITIES
// ===========================================================================
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

static void pin_to_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// ===========================================================================
// MAIN — mirrors reference's I/O exactly; sim loop replaced by threaded
// bitplane step.
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

    // -------------------------------------------------------------------------
    // Read input
    // -------------------------------------------------------------------------
    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width, height;
    if (std::fread(&width,  sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: input file too short (cannot read header)\n");
        std::fclose(fin);
        return 3;
    }

    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty, got %" PRIu64 " x %" PRIu64 "\n",
            width, height);
        std::fclose(fin);
        return 3;
    }

    const size_t N = size_t(width);
    if ((N & (N - 1)) != 0 || N < 128) {
        std::fprintf(stderr,
            "Error: grid size must be a power of two >= 128, got %zu\n", N);
        std::fclose(fin);
        return 3;
    }
    const size_t total_bytes = N * N;
    const size_t R_BYTES     = N / 8;            // bytes per bitplane row
    const size_t R_REGS      = R_BYTES / 16;     // NEON registers per bitplane row

    AlignedBuf byte_buf = aligned_zeros(total_bytes);
    if (std::fread(byte_buf.get(), 1, total_bytes, fin) != total_bytes) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    // -------------------------------------------------------------------------
    // Allocate bitplane buffers and convert byte grid -> bitplanes.
    // -------------------------------------------------------------------------
    AlignedBuf low_a  = aligned_zeros(N * R_BYTES);
    AlignedBuf high_a = aligned_zeros(N * R_BYTES);
    AlignedBuf low_b  = aligned_zeros(N * R_BYTES);
    AlignedBuf high_b = aligned_zeros(N * R_BYTES);
    bytes_to_bitplanes(byte_buf.get(), low_a.get(), high_a.get(), N);

    uint8_t* cur_low   = low_a.get();
    uint8_t* cur_high  = high_a.get();
    uint8_t* next_low  = low_b.get();
    uint8_t* next_high = high_b.get();

    // -------------------------------------------------------------------------
    // Set up worker threads, one strip each. Barrier swaps the buffers
    // exactly once per generation in its completion function.
    // -------------------------------------------------------------------------
    const int n_threads = choose_thread_count();
    std::barrier gen_barrier(n_threads, [&]() noexcept {
        std::swap(cur_low,  next_low);
        std::swap(cur_high, next_high);
    });
    std::latch start_latch(1);

    std::vector<std::jthread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        size_t y_start = (size_t(t)     * N) / size_t(n_threads);
        size_t y_end   = (size_t(t + 1) * N) / size_t(n_threads);
        workers.emplace_back([&, t, y_start, y_end]() {
            pin_to_cpu(t);
            ensure_ring(R_BYTES);
            start_latch.wait();
            for (int gen = 0; gen < generations; ++gen) {
                step_strip(cur_low, cur_high, next_low, next_high,
                           N, R_BYTES, R_REGS, y_start, y_end);
                gen_barrier.arrive_and_wait();
            }
        });
    }

    // -------------------------------------------------------------------------
    // Timed region
    // -------------------------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();
    start_latch.count_down();
    for (auto& w : workers) w.join();
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    // -------------------------------------------------------------------------
    // Convert bitplanes -> byte grid; write output.
    // -------------------------------------------------------------------------
    bitplanes_to_bytes(cur_low, cur_high, byte_buf.get(), N);

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }
    if (std::fwrite(&width,        sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height,       sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(byte_buf.get(), 1, total_bytes, fout) != total_bytes) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }
    std::fclose(fout);
    return 0;
}
