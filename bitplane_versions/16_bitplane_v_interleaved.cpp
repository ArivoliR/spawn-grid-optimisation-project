// Bitplane version 16: vb14 + interleaved V scratch layout.
//
// This keeps vb14's byte-lane NEON H-tree, row-major hot loop, compact state
// transition, huge-page hints, and aligned storage. The only architectural
// change is the vertical-count scratch layout:
//
// Old vb14 layout:
//   five full-row planes: v0 row, v1 row, v2 row, v3 row, v4 row
//
// New vb16 layout:
//   v0[r], v1[r], v2[r], v3[r], v4[r]
//
// The hot slide loop loads/stores all five V vectors for a register every
// output row, so keeping those five vectors adjacent improves locality without
// changing the algorithm. This stays as a local scratch-layout change on top
// of vb14.
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
// What stays the same as vb14:
//   - byte-packed uint8x16_t storage
//   - tree-style sum_of_5 with FA/HA/maj/EOR3
//   - vextq_u8 byte-granular shifts
//   - compact birth/survive predicate
//   - block-H scratch
//   - persistent per-thread scratch + hugepages + 64-byte aligned storage
//   - row-major (k outer, r inner) access pattern
//
// What we deliberately don't add:
//   - CPU pinning (docs: neutral under taskset, slightly negative in one run)
//   - Loop-swap to r outer (bad row-stride locality)
//   - Temporal stripe (correct but slower in earlier target tests)
//
// Build (target, default block size 128):
//   g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
//          bitplane_versions/16_bitplane_v_interleaved.cpp -o /tmp/vb16
//
// Sweep block sizes:
//   g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
//          -DSPAWN_BLOCK_ROWS=64 \
//          bitplane_versions/16_bitplane_v_interleaved.cpp -o /tmp/vb16_b64
//
//   g++-14 -std=c++23 -O3 -mcpu=neoverse-v2+sha3 -pthread \
//          -DSPAWN_BLOCK_ROWS=256 \
//          bitplane_versions/16_bitplane_v_interleaved.cpp -o /tmp/vb16_b256

#include <algorithm>
#include <arm_neon.h>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
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
    const uint8x16_t left_carry = vextq_u8(prev, curr, 15);
    const uint8x16_t right_carry = vextq_u8(curr, next, 1);
    const uint8x16_t s_m2 = vorrq_u8(vshlq_n_u8(curr, 2), vshrq_n_u8(left_carry, 6));
    const uint8x16_t s_m1 = vorrq_u8(vshlq_n_u8(curr, 1), vshrq_n_u8(left_carry, 7));
    const uint8x16_t s_p1 = vorrq_u8(vshrq_n_u8(curr, 1), vshlq_n_u8(right_carry, 7));
    const uint8x16_t s_p2 = vorrq_u8(vshrq_n_u8(curr, 2), vshlq_n_u8(right_carry, 6));
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
    const uint8x16_t borrow3 = vbicq_u8(borrow2, v.b3);
    const uint8x16_t diff4   = veorq_u8(v.b4, borrow3);
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
static inline LH apply_rule_byte(V5 v, uint8x16_t low, uint8x16_t high)
{
    const uint8x16_t nc4 = vmvnq_u8(v.b4);
    const uint8x16_t nc3 = vmvnq_u8(v.b3);
    const uint8x16_t nc2 = vmvnq_u8(v.b2);
    const uint8x16_t nc1 = vmvnq_u8(v.b1);

    const uint8x16_t nc4_nc3 = vandq_u8(nc4, nc3);
    const uint8x16_t born_a  = vandq_u8(vandq_u8(nc2, v.b1), v.b0);
    const uint8x16_t born_b  = vandq_u8(v.b2, nc1);
    const uint8x16_t born    = vandq_u8(nc4_nc3, vorrq_u8(born_a, born_b));

    const uint8x16_t v1_or_v0  = vorrq_u8(v.b1, v.b0);
    const uint8x16_t v1_and_v0 = vandq_u8(v.b1, v.b0);
    const uint8x16_t low_case  = vandq_u8(vandq_u8(nc3, v.b2), v1_or_v0);
    const uint8x16_t high_case = vbicq_u8(vandq_u8(v.b3, nc2), v1_and_v0);
    const uint8x16_t alive_v   = vandq_u8(nc4, vorrq_u8(low_case, high_case));

    const uint8x16_t adult     = vandq_u8(low, high);
    const uint8x16_t adult_r   = vandq_u8(adult, alive_v);
    const uint8x16_t next_high = vorrq_u8(veorq_u8(high, low), adult_r);
    const uint8x16_t next_low  = vorrq_u8(vbicq_u8(vorrq_u8(high, born), low), adult_r);
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
    for (; r + 3 < R_REGS; r += 4) {
        const int r1 = r + 1;
        const int r2 = r + 2;
        const int r3 = r + 3;
        const int r4 = (r + 4 == R_REGS) ? 0 : (r + 4);
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
    AlignedBytes h_store;
    AlignedBytes vcount_store;
    uint8_t* v_b0 = nullptr;
    uint8_t* v_b1 = nullptr;
    uint8_t* v_b2 = nullptr;
    uint8_t* v_b3 = nullptr;
    uint8_t* v_b4 = nullptr;

    void resize(int row_bytes)
    {
        h_store.resize((size_t)(BLOCK_ROWS + 4) * 3 * row_bytes);
        vcount_store.resize(5 * (size_t)row_bytes);
        advise_huge_pages(h_store.data(), h_store.size());
        advise_huge_pages(vcount_store.data(), vcount_store.size());
        v_b0 = vcount_store.data() + 0 * (size_t)row_bytes;
        v_b1 = vcount_store.data() + 1 * (size_t)row_bytes;
        v_b2 = vcount_store.data() + 2 * (size_t)row_bytes;
        v_b3 = vcount_store.data() + 3 * (size_t)row_bytes;
        v_b4 = vcount_store.data() + 4 * (size_t)row_bytes;
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

    uint8_t* h_data = scratch.h_store.data();
    uint8_t* v_data = scratch.vcount_store.data();

    auto h0p = [&](int i) { return h_data + (size_t)(3 * i + 0) * row_bytes; };
    auto h1p = [&](int i) { return h_data + (size_t)(3 * i + 1) * row_bytes; };
    auto h2p = [&](int i) { return h_data + (size_t)(3 * i + 2) * row_bytes; };

    for (int b0 = y0; b0 < y1; b0 += BLOCK_ROWS) {
        const int b1 = std::min(b0 + BLOCK_ROWS, y1);
        const int block_rows = b1 - b0;
        const int h_rows = block_rows + 4;

        // ---- Pass 1: materialise H rows [b0-2 .. b0+block_rows+1]. ----
        for (int i = 0; i < h_rows; ++i) {
            const int y = (b0 - 2 + i + N) & ymask;
            compute_H_row(src.row0(y), src.row1(y), R_REGS,
                          h0p(i), h1p(i), h2p(i));
        }

        // ---- Pass 2: initial V = sum of H[0..4]. Unrolled by 2 in r. ----
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

        // ---- Apply rule to first output row. Unrolled by 2 in r. ----
        {
            const uint8_t* low_row  = src.row0(b0);
            const uint8_t* high_row = src.row1(b0);
            uint8_t* nl_row = dst.row0(b0);
            uint8_t* nh_row = dst.row1(b0);

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

        // ---- Slide V down through the block, applying rule per output row.
        //      Unrolled by 2 in r. Hot path. ----
        for (int k = 1; k < block_rows; ++k) {
            const uint8_t* h_out_0 = h0p(k - 1);
            const uint8_t* h_out_1 = h1p(k - 1);
            const uint8_t* h_out_2 = h2p(k - 1);
            const uint8_t* h_in_0  = h0p(k + 4);
            const uint8_t* h_in_1  = h1p(k + 4);
            const uint8_t* h_in_2  = h2p(k + 4);
            const int y = b0 + k;
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
    for (unsigned t = 0; t < T; ++t) {
        row_lo[t] = (int)((uint64_t)t * N / T);
        row_hi[t] = (int)((uint64_t)(t + 1) * N / T);
    }
    std::vector<ThreadScratch> scratch(T);
    for (auto& ctx : scratch) {
        ctx.resize(grid_a.row_bytes);
    }

    const BitGrid* shared_src = nullptr;
    BitGrid* shared_dst = nullptr;
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
                step_rows_bitplane(*shared_src, *shared_dst, row_lo[t], row_hi[t], scratch[t]);
                bar_done.arrive_and_wait();
            }
        });
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int gen = 0; gen < generations; ++gen) {
        shared_src = cur;
        shared_dst = next;
        bar_start.arrive_and_wait();
        step_rows_bitplane(*cur, *next, row_lo[0], row_hi[0], scratch[0]);
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
