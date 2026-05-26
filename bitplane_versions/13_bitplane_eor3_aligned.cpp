// Bitplane version 13: v10 plus EOR3 and aligned vector storage.
//
// Representation:
//   state bit 0 plane: s0
//   state bit 1 plane: s1
//   ADULT == 3 == s1 & s0
//
// Each uint64_t word stores 64 horizontal cells. Compared to v12, this version
// computes the horizontal 5-cell ADULT sum for each source row once, stores that
// as three bitplanes in a 5-slot ring, then combines the five ring slots
// vertically. This avoids rebuilding the same horizontal sums for neighboring
// output rows.
//
// Compared to bitplane version 09, rule predicates are reduced from explicit
// equality masks to closed-form boolean ranges, and large buffers are hinted to
// Linux transparent huge pages.

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
static constexpr int BLOCK_ROWS = 128;

static inline uint64x2_t vnotq_u64(uint64x2_t x)
{
    return veorq_u64(x, vdupq_n_u64(~0ULL));
}

static inline uint64x2_t vxor3q_u64(uint64x2_t a, uint64x2_t b, uint64x2_t c)
{
#if defined(__ARM_FEATURE_SHA3)
    return veor3q_u64(a, b, c);
#else
    return veorq_u64(veorq_u64(a, b), c);
#endif
}

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

template <class T, size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

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

    void deallocate(T* p, std::size_t) noexcept
    {
        std::free(p);
    }
};

template <class T, class U, size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept
{
    return true;
}

template <class T, class U, size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept
{
    return false;
}

using AlignedWords = std::vector<uint64_t, AlignedAllocator<uint64_t, 64>>;

struct BitGrid {
    int n = 0;
    int row_words = 0;
    AlignedWords s0;
    AlignedWords s1;

    void resize(int side)
    {
        n = side;
        row_words = n / 64;
        s0.assign((size_t)n * row_words, 0);
        s1.assign((size_t)n * row_words, 0);
        advise_huge_pages(s0.data(), s0.size() * sizeof(uint64_t));
        advise_huge_pages(s1.data(), s1.size() * sizeof(uint64_t));
    }

    const uint64_t* row0(int y) const { return s0.data() + (size_t)y * row_words; }
    const uint64_t* row1(int y) const { return s1.data() + (size_t)y * row_words; }
    uint64_t* row0(int y) { return s0.data() + (size_t)y * row_words; }
    uint64_t* row1(int y) { return s1.data() + (size_t)y * row_words; }
};

static void bytes_to_bitgrid(const std::vector<uint8_t>& cells, BitGrid& out)
{
    const int N = out.n;
    const int rw = out.row_words;
    for (int y = 0; y < N; ++y) {
        uint64_t* s0 = out.row0(y);
        uint64_t* s1 = out.row1(y);
        for (int w = 0; w < rw; ++w) {
            uint64_t lo = 0;
            uint64_t hi = 0;
            const size_t base = (size_t)y * N + (size_t)w * 64;
            for (int b = 0; b < 64; ++b) {
                const uint8_t cell = cells[base + b];
                lo |= (uint64_t)(cell & 1u) << b;
                hi |= (uint64_t)((cell >> 1) & 1u) << b;
            }
            s0[w] = lo;
            s1[w] = hi;
        }
    }
}

static void bitgrid_to_bytes(const BitGrid& in, std::vector<uint8_t>& cells)
{
    const int N = in.n;
    const int rw = in.row_words;
    cells.assign((size_t)N * N, 0);
    for (int y = 0; y < N; ++y) {
        const uint64_t* s0 = in.row0(y);
        const uint64_t* s1 = in.row1(y);
        for (int w = 0; w < rw; ++w) {
            uint64_t lo = s0[w];
            uint64_t hi = s1[w];
            const size_t base = (size_t)y * N + (size_t)w * 64;
            for (int b = 0; b < 64; ++b) {
                cells[base + b] = (uint8_t)(((lo >> b) & 1u) | (((hi >> b) & 1u) << 1));
            }
        }
    }
}

static inline uint64_t shifted_adult_word(const uint64_t* adult, int rw, int w, int dx)
{
    const uint64_t prev = adult[(w - 1 + rw) % rw];
    const uint64_t curr = adult[w];
    const uint64_t next = adult[(w + 1) % rw];

    switch (dx) {
        case -2: return (curr << 2) | (prev >> 62);
        case -1: return (curr << 1) | (prev >> 63);
        case  0: return curr;
        case  1: return (curr >> 1) | (next << 63);
        case  2: return (curr >> 2) | (next << 62);
        default: return 0;
    }
}

static inline uint64_t shifted_adult_word_with_neighbors(uint64_t prev, uint64_t curr,
                                                         uint64_t next, int dx)
{
    switch (dx) {
        case -2: return (curr << 2) | (prev >> 62);
        case -1: return (curr << 1) | (prev >> 63);
        case  0: return curr;
        case  1: return (curr >> 1) | (next << 63);
        case  2: return (curr >> 2) | (next << 62);
        default: return 0;
    }
}

static inline void add_mask_to_count(uint64_t mask,
                                     uint64_t& c0, uint64_t& c1, uint64_t& c2,
                                     uint64_t& c3, uint64_t& c4)
{
    uint64_t carry = c0 & mask;
    c0 ^= mask;
    mask = carry;

    carry = c1 & mask;
    c1 ^= mask;
    mask = carry;

    carry = c2 & mask;
    c2 ^= mask;
    mask = carry;

    carry = c3 & mask;
    c3 ^= mask;
    mask = carry;

    c4 ^= mask;
}

static inline void add_mask_to_count_v(uint64x2_t mask,
                                       uint64x2_t& c0, uint64x2_t& c1,
                                       uint64x2_t& c2, uint64x2_t& c3,
                                       uint64x2_t& c4)
{
    uint64x2_t carry = vandq_u64(c0, mask);
    c0 = veorq_u64(c0, mask);
    mask = carry;

    carry = vandq_u64(c1, mask);
    c1 = veorq_u64(c1, mask);
    mask = carry;

    carry = vandq_u64(c2, mask);
    c2 = veorq_u64(c2, mask);
    mask = carry;

    carry = vandq_u64(c3, mask);
    c3 = veorq_u64(c3, mask);
    mask = carry;

    c4 = veorq_u64(c4, mask);
}

static inline void row_sum_5_word(const uint64_t* adult, int rw, int w,
                                  uint64_t& r0, uint64_t& r1, uint64_t& r2)
{
    uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    add_mask_to_count(shifted_adult_word(adult, rw, w, -2), c0, c1, c2, c3, c4);
    add_mask_to_count(shifted_adult_word(adult, rw, w, -1), c0, c1, c2, c3, c4);
    add_mask_to_count(shifted_adult_word(adult, rw, w,  0), c0, c1, c2, c3, c4);
    add_mask_to_count(shifted_adult_word(adult, rw, w,  1), c0, c1, c2, c3, c4);
    add_mask_to_count(shifted_adult_word(adult, rw, w,  2), c0, c1, c2, c3, c4);
    r0 = c0;
    r1 = c1;
    r2 = c2;
}

static inline void row_sum_5_word_neighbors(uint64_t prev, uint64_t curr, uint64_t next,
                                            uint64_t& r0, uint64_t& r1, uint64_t& r2)
{
    uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    add_mask_to_count(shifted_adult_word_with_neighbors(prev, curr, next, -2), c0, c1, c2, c3, c4);
    add_mask_to_count(shifted_adult_word_with_neighbors(prev, curr, next, -1), c0, c1, c2, c3, c4);
    add_mask_to_count(shifted_adult_word_with_neighbors(prev, curr, next,  0), c0, c1, c2, c3, c4);
    add_mask_to_count(shifted_adult_word_with_neighbors(prev, curr, next,  1), c0, c1, c2, c3, c4);
    add_mask_to_count(shifted_adult_word_with_neighbors(prev, curr, next,  2), c0, c1, c2, c3, c4);
    r0 = c0;
    r1 = c1;
    r2 = c2;
}

static inline void row_sum_5_pair_neighbors(uint64x2_t prev, uint64x2_t curr, uint64x2_t next,
                                            uint64x2_t& r0, uint64x2_t& r1, uint64x2_t& r2)
{
    uint64x2_t c0 = vdupq_n_u64(0);
    uint64x2_t c1 = vdupq_n_u64(0);
    uint64x2_t c2 = vdupq_n_u64(0);
    uint64x2_t c3 = vdupq_n_u64(0);
    uint64x2_t c4 = vdupq_n_u64(0);

    add_mask_to_count_v(vorrq_u64(vshlq_n_u64(curr, 2), vshrq_n_u64(prev, 62)), c0, c1, c2, c3, c4);
    add_mask_to_count_v(vorrq_u64(vshlq_n_u64(curr, 1), vshrq_n_u64(prev, 63)), c0, c1, c2, c3, c4);
    add_mask_to_count_v(curr, c0, c1, c2, c3, c4);
    add_mask_to_count_v(vorrq_u64(vshrq_n_u64(curr, 1), vshlq_n_u64(next, 63)), c0, c1, c2, c3, c4);
    add_mask_to_count_v(vorrq_u64(vshrq_n_u64(curr, 2), vshlq_n_u64(next, 62)), c0, c1, c2, c3, c4);

    r0 = c0;
    r1 = c1;
    r2 = c2;
}

static inline void add_rowsum_to_count(uint64_t r0, uint64_t r1, uint64_t r2,
                                       uint64_t& c0, uint64_t& c1, uint64_t& c2,
                                       uint64_t& c3, uint64_t& c4)
{
    uint64_t carry;

    const uint64_t ns0 = c0 ^ r0;
    carry = c0 & r0;
    c0 = ns0;

    const uint64_t c1xr1 = c1 ^ r1;
    const uint64_t ns1 = c1xr1 ^ carry;
    carry = (c1 & r1) | (carry & c1xr1);
    c1 = ns1;

    const uint64_t c2xr2 = c2 ^ r2;
    const uint64_t ns2 = c2xr2 ^ carry;
    carry = (c2 & r2) | (carry & c2xr2);
    c2 = ns2;

    const uint64_t ns3 = c3 ^ carry;
    carry = c3 & carry;
    c3 = ns3;

    c4 ^= carry;
}

static inline void add_rowsum_to_count_v(uint64x2_t r0, uint64x2_t r1, uint64x2_t r2,
                                         uint64x2_t& c0, uint64x2_t& c1,
                                         uint64x2_t& c2, uint64x2_t& c3,
                                         uint64x2_t& c4)
{
    uint64x2_t carry;

    const uint64x2_t ns0 = veorq_u64(c0, r0);
    carry = vandq_u64(c0, r0);
    c0 = ns0;

    const uint64x2_t c1xr1 = veorq_u64(c1, r1);
    const uint64x2_t ns1 = vxor3q_u64(c1, r1, carry);
    carry = vorrq_u64(vandq_u64(c1, r1), vandq_u64(carry, c1xr1));
    c1 = ns1;

    const uint64x2_t c2xr2 = veorq_u64(c2, r2);
    const uint64x2_t ns2 = vxor3q_u64(c2, r2, carry);
    carry = vorrq_u64(vandq_u64(c2, r2), vandq_u64(carry, c2xr2));
    c2 = ns2;

    const uint64x2_t ns3 = veorq_u64(c3, carry);
    carry = vandq_u64(c3, carry);
    c3 = ns3;

    c4 = veorq_u64(c4, carry);
}

static inline void subtract_rowsum_from_count(uint64_t r0, uint64_t r1, uint64_t r2,
                                              uint64_t& c0, uint64_t& c1, uint64_t& c2,
                                              uint64_t& c3, uint64_t& c4)
{
    uint64_t borrow;

    const uint64_t d0 = c0 ^ r0;
    borrow = ~c0 & r0;
    c0 = d0;

    const uint64_t c1xr1 = c1 ^ r1;
    const uint64_t d1 = c1xr1 ^ borrow;
    borrow = (~c1 & r1) | (~c1xr1 & borrow);
    c1 = d1;

    const uint64_t c2xr2 = c2 ^ r2;
    const uint64_t d2 = c2xr2 ^ borrow;
    borrow = (~c2 & r2) | (~c2xr2 & borrow);
    c2 = d2;

    const uint64_t d3 = c3 ^ borrow;
    borrow = ~c3 & borrow;
    c3 = d3;

    c4 ^= borrow;
}

static inline void subtract_rowsum_from_count_v(uint64x2_t r0, uint64x2_t r1, uint64x2_t r2,
                                                uint64x2_t& c0, uint64x2_t& c1,
                                                uint64x2_t& c2, uint64x2_t& c3,
                                                uint64x2_t& c4)
{
    uint64x2_t borrow;

    const uint64x2_t d0 = veorq_u64(c0, r0);
    borrow = vandq_u64(vnotq_u64(c0), r0);
    c0 = d0;

    const uint64x2_t c1xr1 = veorq_u64(c1, r1);
    const uint64x2_t d1 = vxor3q_u64(c1, r1, borrow);
    borrow = vorrq_u64(vandq_u64(vnotq_u64(c1), r1),
                       vandq_u64(vnotq_u64(c1xr1), borrow));
    c1 = d1;

    const uint64x2_t c2xr2 = veorq_u64(c2, r2);
    const uint64x2_t d2 = vxor3q_u64(c2, r2, borrow);
    borrow = vorrq_u64(vandq_u64(vnotq_u64(c2), r2),
                       vandq_u64(vnotq_u64(c2xr2), borrow));
    c2 = d2;

    const uint64x2_t d3 = veorq_u64(c3, borrow);
    borrow = vandq_u64(vnotq_u64(c3), borrow);
    c3 = d3;

    c4 = veorq_u64(c4, borrow);
}

static inline void subtract_mask_from_count(uint64_t mask,
                                            uint64_t& c0, uint64_t& c1, uint64_t& c2,
                                            uint64_t& c3, uint64_t& c4)
{
    uint64_t borrow = mask;
    uint64_t diff;

    diff = c0 ^ borrow;
    borrow = ~c0 & borrow;
    c0 = diff;

    diff = c1 ^ borrow;
    borrow = ~c1 & borrow;
    c1 = diff;

    diff = c2 ^ borrow;
    borrow = ~c2 & borrow;
    c2 = diff;

    diff = c3 ^ borrow;
    borrow = ~c3 & borrow;
    c3 = diff;

    c4 ^= borrow;
}

static inline void subtract_mask_from_count_v(uint64x2_t mask,
                                              uint64x2_t& c0, uint64x2_t& c1,
                                              uint64x2_t& c2, uint64x2_t& c3,
                                              uint64x2_t& c4)
{
    uint64x2_t borrow = mask;
    uint64x2_t diff;

    diff = veorq_u64(c0, borrow);
    borrow = vandq_u64(vnotq_u64(c0), borrow);
    c0 = diff;

    diff = veorq_u64(c1, borrow);
    borrow = vandq_u64(vnotq_u64(c1), borrow);
    c1 = diff;

    diff = veorq_u64(c2, borrow);
    borrow = vandq_u64(vnotq_u64(c2), borrow);
    c2 = diff;

    diff = veorq_u64(c3, borrow);
    borrow = vandq_u64(vnotq_u64(c3), borrow);
    c3 = diff;

    c4 = veorq_u64(c4, borrow);
}

struct ThreadScratch {
    AlignedWords adult_tmp;
    AlignedWords h_store;
    AlignedWords vcount_store;
    uint64_t* vc0 = nullptr;
    uint64_t* vc1 = nullptr;
    uint64_t* vc2 = nullptr;
    uint64_t* vc3 = nullptr;
    uint64_t* vc4 = nullptr;

    void resize(int rw)
    {
        adult_tmp.resize(rw);
        h_store.resize((size_t)(BLOCK_ROWS + 4) * 3 * rw);
        vcount_store.resize(5 * (size_t)rw);
        advise_huge_pages(adult_tmp.data(), adult_tmp.size() * sizeof(uint64_t));
        advise_huge_pages(h_store.data(), h_store.size() * sizeof(uint64_t));
        advise_huge_pages(vcount_store.data(), vcount_store.size() * sizeof(uint64_t));
        vc0 = vcount_store.data() + 0 * (size_t)rw;
        vc1 = vcount_store.data() + 1 * (size_t)rw;
        vc2 = vcount_store.data() + 2 * (size_t)rw;
        vc3 = vcount_store.data() + 3 * (size_t)rw;
        vc4 = vcount_store.data() + 4 * (size_t)rw;
    }
};

static void step_rows_bitplane(const BitGrid& src, BitGrid& dst, int y0, int y1, ThreadScratch& scratch)
{
    const int N = src.n;
    const int rw = src.row_words;
    const int ymask = N - 1;

    uint64_t* adult_tmp = scratch.adult_tmp.data();
    uint64_t* h_data = scratch.h_store.data();
    uint64_t* vc0 = scratch.vc0;
    uint64_t* vc1 = scratch.vc1;
    uint64_t* vc2 = scratch.vc2;
    uint64_t* vc3 = scratch.vc3;
    uint64_t* vc4 = scratch.vc4;

    auto h0 = [&](int i) { return h_data + (size_t)(3 * i + 0) * rw; };
    auto h1 = [&](int i) { return h_data + (size_t)(3 * i + 1) * rw; };
    auto h2 = [&](int i) { return h_data + (size_t)(3 * i + 2) * rw; };

    auto fill_h = [&](int src_y, int hidx) {
        const int yy = src_y & ymask;
        const uint64_t* s0 = src.row0(yy);
        const uint64_t* s1 = src.row1(yy);
        uint64_t* out0 = h0(hidx);
        uint64_t* out1 = h1(hidx);
        uint64_t* out2 = h2(hidx);

        int aw = 0;
        for (; aw + 1 < rw; aw += 2) {
            const uint64x2_t a0 = vld1q_u64(s0 + aw);
            const uint64x2_t a1 = vld1q_u64(s1 + aw);
            vst1q_u64(adult_tmp + aw, vandq_u64(a0, a1));
        }
        for (; aw < rw; ++aw) {
            adult_tmp[aw] = s0[aw] & s1[aw];
        }
        if (rw == 1) {
            row_sum_5_word(adult_tmp, rw, 0, out0[0], out1[0], out2[0]);
            return;
        }

        row_sum_5_word_neighbors(adult_tmp[rw - 1], adult_tmp[0], adult_tmp[1],
                                 out0[0], out1[0], out2[0]);

        int w = 1;
        for (; w + 1 < rw - 1; w += 2) {
            const uint64x2_t prev = vld1q_u64(adult_tmp + w - 1);
            const uint64x2_t curr = vld1q_u64(adult_tmp + w);
            const uint64x2_t next = vld1q_u64(adult_tmp + w + 1);
            uint64x2_t r0, r1, r2;
            row_sum_5_pair_neighbors(prev, curr, next, r0, r1, r2);
            vst1q_u64(out0 + w, r0);
            vst1q_u64(out1 + w, r1);
            vst1q_u64(out2 + w, r2);
        }

        for (; w < rw - 1; ++w) {
            row_sum_5_word_neighbors(adult_tmp[w - 1], adult_tmp[w], adult_tmp[w + 1],
                                     out0[w], out1[w], out2[w]);
        }

        row_sum_5_word_neighbors(adult_tmp[rw - 2], adult_tmp[rw - 1], adult_tmp[0],
                                 out0[rw - 1], out1[rw - 1], out2[rw - 1]);
    };

    for (int b0 = y0; b0 < y1; b0 += BLOCK_ROWS) {
        const int b1 = std::min(b0 + BLOCK_ROWS, y1);
        const int block_rows = b1 - b0;
        const int h_rows = block_rows + 4;

        for (int i = 0; i < h_rows; ++i) {
            fill_h(b0 - 2 + i, i);
        }

        for (int w = 0; w < rw; ++w) {
            uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
            for (int i = 0; i < 5; ++i) {
                add_rowsum_to_count(h0(i)[w], h1(i)[w], h2(i)[w], c0, c1, c2, c3, c4);
            }
            vc0[w] = c0;
            vc1[w] = c1;
            vc2[w] = c2;
            vc3[w] = c3;
            vc4[w] = c4;
        }

        for (int local_y = 0; local_y < block_rows; ++local_y) {
            const int y = b0 + local_y;
            const uint64_t* center0 = src.row0(y);
            const uint64_t* center1 = src.row1(y);
            uint64_t* out0 = dst.row0(y);
            uint64_t* out1 = dst.row1(y);

            int w = 0;
            for (; w + 1 < rw; w += 2) {
                const uint64x2_t c0 = vld1q_u64(vc0 + w);
                const uint64x2_t c1 = vld1q_u64(vc1 + w);
                const uint64x2_t c2 = vld1q_u64(vc2 + w);
                const uint64x2_t c3 = vld1q_u64(vc3 + w);
                const uint64x2_t c4 = vld1q_u64(vc4 + w);

                const uint64x2_t s0 = vld1q_u64(center0 + w);
                const uint64x2_t s1 = vld1q_u64(center1 + w);
                const uint64x2_t adult = vandq_u64(s0, s1);

                const uint64x2_t empty = vnotq_u64(vorrq_u64(s0, s1));
                const uint64x2_t egg = vandq_u64(s0, vnotq_u64(s1));
                const uint64x2_t juvenile = vandq_u64(s1, vnotq_u64(s0));

                const uint64x2_t nc4 = vnotq_u64(c4);
                const uint64x2_t nc3 = vnotq_u64(c3);
                const uint64x2_t nc2 = vnotq_u64(c2);
                const uint64x2_t nc1 = vnotq_u64(c1);
                const uint64x2_t birth_range =
                    vandq_u64(vandq_u64(nc4, nc3),
                              vorrq_u64(vandq_u64(vandq_u64(nc2, c1), c0),
                                        vandq_u64(c2, nc1)));
                const uint64x2_t survive_range =
                    vandq_u64(nc4,
                              vorrq_u64(vandq_u64(vandq_u64(nc3, c2),
                                                  vorrq_u64(c1, c0)),
                                        vandq_u64(vandq_u64(c3, nc2),
                                                  vnotq_u64(vandq_u64(c1, c0)))));

                const uint64x2_t birth = vandq_u64(empty, birth_range);
                const uint64x2_t survive = vandq_u64(adult, survive_range);

                vst1q_u64(out1 + w, vorrq_u64(vorrq_u64(egg, juvenile), survive));
                vst1q_u64(out0 + w, vorrq_u64(vorrq_u64(birth, juvenile), survive));
            }

            for (; w < rw; ++w) {
                const uint64_t c0 = vc0[w];
                const uint64_t c1 = vc1[w];
                const uint64_t c2 = vc2[w];
                const uint64_t c3 = vc3[w];
                const uint64_t c4 = vc4[w];

                const uint64_t s0 = center0[w];
                const uint64_t s1 = center1[w];
                const uint64_t adult = s0 & s1;

                const uint64_t empty = ~(s0 | s1);
                const uint64_t egg = s0 & ~s1;
                const uint64_t juvenile = s1 & ~s0;

                const uint64_t nc4 = ~c4, nc3 = ~c3, nc2 = ~c2, nc1 = ~c1;
                const uint64_t birth_range = nc4 & nc3 & ((nc2 & c1 & c0) | (c2 & nc1));
                const uint64_t survive_range = nc4 & ((nc3 & c2 & (c1 | c0)) |
                                                      (c3 & nc2 & ~(c1 & c0)));

                const uint64_t birth = empty & birth_range;
                const uint64_t survive = adult & survive_range;

                out1[w] = egg | juvenile | survive;
                out0[w] = birth | juvenile | survive;
            }

            if (local_y + 1 == block_rows) {
                continue;
            }

            int uw = 0;
            for (; uw + 1 < rw; uw += 2) {
                uint64x2_t c0 = vld1q_u64(vc0 + uw);
                uint64x2_t c1 = vld1q_u64(vc1 + uw);
                uint64x2_t c2 = vld1q_u64(vc2 + uw);
                uint64x2_t c3 = vld1q_u64(vc3 + uw);
                uint64x2_t c4 = vld1q_u64(vc4 + uw);
                subtract_rowsum_from_count_v(vld1q_u64(h0(local_y) + uw),
                                             vld1q_u64(h1(local_y) + uw),
                                             vld1q_u64(h2(local_y) + uw),
                                             c0, c1, c2, c3, c4);
                add_rowsum_to_count_v(vld1q_u64(h0(local_y + 5) + uw),
                                      vld1q_u64(h1(local_y + 5) + uw),
                                      vld1q_u64(h2(local_y + 5) + uw),
                                      c0, c1, c2, c3, c4);
                vst1q_u64(vc0 + uw, c0);
                vst1q_u64(vc1 + uw, c1);
                vst1q_u64(vc2 + uw, c2);
                vst1q_u64(vc3 + uw, c3);
                vst1q_u64(vc4 + uw, c4);
            }
            for (; uw < rw; ++uw) {
                subtract_rowsum_from_count(h0(local_y)[uw], h1(local_y)[uw], h2(local_y)[uw],
                                           vc0[uw], vc1[uw], vc2[uw], vc3[uw], vc4[uw]);
                add_rowsum_to_count(h0(local_y + 5)[uw], h1(local_y + 5)[uw], h2(local_y + 5)[uw],
                                    vc0[uw], vc1[uw], vc2[uw], vc3[uw], vc4[uw]);
            }
        }
    }
}

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
    if (width == 0 || width != height || (width % 64) != 0) {
        std::fprintf(stderr,
            "Error: grid must be square, non-empty, and divisible by 64, got %" PRIu64 " x %" PRIu64 "\n",
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
        ctx.resize(grid_a.row_words);
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
