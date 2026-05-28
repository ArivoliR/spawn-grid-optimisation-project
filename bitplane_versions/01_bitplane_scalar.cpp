// Version 12: bitplane scalar rewrite with row-parallel persistent workers.
//
// Representation:
//   state bit 0 plane: s0
//   state bit 1 plane: s1
//   ADULT == 3 == s1 & s0
//
// Each uint64_t word stores 64 horizontal cells. The kernel computes one output
// word, i.e. 64 cells, at a time using bitwise arithmetic. This is a separate
// high-ceiling path from the byte-grid hsum versions.

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static constexpr uint8_t EMPTY    = 0;
static constexpr uint8_t EGG      = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT    = 3;

struct BitGrid {
    int n = 0;
    int row_words = 0;
    std::vector<uint64_t> s0;
    std::vector<uint64_t> s1;

    void resize(int side)
    {
        n = side;
        row_words = n / 64;
        s0.assign((size_t)n * row_words, 0);
        s1.assign((size_t)n * row_words, 0);
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

static void step_rows_bitplane(const BitGrid& src, BitGrid& dst, int y0, int y1)
{
    const int N = src.n;
    const int rw = src.row_words;
    const int ymask = N - 1;

    std::vector<uint64_t> adult_rows(5 * (size_t)rw);
    uint64_t* adult[5];
    for (int i = 0; i < 5; ++i) adult[i] = adult_rows.data() + (size_t)i * rw;

    for (int y = y0; y < y1; ++y) {
        const int ys[5] = {
            (y - 2 + N) & ymask,
            (y - 1 + N) & ymask,
            y,
            (y + 1) & ymask,
            (y + 2) & ymask,
        };

        for (int r = 0; r < 5; ++r) {
            const uint64_t* s0 = src.row0(ys[r]);
            const uint64_t* s1 = src.row1(ys[r]);
            for (int w = 0; w < rw; ++w) {
                adult[r][w] = s0[w] & s1[w];
            }
        }

        const uint64_t* center0 = src.row0(y);
        const uint64_t* center1 = src.row1(y);
        uint64_t* out0 = dst.row0(y);
        uint64_t* out1 = dst.row1(y);

        for (int w = 0; w < rw; ++w) {
            uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;

            for (int r = 0; r < 5; ++r) {
                for (int dx = -2; dx <= 2; ++dx) {
                    if (r == 2 && dx == 0) continue;
                    add_mask_to_count(shifted_adult_word(adult[r], rw, w, dx), c0, c1, c2, c3, c4);
                }
            }

            const uint64_t s0 = center0[w];
            const uint64_t s1 = center1[w];
            const uint64_t adult = s0 & s1;
            const uint64_t empty = ~(s0 | s1);
            const uint64_t egg = s0 & ~s1;
            const uint64_t juvenile = s1 & ~s0;

            const uint64_t nc4 = ~c4, nc3 = ~c3, nc2 = ~c2, nc1 = ~c1, nc0 = ~c0;
            const uint64_t eq3 = nc4 & nc3 & nc2 & c1  & c0;
            const uint64_t eq4 = nc4 & nc3 & c2  & nc1 & nc0;
            const uint64_t eq5 = nc4 & nc3 & c2  & nc1 & c0;
            const uint64_t eq6 = nc4 & nc3 & c2  & c1  & nc0;
            const uint64_t eq7 = nc4 & nc3 & c2  & c1  & c0;
            const uint64_t eq8 = nc4 & c3  & nc2 & nc1 & nc0;
            const uint64_t eq9 = nc4 & c3  & nc2 & nc1 & c0;

            const uint64_t birth = empty & (eq3 | eq4 | eq5);
            const uint64_t survive = adult & (eq4 | eq5 | eq6 | eq7 | eq8 | eq9);

            out1[w] = egg | juvenile | survive;
            out0[w] = birth | juvenile | survive;
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
                step_rows_bitplane(*shared_src, *shared_dst, row_lo[t], row_hi[t]);
                bar_done.arrive_and_wait();
            }
        });
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int gen = 0; gen < generations; ++gen) {
        shared_src = cur;
        shared_dst = next;
        bar_start.arrive_and_wait();
        step_rows_bitplane(*cur, *next, row_lo[0], row_hi[0]);
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
