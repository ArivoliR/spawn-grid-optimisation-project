// Version 09: persistent thread pool plus precomputed horizontal adult sums
// and a transition lookup table.
//
// Change from 08_sliding_counts.cpp:
//   - Build hsum[y][x] = adults in row y from x-2..x+2 for every cell.
//   - Compute A from five hsum rows, subtracting center adult.
//   - Replace the transition switch with transition[cell][adult_count].
//
// This remains O(generations * cells). hsum is per-generation scratch data
// derived from the current grid, not memoized state across generations.

#include <algorithm>
#include <array>
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

static constexpr auto make_transition_table()
{
    std::array<std::array<uint8_t, 25>, 4> table{};
    for (int A = 0; A <= 24; ++A) {
        table[EMPTY][A] = (A >= 3 && A <= 5) ? EGG : EMPTY;
        table[EGG][A] = JUVENILE;
        table[JUVENILE][A] = ADULT;
        table[ADULT][A] = (A >= 4 && A <= 9) ? ADULT : EMPTY;
    }
    return table;
}

static constexpr auto TRANSITION = make_transition_table();

static inline uint8_t adult_at(const uint8_t* row, int x)
{
    return row[x] == ADULT;
}

static inline uint8_t initial_hsum5(const uint8_t* row, int N, int mask)
{
    return (uint8_t)(adult_at(row, (N - 2) & mask) +
                     adult_at(row, (N - 1) & mask) +
                     adult_at(row, 0) +
                     adult_at(row, 1 & mask) +
                     adult_at(row, 2 & mask));
}

static void build_hsum_rows(const uint8_t* src, uint8_t* hsum, int N, int y0, int y1)
{
    const int mask = N - 1;
    for (int y = y0; y < y1; ++y) {
        const uint8_t* row = src + (size_t)y * N;
        uint8_t* out = hsum + (size_t)y * N;

        uint8_t h = initial_hsum5(row, N, mask);
        for (int x = 0; x < N; ++x) {
            out[x] = h;
            const int remove_x = (x - 2 + N) & mask;
            const int add_x = (x + 3) & mask;
            h = (uint8_t)(h + adult_at(row, add_x) - adult_at(row, remove_x));
        }
    }
}

static void step_from_hsum_rows(const uint8_t* src, const uint8_t* hsum,
                                uint8_t* dst, int N, int y0, int y1)
{
    const int mask = N - 1;
    for (int y = y0; y < y1; ++y) {
        const int ym2 = (y - 2 + N) & mask;
        const int ym1 = (y - 1 + N) & mask;
        const int yp1 = (y + 1) & mask;
        const int yp2 = (y + 2) & mask;

        const uint8_t* h0 = hsum + (size_t)ym2 * N;
        const uint8_t* h1 = hsum + (size_t)ym1 * N;
        const uint8_t* h2 = hsum + (size_t)y   * N;
        const uint8_t* h3 = hsum + (size_t)yp1 * N;
        const uint8_t* h4 = hsum + (size_t)yp2 * N;
        const uint8_t* row = src + (size_t)y * N;
        uint8_t* out = dst + (size_t)y * N;

        for (int x = 0; x < N; ++x) {
            const uint8_t cell = row[x];
            const int A = h0[x] + h1[x] + h2[x] + h3[x] + h4[x] - adult_at(row, x);
            out[x] = TRANSITION[cell <= ADULT ? cell : EMPTY][A];
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
    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty, got %" PRIu64 " x %" PRIu64 "\n",
            width, height);
        std::fclose(fin);
        return 3;
    }

    const int N = (int)width;
    const size_t Ncells = (size_t)N * N;

    std::vector<uint8_t> grid_a(Ncells), grid_b(Ncells), hsum(Ncells);
    if (std::fread(grid_a.data(), 1, Ncells, fin) != Ncells) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    uint8_t* cur = grid_a.data();
    uint8_t* next = grid_b.data();
    uint8_t* hsum_data = hsum.data();

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    unsigned T = std::min<unsigned>(hw, 8u);
    if ((int)T > N) T = (unsigned)N;

    std::vector<int> row_lo(T), row_hi(T);
    for (unsigned t = 0; t < T; ++t) {
        row_lo[t] = (int)((uint64_t)t * N / T);
        row_hi[t] = (int)((uint64_t)(t + 1) * N / T);
    }

    const uint8_t* shared_src = nullptr;
    uint8_t* shared_dst = nullptr;
    bool stop = false;

    std::barrier bar_start(T);
    std::barrier bar_hsum_done(T);
    std::barrier bar_step_done(T);

    std::vector<std::thread> pool;
    pool.reserve(T - 1);
    for (unsigned t = 1; t < T; ++t) {
        pool.emplace_back([&, t]() {
            for (;;) {
                bar_start.arrive_and_wait();
                if (stop) return;
                build_hsum_rows(shared_src, hsum_data, N, row_lo[t], row_hi[t]);
                bar_hsum_done.arrive_and_wait();
                step_from_hsum_rows(shared_src, hsum_data, shared_dst, N, row_lo[t], row_hi[t]);
                bar_step_done.arrive_and_wait();
            }
        });
    }

    auto t0 = std::chrono::steady_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        shared_src = cur;
        shared_dst = next;
        bar_start.arrive_and_wait();
        build_hsum_rows(cur, hsum_data, N, row_lo[0], row_hi[0]);
        bar_hsum_done.arrive_and_wait();
        step_from_hsum_rows(cur, hsum_data, next, N, row_lo[0], row_hi[0]);
        bar_step_done.arrive_and_wait();
        std::swap(cur, next);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    stop = true;
    bar_start.arrive_and_wait();
    for (auto& th : pool) th.join();

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }
    if (std::fwrite(&width, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(cur, 1, Ncells, fout) != Ncells) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }
    std::fclose(fout);
    return 0;
}
