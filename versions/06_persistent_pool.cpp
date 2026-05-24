// spawn_sim.cpp — Baseline (scalar, multi-threaded) Monster Spawning Grid.
//
// Same on-disk format and CLI as reference/spawn_sim.cpp:
//   spawn_sim <input.bin> <output.bin> [generations]   (generations defaults to 10000)
// Header: little-endian uint64 width, uint64 height; then width*height bytes (row-major).
// Cell states: 0=EMPTY, 1=EGG, 2=JUVENILE, 3=ADULT.
//
// Differences from the reference:
//   - Row-blocked parallelism with a persistent std::thread pool (barriers per generation).
//   - Toroidal x-wrap uses & (N-1); the grid is guaranteed power-of-two.
// The inner per-cell body is deliberately straight-line scalar so SIMD/tiling can replace
// it later without disturbing the threading or I/O contract.

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

// Process rows [y0, y1) of one generation.
// N is the grid side, assumed power-of-two so x-wrap is a mask.
static void step_rows(const uint8_t* src, uint8_t* dst, int N, int y0, int y1)
{
    const int mask = N - 1;
    for (int y = y0; y < y1; ++y) {
        const int ym2 = (y - 2 + N) & mask;
        const int ym1 = (y - 1 + N) & mask;
        const int yp1 = (y + 1)     & mask;
        const int yp2 = (y + 2)     & mask;

        const uint8_t* r0 = src + (size_t)ym2 * N;
        const uint8_t* r1 = src + (size_t)ym1 * N;
        const uint8_t* r2 = src + (size_t)y   * N;
        const uint8_t* r3 = src + (size_t)yp1 * N;
        const uint8_t* r4 = src + (size_t)yp2 * N;
        uint8_t*       rw = dst + (size_t)y   * N;

        for (int x = 0; x < N; ++x) {
            const int xm2 = (x - 2 + N) & mask;
            const int xm1 = (x - 1 + N) & mask;
            const int xp1 = (x + 1)     & mask;
            const int xp2 = (x + 2)     & mask;

            int A =
                (r0[xm2]==ADULT) + (r0[xm1]==ADULT) + (r0[x]==ADULT) + (r0[xp1]==ADULT) + (r0[xp2]==ADULT) +
                (r1[xm2]==ADULT) + (r1[xm1]==ADULT) + (r1[x]==ADULT) + (r1[xp1]==ADULT) + (r1[xp2]==ADULT) +
                (r2[xm2]==ADULT) + (r2[xm1]==ADULT) +                  (r2[xp1]==ADULT) + (r2[xp2]==ADULT) +
                (r3[xm2]==ADULT) + (r3[xm1]==ADULT) + (r3[x]==ADULT) + (r3[xp1]==ADULT) + (r3[xp2]==ADULT) +
                (r4[xm2]==ADULT) + (r4[xm1]==ADULT) + (r4[x]==ADULT) + (r4[xp1]==ADULT) + (r4[xp2]==ADULT);

            const uint8_t cell = r2[x];
            uint8_t next;
            switch (cell) {
                case EMPTY:    next = (A >= 3 && A <= 5) ? EGG   : EMPTY; break;
                case EGG:      next = JUVENILE;                            break;
                case JUVENILE: next = ADULT;                               break;
                case ADULT:    next = (A >= 4 && A <= 9) ? ADULT : EMPTY;  break;
                default:       next = EMPTY;                               break;
            }
            rw[x] = next;
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

    const int    N    = (int)width;
    const size_t Ncells = (size_t)N * N;

    std::vector<uint8_t> grid_a(Ncells), grid_b(Ncells);
    if (std::fread(grid_a.data(), 1, Ncells, fin) != Ncells) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    uint8_t* cur  = grid_a.data();
    uint8_t* next = grid_b.data();

    // Decide thread count: cap to grid rows and to 8 (challenge HW).
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    unsigned T = std::min<unsigned>(hw, 8u);
    if ((int)T > N) T = (unsigned)N;

    // Precompute fixed row bands.
    std::vector<int> row_lo(T), row_hi(T);
    for (unsigned t = 0; t < T; ++t) {
        row_lo[t] = (int)((uint64_t)t       * N / T);
        row_hi[t] = (int)((uint64_t)(t + 1) * N / T);
    }

    // Shared per-generation state.
    const uint8_t* shared_src = nullptr;
    uint8_t*       shared_dst = nullptr;
    bool stop = false;

    // Two barriers: start (workers wait until main publishes src/dst), done (main waits
    // for workers to finish the step). Main thread participates as worker 0, so both
    // barriers include T parties total.
    std::barrier bar_start(T);
    std::barrier bar_done (T);

    std::vector<std::thread> pool;
    pool.reserve(T - 1);
    for (unsigned t = 1; t < T; ++t) {
        pool.emplace_back([&, t]() {
            for (;;) {
                bar_start.arrive_and_wait();
                if (stop) return;
                step_rows(shared_src, shared_dst, N, row_lo[t], row_hi[t]);
                bar_done.arrive_and_wait();
            }
        });
    }

    auto t0 = std::chrono::steady_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        shared_src = cur;
        shared_dst = next;
        bar_start.arrive_and_wait();
        step_rows(cur, next, N, row_lo[0], row_hi[0]);
        bar_done.arrive_and_wait();
        std::swap(cur, next);
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    // Tear down workers.
    stop = true;
    bar_start.arrive_and_wait();
    for (auto& th : pool) th.join();

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }
    if (std::fwrite(&width,  sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(cur,     1, Ncells,           fout) != Ncells) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }
    std::fclose(fout);
    return 0;
}
