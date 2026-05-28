// Version 07: std::execution experiment.
// Change from 04_row_pointers.cpp: use std::for_each(std::execution::par, ...)
// over fixed row chunks each generation instead of managing threads directly.
//
// This is intentionally placed before 08_sliding_counts because it tests a
// parallelization API choice, not a different neighbor-counting algorithm.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <thread>
#include <vector>

static constexpr uint8_t EMPTY    = 0;
static constexpr uint8_t EGG      = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT    = 3;

struct RowChunk {
    int y0;
    int y1;
};

static void step_rows(const uint8_t* src, uint8_t* dst, int N, int y0, int y1)
{
    const int mask = N - 1;
    for (int y = y0; y < y1; ++y) {
        const int ym2 = (y - 2 + N) & mask;
        const int ym1 = (y - 1 + N) & mask;
        const int yp1 = (y + 1) & mask;
        const int yp2 = (y + 2) & mask;

        const uint8_t* r0 = src + (size_t)ym2 * N;
        const uint8_t* r1 = src + (size_t)ym1 * N;
        const uint8_t* r2 = src + (size_t)y   * N;
        const uint8_t* r3 = src + (size_t)yp1 * N;
        const uint8_t* r4 = src + (size_t)yp2 * N;
        uint8_t* rw = dst + (size_t)y * N;

        for (int x = 0; x < N; ++x) {
            const int xm2 = (x - 2 + N) & mask;
            const int xm1 = (x - 1 + N) & mask;
            const int xp1 = (x + 1) & mask;
            const int xp2 = (x + 2) & mask;

            const int A =
                (r0[xm2] == ADULT) + (r0[xm1] == ADULT) + (r0[x] == ADULT) + (r0[xp1] == ADULT) + (r0[xp2] == ADULT) +
                (r1[xm2] == ADULT) + (r1[xm1] == ADULT) + (r1[x] == ADULT) + (r1[xp1] == ADULT) + (r1[xp2] == ADULT) +
                (r2[xm2] == ADULT) + (r2[xm1] == ADULT) +                       (r2[xp1] == ADULT) + (r2[xp2] == ADULT) +
                (r3[xm2] == ADULT) + (r3[xm1] == ADULT) + (r3[x] == ADULT) + (r3[xp1] == ADULT) + (r3[xp2] == ADULT) +
                (r4[xm2] == ADULT) + (r4[xm1] == ADULT) + (r4[x] == ADULT) + (r4[xp1] == ADULT) + (r4[xp2] == ADULT);

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

    std::vector<uint8_t> grid_a(Ncells), grid_b(Ncells);
    if (std::fread(grid_a.data(), 1, Ncells, fin) != Ncells) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    unsigned T = std::min<unsigned>(hw, 8u);
    if ((int)T > N) T = (unsigned)N;

    std::vector<RowChunk> chunks(T);
    for (unsigned t = 0; t < T; ++t) {
        chunks[t].y0 = (int)((uint64_t)t * N / T);
        chunks[t].y1 = (int)((uint64_t)(t + 1) * N / T);
    }

    uint8_t* cur = grid_a.data();
    uint8_t* next = grid_b.data();

    auto t0 = std::chrono::steady_clock::now();
    for (int gen = 0; gen < generations; ++gen) {
        std::for_each(std::execution::par, chunks.begin(), chunks.end(),
            [&](const RowChunk& c) {
                step_rows(cur, next, N, c.y0, c.y1);
            });
        std::swap(cur, next);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

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
