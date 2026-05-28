// Version 03: version 02 plus power-of-two bitmask wrapping.
// Change from 02_unrolled_scalar.cpp: replace modulo wrap with & (N - 1).

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr uint8_t EMPTY    = 0;
static constexpr uint8_t EGG      = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT    = 3;

static void step(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int N)
{
    const int mask = N - 1;
    for (int y = 0; y < N; ++y) {
        const int ym2 = (y - 2 + N) & mask;
        const int ym1 = (y - 1 + N) & mask;
        const int yp1 = (y + 1) & mask;
        const int yp2 = (y + 2) & mask;

        for (int x = 0; x < N; ++x) {
            const int xm2 = (x - 2 + N) & mask;
            const int xm1 = (x - 1 + N) & mask;
            const int xp1 = (x + 1) & mask;
            const int xp2 = (x + 2) & mask;

            const int A =
                (src[(size_t)ym2 * N + xm2] == ADULT) +
                (src[(size_t)ym2 * N + xm1] == ADULT) +
                (src[(size_t)ym2 * N + x]   == ADULT) +
                (src[(size_t)ym2 * N + xp1] == ADULT) +
                (src[(size_t)ym2 * N + xp2] == ADULT) +
                (src[(size_t)ym1 * N + xm2] == ADULT) +
                (src[(size_t)ym1 * N + xm1] == ADULT) +
                (src[(size_t)ym1 * N + x]   == ADULT) +
                (src[(size_t)ym1 * N + xp1] == ADULT) +
                (src[(size_t)ym1 * N + xp2] == ADULT) +
                (src[(size_t)y   * N + xm2] == ADULT) +
                (src[(size_t)y   * N + xm1] == ADULT) +
                (src[(size_t)y   * N + xp1] == ADULT) +
                (src[(size_t)y   * N + xp2] == ADULT) +
                (src[(size_t)yp1 * N + xm2] == ADULT) +
                (src[(size_t)yp1 * N + xm1] == ADULT) +
                (src[(size_t)yp1 * N + x]   == ADULT) +
                (src[(size_t)yp1 * N + xp1] == ADULT) +
                (src[(size_t)yp1 * N + xp2] == ADULT) +
                (src[(size_t)yp2 * N + xm2] == ADULT) +
                (src[(size_t)yp2 * N + xm1] == ADULT) +
                (src[(size_t)yp2 * N + x]   == ADULT) +
                (src[(size_t)yp2 * N + xp1] == ADULT) +
                (src[(size_t)yp2 * N + xp2] == ADULT);

            const uint8_t cell = src[(size_t)y * N + x];
            uint8_t next;
            switch (cell) {
                case EMPTY:    next = (A >= 3 && A <= 5) ? EGG   : EMPTY; break;
                case EGG:      next = JUVENILE;                            break;
                case JUVENILE: next = ADULT;                               break;
                case ADULT:    next = (A >= 4 && A <= 9) ? ADULT : EMPTY;  break;
                default:       next = EMPTY;                               break;
            }
            dst[(size_t)y * N + x] = next;
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

    auto* cur = &grid_a;
    auto* next = &grid_b;

    auto t0 = std::chrono::steady_clock::now();
    for (int gen = 0; gen < generations; ++gen) {
        step(*cur, *next, N);
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
        std::fwrite(cur->data(), 1, Ncells, fout) != Ncells) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }
    std::fclose(fout);
    return 0;
}
