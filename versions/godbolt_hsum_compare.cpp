// Godbolt comparison snippet for v09 vs v10 horizontal-sum styles.
//
// Suggested flags:
//   -std=c++23 -O3 -march=native
//
// Compare:
//   build_hsum_rolling_wrap   -> v09-style rolling update with wrap arithmetic
//   build_hsum_direct_interior -> v10-style independent interior sums

#include <cstdint>

static constexpr uint8_t ADULT = 3;

static inline uint8_t adult_at(const uint8_t* row, int x)
{
    return row[x] == ADULT;
}

// v09
__attribute__((noinline))
void build_hsum_rolling_wrap(const uint8_t* row, uint8_t* out, int N)
{
    const int mask = N - 1;
    uint8_t h = (uint8_t)(adult_at(row, (N - 2) & mask) +
                          adult_at(row, (N - 1) & mask) +
                          adult_at(row, 0) +
                          adult_at(row, 1 & mask) +
                          adult_at(row, 2 & mask));

    for (int x = 0; x < N; ++x) {
        out[x] = h;
        const int remove_x = (x - 2 + N) & mask;
        const int add_x = (x + 3) & mask;
        h = (uint8_t)(h + adult_at(row, add_x) - adult_at(row, remove_x));
    }
}

// v10
__attribute__((noinline))
void build_hsum_direct_interior(const uint8_t* row, uint8_t* out, int N)
{
    for (int x = 2; x < N - 2; ++x) {
        out[x] = (uint8_t)(adult_at(row, x - 2) +
                           adult_at(row, x - 1) +
                           adult_at(row, x) +
                           adult_at(row, x + 1) +
                           adult_at(row, x + 2));
    }
}

// Optional caller so Godbolt cannot discard the functions if you compile whole file.
void call_both(const uint8_t* row, uint8_t* out1, uint8_t* out2, int N)
{
    build_hsum_rolling_wrap(row, out1, N);
    build_hsum_direct_interior(row, out2, N);
}
