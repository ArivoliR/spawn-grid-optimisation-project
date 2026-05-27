// spawn_sim.cpp — Bitplane SIMD + ARMv8.2-SHA3 EOR3 + threaded variant of
// reference/spawn_sim.cpp. I/O format, CLI, error codes, and main()
// structure preserved verbatim. The reference's scalar count_adults +
// step are replaced by a bitplane SIMD step_strip; main()'s serial
// for-gen loop is replaced by a parallel block of worker threads
// synchronised with an atomic-counter barrier.

#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static const int RANGE = 2;

static const uint8_t EMPTY    = 0;
static const uint8_t EGG      = 1;
static const uint8_t JUVENILE = 2;
static const uint8_t ADULT    = 3;

// ---- Byte <-> bitplane conversion (one-shot, outside the timed region) ----
static void bytes_to_bitplanes(const uint8_t* bytes, uint8_t* lo, uint8_t* hi, size_t G) {
    for (size_t i = 0; i < G * G; i += 8) {
        uint8_t l = 0, h = 0;
        for (int j = 0; j < 8; ++j) {
            uint8_t v = bytes[i + j];
            l |= uint8_t((v & 1u) << j);
            h |= uint8_t(((v >> 1) & 1u) << j);
        }
        lo[i >> 3] = l; hi[i >> 3] = h;
    }
}
static void bitplanes_to_bytes(const uint8_t* lo, const uint8_t* hi, uint8_t* bytes, size_t G) {
    for (size_t p = 0; p < (G * G) / 8; ++p) {
        uint8_t l = lo[p], h = hi[p];
        for (int j = 0; j < 8; ++j)
            bytes[p * 8 + j] = uint8_t(((l >> j) & 1u) | (((h >> j) & 1u) << 1));
    }
}

// ---- Row shifts (LSB-first; cross 128-bit register boundary via vextq_u8) ----
static inline uint8x16_t shl_row_1(uint8x16_t p, uint8x16_t c) {
    return vorrq_u8(vshlq_n_u8(c, 1), vshrq_n_u8(vextq_u8(p, c, 15), 7));
}
static inline uint8x16_t shl_row_2(uint8x16_t p, uint8x16_t c) {
    return vorrq_u8(vshlq_n_u8(c, 2), vshrq_n_u8(vextq_u8(p, c, 15), 6));
}
static inline uint8x16_t shr_row_1(uint8x16_t c, uint8x16_t n) {
    return vorrq_u8(vshrq_n_u8(c, 1), vshlq_n_u8(vextq_u8(c, n, 1), 7));
}
static inline uint8x16_t shr_row_2(uint8x16_t c, uint8x16_t n) {
    return vorrq_u8(vshrq_n_u8(c, 2), vshlq_n_u8(vextq_u8(c, n, 1), 6));
}

// ---- Full adder: sum via EOR3, carry via MAJ ----
static inline uint8x16_t fa_sum  (uint8x16_t a, uint8x16_t b, uint8x16_t c) { return veor3q_u8(a, b, c); }
static inline uint8x16_t fa_carry(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
    return vbslq_u8(veorq_u8(a, b), c, vandq_u8(a, b));
}

struct H3 { uint8x16_t h0, h1, h2; };
struct V5 { uint8x16_t b0, b1, b2, b3, b4; };

static inline H3 sum_of_5(uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d, uint8x16_t e) {
    uint8x16_t s1 = fa_sum  (a, b, c), c1 = fa_carry(a, b, c);
    uint8x16_t s2 = fa_sum  (d, e, s1), c2 = fa_carry(d, e, s1);
    return { s2, veorq_u8(c1, c2), vandq_u8(c1, c2) };
}
static inline V5 v5_add_h3(V5 v, H3 h) {
    uint8x16_t s0 = veorq_u8(v.b0, h.h0), c0 = vandq_u8(v.b0, h.h0);
    uint8x16_t s1 = fa_sum  (v.b1, h.h1, c0), c1 = fa_carry(v.b1, h.h1, c0);
    uint8x16_t s2 = fa_sum  (v.b2, h.h2, c1), c2 = fa_carry(v.b2, h.h2, c1);
    uint8x16_t s3 = veorq_u8(v.b3, c2),       c3 = vandq_u8(v.b3, c2);
    return { s0, s1, s2, s3, veorq_u8(v.b4, c3) };
}
// borrow_out(A,B,Bin) = (~A & (B|Bin)) | (B & Bin)
static inline V5 v5_sub_h3(V5 v, H3 h) {
    uint8x16_t d0 = veorq_u8(v.b0, h.h0), b0 = vbicq_u8(h.h0, v.b0);
    uint8x16_t d1 = veor3q_u8(v.b1, h.h1, b0);
    uint8x16_t b1 = vorrq_u8(vbicq_u8(vorrq_u8(h.h1, b0), v.b1), vandq_u8(h.h1, b0));
    uint8x16_t d2 = veor3q_u8(v.b2, h.h2, b1);
    uint8x16_t b2 = vorrq_u8(vbicq_u8(vorrq_u8(h.h2, b1), v.b2), vandq_u8(h.h2, b1));
    uint8x16_t d3 = veorq_u8(v.b3, b2), b3 = vbicq_u8(b2, v.b3);
    return { d0, d1, d2, d3, veorq_u8(v.b4, b3) };
}
// V (5-bit) -= b (1-bit): removes the centre cell to get A from the 25-cell box V.
static inline V5 v5_sub_b(V5 v, uint8x16_t b) {
    uint8x16_t d0 = veorq_u8(v.b0, b); b = vbicq_u8(b, v.b0);
    uint8x16_t d1 = veorq_u8(v.b1, b); b = vbicq_u8(b, v.b1);
    uint8x16_t d2 = veorq_u8(v.b2, b); b = vbicq_u8(b, v.b2);
    uint8x16_t d3 = veorq_u8(v.b3, b); b = vbicq_u8(b, v.b3);
    return { d0, d1, d2, d3, veorq_u8(v.b4, b) };
}

static void compute_H_row(const uint8_t* lo, const uint8_t* hi, size_t R_REGS,
                          uint8_t* h0, uint8_t* h1, uint8_t* h2) {
    auto adult = [&](size_t r) {
        return vandq_u8(vld1q_u8(lo + r * 16), vld1q_u8(hi + r * 16));
    };
    uint8x16_t prev = adult(R_REGS - 1), curr = adult(0);
    for (size_t r = 0; r < R_REGS; ++r) {
        uint8x16_t next = adult((r + 1 == R_REGS) ? 0 : r + 1);
        H3 h = sum_of_5(shl_row_2(prev, curr), shl_row_1(prev, curr), curr,
                        shr_row_1(curr, next), shr_row_2(curr, next));
        vst1q_u8(h0 + r * 16, h.h0);
        vst1q_u8(h1 + r * 16, h.h1);
        vst1q_u8(h2 + r * 16, h.h2);
        prev = curr; curr = next;
    }
}

// ---- Per-thread 5-slot H ring + persistent V (replaces nothing in reference;
//      these are the bookkeeping needed by the vertical sliding-window sum) ----
struct Scratch {
    std::vector<uint8_t> hring[5][3];
    std::vector<uint8_t> v[5];
    size_t cap = 0;
};
static thread_local Scratch tls;
static void ensure_scratch(size_t R_BYTES) {
    if (tls.cap >= R_BYTES) return;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) tls.hring[i][j].assign(R_BYTES, 0);
        tls.v[i].assign(R_BYTES, 0);
    }
    tls.cap = R_BYTES;
}

// step_strip — replaces the reference's count_adults + step (called per
// generation per thread on a strip of rows [y_start, y_end)).
static void step_strip(const uint8_t* cur_low,  const uint8_t* cur_high,
                       uint8_t* next_low, uint8_t* next_high,
                       size_t N, size_t R_BYTES, size_t R_REGS,
                       size_t y_start, size_t y_end)
{
    ensure_scratch(R_BYTES);
    uint8_t* hring[5][3];
    uint8_t* vp[5];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) hring[i][j] = tls.hring[i][j].data();
        vp[i] = tls.v[i].data();
    }
    auto y_wrap = [&](long y) { return size_t((y % long(N) + long(N)) % long(N)); };

    for (int i = 0; i < 5; ++i) {
        size_t y_abs = y_wrap(long(y_start) + (i - 2));
        compute_H_row(cur_low + y_abs * R_BYTES, cur_high + y_abs * R_BYTES, R_REGS,
                      hring[i][0], hring[i][1], hring[i][2]);
    }
    for (size_t r = 0; r < R_REGS; ++r) {
        V5 v{ vld1q_u8(hring[0][0] + r*16), vld1q_u8(hring[0][1] + r*16),
              vld1q_u8(hring[0][2] + r*16), vdupq_n_u8(0), vdupq_n_u8(0) };
        for (int i = 1; i < 5; ++i)
            v = v5_add_h3(v, { vld1q_u8(hring[i][0] + r*16),
                               vld1q_u8(hring[i][1] + r*16),
                               vld1q_u8(hring[i][2] + r*16) });
        vst1q_u8(vp[0] + r*16, v.b0); vst1q_u8(vp[1] + r*16, v.b1);
        vst1q_u8(vp[2] + r*16, v.b2); vst1q_u8(vp[3] + r*16, v.b3);
        vst1q_u8(vp[4] + r*16, v.b4);
    }

    const size_t strip_rows = y_end - y_start;
    for (size_t k = 0; k < strip_rows; ++k) {
        const size_t y = y_start + k, row_off = y * R_BYTES;
        const int slot = int(k) % 5;
        const bool more = (k + 1 < strip_rows);

        for (size_t r = 0; r < R_REGS; ++r) {
            const size_t off = r * 16;
            V5 V{ vld1q_u8(vp[0] + off), vld1q_u8(vp[1] + off),
                  vld1q_u8(vp[2] + off), vld1q_u8(vp[3] + off),
                  vld1q_u8(vp[4] + off) };
            uint8x16_t low  = vld1q_u8(cur_low  + row_off + off);
            uint8x16_t high = vld1q_u8(cur_high + row_off + off);
            V5 A = v5_sub_b(V, vandq_u8(low, high));

            uint8x16_t hiz = vandq_u8(vmvnq_u8(A.b4), vmvnq_u8(A.b3));
            uint8x16_t E = vandq_u8(hiz, vorrq_u8(vbicq_u8(vandq_u8(A.b1, A.b0), A.b2),
                                                   vbicq_u8(A.b2, A.b1)));
            uint8x16_t R = vandq_u8(vmvnq_u8(A.b4),
                vorrq_u8(vbicq_u8(A.b2, A.b3),
                         vandq_u8(A.b3, vbicq_u8(vmvnq_u8(A.b1), A.b2))));
            uint8x16_t hl_R = vandq_u8(vandq_u8(high, low), R);
            vst1q_u8(next_high + row_off + off, vorrq_u8(veorq_u8(high, low), hl_R));
            vst1q_u8(next_low  + row_off + off,
                     vorrq_u8(vbicq_u8(vorrq_u8(high, E), low), hl_R));

            if (more) {
                V = v5_sub_h3(V, { vld1q_u8(hring[slot][0] + off),
                                   vld1q_u8(hring[slot][1] + off),
                                   vld1q_u8(hring[slot][2] + off) });
                vst1q_u8(vp[0] + off, V.b0); vst1q_u8(vp[1] + off, V.b1);
                vst1q_u8(vp[2] + off, V.b2); vst1q_u8(vp[3] + off, V.b3);
                vst1q_u8(vp[4] + off, V.b4);
            }
        }

        if (more) {
            size_t y_new = y_wrap(long(y) + 3);
            compute_H_row(cur_low + y_new * R_BYTES, cur_high + y_new * R_BYTES, R_REGS,
                          hring[slot][0], hring[slot][1], hring[slot][2]);
            for (size_t r = 0; r < R_REGS; ++r) {
                const size_t off = r * 16;
                V5 V{ vld1q_u8(vp[0] + off), vld1q_u8(vp[1] + off),
                      vld1q_u8(vp[2] + off), vld1q_u8(vp[3] + off),
                      vld1q_u8(vp[4] + off) };
                V = v5_add_h3(V, { vld1q_u8(hring[slot][0] + off),
                                   vld1q_u8(hring[slot][1] + off),
                                   vld1q_u8(hring[slot][2] + off) });
                vst1q_u8(vp[0] + off, V.b0); vst1q_u8(vp[1] + off, V.b1);
                vst1q_u8(vp[2] + off, V.b2); vst1q_u8(vp[3] + off, V.b3);
                vst1q_u8(vp[4] + off, V.b4);
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
            "Error: grid must be square and non-empty, got %" PRIu64 " × %" PRIu64 "\n",
            width, height);
        std::fclose(fin);
        return 3;
    }

    int grid_size = (int)width;
    const size_t N = (size_t)grid_size * grid_size;

    std::vector<uint8_t> grid_a(N);
    if (std::fread(grid_a.data(), 1, N, fin) != N) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    // -------------------------------------------------------------------------
    // Simulate
    // -------------------------------------------------------------------------
    // Bitplane buffers (one low + one high per double-buffer slot) replace the
    // reference's grid_a/grid_b byte buffers during the simulation only.
    const size_t R_BYTES = (size_t)grid_size / 8;
    const size_t R_REGS  = R_BYTES / 16;
    if (R_REGS == 0 || ((size_t)grid_size & ((size_t)grid_size - 1)) != 0) {
        std::fprintf(stderr,
            "Error: bitplane SIMD path requires grid_size to be a power of two >= 128\n");
        return 3;
    }
    std::vector<uint8_t> low_a ((size_t)grid_size * R_BYTES);
    std::vector<uint8_t> high_a((size_t)grid_size * R_BYTES);
    std::vector<uint8_t> low_b ((size_t)grid_size * R_BYTES);
    std::vector<uint8_t> high_b((size_t)grid_size * R_BYTES);
    bytes_to_bitplanes(grid_a.data(), low_a.data(), high_a.data(), (size_t)grid_size);

    uint8_t* cur_low   = low_a.data();
    uint8_t* cur_high  = high_a.data();
    uint8_t* next_low  = low_b.data();
    uint8_t* next_high = high_b.data();

    auto t0 = std::chrono::steady_clock::now();

    // Parallel block: replaces reference's serial for-gen loop.
    // n_threads workers each own a row strip; sync between generations via an
    // atomic-counter barrier whose "last arrival" thread swaps the cur/next
    // bitplane pointers (the equivalent of reference's std::swap(cur, next)).
    {
        unsigned hc = std::thread::hardware_concurrency();
        const int n_threads = int(std::min(hc ? hc : 1u, 8u));
        std::atomic<int> arrival{0}, gen_done{0};
        std::atomic<bool> started{false};
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        for (int t = 0; t < n_threads; ++t) {
            size_t ys = ((size_t)t     * (size_t)grid_size) / (size_t)n_threads;
            size_t ye = ((size_t)(t+1) * (size_t)grid_size) / (size_t)n_threads;
            workers.emplace_back([&, ys, ye]() {
                while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
                for (int gen = 0; gen < generations; ++gen) {
                    step_strip(cur_low, cur_high, next_low, next_high,
                               (size_t)grid_size, R_BYTES, R_REGS, ys, ye);
                    if (arrival.fetch_add(1, std::memory_order_acq_rel) + 1 == n_threads) {
                        std::swap(cur_low,  next_low);
                        std::swap(cur_high, next_high);
                        arrival.store(0, std::memory_order_relaxed);
                        gen_done.store(gen + 1, std::memory_order_release);
                    } else {
                        while (gen_done.load(std::memory_order_acquire) <= gen)
                            std::this_thread::yield();
                    }
                }
            });
        }
        started.store(true, std::memory_order_release);
        for (auto& w : workers) w.join();
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    // Convert final bitplanes back into grid_a (reused as the byte fwrite buffer).
    bitplanes_to_bytes(cur_low, cur_high, grid_a.data(), (size_t)grid_size);

    // -------------------------------------------------------------------------
    // Write output
    // -------------------------------------------------------------------------
    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }

    if (std::fwrite(&width,       sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height,      sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(grid_a.data(), 1, N, fout) != N) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }

    std::fclose(fout);
    return 0;
}
