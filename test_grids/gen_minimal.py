#!/usr/bin/env python3
"""Minimal stdlib-only generator for small spawn-grid test inputs.

Matches the seed / generator semantics of generate.py closely enough that
the same patterns are produced. Used here because numpy isn't installed.
"""
import argparse
import os
import random
import struct
from pathlib import Path

EMPTY, EGG, JUVENILE, ADULT = 0, 1, 2, 3

def write_grid(path, n, cells_bytes):
    assert len(cells_bytes) == n * n
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', n, n))
        f.write(cells_bytes)

def gen_random_low(rng, n):
    buf = bytearray(n * n)
    for i in range(n * n):
        if rng.random() < 0.05:
            buf[i] = ADULT
    return bytes(buf)

def gen_random_high(rng, n):
    buf = bytearray(n * n)
    for i in range(n * n):
        if rng.random() < 0.50:
            buf[i] = ADULT
    return bytes(buf)

def gen_structured(rng, n):
    buf = bytearray(n * n)
    period, block = 10, 3
    for y in range(n):
        if y % period >= block:
            continue
        row_off = y * n
        for x in range(n):
            if x % period < block:
                buf[row_off + x] = ADULT
    return bytes(buf)

def gen_boundary_stress(rng, n):
    buf = bytearray(n * n)
    density = 0.75
    border = 4
    def fill(ys, xs):
        for y in ys:
            row_off = y * n
            for x in xs:
                if rng.random() < density:
                    buf[row_off + x] = ADULT
    fill(range(0, border),         range(n))
    fill(range(n - border, n),     range(n))
    fill(range(n),                 range(0, border))
    fill(range(n),                 range(n - border, n))
    return bytes(buf)

GRIDS = [
    ('public_1_random_low',      gen_random_low),
    ('public_2_random_high',     gen_random_high),
    ('public_3_structured',      gen_structured),
    ('public_5_boundary_stress', gen_boundary_stress),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default=str(Path(__file__).parent))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sizes', default='512,2048')
    args = p.parse_args()
    sizes = [int(s) for s in args.sizes.split(',')]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for size in sizes:
        print(f"{size}x{size}:")
        for name, fn in GRIDS:
            path = out_dir / f"{name}_{size}.bin"
            cells = fn(rng, size)
            write_grid(path, size, cells)
            n_adult = cells.count(ADULT)
            print(f"  {path.name}: {n_adult*100/(size*size):.1f}% adult, {(16+size*size)/1024**2:.1f} MiB")

if __name__ == '__main__':
    main()
