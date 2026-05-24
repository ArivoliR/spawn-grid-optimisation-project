#!/usr/bin/env bash
# build.sh — Builds the root-level spawn_sim baseline.
# Override the compiler with CXX, e.g.: CXX=clang++-18 bash build.sh
set -euo pipefail

CXX="${CXX:-g++-14}"
CXXFLAGS="${CXXFLAGS:--std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra}"
OUTPUT="${OUTPUT:-spawn_sim}"

echo "Building spawn_sim with $CXX ..."
"$CXX" $CXXFLAGS spawn_sim.cpp -o "$OUTPUT" -lpthread
echo "Done: $OUTPUT"
