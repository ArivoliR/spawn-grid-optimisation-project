#!/usr/bin/env bash
# build.sh — Build the optimised spawn_sim binary.
set -euo pipefail

CXX="${CXX:-g++-14}"
CXXFLAGS="-std=c++23 -O3 -mcpu=neoverse-v2+sha3 -Wall -Wextra"
OUTPUT="${OUTPUT:-spawn_sim}"

echo "Building optimised implementation with $CXX ..."
"$CXX" $CXXFLAGS spawn_sim.cpp -o "$OUTPUT" -lpthread
echo "Done: $OUTPUT"
