#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-/tmp/spawn_target_profile}"
GENS="${GENS:-10000}"
SIZE="${SIZE:-32768}"
CPUS="${CPUS:-0-7}"
CXX="${CXX:-g++-14}"
CXXFLAGS="${CXXFLAGS:--std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra -pthread}"

mkdir -p "$WORKDIR"

INPUT="$WORKDIR/input_${SIZE}_boundary.bin"
VB03_BIN="$WORKDIR/vb03"
VB04_BIN="$WORKDIR/vb04"
VB03_OUT="$WORKDIR/vb03_${SIZE}_${GENS}.bin"
VB04_OUT="$WORKDIR/vb04_${SIZE}_${GENS}.bin"
VB03_LOG="$WORKDIR/vb03_${SIZE}_${GENS}.log"
VB04_LOG="$WORKDIR/vb04_${SIZE}_${GENS}.log"

echo "Target profile workdir: $WORKDIR"
echo "Host: $(uname -a)"
echo "Compiler: $($CXX --version | head -1)"
echo "CPU list: $CPUS"
echo "Grid: ${SIZE}x${SIZE}, generations: $GENS"

echo
echo "Building vb03 and vb04..."
"$CXX" $CXXFLAGS "$ROOT/bitplane_versions/03_bitplane_ring_split.cpp" -o "$VB03_BIN"
"$CXX" $CXXFLAGS "$ROOT/bitplane_versions/04_bitplane_tree_reduce.cpp" -o "$VB04_BIN"

if [[ ! -f "$INPUT" ]]; then
  echo
  echo "Generating boundary-stress input: $INPUT"
  python3 - "$INPUT" "$SIZE" <<'PY'
import struct
import sys

path = sys.argv[1]
n = int(sys.argv[2])
band = max(1, n // 64)

row_edge = bytes([3]) * n
row_mid = bytes([3]) * band + bytes([0]) * (n - 2 * band) + bytes([3]) * band

with open(path, "wb") as f:
    f.write(struct.pack("<QQ", n, n))
    for y in range(n):
        f.write(row_edge if y < band or y >= n - band else row_mid)
PY
else
  echo
  echo "Reusing existing input: $INPUT"
fi

run_one() {
  local name="$1"
  local bin="$2"
  local out="$3"
  local log="$4"

  echo
  echo "Running $name..."
  if command -v perf >/dev/null 2>&1; then
    /usr/bin/time -v \
      perf stat -d -d -d \
      taskset -c "$CPUS" "$bin" "$INPUT" "$out" "$GENS" \
      >"$log" 2>&1
  else
    /usr/bin/time -v \
      taskset -c "$CPUS" "$bin" "$INPUT" "$out" "$GENS" \
      >"$log" 2>&1
  fi
  cat "$log"
}

run_one "vb03" "$VB03_BIN" "$VB03_OUT" "$VB03_LOG"
run_one "vb04" "$VB04_BIN" "$VB04_OUT" "$VB04_LOG"

echo
echo "Comparing vb03 and vb04 outputs..."
cmp "$VB03_OUT" "$VB04_OUT"
echo "Outputs match."

echo
echo "Summary:"
grep -H -m1 -E '[0-9]+(\.[0-9]+)? ms' "$VB03_LOG" "$VB04_LOG" || true
grep -H -E 'Elapsed \\(wall clock\\)|Maximum resident set size|User time|System time' "$VB03_LOG" "$VB04_LOG" || true
