#!/usr/bin/env bash
# Build and run the Treiber stack smoke test.
#   - once without TSan (checks the stack logic on its own)
#   - RUNS times with the fork's TSan (must print no warnings at all)
#
# Usage:  ./build.sh [runs]        default: 5 runs
#         CXX=/path/to/clang++ ./build.sh   to use a different compiler
set -euo pipefail

RUNS="${1:-5}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # tsan-aba-eval/treiber
ROOT="$(cd "$HERE/../.." && pwd)"                      # llvm-project
CXX="${CXX:-$HOME/llvm-build/bin/clang++}"
VARIANT="${VARIANT:-Leak}"
INC="$ROOT/compiler-rt/test/tsan/Inputs"
OUT="${TMPDIR:-/tmp}/aba_treiber"
mkdir -p "$OUT"

FLAGS=(-std=c++20 -O1 -g -Wall -Wextra -I "$INC" -DABA_TREIBER_VARIANT="$VARIANT")

echo "== compiler: $CXX  variant: $VARIANT"

echo "== plain build (no TSan)"
"$CXX" "${FLAGS[@]}" "$HERE/smoke.cpp" -o "$OUT/smoke_plain"
"$OUT/smoke_plain" || echo "plain run failed (exit $?)"

echo "== TSan build"
"$CXX" "${FLAGS[@]}" -fsanitize=thread "$HERE/smoke.cpp" -o "$OUT/smoke_tsan"

failed=0
for i in $(seq 1 "$RUNS"); do
  log="$OUT/run_$i.log"
  status=0
  "$OUT/smoke_tsan" >"$log" 2>&1 || status=$?
  if grep -q "WARNING: ThreadSanitizer" "$log"; then
    echo "run $i: TSan warning (log: $log)"
    grep "WARNING: ThreadSanitizer" "$log" | sort | uniq -c
    failed=1
  elif [ "$status" -ne 0 ]; then
    echo "run $i: exit code $status (log: $log)"
    tail -n 3 "$log"
    failed=1
  else
    echo "run $i: ok"
  fi
done

if [ "$failed" -eq 0 ]; then
  echo "== PASS: $RUNS/$RUNS TSan runs clean"
else
  echo "== FAIL: see the logs above"
  exit 1
fi
