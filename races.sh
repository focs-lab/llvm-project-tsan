#!/usr/bin/env bash
#
# races.sh
#
# Builds each RACE_CASE against baseline (-tsan-enable-l0=false) and your
# L0 filter (-tsan-enable-l0=true), runs N trials of each, and writes ONE
# consolidated report.txt covering:
#   - detection rate (did a warning fire at all)
#   - tally of warning "kinds" TSan reported
#   - tally of *which source lines* were reported as the racing accesses
#     (this is the sharper signal: even when detection rate ties, the line
#     numbers reveal whether the *intended* racy access was ever seen by
#     TSan's real runtime, or only a coincidental first-touch elsewhere)
#
# Only binaries + report.txt are kept on disk; all per-run output is
# captured in memory / scratch temp files that get cleaned up automatically.

set -uo pipefail

# ------------------------- CONFIG -------------------------
CLANGXX="./build/bin/clang++"
SRC="race.cpp"
CASES=(1 2 3 4) # skip 5 for now
TRIALS=50
OUTDIR="race_results"
REPORT="$OUTDIR/report.txt"
SDKROOT_FLAG=(-isysroot "$(xcrun --show-sdk-path 2>/dev/null)")
# ------------------------------------------------------------

mkdir -p "$OUTDIR"
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

: > "$REPORT"
log() { echo "$@" | tee -a "$REPORT"; }

log "== Building binaries =="
for case in "${CASES[@]}"; do
  base_bin="$OUTDIR/race${case}_base"
  l0_bin="$OUTDIR/race${case}_l0"

  "$CLANGXX" "${SDKROOT_FLAG[@]}" -fsanitize=thread -mllvm -tsan-enable-l0=false \
    -DRACE_CASE=$case -O1 -g "$SRC" -o "$base_bin" \
    || { echo "BUILD FAILED: $base_bin"; exit 1; }

  "$CLANGXX" "${SDKROOT_FLAG[@]}" -fsanitize=thread -mllvm -tsan-enable-l0=true \
    -DRACE_CASE=$case -O1 -g "$SRC" -o "$l0_bin" \
    || { echo "BUILD FAILED: $l0_bin"; exit 1; }
done
log "Build OK."
log ""

# run_trials <binary> <kinds_file> <lines_file> -> echoes hit count
run_trials() {
  local binary=$1
  local kinds_file=$2
  local lines_file=$3
  local hits=0

  : > "$kinds_file"
  : > "$lines_file"

  for i in $(seq 1 "$TRIALS"); do
    out=$(TSAN_OPTIONS="halt_on_error=0" "$binary" 2>&1)
    if echo "$out" | grep -q "WARNING: ThreadSanitizer"; then
      hits=$((hits+1))
      echo "$out" | grep "WARNING: ThreadSanitizer:" >> "$kinds_file"
      # Extract "race.cpp:NN" style locations from the stack traces
      echo "$out" | grep -oE "race\.cpp:[0-9]+" >> "$lines_file"
    fi
  done

  echo "$hits"
}

declare -A BASE_HITS L0_HITS

for case in "${CASES[@]}"; do
  base_bin="$OUTDIR/race${case}_base"
  l0_bin="$OUTDIR/race${case}_l0"
  base_kinds="$SCRATCH/base${case}.kinds"
  base_lines="$SCRATCH/base${case}.lines"
  l0_kinds="$SCRATCH/l0${case}.kinds"
  l0_lines="$SCRATCH/l0${case}.lines"

  BASE_HITS[$case]=$(run_trials "$base_bin" "$base_kinds" "$base_lines")
  L0_HITS[$case]=$(run_trials "$l0_bin" "$l0_kinds" "$l0_lines")

  {
    echo "---- case $case ----"
    echo "detection rate: base=${BASE_HITS[$case]}/$TRIALS   l0=${L0_HITS[$case]}/$TRIALS"
    echo ""
    echo "warning kinds (base):"
    sort "$base_kinds" | uniq -c | sort -rn | sed 's/^/  /'
    echo "warning kinds (l0):"
    sort "$l0_kinds" | uniq -c | sort -rn | sed 's/^/  /'
    echo ""
    echo "reported source lines (base):"
    sort "$base_lines" | uniq -c | sort -rn | sed 's/^/  /'
    echo "reported source lines (l0):"
    sort "$l0_lines" | uniq -c | sort -rn | sed 's/^/  /'
    echo ""
    if diff -q <(sort "$base_lines" | uniq -c) <(sort "$l0_lines" | uniq -c) > /dev/null; then
      echo "  => SAME lines reported at same rates: filter had no observable effect on detection."
    else
      echo "  => DIFFERENT lines/rates reported — filter is changing which accesses TSan actually sees:"
      diff <(sort "$base_lines" | uniq -c) <(sort "$l0_lines" | uniq -c) | sed 's/^/     /'
    fi
    echo ""
  } >> "$REPORT"
done

log "== Summary =="
printf "%-8s %-20s %-20s\n" "case" "base (hits/trials)" "l0 (hits/trials)" >> "$REPORT"
for case in "${CASES[@]}"; do
  printf "%-8s %-20s %-20s\n" "$case" "${BASE_HITS[$case]}/$TRIALS" "${L0_HITS[$case]}/$TRIALS" >> "$REPORT"
done

echo
echo "Full report written to: $REPORT"
echo "Binaries kept under:    $OUTDIR/"
cat "$REPORT" | tail -n +$(( $(grep -n "== Summary ==" "$REPORT" | cut -d: -f1) ))