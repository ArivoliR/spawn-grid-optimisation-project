#!/usr/bin/env bash
# run_32k_verify.sh — Detached harness: runs the reference on the 32K input
# (~25 hours), then bit-identity-checks our output against it. Writes a
# final PASS/FAIL line to 32k_verify_result.txt.
#
# Run via:  setsid nohup bash run_32k_verify.sh >32k_verify.log 2>&1 </dev/null &

set -u

cd "$(dirname "$0")"

LOG=32k_verify.log
RESULT=32k_verify_result.txt
REF_OUTPUT=test_grids/public_1_random_low_32768.expected.bin
MY_OUTPUT=test_grids/mine_32768_output.bin
INPUT=test_grids/public_1_random_low_32768.bin

echo "=== 32K verification job ===" | tee -a "$LOG"
echo "started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG"
echo "host:    $(hostname)"          | tee -a "$LOG"
echo "pid:     $$"                   | tee -a "$LOG"

if [[ ! -f "$MY_OUTPUT" ]]; then
    echo "FAIL: my output not found at $MY_OUTPUT" | tee -a "$LOG" "$RESULT"
    exit 1
fi
if [[ ! -f "$INPUT" ]]; then
    echo "FAIL: input not found at $INPUT" | tee -a "$LOG" "$RESULT"
    exit 1
fi

echo "Running reference (estimated ~25 hours) ..." | tee -a "$LOG"
echo "ref start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG"
REF_TIME=$(taskset -c 0-7 ./spawn_sim_ref "$INPUT" "$REF_OUTPUT" 2>&1 | tail -1)
RC=$?
echo "ref done:  $(date -u '+%Y-%m-%d %H:%M:%S UTC') (exit $RC)" | tee -a "$LOG"
echo "ref time:  $REF_TIME" | tee -a "$LOG"

if [[ $RC -ne 0 ]]; then
    echo "FAIL: reference exited non-zero ($RC)" | tee -a "$LOG" "$RESULT"
    exit 1
fi

echo "Comparing my output vs reference output ..." | tee -a "$LOG"
if python3 harness/verify.py "$REF_OUTPUT" "$MY_OUTPUT" >>"$LOG" 2>&1; then
    {
        echo "PASS"
        echo "32K x 32K x 10000 generations: bit-identical to reference."
        echo "reference time: $REF_TIME"
        echo "mine time:      138978.983 ms"
        echo "completed:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    } | tee "$RESULT"
else
    {
        echo "FAIL"
        echo "verify.py reported a mismatch. See $LOG for the first differing byte."
        echo "reference time: $REF_TIME"
        echo "completed:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    } | tee "$RESULT"
fi
