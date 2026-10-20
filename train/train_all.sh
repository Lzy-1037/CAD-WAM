#!/usr/bin/env bash
# Train every Qwen3 size one after another on GPUs 4-7, each starting only when the previous
# one has finished.  Total ~5 hours.
#
#
# Env overrides:
#   MODELS="Qwen3-1.7B Qwen3-4B"   train a subset / different order
#   STOP_ON_FAIL=1                 abort the whole sequence if one model fails
#                                  (default: keep going, so one bad run does not waste the night)
#   EXTRA="--lr 1e-5 --new-lr 5e-5"  identical optimisation for every size (scaling table)
set -uo pipefail                  # deliberately NOT -e: a failed model must not kill the loop

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate verl
cd $REMOTE/project

MODELS="${MODELS:-Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
EXTRA="${EXTRA:-}"
SUMMARY=/datasets/anon/new_cadseq/checkpoints/train_all_$(date +%Y%m%d_%H%M%S).log

echo "sequence: ${MODELS}" | tee -a "${SUMMARY}"
echo "extra args: ${EXTRA:-<none>}" | tee -a "${SUMMARY}"
t_all=$(date +%s)
rc_all=0
for m in ${MODELS}; do
    echo "" | tee -a "${SUMMARY}"
    echo "=== $(date '+%F %T')  START ${m}" | tee -a "${SUMMARY}"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    bash train_cad_sft.sh "${m}" ${EXTRA}
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ "${rc}" -eq 0 ]; then
        echo "=== $(date '+%F %T')  OK    ${m}  ($((dt/60)) min)" | tee -a "${SUMMARY}"
    else
        echo "=== $(date '+%F %T')  FAIL  ${m}  (exit ${rc}, $((dt/60)) min)" | tee -a "${SUMMARY}"
        rc_all=1
        [ "${STOP_ON_FAIL}" -eq 1 ] && { echo "STOP_ON_FAIL=1, aborting" | tee -a "${SUMMARY}"; break; }
    fi
done

echo "" | tee -a "${SUMMARY}"
echo "=== all done in $(( ($(date +%s) - t_all) / 60 )) min" | tee -a "${SUMMARY}"
echo "checkpoints:" | tee -a "${SUMMARY}"
ls -d /datasets/anon/new_cadseq/checkpoints/Qwen3-*/*/final 2>/dev/null | tee -a "${SUMMARY}"
echo "summary written to ${SUMMARY}"
exit "${rc_all}"
