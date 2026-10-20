#!/usr/bin/env bash
# Evaluate the four Qwen3ForCADActionState checkpoints (0.6B / 1.7B / 4B / 8B) one after another.
#   cd $REMOTE/project && bash cad_eval/run_all_eval.sh [extra rollout args]
# Env knobs (optional): GPUS=0,1,2,3,4,5,6,7  GEOM_WORKERS=6  PHASE=all|rollout|eval
cd $REMOTE/project

export GPU_PYTHON=$CONDA_ENV_verl/bin/python      # torch + transformers
export GEOM_PYTHON=$CONDA_ENV_cadeval/bin/python  # cadquery + trimesh + scipy
export GPUS="${GPUS:-4,5,6,7}"

CKPT_ROOT=/datasets/anon/new_cadseq/checkpoints
RUNS=(
  "0_6B  ${CKPT_ROOT}/Qwen3-0.6B/20260908_121442/final"
  "1_7B  ${CKPT_ROOT}/Qwen3-1.7B/20260908_123759/final"
  "4B    ${CKPT_ROOT}/Qwen3-4B/20260909_032856/final"
  "8B    ${CKPT_ROOT}/Qwen3-8B/20260908_131016/final"
)

status=()
for entry in "${RUNS[@]}"; do
  read -r name ckpt <<< "${entry}"
  echo "=================== ${name}: ${ckpt} ==================="
  if bash cad_eval/run_eval.sh "${ckpt}" "${name}" "$@"; then
    status+=("${name}: OK")
  else
    status+=("${name}: FAILED (see cad_eval/runs/${name}/*.log)")
  fi
done

echo
echo "=================== summary ==================="
printf '%s\n' "${status[@]}"
for entry in "${RUNS[@]}"; do
  read -r name _ <<< "${entry}"
  rep=cad_eval/runs/${name}/report.md
  [ -f "${rep}" ] && { echo "--- ${name}"; sed -n '/invalidity ratio/p;/^| CD/p;/^| NC/p;/^| HD/p' "${rep}"; }
done
