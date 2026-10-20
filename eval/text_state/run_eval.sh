#!/usr/bin/env bash
# End-to-end test-set evaluation of a fine-tuned text-state checkpoint (Qwen3 or Qwen2.5, plain causal LM).
#   bash run_eval.sh <checkpoint_dir | gt> [run_name] [extra rollout args...]
# e.g.
#   bash run_eval.sh <ckpt> smoke --limit 200              # quick subset
#   bash run_eval.sh gt oracle_text_v5                     # ground-truth actions: pipeline test + floor
#   bash run_eval.sh <ckpt> v5_text --engine vllm          # vllm generation (much faster, needs vllm in GPU env)
# v5 (same test set, references, scoring and step limit as cad_eval/run_eval.sh for the latent model):
#   DATASET=dataset/cadseq_stepwise_text_v5 REF=$REMOTE/project/eval_refs/deepcad_v5_test \
#   bash text_cad_eval/run_eval.sh <ckpt> v5_text_qwen25_1_5B
set -xeuo pipefail

CKPT="${1:?usage: run_eval.sh <checkpoint_dir | gt> [run_name] [extra args]}"
RUN_NAME="${2:-$(date +%Y%m%d_%H%M%S)}"
shift 2 2>/dev/null || shift 1

PROJ="${PROJ:-$REMOTE/text_project}"             # cadseq/, text_state.py, dataset/, text_cad_eval/
DATASET="${DATASET:-dataset/cadseq_stepwise_text}"                 # v5: dataset/cadseq_stepwise_text_v5
REF="${REF:-$REMOTE/project/DeepCAD/official_step}"   # v5: new_cadseq/eval_refs/deepcad_v5_test
# another test file of the same dataset, e.g. the 80 models the 7,962-model test set adds:
#   TEST=${PROJ}/dataset/cadseq_stepwise_text_v5_count/test80extra.parquet
TEST="${TEST:-${PROJ}/${DATASET}/test.parquet}"
OUT="${OUT:-${PROJ}/eval/text_state/runs/${RUN_NAME}}"

# GEOM_PYTHON runs the geometry workers (cadquery): the env that built the dataset states (cadquery27);
# SCORE_PYTHON runs paper_eval.py (cadquery/OCP + trimesh + scipy), e.g. the cadeval env; defaults to GEOM_PYTHON.
GEOM_PYTHON="${GEOM_PYTHON:-$(command -v python)}"
SCORE_PYTHON="${SCORE_PYTHON:-${GEOM_PYTHON}}"
GPU_PYTHON="${GPU_PYTHON:-$(command -v python)}"     # env with torch + transformers (+ vllm for --engine vllm)
GPUS="${GPUS:-4,5,6,7}"
GEOM_WORKERS="${GEOM_WORKERS:-6}"
PHASE="${PHASE:-all}"               # all | rollout | eval
MAX_STEPS="${MAX_STEPS:-20}"        # same limit as the latent rollout (ground truth never exceeds 10)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6144}"   # answer = action + state (longest seen 3901 tokens)
MESH_SCALE="${MESH_SCALE:-unit}"    # unit (default, same as cad_eval) | native (verbatim CADFS)
REPORT_TAG="${REPORT_TAG:-}"
ORACLE_ARGS=(); [ "${CKPT}" = "gt" ] && ORACLE_ARGS=(--oracle)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT}"

set +x
fail=0
need_file() { [ -e "$1" ] || { echo "MISSING  $1"; echo "         -> $2"; fail=1; }; }
need_mod()  { "$1" -c "$2" >/dev/null 2>&1 || { echo "MISSING  $3  (in $1)"; echo "         -> $4"; fail=1; }; }
echo "--- pre-flight -------------------------------------------------------------"
[ "${CKPT}" = "gt" ] || [ "${PHASE}" = "eval" ] || need_file "${CKPT}/config.json" "point CKPT at a trained checkpoint dir (step-N/, epoch-N/ or final/), or 'gt'"
need_file "${TEST}" "build the dataset first (text_datagen/build_text_v5.py) or set DATASET=..."
[ "${PHASE}" = "rollout" ] || need_file "${REF}" "reference STEP dir (v5: $REMOTE/project/eval_refs/deepcad_v5_test)"
need_file "${PROJ}/cadseq/__init__.py" "copy the cadseq/ package to ${PROJ}/cadseq/"
need_file "${PROJ}/text_state.py" "copy stepwise_dataset/text_state.py to ${PROJ}/text_state.py"
need_mod "${GPU_PYTHON}" "import torch, transformers, pandas" "torch/transformers/pandas" "GPU_PYTHON=/path/to/verl/bin/python"
need_mod "${GEOM_PYTHON}" "
import sys; sys.path.insert(0, '${PROJ}')
import cadquery, cadseq, text_state" "cadquery + cadseq + text_state importable by GEOM_PYTHON" \
    "GEOM_PYTHON must be the cadquery env that built the dataset (cadquery27); check ${PROJ}/cadseq/ and ${PROJ}/text_state.py"
[ "${PHASE}" = "rollout" ] || need_mod "${SCORE_PYTHON}" "import cadquery, trimesh, scipy" "cadquery/trimesh/scipy" \
    "SCORE_PYTHON must have cadquery + trimesh + scipy (e.g. the cadeval env)"
if [ "${fail}" -ne 0 ]; then echo "pre-flight failed; fix the items above and re-run." >&2; exit 1; fi
echo "pre-flight OK   GPU_PYTHON=${GPU_PYTHON}"
echo "                GEOM_PYTHON=${GEOM_PYTHON}   ($("${GEOM_PYTHON}" -c 'import cadquery, OCP; print("cadquery", cadquery.__version__, "OCP", OCP.__version__)' 2>/dev/null))"
echo "                SCORE_PYTHON=${SCORE_PYTHON}"
echo "                model=$( [ "${CKPT}" = gt ] && echo 'ground-truth actions (oracle)' || echo ${CKPT} )"
echo "                test=${TEST}"
echo "                ref=${REF}   out=${OUT}   max_steps=${MAX_STEPS}   mesh_scale=${MESH_SCALE}"
set -x

if [ "${PHASE}" != "eval" ]; then
set +x
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NSHARD=${#GPU_ARR[@]}
echo "launching ${NSHARD} rollout shard(s) on GPU(s) ${GPUS}"
set -x
pids=()
for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" "${GPU_PYTHON}" "${PROJ}/eval/text_state/rollout.py" \
        --model "${CKPT}" --test "${TEST}" --out "${OUT}" \
        --cadseq-root "${PROJ}" --geom-python "${GEOM_PYTHON}" --geom-workers "${GEOM_WORKERS}" \
        --batch 16 --max-steps "${MAX_STEPS}" --max-new-tokens "${MAX_NEW_TOKENS}" \
        --shard "${i}" --num-shards "${NSHARD}" \
        ${ORACLE_ARGS[@]+"${ORACLE_ARGS[@]}"} \
        "$@" > >(tee "${OUT}/rollout_shard${i}.log") 2>&1 &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
[ "${rc}" -eq 0 ] || { echo "ERROR: a rollout shard failed; see ${OUT}/rollout_shard*.log" >&2; exit 1; }
if [ "${NSHARD}" -gt 1 ]; then   # one shard writes rollout.jsonl / rollout_summary.json itself
"${GPU_PYTHON}" "${PROJ}/eval/text_state/merge_shards.py" --out "${OUT}" --num-shards "${NSHARD}" 2>&1 | tee "${OUT}/rollout.log"
fi
fi

if [ "${PHASE}" = "rollout" ]; then
    echo "generation done: ${OUT}/pred_step/ + rollout.jsonl ; score later with: PHASE=eval bash text_cad_eval/run_eval.sh ${CKPT} ${RUN_NAME}"
    exit 0
fi

SFX="${REPORT_TAG:+_${REPORT_TAG}}"
"${SCORE_PYTHON}" "${PROJ}/eval/text_state/paper_eval.py" \
    --pred-dir "${OUT}/pred_step" \
    --ref-dir "${REF}" \
    --ids "${IDS}" \
    --system "$(basename "${OUT}")" \
    --out "${OUT}/paper_metrics.json" \
    --workers 16
echo "Report: ${OUT}/report${SFX}.md   (also report${SFX}.json, metrics${SFX}.csv, rollout.jsonl, pred_step/)"
