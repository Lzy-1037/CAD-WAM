#!/usr/bin/env bash
# End-to-end test-set evaluation of a trained Qwen3ForCADActionState checkpoint.
#   bash run_eval.sh <checkpoint_dir | gt> [run_name] [extra rollout args...]
# e.g.
#   bash run_eval.sh /datasets/anon/new_cadseq/checkpoints/cad_sft_20260904_010203/final exp1
#   bash run_eval.sh <ckpt> smoke --limit 40           # quick subset
#   bash run_eval.sh gt oracle_v5                       # ground-truth actions (pipeline check + floor)
# v5:
#   DATASET=dataset/cadseq_stepwise_v5 REF=$REMOTE/project/eval_refs/deepcad_v5_test \
#   bash cad_eval/run_eval.sh <ckpt> v5_1_7B
# v6 (the encoder is taken from the dataset, see ENCODER_CKPT below):
#   DATASET=dataset/cadseq_stepwise_v6 REF=$REMOTE/project/eval_refs/deepcad_v5_test \
#   bash cad_eval/run_eval.sh <ckpt> v6_qwen25
# Score an existing run against a second reference without overwriting the first report:
#   PHASE=eval REF=<oracle run>/pred_step REPORT_TAG=cadseq bash cad_eval/run_eval.sh <ckpt> v5_1_7B
set -xeuo pipefail

CKPT="${1:?usage: run_eval.sh <checkpoint_dir | gt> [run_name] [extra args]}"
RUN_NAME="${2:-$(date +%Y%m%d_%H%M%S)}"
shift 2 2>/dev/null || shift 1

PROJ="${PROJ:-$REMOTE/project}"             # holds cadwam/, cadseq/, dataset/, cad_eval/
ENCODER="${ENCODER:-$REMOTE/code/pc_latent_encoder}"  # point cloud -> 1024-d latent
# reference STEP files named <model_id>.step: DeepCAD/official_step = the 890 v2-v4 test models,
# eval_refs/deepcad_v5_test = the 7,882 v5 test models (official STEP; corrected STEP for the
# 123 models hit by DeepCAD's plane bug)
REF="${REF:-${PROJ}/DeepCAD/official_step}"
# rollout uses only the prompts / GT actions from this file (the stored latents are not
# read -- z_t comes from the live encoder), but keep it on the version the model was
# trained on so prompt text can never silently diverge.
# must be the dataset the model was trained on: its test_stats.json tells the rollout which prompt
# lines to rebuild each step.  e.g.  DATASET=dataset/cadseq_stepwise_v3cb bash run_eval.sh <ckpt> <name>
DATASET="${DATASET:-dataset/cadseq_stepwise_v3}"
# another test file of the same dataset (same prompt format and encoder), e.g. the 7,962-model set:
#   TEST=${PROJ}/dataset/cadseq_stepwise_v6/test7962.parquet
TEST="${TEST:-${PROJ}/${DATASET}/test.parquet}"
OUT="${OUT:-${PROJ}/eval/runs/${RUN_NAME}}"

# GEOM_PYTHON runs the rollout's geometry workers (cadquery): use the env that generated the training
# clouds (cadquery 2.7.0 / OCP 7.8.1.1) so the encoder sees the same tessellation.
# SCORE_PYTHON runs paper_eval.py (cadquery/OCP + trimesh + scipy); defaults to GEOM_PYTHON.
GEOM_PYTHON="${GEOM_PYTHON:-$(command -v python)}"
SCORE_PYTHON="${SCORE_PYTHON:-${GEOM_PYTHON}}"
GPU_PYTHON="${GPU_PYTHON:-$(command -v python)}"

# GPUs for the rollout: one independent shard per card (data parallel, ~N x faster).
# Cards 4-7 by default, leaving 0-3 free; GPUS=0,1,2,3,4,5,6,7 to use the whole machine.
GPUS="${GPUS:-4,5,6,7}"
GEOM_WORKERS="${GEOM_WORKERS:-6}"   # geometry subprocesses PER SHARD (4 shards x 6 = 24 procs)
PHASE="${PHASE:-all}"               # all | rollout (generate only) | eval (score an existing run)
# per action; the longest ground-truth action is 1705 tokens (v5 test) -- 512 truncated 283 models
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
REPORT_TAG="${REPORT_TAG:-}"        # non-empty: write metrics_<tag>.csv / report_<tag>.md instead
# unit: scale shapes to max extent 1 before meshing (default; see metrics.py) | native: verbatim CADFS
MESH_SCALE="${MESH_SCALE:-unit}"
ORACLE=0; [ "${CKPT}" = "gt" ] && ORACLE=1
# SetVAE checkpoint of the live encoder: must be the one that built DATASET's latents.  Default = the file
# recorded as "encoder" in DATASET's test_stats.json (v5 -> setvae_verified_random_best.pt, v6 ->
# setvae_v6_deepcad_best.pt), else pc_latent_encoder's DEFAULT_CKPT; ENCODER_CKPT=<file> overrides.
# The pre-flight checks its md5 against the stats' encoder_md5 in every case.
STATS="${TEST%.parquet}_stats.json"
[ -f "${STATS}" ] || STATS="$(dirname "${TEST}")/test_stats.json"
ENCODER_CKPT="${ENCODER_CKPT:-}"
if [ -z "${ENCODER_CKPT}" ] && [ -f "${STATS}" ]; then
    ENCODER_CKPT=$("${GPU_PYTHON}" -c "import json, sys; print(json.load(open(sys.argv[1])).get('encoder') or '')" \
                   "${STATS}" 2>/dev/null || true)
fi
if [ -z "${ENCODER_CKPT}" ]; then
    ENCODER_CKPT="${ENCODER}/$("${GPU_PYTHON}" -c "import sys; sys.path.insert(0, sys.argv[1]); from encoder import DEFAULT_CKPT; print(DEFAULT_CKPT)" \
                   "${ENCODER}" 2>/dev/null || true)"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT}"

# ---- pre-flight: report EVERY missing piece at once, with the fix for each ---------------------
set +x
fail=0
need_file() { [ -e "$1" ] || { echo "MISSING  $1"; echo "         -> $2"; fail=1; }; }
need_mod()  {  # need_mod <python> <import stmt> <label> <fix hint>
    "$1" -c "$2" >/dev/null 2>&1 || { echo "MISSING  $3  (in $1)"; echo "         -> $4"; fail=1; }; }

echo "--- pre-flight -------------------------------------------------------------"
need_file "${TEST}" "rsync dataset/$(basename $(dirname ${TEST}))/ to ${PROJ}/dataset/"
if [ "${ORACLE}" -eq 0 ] && [ "${PHASE}" != "eval" ]; then
need_file "${CKPT}/config.json" "point CKPT at a trained checkpoint dir (epoch-N/ or final/), or 'gt' for the oracle"
need_file "${ENCODER}/encoder.py" \
    "rsync the pc_latent_encoder project to $(dirname ${ENCODER})/  (see README: 'Required transfers')"
# the encoder must be the one the training latents were built with (ENCODER_CKPT, resolved above);
# rollout.py prints what it loaded
need_file "${ENCODER_CKPT}" \
    "copy the SetVAE checkpoint that built ${DATASET} to this path, or set ENCODER_CKPT=<file>"
# ...and byte-identical to the checkpoint recorded in the test set's stats: a same-named file with other
# contents, or an override pointing at another encoder, would otherwise pass silently.
need_mod "${GPU_PYTHON}" "
import os, json, hashlib
p = '${STATS}'
want = json.load(open(p)).get('encoder_md5') if os.path.isfile(p) else None
if want:
    got = hashlib.md5(open('${ENCODER_CKPT}', 'rb').read()).hexdigest()
    assert got == want, ('${ENCODER_CKPT}', got, want)" \
    "encoder ${ENCODER_CKPT} != the encoder that built $(dirname ${TEST})" \
    "set ENCODER_CKPT to the file whose md5 is 'encoder_md5' in ${STATS}"
# ...and the checkpoint must be evaluated on the dataset variant it was trained on: variants differ in
# their prompt lines (cad_prompt.py) and the rollout rebuilds whatever DATASET's test_stats.json says.
need_mod "${GPU_PYTHON}" "
import json, os
c = os.path.join(os.path.dirname(os.path.abspath('${CKPT}')), 'config.json')
tr = json.load(open(c)).get('train', '') if os.path.isfile(c) else ''
assert not tr or os.path.dirname(tr).rstrip('/').endswith('${DATASET}'.rstrip('/')), (tr, '${DATASET}')" \
    "checkpoint was trained on a different dataset than DATASET=${DATASET}" \
    "set DATASET to the dir in $(dirname ${CKPT})/config.json 'train', e.g. DATASET=dataset/cadseq_stepwise_v5 bash run_eval.sh ..."
need_file "${PROJ}/cadwam/__init__.py" "the cadwam package must sit at ${PROJ}/cadwam/"
need_mod "${GPU_PYTHON}" "import torch, transformers" \
    "torch/transformers" "use the GPU env: GPU_PYTHON=/path/to/verl/bin/python"
fi
if [ "${PHASE}" != "rollout" ]; then
need_file "${REF}" "reference STEP dir (v5 test: ${PROJ}/eval_refs/deepcad_v5_test)"
fi
need_file "${PROJ}/cadseq/__init__.py" "the cadseq package must sit at ${PROJ}/cadseq/"
need_file "${PROJ}/eval/cad_prompt.py" "sync cad_eval/cad_prompt.py (prompt format shared with the dataset builder)"

need_mod "${GPU_PYTHON}" "import pandas, numpy" \
    "pandas/numpy" "use the GPU env: GPU_PYTHON=/path/to/verl/bin/python"
need_mod "${GEOM_PYTHON}" "import cadquery" "cadquery" \
    "GEOM_PYTHON must point at a CadQuery env (the one that generated the training clouds, e.g. cadquery27)"
if [ "${PHASE}" != "rollout" ]; then
need_mod "${SCORE_PYTHON}" "import cadquery, trimesh, scipy, numpy" "cadquery/trimesh/scipy" \
    "SCORE_PYTHON must have cadquery + trimesh + scipy (e.g. the cadeval env)"
fi
need_mod "${GEOM_PYTHON}" "
import sys; sys.path.insert(0, '${PROJ}')
import cadseq" "cadseq importable by GEOM_PYTHON" "check ${PROJ}/cadseq/ and that its deps are in the geometry env"

if [ "${fail}" -ne 0 ]; then
    echo "----------------------------------------------------------------------------"
    echo "pre-flight failed; fix the items above and re-run." >&2
    exit 1
fi
echo "pre-flight OK   GPU_PYTHON=${GPU_PYTHON}"
echo "                GEOM_PYTHON=${GEOM_PYTHON}   ($("${GEOM_PYTHON}" -c 'import cadquery, OCP; print("cadquery", cadquery.__version__, "OCP", OCP.__version__)' 2>/dev/null))"
echo "                SCORE_PYTHON=${SCORE_PYTHON}"
echo "                model=$( [ ${ORACLE} -eq 1 ] && echo 'ground-truth actions (oracle)' || echo ${CKPT} )"
[ "${ORACLE}" -eq 1 ] || echo "                encoder=${ENCODER_CKPT}"
echo "                test=${TEST}"
echo "                ref=${REF}${REPORT_TAG:+  (report tag: ${REPORT_TAG})}   out=${OUT}"
echo "----------------------------------------------------------------------------"
set -x

# ---- phase 1: closed-loop rollout, data-parallel over GPUS -------------------------------------
if [ "${PHASE}" != "eval" ]; then
# One independent process per GPU, each handling models[i::N]; STEP files land in the shared
# pred_step/ (filenames are model ids) and the per-shard jsonl files are merged afterwards.
set +x
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NSHARD=${#GPU_ARR[@]}
echo "launching ${NSHARD} rollout shard(s) on GPU(s) ${GPUS}"
ORACLE_ARGS=(); [ "${ORACLE}" -eq 1 ] && ORACLE_ARGS=(--oracle)
set -x
pids=()
for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" "${GPU_PYTHON}" "${PROJ}/eval/rollout.py" \
        --model "${CKPT}" \
        --test "${TEST}" \
        --out "${OUT}" \
        --encoder "${ENCODER}" \
        --encoder-ckpt "${ENCODER_CKPT}" \
        --cadseq-root "${PROJ}" \
        --geom-python "${GEOM_PYTHON}" \
        --geom-workers "${GEOM_WORKERS}" \
        --batch 16 \
        --max-steps 20 \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --geo-frame current \
        --shard "${i}" --num-shards "${NSHARD}" \
        ${ORACLE_ARGS[@]+"${ORACLE_ARGS[@]}"} \
        "$@" > >(tee "${OUT}/rollout_shard${i}.log") 2>&1 &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
[ "${rc}" -eq 0 ] || { echo "ERROR: a rollout shard failed; see ${OUT}/rollout_shard*.log" >&2; exit 1; }

# barrier: every shard has finished and exited 0 before anything below runs
if [ "${NSHARD}" -gt 1 ]; then   # one shard writes rollout.jsonl / rollout_summary.json itself
"${GPU_PYTHON}" "${PROJ}/eval/merge_shards.py" --out "${OUT}" --num-shards "${NSHARD}" \
    2>&1 | tee "${OUT}/rollout.log"
fi
fi

if [ "${PHASE}" = "rollout" ]; then
    set +x
    echo "generation done: ${OUT}/pred_step/ + rollout.jsonl"
    echo "score it later with:  PHASE=eval bash cad_eval/run_eval.sh ${CKPT} ${RUN_NAME}"
    exit 0
fi

# ---- phase 2: metrics vs the reference STEPs (geometry env) ------------------------------------
SFX="${REPORT_TAG:+_${REPORT_TAG}}"
"${SCORE_PYTHON}" "${PROJ}/eval/paper_eval.py" \
    --pred-dir "${OUT}/pred_step" \
    --ref-dir "${REF}" \
    --ids "${IDS}" \
    --system "$(basename "${OUT}")" \
    --out "${OUT}/paper_metrics.json" \
    --workers 16

echo "Report: ${OUT}/report${SFX}.md   (also report${SFX}.json, metrics${SFX}.csv, rollout.jsonl, pred_step/)"
