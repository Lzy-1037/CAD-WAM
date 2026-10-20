#!/usr/bin/env bash
#
#   bash train_cad_sft.sh <Qwen3-0.6B|Qwen3-1.7B|Qwen3-4B|Qwen3-8B|Qwen2.5-1.5B-Instruct> [extra trainer args...]
#
# Qwen2.5-1.5B-Instruct = the backbone Pointer-CAD fine-tunes (its released config uses Qwen2.5-*-Instruct);
# the trainer picks the model class (Qwen3 / Qwen2) from the checkpoint's config.json.
#
# Checkpoints go to  ${CKPT_ROOT}/<model>/<timestamp>/  so runs of different sizes never mix and a
# re-run of the same size never overwrites an earlier one, e.g.
#   /datasets/anon/new_cadseq/checkpoints/Qwen3-4B/20260908_143000/{epoch-N,final}/
#
# One script rather than four copies: the only per-size differences are the micro-batch (memory) and
# the two learning rates, kept in the table below.  GLOBAL_BATCH is 64 for every size, so the
# optimisation schedule (374 steps/epoch) is identical and the sizes stay comparable.
set -xeuo pipefail

MODEL_NAME="${1:?usage: train_cad_sft.sh <Qwen3-0.6B|Qwen3-1.7B|Qwen3-4B|Qwen3-8B|Qwen2.5-1.5B-Instruct> [extra args]}"
shift

# ---- per-size settings (validated before anything else, so a typo fails immediately) -----------
# MICRO_BATCH is chosen so MICRO_BATCH * 4 GPUs * accum = GLOBAL_BATCH (64) exactly.
#
# It does NOT scale up for smaller models.  Peak memory here is dominated by the LOGITS tensor,
# micro x seq x vocab(151936), plus the fp32 copy cross_entropy makes -- 6 bytes/element, and
# completely independent of hidden_size -- the same for a 0.6B as for an 8B.
# The TRAIN split reaches 4571 tokens (13 samples > 4096), so batches do hit MAX_LENGTH=4096, and
# during the CE backward three logits-sized tensors coexist (bf16 forward + fp32 upcast + fp32 grad)
# ~= 6.2 GB per unit of micro-batch:
#     micro 8 -> ~50 GB of logits alone; 18.5 GB of that is the grad, which is what OOMed the 4B
#     micro 4 -> ~25 GB, leaving room for states + activations on every size
# Peaks with 8-way FSDP on 79 GB cards at micro 4: 0.6B 29G  1.7B 33G  4B 41G  8B 54G.
# Do NOT raise MICRO_BATCH without also lowering MAX_LENGTH -- memory is set by the longest
# sequence in a batch, not the average (mean is only ~1000 tokens, so most steps use far less).
# LR/NEW_LR follow the usual "smaller model tolerates a larger step" rule; for a strict scaling
# table in the paper, override them so every size trains with identical optimisation:
#   bash train_cad_sft.sh Qwen3-4B --lr 1e-5 --new-lr 5e-5
case "${MODEL_NAME}" in
  Qwen3-0.6B) MICRO_BATCH=4;  LR=3e-5;   NEW_LR=1.5e-4 ;;
  Qwen3-1.7B) MICRO_BATCH=4;  LR=2e-5;   NEW_LR=1e-4   ;;
  Qwen3-4B)   MICRO_BATCH=4;  LR=1.5e-5; NEW_LR=7e-5   ;;
  Qwen3-8B)   MICRO_BATCH=4;  LR=1e-5;   NEW_LR=5e-5   ;;
  Qwen2.5-1.5B-Instruct) MICRO_BATCH=4; LR=2e-5; NEW_LR=1e-4 ;;   # same as Qwen3-1.7B (similar size)
  *) echo "ERROR: unknown model '${MODEL_NAME}'" >&2; exit 1 ;;
esac
# MICRO_BATCH_OVERRIDE=<n> overrides the table; GLOBAL_BATCH stays 64 (accumulation grows), so optimisation is
# unchanged.  Needed for 8B on 4 GPUs at MAX_LENGTH=5632: estimated peak ~83 GB at micro 4 (fp32
# params+grads+Adam sharded 4-way ~33 GB + logits of a 5.4k-token batch ~33 GB + activations), ~60 GB at 2.
MICRO_BATCH="${MICRO_BATCH_OVERRIDE:-${MICRO_BATCH}}"

cd $REMOTE/project

MODEL_PATH=/datasets/anon/${MODEL_NAME}
# v3 = the v2 records (per-step self-normalised frames) with latents re-encoded by the all-entities
# SetVAE (pc_latent_encoder DEFAULT_CKPT = setvae_allcad_all_best.pt).  Evaluate v3-trained models
# only with that encoder; models trained on v2 need HOLDOUT_CKPT instead.
# Other variants:  DATASET=dataset/cadseq_stepwise_v3cb bash train_cad_sft.sh Qwen3-1.7B
DATASET="${DATASET:-dataset/cadseq_stepwise_v3}"
TRAIN_FILE=${DATASET}/train.parquet
# v5+ has a real validation split (DeepCAD official); v2-v4 only had test.  The trainer scores
# --eval-samples (1024) records of it per epoch.
VAL_FILE=${DATASET}/validation.parquet
[ -f "${VAL_FILE}" ] || VAL_FILE=${DATASET}/test.parquet

CKPT_ROOT=/datasets/anon/new_cadseq/checkpoints
# The run folder says what makes it special: "_lambda<w>" whenever the state-loss weight is not 1
# (e.g. CAD_STATE_LOSS_WEIGHT=0 -> 20260918_153000_lambda0), plus an optional free tag RUN_TAG=<text>.
CAD_STATE_LOSS_WEIGHT="${CAD_STATE_LOSS_WEIGHT:-1.0}"
LAMBDA_TAG=$(awk -v w="${CAD_STATE_LOSS_WEIGHT}" 'BEGIN { if (w + 0 != 1) printf "_lambda%g", w }')
RUN_TAG="${RUN_TAG:-}"
OUT_DIR=${CKPT_ROOT}/${MODEL_NAME}/$(date +%Y%m%d_%H%M%S)${LAMBDA_TAG}${RUN_TAG:+_${RUN_TAG}}

# All 8 cards by default; GPUS=4,5,6,7 falls back to half the machine.  GLOBAL_BATCH is kept at 64
# whatever the GPU count, so the optimisation (375 steps/epoch) and the learning rates are unchanged
# -- more GPUs buy wall-clock, not a different schedule.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC=$(awk -F, '{print NF}' <<< "${GPUS}")

GLOBAL_BATCH=64
if (( GLOBAL_BATCH % (MICRO_BATCH * NPROC) != 0 )); then
    echo "ERROR: GLOBAL_BATCH ${GLOBAL_BATCH} is not divisible by MICRO_BATCH ${MICRO_BATCH} x ${NPROC} GPUs" >&2
    exit 1
fi
# v3/v3cb (24k records, 375 steps/epoch) used 3 epochs.  v5 has 475,656 train records (7,432
# steps/epoch at GLOBAL_BATCH 64): one epoch already sees ~6.6x the samples of those 3 epochs, so
# run v5 with EPOCHS=1 and watch the step-level validation (EVAL_EVERY) instead.
EPOCHS="${EPOCHS:-3}"
EVAL_EVERY="${EVAL_EVERY:-0}"     # extra validation every N optimizer steps (0 = epoch ends only)
SAVE_EVERY="${SAVE_EVERY:-0}"     # extra checkpoint <out>/step-N every N steps (0 = epoch ends + final)
MAX_LENGTH="${MAX_LENGTH:-4096}"  # v3 train max is 4571 tokens: the 13 samples above 4096 lose the end of their answer
# v5: max 5406 tokens (72 train records > 4096, 9 of them with the PROMPT alone > 4096, which would cut
# the geo tokens and crash) -> MAX_LENGTH=5632.  Only batches holding such a record get longer.
SAVE_DTYPE=bf16        # 0.6B ~1.2GB, 1.7B ~3.5GB, 4B ~8GB, 8B ~16.4GB per checkpoint

export CUDA_VISIBLE_DEVICES="${GPUS}"
export CAD_STATE_LOSS_WEIGHT="${CAD_STATE_LOSS_WEIGHT:-1.0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

mkdir -p "${OUT_DIR}"

# ---- pre-flight ---------------------------------------------------------------------------------
set +x
fail=0
need() { [ -e "$1" ] || { echo "MISSING  $1"; echo "         -> $2"; fail=1; }; }
echo "--- pre-flight -------------------------------------------------------------"
need "${MODEL_PATH}/config.json" "download it: python download_qwen3.py --models ${MODEL_NAME}"
need "${TRAIN_FILE}" "rsync ${DATASET}/ to ${PWD}/dataset/"
need "${VAL_FILE}"   "rsync ${DATASET}/ to ${PWD}/dataset/"
need cadwam/__init__.py "the cadwam package must sit in ${PWD}"
need train_sft_cad_state.py "the trainer must sit in ${PWD}"
python - "${MODEL_PATH}" <<'PY' || fail=1
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
print(f"backbone  hidden={c['hidden_size']} layers={c['num_hidden_layers']} "
      f"tie_word_embeddings={c.get('tie_word_embeddings')}  "
      f"-> geo/state heads resize to hidden={c['hidden_size']}")
PY
[ "${fail}" -eq 0 ] || { echo "pre-flight failed" >&2; exit 1; }
echo "model    ${MODEL_NAME}  (${MODEL_PATH})"
echo "data     ${DATASET}"
echo "out      ${OUT_DIR}"
echo "gpus     ${GPUS}  (${NPROC} ranks)"
echo "batch    micro ${MICRO_BATCH} x ${NPROC} GPUs x accum $((GLOBAL_BATCH / (MICRO_BATCH * NPROC))) = ${GLOBAL_BATCH}"
echo "lr       backbone ${LR}   new modules ${NEW_LR}   state_loss_weight ${CAD_STATE_LOSS_WEIGHT}"
echo "schedule epochs ${EPOCHS}   eval_every ${EVAL_EVERY}   save_every ${SAVE_EVERY}   max_length ${MAX_LENGTH}"
df -h "${CKPT_ROOT}" | tail -1
echo "----------------------------------------------------------------------------"
set -x

torchrun --standalone --nproc_per_node="${NPROC}" train_sft_cad_state.py \
    --model "${MODEL_PATH}" \
    --train "${TRAIN_FILE}" \
    --val "${VAL_FILE}" \
    --out "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --eval-every "${EVAL_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --micro-batch "${MICRO_BATCH}" \
    --global-batch "${GLOBAL_BATCH}" \
    --max-length "${MAX_LENGTH}" \
    --lr "${LR}" \
    --new-lr "${NEW_LR}" \
    --state-loss-weight "${CAD_STATE_LOSS_WEIGHT}" \
    --save-dtype "${SAVE_DTYPE}" \
    --save-every-epoch \
    --warmup-ratio 0.03 --clip 1.0 --seed 0 --log-every 10 \
    "$@" 2>&1 | tee -a "${OUT_DIR}/train.log"

echo "Done. ${OUT_DIR}  (epoch-N/, final/ — load with Qwen3ForCADActionState.from_pretrained)"
