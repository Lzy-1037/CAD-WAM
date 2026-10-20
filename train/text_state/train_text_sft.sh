#!/usr/bin/env bash
#   train_text_sft.sh  train_sft_text.py  dataset/<DATASET>/{train,validation|test}.parquet
#
#   bash train_text_sft.sh [extra trainer args...]
# Every setting below is an environment variable.  The v5 text-state baseline (compared with the
# hidden-state Qwen2.5 run: same backbone, data, split, 1 epoch, lr 2e-5, ~64 records per step):
#   MODEL_NAME=Qwen2.5-1.5B-Instruct DATASET=dataset/cadseq_stepwise_text_v5 EPOCHS=1 LR=2e-5 \
#   EVAL_EVERY=500 SAVE_EVERY=2000 MAX_LENGTH=<from verify E> GPUS=4,5,6,7 bash train_text_sft.sh
set -xeuo pipefail

cd $REMOTE/text_project

MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_PATH=/datasets/anon/${MODEL_NAME}
DATASET="${DATASET:-dataset/cadseq_stepwise_text}"
TRAIN_FILE=${DATASET}/train.parquet
VAL_FILE=${DATASET}/validation.parquet            # v5 has a validation split; the old dataset only test
[ -f "${VAL_FILE}" ] || VAL_FILE=${DATASET}/test.parquet

CKPT_ROOT=/datasets/anon/text_cadseq
OUT_DIR=${CKPT_ROOT}/${MODEL_NAME}/$(date +%Y%m%d_%H%M%S)

GPUS="${GPUS:-4,5,6,7}"
NPROC=$(awk -F, '{print NF}' <<< "${GPUS}")
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true    # rank 0 tokenises the whole split once (dataloader workers never tokenise)
export OMP_NUM_THREADS=8

EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
EVAL_EVERY="${EVAL_EVERY:-0}"          # extra validation every N optimizer steps (0 = epoch ends only)
SAVE_EVERY="${SAVE_EVERY:-0}"          # extra checkpoint <out>/step-N every N steps
SAVE_DTYPE=bf16
MAX_LENGTH="${MAX_LENGTH:-10240}"      # old text dataset max 9454 tokens (Qwen3); v5: take the max from verify_text_v5 E
                                       # (--truncation error: a longer record stops the run instead of being cut)

# ---- packed mode (default when flash_attn is installed in the env) ----------------------------
PACK_LENGTH="${PACK_LENGTH:-16384}"    # tokens per packed bin = one micro-batch per GPU; must be >= MAX_LENGTH
TOKENS_PER_STEP="${TOKENS_PER_STEP:-131072}"   # optimiser step in tokens; accum = TOKENS_PER_STEP / (PACK_LENGTH x GPUs)
                                       # 131072 on 4 GPUs = 2 bins x 4 GPUs ~= 60 records of ~2.1k tokens

# ---- padded fallback (no flash_attn): records per GPU / per step ------------------------------
MICRO_BATCH="${MICRO_BATCH:-2}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"

mkdir -p "${OUT_DIR}"
for f in "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${VAL_FILE}" train_sft_text.py; do
    [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
(( PACK_LENGTH >= MAX_LENGTH )) || { echo "ERROR: PACK_LENGTH ${PACK_LENGTH} < MAX_LENGTH ${MAX_LENGTH}" >&2; exit 1; }
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || \
    echo "WARNING: flash_attn not installed -> padded sdpa mode (2-3x slower)"
set +x
echo "model    ${MODEL_NAME}  (${MODEL_PATH})"
echo "data     ${TRAIN_FILE} | val ${VAL_FILE}"
echo "out      ${OUT_DIR}"
echo "gpus     ${GPUS}  (${NPROC} ranks)"
echo "schedule epochs ${EPOCHS}  lr ${LR}  eval_every ${EVAL_EVERY}  save_every ${SAVE_EVERY}  max_length ${MAX_LENGTH}"
echo "batch    packed: ${TOKENS_PER_STEP} tokens/step in bins of ${PACK_LENGTH} | padded: ${GLOBAL_BATCH} records (micro ${MICRO_BATCH})"
df -h "${CKPT_ROOT}" | tail -1
set -x

torchrun --standalone --nproc_per_node="${NPROC}" train_sft_text.py \
    --model "${MODEL_PATH}" \
    --train "${TRAIN_FILE}" \
    --val "${VAL_FILE}" \
    --out "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --eval-every "${EVAL_EVERY}" --save-every "${SAVE_EVERY}" \
    --pack auto --pack-length "${PACK_LENGTH}" --tokens-per-step "${TOKENS_PER_STEP}" \
    --micro-batch "${MICRO_BATCH}" --global-batch "${GLOBAL_BATCH}" \
    --max-length "${MAX_LENGTH}" --truncation error \
    --lr "${LR}" \
    --save-dtype "${SAVE_DTYPE}" \
    --save-every-epoch \
    --warmup-ratio 0.03 --clip 1.0 --seed 0 --log-every 10 \
    "$@" \
    2>&1 | tee -a "${OUT_DIR}/train.log"

echo "Done. Checkpoints under ${OUT_DIR} (step-N/, epoch-N/, final/ — load with AutoModelForCausalLM.from_pretrained)"
