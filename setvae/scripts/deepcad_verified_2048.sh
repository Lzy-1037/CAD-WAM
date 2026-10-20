#!/usr/bin/env bash
# SetVAE on the step point clouds of the 174,582 geometry-verified, Recap-annotated DeepCAD models,
#
#
# Resume: run the same command again (--resume_latest).
#
# Data: datasets/deepcad_verified_pc = 351,428 clouds (train 317,711 / validation 17,639 / test 16,078),
# 4096 points each, in the frame the closed-loop rollout feeds the encoder.
#
# TRAIN_SPLIT
#   train+validation+test  (default) every one of the 174,582 models, as requested.  The encoder then
#                          has seen the test geometry, and validation curves measure fit, not held-out.
#   train                  only the DeepCAD train split; validation is genuinely held out and the
#                          paper can state that no test geometry was used in training.
#
# Architecture is identical to the earlier SetVAE runs (latent 1024 = 64 x 16).  Batch 64 per GPU
# (global 256): 1,372 optimizer steps / epoch on all splits (1,241 on train only); 160 epochs is
# ~220k (~199k) steps, the same order as the previous 212k-step runs.  KL warm-up keeps its 25% share.
set -euo pipefail
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
PROJ="${PROJ:-$REMOTE/code/SetVAE}"
DATA="${DATA:-${PROJ}/datasets/deepcad_verified_pc}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train+validation+test}"
OUT="${OUT:-/datasets/anon/setvae/checkpoints/deepcad_verified_2048_${TRAIN_SPLIT//+/_}}"
GPUS="${GPUS:-4,5,6,7}"
NPROC=$(awk -F, '{print NF}' <<< "${GPUS}")
BATCH="${BATCH:-64}"
EPOCHS="${EPOCHS:-160}"
KL_WARMUP=$(( EPOCHS / 4 ))

source "${CONDA_SH}"
set +u; conda activate setvae; set -u      # conda activation hooks reference unset variables
cd "${PROJ}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export NCCL_NVLS_ENABLE=0      # NVLink SHARP fails on this NVSwitch setup (CUDA error 401); plain NVLink P2P is used instead

# ---- pre-flight: report every problem at once ------------------------------------------------------
fail=0
for part in ${TRAIN_SPLIT//+/ } validation; do
    for f in "${DATA}/${part}.npy" "${DATA}/${part}_ids.txt"; do
        [ -f "$f" ] || { echo "MISSING $f -> regenerate it: see datagen/README_H800.md"; fail=1; }
    done
done
python -c "from metrics.StructuralLosses.nn_distance import nn_distance" 2>/dev/null \
    || { echo "MISSING compiled StructuralLosses -> bash scripts/setup_env.sh"; fail=1; }
[ "${fail}" -eq 0 ] || { echo "pre-flight failed" >&2; exit 1; }
echo "data  ${DATA}  (train on: ${TRAIN_SPLIT})"
echo "out   ${OUT}"
echo "gpus  ${GPUS} (${NPROC} ranks) | batch ${BATCH}/GPU = $(( BATCH * NPROC )) global | epochs ${EPOCHS}, KL warm-up ${KL_WARMUP}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "${GPUS}"
mkdir -p "${OUT}"

torchrun --standalone --nproc_per_node="${NPROC}" train_deepcad.py \
  --data_dir "${DATA}" --train_split "${TRAIN_SPLIT}" --out_dir "${OUT}" \
  --dataset_type shapenet15k --log_name deepcad_verified_2048 \
  --input_dim 3 --n_points 2048 \
  --init_dim 32 --n_mixtures 4 --fixed_gmm --train_gmm \
  --z_dim 16 --z_scales 1 1 2 4 8 16 32 \
  --hidden_dim 192 --num_heads 8 --isab_inds 16 --slot_att --ln \
  --standardize global \
  --batch_size "${BATCH}" --accum 1 --num_workers 8 \
  --optimizer adam --lr 1e-3 --beta1 0.9 --beta2 0.999 --weight_decay 0 \
  --warmup_steps 1400 --min_lr_ratio 0.01 --epochs "${EPOCHS}" \
  --matcher chamfer --beta 1.0 --kl_warmup_epochs "${KL_WARMUP}" \
  --max_grad_norm 1000 --bf16 --seed 0 \
  --val_freq 5 --preview_freq 5 --ckpt_freq 10 --n_gen_eval 256 --log_freq 500 \
  --resume_latest \
  "$@" 2>&1 | tee -a "${OUT}/train.log"
