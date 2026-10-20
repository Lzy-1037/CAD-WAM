#!/usr/bin/env bash
#
#   (resume after an interruption: run the same command again -- --resume_latest)
#
# Why: the v5 SetVAE was trained on a random 90/5/5 model split, so its training clouds contained most
# DeepCAD-test shapes -- the same test shapes the CAD models are evaluated on.  v6 trains on the DeepCAD
# train split only (317,711 clouds of 157,945 models); validation (17,639) selects best.pt; the DeepCAD
# test split (16,078 clouds of 7,882 models) is never seen.
#
# Hyper-parameters, chosen from the v5 run's curves (v5: 160 epochs, val CD still falling 44.3 -> 42.7 over
# the last 30 epochs as the lr reached its floor, train 41.7 vs val 42.7 = no overfitting, stable at lr 1e-3):
#   global batch 256 = 32 per GPU x 8 GPUs (as v5)         -> 1,241 optimizer steps / epoch
#   lr 1e-3 (Adam), warm-up 1,400 steps, cosine to 1%      (as v5, stable)
#   300 epochs (~372k steps, v5: 160 / ~197k)              <- the change: a longer cosine schedule
#   KL weight 0 -> 1 over the first 25% = 75 epochs ; bf16 ; grad clip 1000 ; validation every 5 epochs
# Architecture unchanged (hidden 192, latent 1024 = 64 x 16: the Qwen geometry input expects 1024).
# Memory: ~7 GB per GPU at 32/GPU -- 80 GB cards are far from the limit; a larger batch would need a new lr
# and more steps for the same result.
# Expected time: ~100-120 s / epoch on 8 GPUs -> ~8.5-10 h.
set -euo pipefail
PROJ="${PROJ:-$REMOTE/code/SetVAE}"
export PROJ
export DATA="${DATA:-${PROJ}/datasets/deepcad_verified_pc}"            # DeepCAD-split packing (repack_clouds.py)
export TRAIN_SPLIT=train
export OUT="${OUT:-/datasets/anon/setvae/checkpoints/v6_deepcad_train}"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export BATCH="${BATCH:-32}"
export EPOCHS="${EPOCHS:-300}"

# the packing must be the DeepCAD official split, not the random one
python3 - "${DATA}" "${PROJ}/datagen/models.tsv" <<'PY' || { echo "ERROR: ${DATA} is not packed with the DeepCAD split (run datagen/repack_clouds.py)" >&2; exit 1; }
import hashlib, json, sys
meta = json.load(open(sys.argv[1] + "/meta.json"))
want = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert meta.get("split_file_sha256") == want, (meta.get("split_file"), meta.get("split_file_sha256"), want)
print(f"data  {sys.argv[1]}: DeepCAD official split (models.tsv sha256 {want[:16]}...), "
      f"train {meta['splits']['train']['clouds']} / validation {meta['splits']['validation']['clouds']} / "
      f"test {meta['splits']['test']['clouds']} clouds")
PY

exec bash "${PROJ}/scripts/deepcad_verified_2048.sh" "$@"
