# CAD-WAM

Stepwise text-to-CAD generation with a geometric world model: at every step the model predicts the next
construction action **and** the geometric state that action will produce, and is re-conditioned on the
solid its own actions have actually built.

The loop is closed through a real CAD kernel — each predicted action is executed, the resulting solid is
encoded, and that encoding is fed back as the state for the next step — so a model is never asked to
continue from a history it has not seen built.

## Repository layout

| directory | contents |
|---|---|
| `cadseq/` | the action representation and its executor: sketch and extrude actions, boolean operations, and the step function `(state, action) -> state` that defines every transition |
| `cadwam/` | the model: a Qwen2.5 / Qwen3 causal LM extended with a geometric-state embedding (latent → 16 tokens) and a next-state prediction head; the loss is action cross-entropy plus `lambda` times the state term |
| `setvae/` | the point-cloud encoder that turns a solid into the 1024-d state vector, with the per-step cloud generation used to train it |
| `datagen/` | dataset construction: `stepwise/` builds the per-step records (prompt, action, encoded state), `text_state/` builds the variant whose state is written as text, `test_set/` defines the evaluation set |
| `train/` | supervised fine-tuning entry points for the model and for the text-state ablation |
| `eval/` | the closed-loop rollout and `paper_eval.py`, the metric suite: invalidity ratio, Chamfer and Hausdorff distance, normal consistency, edge Chamfer distance, volumetric IoU and Euler-characteristic match |

## Installation

Two environments are required and cannot be merged: the CAD kernel bindings that come with CadQuery and
the deep-learning stack pin incompatible libraries. See `requirements.txt` for both lists. The rollout
uses them together — the model runs in the learning environment, the kernel in a geometry subprocess.

## Pipeline

```bash
# 1. dataset: per-step records with the state of the partial solid after every action
python datagen/stepwise/build_stepwise_v5.py --help

# 2. state encoder: per-step point clouds, then the encoder itself
python setvae/datagen/generate_clouds.py --help
bash  setvae/scripts/setvae_v6_deepcad_split.sh   # needs the SetVAE Chamfer/EMD CUDA extension

# 3. train the action + state model
python train/train_sft_cad_state.py --help

# 4. closed-loop generation on the test set, then scoring
bash eval/run_eval.sh <checkpoint> <run name>
python eval/paper_eval.py --pred-dir <run>/pred_step --ref-dir <reference STEP dir> \
    --ids datagen/test_set/ids_test7962.txt --system "<name>" --out <run>/paper_metrics.json
```

The same executor produces the states supervised during training and the states fed back during
generation, so training and evaluation cannot drift apart.

## Evaluation protocol

`eval/paper_eval.py` scores every system identically: each solid is fitted to `[-1,1]^3` and tessellated
at a fixed deflection, 8,192 points are sampled per shape for the distance metrics and 100,000 for the
edge metric, and each metric keeps the normalisation of the work that introduced it. Invalid predictions
are counted in the invalidity ratio and excluded from the geometric metrics, so the two must be read
together. `eval/paper_table.py` turns the resulting JSON files into a table.


## Licence and attribution

This project is released under the MIT License (`LICENSE`). It contains code derived from SetVAE,
CADFS and DeepCAD, all MIT-licensed; their notices and licence texts are kept in `THIRD_PARTY.md` and
`licenses/`.
