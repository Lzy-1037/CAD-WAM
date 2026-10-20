# text_cad_eval — closed-loop test-set evaluation of the text-state Qwen3-8B (plain SFT)

Input: a fine-tuned checkpoint directory (`epoch-N/` or `final/` from `train_sft_text.py`).
Output: predicted STEP files + the same CADFS geometric report as `cad_eval` (CD / NC / HD, IR)
against the reference STEPs, plus a next-state (world-model) diagnostic section.

```
test.parquet (890 models, cadseq_stepwise_text)
    │  for each model, closed loop:
    │      state_0 = {"solids":0}
    │      answer_t = model.generate(instruction + description + "Current state:\n" + state_t)
    │                 -> {"action": A_t, "state": S_pred}
    │      solid_t  = cadseq.Executor.step(solid_{t-1}, A_t)         ← real kernel, worker process
    │      state_{t+1} = text_state.encode_state(solid_t)             ← REAL state fed back
    │      until A_t == {"kind":"end"} / rejected action / unparsable answer / max steps
    ▼
pred_step/<model_id>.step ──compare──▶ CD, NC, HD + invalidity ratio ──▶ report.md / report.json / metrics.csv
```

The model's predicted `state` is compared with the executed one (exact JSON match, solids / faces
count match, bbox and volume error) and reported, but it is **never fed back** — the loop is closed
through the real executor, exactly the protocol the dataset was built with.  Checkpoints trained
with `--action-only-loss` answer `{"action": ...}` only; that is accepted (no state diagnostics).

## Files

| file | env | role |
|---|---|---|
| `rollout.py` | GPU env (torch + transformers; `--engine vllm` needs vllm) | phase 1: closed-loop generation → `pred_step/*.step`, `rollout.jsonl` |
| `geometry_worker.py` | CadQuery env | stateless JSON-line worker: executes actions, encodes the text state, exports STEP |
| `metrics.py` | CadQuery env | CADFS official CD / NC / HD (identical copy of `cad_eval/metrics.py`) |
| `merge_shards.py`, `progress.py` | — | helpers (data-parallel shards, progress lines) |
| `run_eval.sh` | — | pre-flight checks + both phases |

## Server layout (`$REMOTE/text_project`)

```
text_cad_eval/                       this directory
cadseq/                              the cadseq package (unchanged)
text_state.py                        = stepwise_dataset/text_state.py (state encoder used in training)
dataset/cadseq_stepwise_text/test.parquet
```
Reference STEPs: `eval_refs/official_step` (890 files) — the latent project already has them at
`$REMOTE/project/DeepCAD/official_step`, which is the default `REF`; override with
`REF=/path bash run_eval.sh ...`.  `eval_refs/cadseq_step` gives the second reference (oracle CD = 0).

## Usage

```bash
conda activate verl                         # GPU env; GEOM_PYTHON must have cadquery + trimesh + scipy
GEOM_PYTHON=/path/to/cadquery-env/bin/python \
bash text_cad_eval/run_eval.sh /datasets/anon/text_cadseq/text_sft_<ts>/final exp1

bash text_cad_eval/run_eval.sh <ckpt> smoke --limit 40          # quick subset
bash text_cad_eval/run_eval.sh gt oracle --oracle               # ground-truth actions: pipeline test + upper bound
bash text_cad_eval/run_eval.sh <ckpt> fast --engine vllm        # vllm generation (answers are ~800 tokens; much faster)
PHASE=eval bash text_cad_eval/run_eval.sh <ckpt> exp1           # re-score an existing run
```
Env knobs: `GPUS=4,5,6,7` (one rollout shard per GPU), `GEOM_WORKERS=6` (per shard), `PHASE=all|rollout|eval`.
Rollout flags: `--max-new-tokens 4096` (action + state answers reach ~3900 tokens), `--max-steps 12`,
`--geom-timeout 60`.

## Metrics

Identical to `cad_eval` (CADFS official protocol: 10k surface samples, independent per-cloud
normalisation): **CD** (×1000, lower better), **NC** (higher better), **HD** (lower better), **IR**
(fraction of test models with no usable STEP), CD success rates over scored and over all models,
exact action-count match, rollout status histogram (`ok_end`, `parse_error`, `action_rejected`,
`action_timeout`, `max_steps`, `empty_end`, `export_failed`).

Additional section — **next-state prediction** (over executed steps whose answer carried a state):
exact JSON match rate, solids / faces count match rates, median bbox error (max corner deviation /
bbox diagonal) and median relative volume error.

Oracle rows to quote (measured with `cad_eval`, all 890 test models): vs `official_step`
CD median 0.38 / NC 0.964 / HD 0.044 — the representation floor a perfect model would reach; vs
`cadseq_step` all zero.  A 12-model local oracle run of this pipeline reproduced that range
(CD median 0.52, IR 0%).
