# cad_eval — closed-loop test-set evaluation of the trained Qwen3ForCADActionState

**Which file produces the paper's numbers:** `paper_eval.py` (one protocol for every system: IR, CD, HD,
NC, ECD, IoU, Euler match) and `paper_table.py`, which formats its JSON output into the tables.
`metrics.py` holds only the two functions `paper_eval.py` inherits from CADFS.

Input: a trained checkpoint directory. Output: predicted STEP files + a geometric-similarity report
(CD / NC / HD, CADFS official protocol) against the reference STEPs.

```
test.parquet (890 models for v2-v4, 7,882 for v5)
    │  for each model, closed loop:
    │      z_0 = zeros(1024)
    │      action_t = model.generate(prompt, geo_latent=z_t)      ← Action Head
    │      solid_t  = cadseq.Executor.step(solid_{t-1}, action_t)  ← real kernel, in a worker process
    │      z_{t+1}  = SetVAE_encoder(4096 pts of solid_t)          ← pc_latent_encoder
    │      until {"kind":"end"} / rejected action / max steps
    ▼
pred_step/<model_id>.step ──compare──▶ CD, NC, HD + invalidity ratio ──▶ report.md / report.json / metrics.csv
                                ▲
                    reference STEP (DeepCAD official_step or cadseq_step)
```

The model's own `pred_next_state` is recorded for diagnostics but is **not** fed back — the loop is
closed through the real executor and encoder, exactly the protocol the dataset was built with.

## Files

| file | env | role |
|---|---|---|
| `rollout.py` | GPU env (torch, transformers) | phase 1: closed-loop generation → `pred_step/*.step`, `rollout.jsonl` |
| `geometry_worker.py` | CadQuery env | stateless JSON-line worker: executes actions, samples clouds, exports STEP |
| `metrics.py` | CadQuery env | CADFS official CD / NC / HD (+ optional ECD, NC_unnorm) |
| `run_eval.sh` | — | runs both phases |

`rollout.py` runs the LLM in one env and spawns `geometry_worker.py` under `--geom-python`, so
CadQuery and torch never need to coexist. It also isolates OCC crashes/hangs on malformed
LLM-generated actions (a dead worker is restarted, the model is just marked failed).

## Usage

```bash
# phase 1 — GPU env
python rollout.py --model /datasets/anon/new_cadseq/checkpoints/cad_sft_*/final \
    --test dataset/cadseq_stepwise/test.parquet --out runs/exp1 \
    --encoder $REMOTE/code/pc_latent_encoder \
    --cadseq-root $REMOTE/project \
    --geom-python /path/to/cadquery-env/bin/python --geom-workers 8 --batch 16

# phase 2 — CadQuery env
```

Useful flags: `--limit N` (quick subset), `--oracle` (replace the LLM with ground-truth actions:
pipeline test + upper-bound baseline), `--geo-frame zeros` (ablation: never feed geometry back).

## Metrics — CADFS official protocol

Sampling: STEP → `BRepMesh_IncrementalMesh(shape, 0.1)` → STL → `trimesh.sample_surface(10000)`,
normals from the sampled faces. Each cloud is then normalised **independently** (centre on its mean,
divide by max |coordinate|), so CD/NC/HD are invariant to translation and uniform scale.

| metric | definition | direction |
|---|---|---|
| **CD** | bidirectional mean **squared** NN distance × 1000 | lower better |
| **NC** | mean of the two directional mean normal dot products | higher better (→1) |
| **HD** | symmetric Hausdorff, raw (unsquared) distances | lower better |
| **IR** | invalidity ratio = fraction of test models with no usable STEP | lower better |

Two metrics from the CADFS module are computed but **off by default**, because they are degenerate
here (verified, not assumed):

* **ECD** — the official edge radius is 0.004, but the NN spacing at 10k points is ~0.010, so only
  **6 of 10000** points get flagged as edges (79 at 50k). The score is noise below ~200k points.
  Enable with `--ecd --num-points 200000` if you need it.
* **NC_unnorm** — matches neighbours on the *raw* clouds, so it only means something when both STEPs
  share a coordinate frame. Predictions are in cadseq units, DeepCAD references are pre-normalised
  (a pure uniform scale apart), which makes the raw-frame matching meaningless. Enable with
  `--nc-unnorm` only if you deliberately align frames first.

## Choosing the reference

**v5 (cadseq_stepwise_v5, 7,882 DeepCAD-official test models)** — reference
`eval_refs/deepcad_v5_test/` (DeepCAD official STEP; the corrected STEP for the 119 test models hit
by DeepCAD's plane bug). Every v5 model's cadseq program was verified to reproduce that geometry, so
a perfect model differs from the reference only by sampling noise. Second reference = the STEP files
of the oracle run itself (`<oracle run>/pred_step`, CD exactly 0 for a perfect model), scored with
`PHASE=eval REPORT_TAG=cadseq`, so it never overwrites the main report.

Measured oracle, 300 v5 test models, IR 0.00% (default `--mesh-scale unit`):

| reference | CD mean | CD median | CD p90 | CD <= 0.5 | NC median |
|---|---|---|---|---|---|
| `eval_refs/deepcad_v5_test` | 0.467 | 0.401 | 0.937 | 62.0% | 0.972 |
| oracle `pred_step` (cadseq) | 0.000 | 0.000 | 0.000 | 100% | 1.000 |

The ~0.4 floor against the official reference is **sampling noise**, not a representation gap: the
reference scored against itself with another sampling seed gives CD median 0.48. Differences below
that level between two models mean nothing.

**Mesh scale (`--mesh-scale`, `MESH_SCALE`).** CADFS meshes with an ABSOLUTE deflection of 0.1, which
assumes DeepCAD's normalised STEPs (max extent ~1). cadseq predictions are in raw DeepCAD units (median
extent 0.08), so under the verbatim protocol their curved faces became coarse polygons and the oracle
scored CD mean 1.72 (p90 1.94) against geometry that is exactly a scaled copy of the reference. The
default `unit` scales every shape to max extent 1 before meshing; CD/NC/HD are scale-invariant, so
only the mesh resolution changes. `native` reproduces the verbatim protocol (and every run scored
before 2026-09-16). Re-score an old run without re-generating: `PHASE=eval bash run_eval.sh <ckpt> <run>`.

**v2-v4 (890 test models)** — `DeepCAD/official_step` (headline) and `DeepCAD/cadseq_step`. The oracle
row measured with the verbatim protocol (CD median 0.3797 / NC 0.9638 vs official_step) mixes the
mesh-scale artefact and sampling noise; it is not a cadseq-vs-DeepCAD gap.

## Interpreting the report

`report.md` gives mean/median/p25/p75/p90 per metric over the *scored* models, CD success rates both
over scored models and over all 890 (so failures are never hidden), the invalidity ratio, the rollout
status histogram (`ok_end`, `parse_error`, `action_rejected`, `max_steps`, `empty_end`,
`export_failed`), and the fraction of models where the predicted action count matched ground truth.
`metrics.csv` has one row per model for significance testing or per-model inspection.

## Note on the geometric-state frame

During training, every intermediate cloud was normalised in the **final** model's frame (that frame is
knowable offline). At inference the final model does not exist yet, so `--geo-frame current` normalises
each partial solid in **its own** bbox frame. The two coincide at the last step of every model (partial
= final) and for all single-action models, but differ at intermediate steps of multi-action models.
`--geo-frame zeros` (never feed geometry back) quantifies how much the geometric channel contributes
at all; comparing the two isolates the cost of this frame mismatch.
