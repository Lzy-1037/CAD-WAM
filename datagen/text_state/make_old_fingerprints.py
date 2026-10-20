#!/usr/bin/env python
"""(local, run once) Fingerprints of the old cadseq_stepwise_text states for the v5 reproducibility check.

For every model that is both in the old text dataset and in v5 (models.tsv) this stores
  - the sha256 of its canonical action list (json with sorted keys), so verify only compares models
    whose program is the same in both datasets, and
  - per state string state_0..state_T: [exact, coarse] fingerprints (text_prompt.state_fingerprints).
verify_text_v5.py D then requires the v5 states of those models to match at 3 significant digits.

  python text_datagen/make_old_fingerprints.py   (any env with pandas)
"""
import hashlib
import json
import os

import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from text_prompt import state_fingerprints  # noqa: E402
ROOT = "$DATA"
OLD = f"{ROOT}/CAD_DSL_NEW/dataset/cadseq_stepwise_text"
MODELS = f"{ROOT}/SetVAE/datagen/models.tsv"


def canon_actions(action_strings):
    acts = [json.loads(s) for s in action_strings]
    return hashlib.sha256(json.dumps(acts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    v5 = {l.split("\t")[0] for l in open(MODELS).read().splitlines()[1:] if l}
    out = {"source": OLD, "note": "state_k = state string after k actions; actions = sha256 of sorted-key action list",
           "models": {}}
    for split in ("train", "test"):
        df = pd.read_parquet(f"{OLD}/{split}.parquet", columns=["model_id", "step_idx", "is_end", "action", "state"])
        for mid, g in df.groupby("model_id", sort=False):
            if mid not in v5:
                continue
            g = g.sort_values("step_idx")
            out["models"][mid] = {
                "actions": canon_actions(g.loc[~g.is_end, "action"]),
                "states": [list(state_fingerprints(s)) for s in g["state"]],
            }
    path = os.path.join(HERE, "old_text_fingerprints.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    n_states = sum(len(v["states"]) for v in out["models"].values())
    print(f"{len(out['models'])} shared models, {n_states} states -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
