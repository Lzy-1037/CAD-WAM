#!/usr/bin/env python
"""Classify every state that differs between the old cadseq_stepwise_text and cadseq_stepwise_text_v5.

  noise      same structure, same face order, every number within float-noise tolerance
  reordered  same faces within tolerance, but face_list order differs (equal-area faces swapped)
  REAL       anything else (different keys / counts / types / numbers beyond tolerance) -> printed in full

Tolerance: 5 significant digits are stored, so a one-unit flip in the last digit is <= 1e-4 relative;
|a - b| <= 2e-4 * max(|a|, |b|) + 2e-6 * bbox diagonal (text_state writes |x| < 1e-6 * diagonal as 0).

  conda activate verl && python text_datagen/diff_old_states.py
"""
import argparse
import json
import math
import os

import pandas as pd

CODE = "$REMOTE/code"
REL, ABS_DIAG = 2e-4, 2e-6


def close(a, b, atol):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= REL * max(abs(a), abs(b)) + atol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], atol) for k in a)
    return a == b


def diag_of(s):
    bb = s.get("bbox")
    return math.dist(bb["min"], bb["max"]) if bb else 0.0


def same_faces_any_order(fa, fb, atol):
    rest = list(fb)
    for f in fa:
        j = next((j for j, g in enumerate(rest) if close(f, g, atol)), None)
        if j is None:
            return False
        rest.pop(j)
    return not rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=f"{CODE}/text_cadseq/dataset/cadseq_stepwise_text")
    ap.add_argument("--new", default=f"{CODE}/text_cadseq/dataset/cadseq_stepwise_text_v5")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    old = pd.concat([pd.read_parquet(os.path.join(a.old, f"{s}.parquet"), columns=["model_id", "step_idx", "state"])
                     for s in ("train", "test")])
    ids = set(old.model_id)
    new = pd.concat([pd.read_parquet(os.path.join(a.new, f"{s}.parquet"), columns=["model_id", "step_idx", "state"],
                                     filters=[("model_id", "in", sorted(ids))])
                     for s in ("train", "validation", "test")])
    m = old.merge(new, on=["model_id", "step_idx"], suffixes=("_old", "_new"))
    d = m[m.state_old != m.state_new]
    counts, shown = {"noise": 0, "reordered": 0, "REAL": 0}, 0
    models = {"noise": set(), "reordered": set(), "REAL": set()}
    for _, r in d.iterrows():
        so, sn = json.loads(r.state_old), json.loads(r.state_new)
        atol = ABS_DIAG * max(diag_of(so), diag_of(sn))
        if close(so, sn, atol):
            kind = "noise"
        elif (so.keys() == sn.keys() and close({k: v for k, v in so.items() if k != "face_list"},
                                              {k: v for k, v in sn.items() if k != "face_list"}, atol)
              and same_faces_any_order(so.get("face_list", []), sn.get("face_list", []), atol)):
            kind = "reordered"
        else:
            kind = "REAL"
        counts[kind] += 1
        models[kind].add(r.model_id)
        if kind != "noise" and shown < a.show:
            shown += 1
            print(f"--- {kind}: {r.model_id} step {r.step_idx}")
            for k in so:
                if k != "face_list" and so[k] != sn.get(k):
                    print(f"    {k}: old {so[k]}  new {sn.get(k)}")
            fo, fn = so.get("face_list", []), sn.get("face_list", [])
            for i, (x, y) in enumerate(zip(fo, fn)):
                if x != y:
                    print(f"    face {i} old {x}\n    face {i} new {y}")
            if len(fo) != len(fn):
                print(f"    face_list length old {len(fo)} new {len(fn)}")
    print(f"compared {len(m)} states of {m.model_id.nunique()} models | identical {len(m) - len(d)} | differing {len(d)}")
    for k in counts:
        print(f"  {k:9s} {counts[k]:4d} states in {len(models[k])} models")
    print("RESULT: OK (all differences are float noise)" if counts["REAL"] == 0 else "RESULT: REAL DIFFERENCES")


if __name__ == "__main__":
    main()
