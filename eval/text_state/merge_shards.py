#!/usr/bin/env python
"""Merge sharded rollout outputs into the single rollout.jsonl / rollout_summary.json that
paper_eval.py expects.  STEP files already share <run>/pred_step/ (names are model ids).

  python merge_shards.py --out runs/exp1 --num-shards 4
"""
import argparse
import glob
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    a = ap.parse_args()

    recs, missing = [], []
    for i in range(a.num_shards):
        p = os.path.join(a.out, f"rollout_shard{i}.jsonl")
        if not os.path.isfile(p):
            missing.append(p); continue
        recs += [json.loads(l) for l in open(p)]
    if missing:
        raise SystemExit(f"ERROR: shard output(s) missing, a rollout process probably died:\n  "
                         + "\n  ".join(missing))
    seen = set()
    recs = [r for r in recs if not (r["model_id"] in seen or seen.add(r["model_id"]))]
    recs.sort(key=lambda r: r["model_id"])
    with open(os.path.join(a.out, "rollout.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    sums = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.out, "rollout_summary_shard*.json")))]
    n_step = sum(1 for r in recs if r.get("has_step"))
    st = Counter(r["status"] for r in recs)
    merged = {"n_models": len(recs), "n_with_step": n_step,
              "invalidity_ratio": 1 - n_step / len(recs) if recs else float("nan"),
              "status_counts": dict(st),
              "oracle": sums[0]["oracle"] if sums else None,
              "engine": sums[0].get("engine") if sums else None,
              "model": sums[0]["model"] if sums else None,
              "num_shards": a.num_shards,
              "seconds_per_shard": [round(s["seconds"], 1) for s in sums],
              "seconds_wall_max": round(max((s["seconds"] for s in sums), default=0), 1)}
    json.dump(merged, open(os.path.join(a.out, "rollout_summary.json"), "w"), indent=2)
    print(json.dumps(merged, indent=2))
    print(f"merged {a.num_shards} shards -> {a.out}/rollout.jsonl ({len(recs)} models, {n_step} with STEP)")


if __name__ == "__main__":
    main()
