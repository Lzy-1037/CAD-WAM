#!/usr/bin/env python
"""Merge a small extra rollout into an existing one, so a finished 7,882-model run becomes a 7,962-model run.

  python cad_eval/topup_run.py --base cad_eval/runs/v6_qwen25 \
                               --extra cad_eval/runs/v6_qwen25_extra80 \
                               --out  cad_eval/runs/v6_qwen25_7962
  PHASE=eval OUT=$PWD/cad_eval/runs/v6_qwen25_7962 REF=$PWD/eval_refs/deepcad_v6_test7962 \
      bash cad_eval/run_eval.sh <ckpt> v6_qwen25_7962

Both runs must come from the same checkpoint, encoder and decoding settings -- the script checks the
model path and encoder recorded in their rollout summaries and refuses if they differ.  Nothing is
modified in place: the merged run is a new directory of hard links (falling back to copies).
"""
import argparse
import glob
import json
import os
import shutil
from collections import Counter


def load_summary(run):
    for p in (os.path.join(run, 'rollout_summary.json'), *sorted(glob.glob(os.path.join(run, 'rollout_summary_shard*.json')))):
        if os.path.isfile(p):
            return json.load(open(p))
    raise SystemExit(f'{run}: no rollout_summary*.json')


def records(run):
    p = os.path.join(run, 'rollout.jsonl')
    if os.path.isfile(p):
        return [json.loads(l) for l in open(p)]
    shards = sorted(glob.glob(os.path.join(run, 'rollout_shard*.jsonl')))
    if not shards:
        raise SystemExit(f'{run}: neither rollout.jsonl nor rollout_shard*.jsonl')
    return [json.loads(l) for s in shards for l in open(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='the finished 7,882-model run')
    ap.add_argument('--extra', required=True, help='the run over the 80 added models')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    sb, se = load_summary(a.base), load_summary(a.extra)
    for k in ('model', 'encoder_ckpt', 'geo_frame'):
        assert sb.get(k) == se.get(k), f'{k} differs: {sb.get(k)} vs {se.get(k)} -- these runs cannot be merged'
    rb, re_ = records(a.base), records(a.extra)
    ids_b = {r['model_id'] for r in rb}
    ids_e = {r['model_id'] for r in re_}
    assert not (ids_b & ids_e), f'{len(ids_b & ids_e)} models appear in both runs, e.g. {sorted(ids_b & ids_e)[:3]}'
    recs = sorted(rb + re_, key=lambda r: r['model_id'])

    os.makedirs(os.path.join(a.out, 'pred_step'), exist_ok=True)
    n_step = 0
    for run in (a.base, a.extra):
        src = os.path.join(run, 'pred_step')
        for f in sorted(os.listdir(src)) if os.path.isdir(src) else []:
            if not f.endswith('.step'):
                continue
            dst = os.path.join(a.out, 'pred_step', f)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(src, f), dst)
                except OSError:
                    shutil.copy(os.path.join(src, f), dst)
            n_step += 1
    with open(os.path.join(a.out, 'rollout.jsonl'), 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')

    st = Counter(r['status'] for r in recs)
    with_step = sum(1 for r in recs if r.get('has_step'))
    merged = {**{k: sb.get(k) for k in ('model', 'encoder_ckpt', 'encoder_epoch', 'oracle', 'geo_frame',
                                        'prompt_format')},
              'n_models': len(recs), 'n_with_step': with_step,
              'invalidity_ratio': 1 - with_step / len(recs) if recs else float('nan'),
              'status_counts': dict(st),
              'merged_from': {'base': os.path.abspath(a.base), 'base_models': len(ids_b),
                              'extra': os.path.abspath(a.extra), 'extra_models': len(ids_e)}}
    json.dump(merged, open(os.path.join(a.out, 'rollout_summary.json'), 'w'), indent=2)
    print(json.dumps(merged, indent=2))
    print(f'merged {len(ids_b)} + {len(ids_e)} = {len(recs)} models, {n_step} STEP files -> {a.out}')


if __name__ == '__main__':
    main()
