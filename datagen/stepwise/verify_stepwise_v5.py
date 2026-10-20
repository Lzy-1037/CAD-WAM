#!/usr/bin/env python
"""Verify cadseq_stepwise_v5 against its inputs, independently of the builder's bookkeeping.

A. Splits    each parquet holds exactly the models.tsv models of that split (disjoint, none missing),
             every model has records step 0..n with is_end only on the last; counts match the stats.
B. Records   FULL coverage: every answer == programs.jsonl action (then end), every description ==
             descriptions.jsonl, every question == cad_prompt.build_prompt(count, bbox from solid_truth).
C. States    geo_idx / target_idx chains: step 0 -> row 0 (zeros), each (model, step) owns one distinct
             row, end record targets its own last state; latents finite, only row 0 all-zero.
D. Encoder   stats encoder_md5 == --ckpt; N random states re-encoded from their clouds with
             SetVAEEncoder.encode(single cloud) -- the exact rollout call -- equal the stored latents.
Also prints the state baselines for SFT val state_mse (zero / copy-current / train-mean predictors),
and whether the eval guard (pc_latent_encoder DEFAULT_CKPT md5) and new_cadseq/cad_eval/cad_prompt.py
already match this dataset (informational, not part of the verdict).

  conda activate verl && CUDA_VISIBLE_DEVICES=4 python qwen_datagen/verify_stepwise_v5.py
Ends with RESULT: OK (exit 0) or RESULT: MISMATCH (exit 1).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_stepwise_v5 import (CODE, LATENT_DIM, SPLITS, cloud_parts, load_descriptions, load_models,  # noqa: E402
                               load_programs, md5, sha256, state_layout)
from cad_prompt import build_prompt  # noqa: E402

LATENT_TOL = 1e-3      # GPU batch-size numerics; a wrong cloud or subset differs by ~0.1-1 per dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=f'{CODE}/new_cadseq/dataset/cadseq_stepwise_v5')
    ap.add_argument('--datagen', default=f'{CODE}/SetVAE/datagen')
    ap.add_argument('--clouds', default=f'{CODE}/SetVAE/datasets/deepcad_verified_pc_random')
    ap.add_argument('--descriptions', default=os.path.join(HERE, 'descriptions.jsonl'))
    ap.add_argument('--encoder', default=f'{CODE}/pc_latent_encoder')
    ap.add_argument('--ckpt', default=f'{CODE}/pc_latent_encoder/checkpoints/setvae_verified_random_best.pt')
    ap.add_argument('--cad-eval', default=f'{CODE}/new_cadseq/cad_eval')
    ap.add_argument('--n-reencode', type=int, default=2000)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    import pandas as pd

    bad = []

    def check(ok, msg):
        print(('  ok   ' if ok else '  FAIL ') + msg, flush=True)
        if not ok:
            bad.append(msg)

    models = load_models(os.path.join(a.datagen, 'models.tsv'))
    acts = load_programs(os.path.join(a.datagen, 'programs.jsonl'), {m[0] for m in models})
    desc = load_descriptions(a.descriptions)
    with np.load(os.path.join(a.datagen, 'solid_truth.npz')) as z:     # NpzFile re-reads on every [key]
        truth = {k: z[k] for k in z.files}
    index, truth_pos, n_rows = state_layout(models)
    split_of = {m: s for m, s, _ in models}
    nsteps = {m: n for m, _, n in models}
    stats = {s: json.load(open(os.path.join(a.data, f'{s}_stats.json'))) for s in SPLITS}
    pf = stats['test']['prompt_format']
    lat = np.load(os.path.join(a.data, 'latents.npy'), mmap_mode='r')

    print('A. splits')
    frames = {}
    for s in SPLITS:
        df = pd.read_parquet(os.path.join(a.data, f'{s}.parquet'))
        frames[s] = df
        want = sorted(m for m, sp, _ in models if sp == s)
        got = sorted(df.model_id.unique())
        check(got == want, f'{s}: {len(got)} models == models.tsv {s} ({len(want)})')
        check((df.deepcad_split == s).all(), f'{s}: deepcad_split column')
        g = df.groupby('model_id')
        per = g.step_idx.agg(['count', 'min', 'max'])
        exp_n = per.index.map(nsteps)
        check(bool(((per['count'] == exp_n + 1) & (per['min'] == 0) & (per['max'] == exp_n)).all()),
              f'{s}: every model has records 0..n_steps')
        check(bool((df.is_end == (df.step_idx == df.n_steps)).all() and (df.n_steps == df.model_id.map(nsteps)).all()),
              f'{s}: is_end / n_steps')
        st = stats[s]['counts'][s]
        check(st['records'] == len(df) and st['models'] == len(got), f'{s}: {len(df)} records match {s}_stats.json')
    all_ids = [set(frames[s].model_id) for s in SPLITS]
    check(not (all_ids[0] & all_ids[1] or all_ids[0] & all_ids[2] or all_ids[1] & all_ids[2]), 'splits disjoint')
    check(sum(len(frames[s]) for s in SPLITS) == len(models) + len(index),
          f'total records {sum(len(frames[s]) for s in SPLITS)} == models + states ({len(models)} + {len(index)})')

    print('B. records (full coverage)')
    for s in SPLITS:
        df = frames[s]
        n_ans = n_desc = n_q = 0
        for mid, t, n, d, q, ans in zip(df.model_id, df.step_idx, df.n_steps, df.description, df.question, df.answer):
            n_ans += ans != (acts[mid][t] if t < n else '{"kind":"end"}')
            n_desc += d != desc[mid]
            bb = None if t == 0 else truth[f'{s}_bbox'][truth_pos[f'{mid}_step{t:02d}'][1]].tolist()
            n_q += q != build_prompt(desc[mid], pf, n_done=t, bbox=bb)
        check(n_ans == 0, f'{s}: answers == programs.jsonl ({n_ans} differ)')
        check(n_desc == 0, f'{s}: descriptions == descriptions.jsonl ({n_desc} differ)')
        check(n_q == 0, f'{s}: questions == build_prompt({pf}) ({n_q} differ)')

    print('C. states')
    check(lat.shape == (n_rows, LATENT_DIM), f'latents.npy {lat.shape} == ({n_rows}, {LATENT_DIM})')
    geo_rows = []
    for s in SPLITS:
        df = frames[s]
        exp_geo = np.array([0 if t == 0 else index[f'{m}_step{t:02d}'] for m, t in zip(df.model_id, df.step_idx)])
        exp_tgt = np.array([index[f'{m}_step{min(t + 1, n):02d}'] for m, t, n in zip(df.model_id, df.step_idx, df.n_steps)])
        check(bool((df.geo_idx.values == exp_geo).all() and (df.target_idx.values == exp_tgt).all()),
              f'{s}: geo_idx / target_idx chains')
        geo_rows.append(df.geo_idx.values[df.step_idx.values > 0])
    geo_rows = np.concatenate(geo_rows)
    check(len(geo_rows) == len(np.unique(geo_rows)) == n_rows - 1, 'each state row used by exactly one (model, step)')
    zero_rows, finite = [], True
    for i in range(0, n_rows, 65536):
        blk = np.asarray(lat[i: i + 65536])
        finite &= bool(np.isfinite(blk).all())
        zero_rows += list(np.flatnonzero(~blk.any(1)) + i)
    check(finite, 'latents finite')
    check(zero_rows == [0], f'only row 0 is all-zero ({len(zero_rows)} zero rows)')

    print('D. encoder')
    got_md5 = md5(a.ckpt)
    check(all(stats[s]['encoder_md5'] == got_md5 for s in SPLITS), f'stats encoder_md5 == md5({os.path.basename(a.ckpt)}) {got_md5}')
    sys.path.insert(0, a.encoder)
    from encoder import DEFAULT_CKPT, SetVAEEncoder
    enc = SetVAEEncoder(a.ckpt, device=a.device)
    loc = {}
    for npy, ids in cloud_parts(a.clouds):
        loc.update({k: (npy, i) for i, k in enumerate(ids)})
    keys = sorted(index)
    pick = np.random.default_rng(12345).choice(len(keys), min(a.n_reencode, len(keys)), replace=False)
    arrs, diffs, pair = {}, [], []
    for j in pick:
        npy, i = loc[keys[j]]
        arr = arrs.setdefault(npy, np.load(npy, mmap_mode='r'))
        z = enc.encode(np.asarray(arr[i], np.float32))
        diffs.append(float(np.abs(z - lat[index[keys[j]]]).max()))
    other = lat[index[keys[pick[0]]]]
    for j in pick[1:201]:
        pair.append(float(np.abs(lat[index[keys[j]]] - other).max()))
    diffs = np.array(diffs)
    check(diffs.max() <= LATENT_TOL, f'{len(diffs)} re-encoded states == stored latents '
          f'(max |diff| {diffs.max():.2e}, median {np.median(diffs):.2e}; tol {LATENT_TOL:g}; '
          f'different shapes differ by median {np.median(pair):.2f})')

    print('baselines for val state_mse (mean over dims, validation records)')
    df = frames['validation']
    tgt = np.asarray(lat[df.target_idx.values])
    cur = np.asarray(lat[df.geo_idx.values])
    tr = frames['train']
    mean_tr = np.zeros(LATENT_DIM, np.float64)
    rows_tr = tr.target_idx.values
    for i in range(0, len(rows_tr), 65536):
        mean_tr += np.asarray(lat[np.sort(rows_tr[i: i + 65536])], np.float64).sum(0)
    mean_tr /= len(rows_tr)
    print(f'  zero predictor        {np.mean(tgt ** 2):.5f}')
    print(f'  copy current state    {np.mean((tgt - cur) ** 2):.5f}')
    print(f'  train-mean target     {np.mean((tgt - mean_tr) ** 2):.5f}')

    print('info (not part of the verdict)')
    dp = os.path.join(a.encoder, DEFAULT_CKPT)
    print(f'  eval guard: pc_latent_encoder DEFAULT_CKPT = {DEFAULT_CKPT} -> '
          + ('matches this dataset' if os.path.isfile(dp) and md5(dp) == got_md5 else
             'DOES NOT match this dataset (run_eval.sh would refuse; see README step 2)'))
    cp = os.path.join(a.cad_eval, 'cad_prompt.py')
    same = os.path.isfile(cp) and sha256(cp) == sha256(os.path.join(HERE, 'cad_prompt.py'))
    print(f'  {cp}: ' + ('identical to qwen_datagen/cad_prompt.py' if same else
                          'missing or different -- copy qwen_datagen/cad_prompt.py there before evaluating'))
    print('RESULT: OK' if not bad else f'RESULT: MISMATCH ({len(bad)} checks failed)')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
