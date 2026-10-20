#!/usr/bin/env python
"""Random model-level train / validation / test split for SetVAE (instead of DeepCAD's official split).

The unit is the CAD MODEL: all construction steps of a model (intermediate and final solids) go to the
same split, so near-identical geometry of one model never straddles two splits.  Deterministic: fixed
seed over the sorted model ids (same output on any machine with Python 3).

    python datagen/make_random_split.py                     # 90 / 5 / 5, seed 0
    python datagen/make_random_split.py --ratios 0.8 0.1 0.1 --seed 1
 -> datagen/models_random.tsv  (same format as models.tsv; pass it to pack_clouds.py and verify_clouds.py)
"""
import argparse
import collections
import hashlib
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=os.path.join(HERE, 'models.tsv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'models_random.tsv'))
    ap.add_argument('--ratios', type=float, nargs=3, default=[0.90, 0.05, 0.05], metavar=('TRAIN', 'VAL', 'TEST'))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    assert abs(sum(a.ratios) - 1) < 1e-9, 'ratios must sum to 1'
    lines = open(a.models).read().splitlines()
    rows = sorted((l.split('\t') for l in lines[1:] if l), key=lambda r: r[0])
    steps = {mid: int(n) for mid, _, n in rows}
    order = [mid for mid, _, _ in rows]
    random.Random(a.seed).shuffle(order)
    n_val, n_test = round(len(order) * a.ratios[1]), round(len(order) * a.ratios[2])
    n_train = len(order) - n_val - n_test
    split = {m: 'train' for m in order[:n_train]}
    split.update({m: 'validation' for m in order[n_train:n_train + n_val]})
    split.update({m: 'test' for m in order[n_train + n_val:]})
    with open(a.out, 'w') as f:
        f.write(lines[0] + '\n')
        for mid, _, n in rows:
            f.write(f'{mid}\t{split[mid]}\t{n}\n')
    models = collections.Counter(split.values())
    clouds = collections.Counter()
    for m, s in split.items():
        clouds[s] += steps[m]
    for s in ('train', 'validation', 'test'):
        print(f'{s:10s} {models[s]:6d} models  {clouds[s]:6d} point clouds')
    print(f'seed {a.seed}, ratios {a.ratios} -> {a.out}')
    print(f'sha256 {hashlib.sha256(open(a.out, "rb").read()).hexdigest()}')


if __name__ == '__main__':
    main()
