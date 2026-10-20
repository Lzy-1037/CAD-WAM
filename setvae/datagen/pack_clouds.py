#!/usr/bin/env python
"""Step 2 (server): pack the generated shards into datasets/deepcad_verified_pc, the files train_deepcad.py reads.

    python datagen/pack_clouds.py                                     # DeepCAD official split
    python datagen/pack_clouds.py --models datagen/models_random.tsv \
        --out datasets/deepcad_verified_pc_random                     # random split
 -> datasets/deepcad_verified_pc/{train,validation,test}.npy  float32 [N, 4096, 3]
                                 {train,validation,test}_ids.txt, meta.json
Rows are ordered exactly like the local dataset (model id ascending, steps ascending).  Needs only numpy.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SPLITS = ('train', 'validation', 'test')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', default=os.path.join(PROJ, 'datasets', 'deepcad_verified_geom'))
    ap.add_argument('--models', default=os.path.join(HERE, 'models.tsv'))
    ap.add_argument('--out', default=os.path.join(PROJ, 'datasets', 'deepcad_verified_pc'))
    ap.add_argument('--n-points', type=int, default=4096)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    rows = {s: [] for s in SPLITS}
    models = {s: 0 for s in SPLITS}
    for line in sorted(open(a.models).read().splitlines()[1:]):
        if not line:
            continue
        mid, split, n = line.split('\t')
        rows[split] += [f'{mid}_step{k:02d}' for k in range(1, int(n) + 1)]
        models[split] += 1
    where = {key: (s, i) for s in SPLITS for i, key in enumerate(rows[s])}
    arr = {s: np.lib.format.open_memmap(os.path.join(a.out, f'{s}.npy'), mode='w+', dtype=np.float32,
                                        shape=(len(rows[s]), a.n_points, 3)) for s in SPLITS}
    filled = {s: np.zeros(len(rows[s]), bool) for s in SPLITS}
    shards = sorted(glob.glob(os.path.join(a.geom, 'clouds', 'shard_*.npz')))
    for k, sh in enumerate(shards):
        z = np.load(sh)
        for key in z.files:
            if key in where:
                s, i = where[key]
                arr[s][i] = z[key]; filled[s][i] = True
        if (k + 1) % 25 == 0 or k + 1 == len(shards):
            print(f'  {k + 1}/{len(shards)} shards ({time.time() - t0:.0f}s)', flush=True)
    stats = {}
    for s in SPLITS:
        assert filled[s].all(), f'{s}: {int((~filled[s]).sum())} clouds missing -- rerun generate_clouds.py'
        mx, finite = 0.0, True
        for i in range(0, len(arr[s]), 8192):
            c = np.asarray(arr[s][i:i + 8192])
            finite &= bool(np.isfinite(c).all()); mx = max(mx, float(np.abs(c).max()))
        assert finite and mx <= 0.7501, f'{s}: finite={finite} max|coord|={mx}'
        arr[s].flush()
        open(os.path.join(a.out, f'{s}_ids.txt'), 'w').write('\n'.join(rows[s]) + '\n')
        stats[s] = {'models': models[s], 'clouds': len(rows[s]), 'max_abs_coord': round(mx, 6)}
    import hashlib
    json.dump({'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'points_per_cloud': a.n_points,
               'split_file': os.path.basename(a.models),
               'split_file_sha256': hashlib.sha256(open(a.models, 'rb').read()).hexdigest(),
               'format': 'float32 [N, P, 3] memmap + ids', 'splits': stats,
               'frame': "each solid's own bbox: centre 0, max half-extent 0.75"},
              open(os.path.join(a.out, 'meta.json'), 'w'), indent=2)
    print(json.dumps(stats, indent=1))
    print(f'done in {time.time() - t0:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
