#!/usr/bin/env python
"""Re-pack an existing packed cloud dataset under another split file, without regenerating anything.

The clouds are keyed by "<model_id>_stepNN", so a packing made for one split (e.g. the random split of
datasets/deepcad_verified_pc_random) can be rearranged into another (default: DeepCAD's official split,
datagen/models.tsv) by copying rows.  Output layout = pack_clouds.py (same row order, ids files, meta.json).

After writing, every destination row is compared byte for byte with its source row.

  conda activate setvae
  python datagen/repack_clouds.py                                   # random pack -> DeepCAD-split pack
  python datagen/verify_clouds.py --skip-solids                     # A (ids) + C (fingerprints)
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SPLITS = ('train', 'validation', 'test')


def dest_rows(models_tsv):
    """Row order of every split exactly as pack_clouds.py writes it for this split file."""
    rows, models = {s: [] for s in SPLITS}, {s: 0 for s in SPLITS}
    for line in sorted(open(models_tsv).read().splitlines()[1:]):
        if line:
            mid, split, n = line.split('\t')
            rows[split] += [f'{mid}_step{k:02d}' for k in range(1, int(n) + 1)]
            models[split] += 1
    return rows, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(PROJ, 'datasets', 'deepcad_verified_pc_random'),
                    help='existing packed dataset (any split names: every <name>.npy + <name>_ids.txt is read)')
    ap.add_argument('--models', default=os.path.join(HERE, 'models.tsv'), help='split file of the new packing')
    ap.add_argument('--out', default=os.path.join(PROJ, 'datasets', 'deepcad_verified_pc'))
    ap.add_argument('--chunk', type=int, default=8192)
    a = ap.parse_args()
    t0 = time.time()
    assert os.path.abspath(a.src) != os.path.abspath(a.out), '--out must differ from --src'
    for s in SPLITS:
        assert not os.path.exists(os.path.join(a.out, f'{s}.npy')), f'{a.out}/{s}.npy exists -- use a new --out'
    os.makedirs(a.out, exist_ok=True)

    rows, models = dest_rows(a.models)
    where = {key: (s, i) for s in SPLITS for i, key in enumerate(rows[s])}
    parts = []
    for f in sorted(glob.glob(os.path.join(a.src, '*_ids.txt'))):
        npy = f[: -len('_ids.txt')] + '.npy'
        if os.path.isfile(npy):
            parts.append((npy, open(f).read().split()))
    assert parts, f'no <name>.npy + <name>_ids.txt pairs in {a.src}'
    n_pts = np.load(parts[0][0], mmap_mode='r').shape[1]
    arr = {s: np.lib.format.open_memmap(os.path.join(a.out, f'{s}.npy'), mode='w+', dtype=np.float32,
                                        shape=(len(rows[s]), n_pts, 3)) for s in SPLITS}
    filled = {s: np.zeros(len(rows[s]), np.int32) for s in SPLITS}

    def plan(ids):
        """per source row: destination split index and row (-1 if the key is not in the new split file)"""
        ds = np.array([SPLITS.index(where[k][0]) if k in where else -1 for k in ids])
        di = np.array([where[k][1] if k in where else -1 for k in ids])
        return ds, di

    # ---- copy: read every source part sequentially, scatter rows to their destination -----------------
    for npy, ids in parts:
        src = np.load(npy, mmap_mode='r')
        assert src.shape[0] == len(ids) and src.shape[1:] == (n_pts, 3), (npy, src.shape)
        ds, di = plan(ids)
        for c in range(0, len(ids), a.chunk):
            block = np.asarray(src[c: c + a.chunk])
            for si, s in enumerate(SPLITS):
                m = ds[c: c + a.chunk] == si
                if m.any():
                    idx = di[c: c + a.chunk][m]
                    arr[s][idx] = block[m]
                    filled[s][idx] += 1
        print(f'  copied {os.path.basename(npy)} ({len(ids)} clouds, {time.time() - t0:.0f}s)', flush=True)
    for s in SPLITS:
        assert (filled[s] == 1).all(), (f'{s}: {int((filled[s] == 0).sum())} clouds missing, '
                                        f'{int((filled[s] > 1).sum())} duplicated in {a.src}')
        arr[s].flush()

    # ---- verify: every destination row equals its source row, byte for byte ---------------------------
    n_cmp = 0
    for npy, ids in parts:
        src = np.load(npy, mmap_mode='r')
        ds, di = plan(ids)
        for c in range(0, len(ids), a.chunk):
            block = np.asarray(src[c: c + a.chunk])
            for si, s in enumerate(SPLITS):
                m = ds[c: c + a.chunk] == si
                if m.any():
                    assert np.array_equal(np.asarray(arr[s][di[c: c + a.chunk][m]]), block[m]), f'{npy}: row mismatch'
                    n_cmp += int(m.sum())
    print(f'  verified {n_cmp} rows byte-identical to the source ({time.time() - t0:.0f}s)', flush=True)

    stats = {}
    for s in SPLITS:
        mx, finite = 0.0, True
        for i in range(0, len(arr[s]), a.chunk):
            c = np.asarray(arr[s][i: i + a.chunk])
            finite &= bool(np.isfinite(c).all()); mx = max(mx, float(np.abs(c).max()))
        assert finite and mx <= 0.7501, f'{s}: finite={finite} max|coord|={mx}'
        open(os.path.join(a.out, f'{s}_ids.txt'), 'w').write('\n'.join(rows[s]) + '\n')
        stats[s] = {'models': models[s], 'clouds': len(rows[s]), 'max_abs_coord': round(mx, 6)}
    src_meta = os.path.join(a.src, 'meta.json')
    json.dump({'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'points_per_cloud': n_pts,
               'split_file': os.path.basename(a.models),
               'split_file_sha256': hashlib.sha256(open(a.models, 'rb').read()).hexdigest(),
               'format': 'float32 [N, P, 3] memmap + ids', 'splits': stats,
               'frame': "each solid's own bbox: centre 0, max half-extent 0.75",
               'repacked_from': os.path.abspath(a.src),
               'source_meta': json.load(open(src_meta)) if os.path.isfile(src_meta) else None},
              open(os.path.join(a.out, 'meta.json'), 'w'), indent=2)
    print(json.dumps(stats, indent=1))
    print(f'RESULT: OK -- {n_cmp} clouds repacked byte-identically, {time.time() - t0:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
