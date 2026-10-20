#!/usr/bin/env python
"""Step 5 (server): verify the regenerated, packed dataset against the local one.

    python datagen/verify_clouds.py                                              # DeepCAD split
    python datagen/verify_clouds.py --models datagen/models_random.tsv \\
        --data datasets/deepcad_verified_pc_random                                # random split

Works for any split file (--models, the one given to pack_clouds.py):
A. Samples      every split's ids list is exactly what the split file prescribes (same rows, same order),
                and together the splits contain each of the 351,428 step clouds exactly once.
B. Every solid  FULL COVERAGE: the exact bounding box and volume of each of the 351,428 step solids,
                recorded while generating, equal the local truth (solid_truth.npz).  They come from the
                CAD kernel, not from point sampling, so they reproduce across runs (measured: bbox <= 3e-13
                of the diagonal, volume <= 1e-13 relative; tolerances 1e-6; a wrong program or step
                changes them by orders of magnitude more).
C. Sampling     2,000 random clouds are bit-identical to the local ones, or -- for the few percent hit by
                OpenCASCADE's non-deterministic tessellation -- their 512 local reference points lie on
                the regenerated cloud (mean squared nearest distance <= 3e-3).
Needs only numpy.  Exit code 0 = OK, 1 = MISMATCH.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SPLITS = ('train', 'validation', 'test')
CLOUD_TOL = 3e-3


def split_ids(models_tsv):
    """Row order of every split, exactly as pack_clouds.py writes it for this split file."""
    rows = {s: [] for s in SPLITS}
    for line in sorted(open(models_tsv).read().splitlines()[1:]):
        if line:
            mid, split, n = line.split('\t')
            rows[split] += [f'{mid}_step{k:02d}' for k in range(1, int(n) + 1)]
    return rows


def load_server_solids(root):
    boxes, vols, errs = {}, {}, {}
    files = sorted(glob.glob(os.path.join(root, 'solids', 'shard_*.json')))
    if not files:
        return None
    for f in files:
        d = json.load(open(f))
        boxes.update(d['bboxes']); vols.update(d.get('volumes', {})); errs.update(d.get('errors', {}))
    return boxes, vols, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(PROJ, 'datasets', 'deepcad_verified_pc'))
    ap.add_argument('--models', default=os.path.join(HERE, 'models.tsv'),
                    help='split file that was given to pack_clouds.py')
    ap.add_argument('--solids', default=os.path.join(PROJ, 'datasets', 'deepcad_verified_geom'),
                    help='directory containing solids/shard_*.json (from generate_clouds.py)')
    ap.add_argument('--bbox-tol', type=float, default=1e-6, help='max |bbox diff| / bbox diagonal')
    ap.add_argument('--vol-tol', type=float, default=1e-6, help='max |volume diff| / volume')
    ap.add_argument('--allow-missing', action='store_true', help='testing on a partial dataset only')
    ap.add_argument('--skip-solids', action='store_true',
                    help='skip B -- for a dataset made by repack_clouds.py: the clouds are byte copies of an already '
                         'verified packing, and the solids/ records of the generation run are not needed again')
    a = ap.parse_args()
    ok = True
    canonical = split_ids(os.path.join(HERE, 'models.tsv'))          # the 351,428 clouds, truth layout
    expected = split_ids(a.models)

    # ---------------- A. samples ----------------
    kind = 'DeepCAD official split' if expected == canonical else 'custom split'
    print(f'A. samples -- split file {os.path.basename(a.models)} ({kind}), '
          f'sha256 {hashlib.sha256(open(a.models, "rb").read()).hexdigest()[:16]}...')
    where = {}
    for s in SPLITS:
        ids = open(os.path.join(a.data, f'{s}_ids.txt')).read().split()
        arr = np.load(os.path.join(a.data, f'{s}.npy'), mmap_mode='r')
        for i, k in enumerate(ids):
            where[k] = (s, arr, i)
        if not a.allow_missing:
            same = ids == expected[s] and arr.shape[0] == len(expected[s])
            ok &= same
            print(f'   {s:10s} ids as prescribed by the split file: {ids == expected[s]} | '
                  f'clouds {arr.shape[0]} (expected {len(expected[s])})')
    if not a.allow_missing:
        all_truth = sorted(k for s in SPLITS for k in canonical[s])
        packed = sorted(k for s in SPLITS for k in open(os.path.join(a.data, f'{s}_ids.txt')).read().split())
        once = packed == all_truth
        ok &= once
        print(f'   every one of the {len(all_truth)} step clouds present exactly once across splits: {once}')

    # ---------------- B. every step solid ----------------
    print('B. every step solid (exact bbox + volume vs local truth)')
    srv = None if a.skip_solids else load_server_solids(a.solids)
    if a.skip_solids:
        print('   skipped (--skip-solids): repacked byte copies of a packing that passed B')
    if srv is None and not a.skip_solids:
        print(f'   no {a.solids}/solids/*.json -- these shards come from an older generate_clouds.py.\n'
              f'   Fix: python datagen/generate_clouds.py --solids-only --out datasets/deepcad_verified_solids\n'
              f'        python datagen/verify_clouds.py ... --solids datasets/deepcad_verified_solids')
        raise SystemExit(1)
    boxes, vols, errs = srv if srv is not None else ({}, {}, {})
    truth = np.load(os.path.join(HERE, 'solid_truth.npz'))
    worst = []
    for s in (() if a.skip_solids else SPLITS):
        tb, tv = truth[f'{s}_bbox'], truth[f'{s}_volume']
        assert len(tb) == len(canonical[s]), 'solid_truth.npz does not match models.tsv'
        present = np.array([k in boxes and k in vols for k in canonical[s]])
        n_missing = int((~present).sum())
        idx = np.flatnonzero(present)
        sb = np.array([boxes[canonical[s][i]] for i in idx], np.float64).reshape(-1, 6)
        sv = np.array([vols[canonical[s][i]] for i in idx], np.float64)
        diag = np.linalg.norm(tb[idx, 3:] - tb[idx, :3], axis=1)
        e_box = np.abs(sb - tb[idx]).max(1) / np.maximum(diag, 1e-300)
        e_vol = np.abs(sv - tv[idx]) / np.maximum(np.abs(tv[idx]), 1e-300)
        bad = (e_box > a.bbox_tol) | (e_vol > a.vol_tol)
        for j in np.flatnonzero(bad)[:10]:
            worst.append((canonical[s][idx[j]], e_box[j], e_vol[j]))
        if n_missing and not a.allow_missing:
            ok = False
        ok &= not bad.any()
        print(f'   {len(idx):6d} solids checked | missing {n_missing:5d} | '
              f'bbox over tol {int((e_box > a.bbox_tol).sum())} (max {e_box.max() if len(idx) else 0:.1e}) | '
              f'volume over tol {int((e_vol > a.vol_tol).sum())} (max {e_vol.max() if len(idx) else 0:.1e})')
    if errs:
        print(f'   generation errors recorded: {len(errs)}, e.g. {list(errs.items())[:3]}')
        ok = False
    for key, eb, ev in worst:
        print(f'   MISMATCH {key}: bbox rel diff {eb:.2e}, volume rel diff {ev:.2e}')

    # ---------------- C. point sampling ----------------
    print('C. point sampling (2,000 random clouds)')
    fp = json.load(open(os.path.join(HERE, 'fingerprints.json')))
    ref = np.load(os.path.join(HERE, 'reference_subsample.npz'))
    exact, dists, missing = 0, [], 0
    for key, (_, h) in fp['clouds'].items():
        if key not in where:
            missing += 1
            continue
        _, arr, i = where[key]
        cloud = np.ascontiguousarray(arr[i], dtype=np.float32)
        if hashlib.sha256(cloud.tobytes()).hexdigest() == h:
            exact += 1
            continue
        r = ref[key].astype(np.float32)
        dists.append(float(((r[:, None, :] - cloud[None, :, :]) ** 2).sum(-1).min(1).mean()))
    if missing and not a.allow_missing:
        ok = False
    print(f'   checked {len(fp["clouds"]) - missing} | bit-identical {exact} | differing {len(dists)}')
    if dists:
        d = np.array(dists)
        print(f'   differing clouds: distance median {np.median(d):.2e}, max {d.max():.2e} '
              f'(tolerance {CLOUD_TOL:.0e}) -> over tolerance {int((d > CLOUD_TOL).sum())}')
        ok &= bool((d <= CLOUD_TOL).all())

    print('RESULT: OK -- the regenerated dataset matches the local one' if ok else
          'RESULT: MISMATCH -- do not train; send this output')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
