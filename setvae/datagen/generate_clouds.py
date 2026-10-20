#!/usr/bin/env python
"""Step 1 (server): regenerate every step point cloud from the verified cadseq programs.

Run with a Python that has CadQuery 2.7.0 (see README_H800.md):
    python datagen/generate_clouds.py --workers 48
 -> datasets/deepcad_verified_geom/clouds/shard_XXXX.npz   4096-point clouds
    datasets/deepcad_verified_geom/solids/shard_XXXX.json  exact bbox + volume of every step solid
    (resumable: finished shards are skipped)

    python datagen/generate_clouds.py --solids-only --out datasets/deepcad_verified_solids
 -> only the solids/*.json (no point sampling; several times faster)

Uses the bundled datagen/cadseq and datagen/geometry_worker.py -- the exact code that produced the local
dataset -- with the same request (4096 points, frame "current", seed 0).
"""
import argparse
import base64
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
os.environ['CADSEQ_ROOT'] = HERE                  # geometry_worker imports cadseq from datagen/
sys.path.insert(0, HERE)
import geometry_worker as gw  # noqa: E402

N_POINTS = 4096
_cache = {}


def run(job):
    mid, k, acts, n_points = job
    out = gw.handle({'cmd': 'exec', 'actions': acts, 'n_points': n_points, 'frame': 'current',
                     'seed': 0, 'half_extent': 0.75}, _cache)
    key = f'{mid}_step{k:02d}'
    if not out.get('ok'):
        return key, None, None, None, out.get('error')
    pts = (np.frombuffer(base64.b64decode(out['points_b64']), np.float32).reshape(n_points, 3)
           if n_points else None)
    return key, pts, out['bbox'], out['volume'], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--programs', default=os.path.join(HERE, 'programs.jsonl'))
    ap.add_argument('--models', default=os.path.join(HERE, 'models.tsv'))
    ap.add_argument('--out', default=os.path.join(PROJ, 'datasets', 'deepcad_verified_geom'))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument('--models-per-shard', type=int, default=1000)
    ap.add_argument('--solids-only', action='store_true', help='bbox + volume only, no point clouds')
    ap.add_argument('--only', help='comma-separated model ids (testing)')
    a = ap.parse_args()
    n_points = 0 if a.solids_only else N_POINTS
    ids = sorted(l.split('\t')[0] for l in open(a.models).read().splitlines()[1:] if l)
    if a.only:
        want = set(a.only.split(','))
        ids = [i for i in ids if i in want]
    prog = {}
    with open(a.programs) as f:
        for line in f:
            d = json.loads(line)
            prog[d['model_id']] = d['program']['actions']
    missing = [i for i in ids if i not in prog]
    assert not missing, f'{len(missing)} models have no program, e.g. {missing[:3]}'
    os.makedirs(os.path.join(a.out, 'solids'), exist_ok=True)
    if not a.solids_only:
        os.makedirs(os.path.join(a.out, 'clouds'), exist_ok=True)
    shards = [ids[i:i + a.models_per_shard] for i in range(0, len(ids), a.models_per_shard)]
    print(f'{len(ids)} models in {len(shards)} shards, {a.workers} workers'
          f'{" (solids only)" if a.solids_only else ""}', flush=True)
    t0, done, failed = time.time(), 0, {}
    with Pool(a.workers) as pool:
        for si, shard in enumerate(shards):
            cp = os.path.join(a.out, 'clouds', f'shard_{si:04d}.npz')
            sp = os.path.join(a.out, 'solids', f'shard_{si:04d}.json')
            if os.path.exists(sp) and (a.solids_only or os.path.exists(cp)):
                continue
            jobs = [(m, k, prog[m][:k], n_points) for m in shard for k in range(1, len(prog[m]) + 1)]
            clouds, boxes, vols, errs = {}, {}, {}, {}
            for key, pts, bb, vol, err in pool.imap_unordered(run, jobs, chunksize=8):
                if err is not None:
                    errs[key] = err
                    continue
                boxes[key], vols[key] = bb, vol
                if pts is not None:
                    clouds[key] = pts
            if not a.solids_only:
                np.savez(cp + '.tmp.npz', **clouds)
                os.replace(cp + '.tmp.npz', cp)
            json.dump({'bboxes': boxes, 'volumes': vols, 'errors': errs}, open(sp + '.tmp', 'w'))
            os.replace(sp + '.tmp', sp)
            done += len(boxes); failed.update(errs)
            el = time.time() - t0
            print(f'  shard {si + 1}/{len(shards)} | {done} solids, {len(failed)} failed | '
                  f'{el / 60:.1f} min ({done / max(el, 1):.0f}/s)', flush=True)
    print(f'done: {done} new solids, {len(failed)} failed {list(failed.items())[:3]} -> {a.out}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
