#!/usr/bin/env python
"""Build cadseq_stepwise_v5 on the server: the Qwen3 SFT dataset encoded with the H800 SetVAE.

v5 = the v3cb / v4 record recipe on the full verified + annotated corpus (174,582 models):
  split        DeepCAD's official split, exactly datagen/models.tsv (train 157,945 / validation 8,755 /
               test 7,882 models) -- no de-duplication (the division we fixed)
  answers      datagen/programs.jsonl actions, then {"kind":"end"}
  prompts      cad_prompt.build_prompt with count + bbox lines; bbox = exact step-solid bbox from
               datagen/solid_truth.npz (the numbers geometry_worker reports during the rollout)
  description  Recap-OmniCAD prompt_exp minus <v> tags (descriptions.jsonl, exported locally)
  states       the packed 4096-point clouds (any packing: ids are looked up, the split used for SetVAE
               does not matter) encoded by the new SetVAE; row 0 of latents.npy is the zero state

Every cloud is encoded with the SAME 2048-point subset the rollout uses (SetVAEEncoder.encode on a single
cloud draws it from default_rng(0)), so stored latents equal what the encoder gives at inference instead
of depending on the row's position inside a batch (v3/v4 had that ~1% subsampling noise).

  conda activate verl      # torch + pandas + pyarrow; the env the rollout runs the encoder in
  CUDA_VISIBLE_DEVICES=4 python qwen_datagen/build_stepwise_v5.py
Then run verify_stepwise_v5.py.  Output ~1.4 GB latents + parquets.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cad_prompt import build_prompt  # noqa: E402

CODE = '$REMOTE/code'
SPLITS = ('train', 'validation', 'test')
LATENT_DIM = 1024
PROMPT_FORMAT = {'count': True, 'bbox': True}
EXPECTED = {'train': (157945, 317711), 'validation': (8755, 17639), 'test': (7882, 16078)}  # models, clouds


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_models(models_tsv):
    """[(model_id, split, n_steps)] sorted by model id -- the order solid_truth.npz rows follow."""
    rows = []
    for line in sorted(open(models_tsv).read().splitlines()[1:]):
        if line:
            mid, split, n = line.split('\t')
            rows.append((mid, split, int(n)))
    return rows


def load_programs(path, wanted):
    acts = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r['model_id'] in wanted:
                acts[r['model_id']] = [json.dumps(x, separators=(',', ':')) for x in r['program']['actions']]
    return acts


def load_descriptions(path):
    with open(path) as f:
        return {r['model_id']: r['description'] for r in map(json.loads, f)}


def state_layout(models):
    """key -> latents row (1..), and key -> (split, row in solid_truth.npz), both in canonical order."""
    index, truth_pos, per_split, n = {}, {}, {s: 0 for s in SPLITS}, 1
    for s in SPLITS:
        for mid, sp, k_max in models:
            if sp != s:
                continue
            for k in range(1, k_max + 1):
                key = f'{mid}_step{k:02d}'
                index[key] = n; n += 1
                truth_pos[key] = (s, per_split[s]); per_split[s] += 1
    return index, truth_pos, n


def cloud_parts(clouds_dir):
    """Every <name>.npy + <name>_ids.txt pair in a packed cloud dir (split names are irrelevant)."""
    parts = []
    for f in sorted(os.listdir(clouds_dir)):
        if f.endswith('_ids.txt'):
            stem = os.path.join(clouds_dir, f[:-len('_ids.txt')])
            if os.path.isfile(stem + '.npy'):
                parts.append((stem + '.npy', open(stem + '_ids.txt').read().split()))
    return parts


def rollout_subset(n_in, n_keep):
    """The subset SetVAEEncoder.prepare draws for a single cloud with seed 0 (what the rollout calls)."""
    if n_in <= n_keep:
        return np.arange(n_in)
    return np.sort(np.random.default_rng(0).choice(n_in, n_keep, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datagen', default=f'{CODE}/SetVAE/datagen',
                    help='models.tsv, programs.jsonl, solid_truth.npz')
    ap.add_argument('--clouds', default=f'{CODE}/SetVAE/datasets/deepcad_verified_pc_random')
    ap.add_argument('--descriptions', default=os.path.join(HERE, 'descriptions.jsonl'))
    ap.add_argument('--encoder', default=f'{CODE}/pc_latent_encoder')
    ap.add_argument('--ckpt', default=f'{CODE}/pc_latent_encoder/checkpoints/setvae_verified_random_best.pt')
    ap.add_argument('--out', default=f'{CODE}/new_cadseq/dataset/cadseq_stepwise_v5')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--rows-per-group', type=int, default=50000)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    if os.path.exists(os.path.join(a.out, 'train.parquet')) and not a.overwrite:
        sys.exit(f'{a.out}/train.parquet exists -- pass --overwrite to rebuild')
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    name = os.path.basename(a.out.rstrip('/'))
    models_tsv = os.path.join(a.datagen, 'models.tsv')
    programs = os.path.join(a.datagen, 'programs.jsonl')
    truth_npz = os.path.join(a.datagen, 'solid_truth.npz')

    # ---- inputs -------------------------------------------------------------------------------
    models = load_models(models_tsv)
    counts = {s: sum(1 for m in models if m[1] == s) for s in SPLITS}
    acts = load_programs(programs, {m[0] for m in models})
    desc = load_descriptions(a.descriptions)
    print(f'{len(models)} models {counts} | {len(acts)} programs | {len(desc)} descriptions', flush=True)
    for mid, sp, n in models:
        assert mid in acts, f'{mid}: no program'
        assert len(acts[mid]) == n, f'{mid}: {len(acts[mid])} actions, models.tsv says {n}'
        assert mid in desc and desc[mid], f'{mid}: no description'
    index, truth_pos, n_rows = state_layout(models)
    truth = np.load(truth_npz)
    bbox = {s: truth[f'{s}_bbox'] for s in SPLITS}
    for s in SPLITS:
        assert len(bbox[s]) == sum(n for _, sp, n in models if sp == s), f'solid_truth.npz vs models.tsv ({s})'
        assert (counts[s], len(bbox[s])) == EXPECTED[s], f'{s}: {counts[s]} models / {len(bbox[s])} states, expected {EXPECTED[s]}'

    # ---- encode every state -------------------------------------------------------------------
    sys.path.insert(0, a.encoder)
    from encoder import SetVAEEncoder
    enc = SetVAEEncoder(a.ckpt, device=a.device)
    print(f'encoder {a.ckpt} (epoch {enc.ckpt_epoch}, {enc.n_points} points) on {enc.device}', flush=True)
    lat = np.lib.format.open_memmap(os.path.join(a.out, 'latents.npy'), mode='w+', dtype=np.float32,
                                    shape=(n_rows, LATENT_DIM))
    lat[0] = 0.0
    filled = np.zeros(n_rows, bool); filled[0] = True
    parts = cloud_parts(a.clouds)
    assert parts, f'no <name>.npy + <name>_ids.txt pairs in {a.clouds}'
    done, t1 = 0, time.time()
    for npy, ids in parts:
        arr = np.load(npy, mmap_mode='r')
        assert arr.shape[0] == len(ids) and arr.shape[2] == 3, (npy, arr.shape, len(ids))
        sel = torch.from_numpy(rollout_subset(arr.shape[1], enc.n_points)).to(enc.device)
        rows = np.array([index.get(k, -1) for k in ids])
        assert (rows > 0).all(), f'{npy}: {int((rows < 0).sum())} clouds not in models.tsv'
        for i in range(0, len(ids), a.batch):
            x = torch.from_numpy(np.array(arr[i: i + a.batch])).to(enc.device)[:, sel]
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=enc.device)
            z = enc.encode_tensor(x, mask).float().cpu().numpy()
            r = rows[i: i + a.batch]
            assert not filled[r].any(), f'{npy}: a state appears twice'
            lat[r] = z; filled[r] = True
            done += len(r)
            if (done // a.batch) % 100 == 0:
                print(f'  encoded {done}/{n_rows-1} ({done/max(time.time()-t1, 1e-9):.0f}/s)', flush=True)
    assert filled.all(), f'{int((~filled).sum())} states have no cloud in {a.clouds}'
    assert np.isfinite(lat).all(), 'non-finite latent'
    lat.flush()
    print(f'encoded {done} states in {(time.time()-t1)/60:.1f} min', flush=True)

    # ---- records ------------------------------------------------------------------------------
    schema = pa.schema([('model_id', pa.string()), ('step_idx', pa.int32()), ('n_steps', pa.int32()),
                        ('is_end', pa.bool_()), ('deepcad_split', pa.string()), ('description', pa.string()),
                        ('question', pa.string()), ('answer', pa.string()),
                        ('geo_idx', pa.int64()), ('target_idx', pa.int64()),
                        ('enable_thinking', pa.bool_()), ('data_source', pa.string())])
    rec_counts = {}
    for s in SPLITS:
        writer = pq.ParquetWriter(os.path.join(a.out, f'{s}.parquet'), schema, compression='zstd')
        buf, n_rec, n_mod = {c: [] for c in schema.names}, 0, 0
        for mid, sp, n in models:
            if sp != s:
                continue
            targets = acts[mid] + ['{"kind":"end"}']
            idx = [0] + [index[f'{mid}_step{k:02d}'] for k in range(1, n + 1)]
            for t, tgt in enumerate(targets):
                bb = None if t == 0 else bbox[s][truth_pos[f'{mid}_step{t:02d}'][1]].tolist()
                buf['model_id'].append(mid); buf['step_idx'].append(t); buf['n_steps'].append(n)
                buf['is_end'].append(t == n); buf['deepcad_split'].append(s)
                buf['description'].append(desc[mid])
                buf['question'].append(build_prompt(desc[mid], PROMPT_FORMAT, n_done=t, bbox=bb))
                buf['answer'].append(tgt)
                buf['geo_idx'].append(idx[t]); buf['target_idx'].append(idx[min(t + 1, n)])
                buf['enable_thinking'].append(False)
                buf['data_source'].append(f'Snitro/Recap-OmniCAD/{name}')
                n_rec += 1
            n_mod += 1
            if len(buf['model_id']) >= a.rows_per_group:
                writer.write_table(pa.Table.from_pydict(buf, schema=schema))
                buf = {c: [] for c in schema.names}
        if buf['model_id']:
            writer.write_table(pa.Table.from_pydict(buf, schema=schema))
        writer.close()
        rec_counts[s] = {'models': n_mod, 'records': n_rec, 'states': len(bbox[s])}
        print(f'{s}: {n_mod} models, {n_rec} records', flush=True)

    stats = {'version': 5, 'dataset': name, 'prompt_format': PROMPT_FORMAT,
             'latents': 'latents.npy (memmap, row 0 = zero state; parquet holds geo_idx/target_idx)',
             'latent_dim': LATENT_DIM, 'n_states': n_rows, 'counts': rec_counts,
             'num_geo_tokens': 16, 'geo_placeholder': '<|image_pad|>',
             'encoder': os.path.abspath(a.ckpt), 'encoder_md5': md5(a.ckpt), 'encoder_epoch': enc.ckpt_epoch,
             'encoder_subset': 'one fixed 2048-point subset = SetVAEEncoder.encode(single cloud, seed 0), as in the rollout',
             'clouds': os.path.abspath(a.clouds), 'n_points_per_cloud': int(np.load(parts[0][0], mmap_mode='r').shape[1]),
             'frame': "each step solid's own bbox (centre 0, max half-extent 0.75) -- rollout --geo-frame current",
             'split': None, 'split_source': 'DeepCAD official split (datagen/models.tsv), no de-duplication',
             'selection': 'DeepCAD models whose cadseq program reproduces the geometry (strict d_surface < 1e-6 '
                          'or fixed-plane reference) AND have a Recap-OmniCAD prompt_exp annotation',
             'inputs_sha256': {'models.tsv': sha256(models_tsv), 'programs.jsonl': sha256(programs),
                               'solid_truth.npz': sha256(truth_npz), 'descriptions.jsonl': sha256(a.descriptions),
                               'cad_prompt.py': sha256(os.path.join(HERE, 'cad_prompt.py'))},
             'z0_convention': 'zeros(1024) for the empty state before the first action',
             'seconds': round(time.time() - t0, 1)}
    for s in SPLITS:
        json.dump({**stats, 'split': s}, open(os.path.join(a.out, f'{s}_stats.json'), 'w'), indent=2)
    print(f'done in {(time.time()-t0)/60:.1f} min -> {a.out}')


if __name__ == '__main__':
    main()
