#!/usr/bin/env python
"""Build the 7,962-model evaluation set: every DeepCAD test model that has a Recap-OmniCAD annotation.

  conda activate verl && cd $REMOTE/project
  python qwen_datagen/build_test7962.py            # after copying this kit into qwen_datagen/

Why 7,962: DeepCAD's official test split has 8,052 models; 7,962 of them carry Recap-OmniCAD text
(the annotations our model is trained on) -- the same set Pointer-CAD reports on.  Our v6 dataset
tests on 7,882 of those, i.e. the ones whose cadseq program also reproduced the official geometry;
defining the test set that way filters by our own converter's success, which this file undoes.

Writes (nothing is overwritten; the v6 dataset dir gains two files):
  <v6>/test7962.parquet + test7962_stats.json   all 7,962 models (rollout input)
  <v6>/test80extra.parquet + test80extra_stats.json   only the 80 added models, so an existing
                                                7,882-model run can be topped up instead of redone
  <refs-out>/                                   7,950 reference STEPs = 7,882 verified + 68 added
                                                (12 of the 80 have no usable reference, see below)

The 80 added models are unverified: our cadseq program did not reproduce DeepCAD's official STEP
(42 tessellation, 14 mismatch, 13 rejected by the executor, 10 timeout/compare-failed, 1 no
reference).  Their rows therefore carry verified=False, their reference is DeepCAD's official STEP
(the plane-bug-corrected file for 16 of them), and 12 cannot be scored at all: 2 have no reference
file and 10 have one that OpenCASCADE cannot mesh.  Those 12 stay in the id list and in the
invalidity ratio but are excluded from CD/NC/ECD -- paper_eval.py reports them as missing references.

Only step 0 of an added model has a dataset-exact prompt (prompt_exact=True): later steps need the
bbox of the partially built solid, which is what the rollout computes for itself, so the stored
question for t>=1 uses "Current bounding box: empty".  The rollout only ever reads step 0.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = '$REMOTE/code'
COLS = ['model_id', 'step_idx', 'n_steps', 'is_end', 'deepcad_split', 'description', 'question',
        'answer', 'geo_idx', 'target_idx', 'enable_thinking', 'data_source']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v6', default=f'{CODE}/new_cadseq/dataset/cadseq_stepwise_v6')
    ap.add_argument('--extra', default=os.path.join(HERE, 'extra80.jsonl'))
    ap.add_argument('--ids', default=os.path.join(HERE, 'ids_test7962.txt'))
    ap.add_argument('--refs-extra', default=os.path.join(HERE, 'refs_extra'))
    ap.add_argument('--refs-src', default=f'{CODE}/new_cadseq/eval_refs/deepcad_v5_test')
    ap.add_argument('--refs-out', default=f'{CODE}/new_cadseq/eval_refs/deepcad_v6_test7962')
    ap.add_argument('--cad-prompt-dir', default=None, help='dir holding cad_prompt.py (default: this dir)')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    sys.path.insert(0, a.cad_prompt_dir or HERE)
    from cad_prompt import build_prompt

    out_all = os.path.join(a.v6, 'test7962.parquet')
    out_extra = os.path.join(a.v6, 'test80extra.parquet')
    for p in (out_all, out_extra):
        if os.path.exists(p) and not a.overwrite:
            sys.exit(f'{p} exists -- pass --overwrite to rebuild')

    # ---- inputs -----------------------------------------------------------------------------
    stats = json.load(open(os.path.join(a.v6, 'test_stats.json')))
    pf = stats['prompt_format']
    base = pd.read_parquet(os.path.join(a.v6, 'test.parquet'))
    base_ids = set(base.model_id.unique())
    want = set(open(a.ids).read().split())
    extra = [json.loads(l) for l in open(a.extra)]
    extra_ids = {r['model_id'] for r in extra}
    assert len(base_ids) == 7882, f'{a.v6}/test.parquet holds {len(base_ids)} models, expected 7882'
    assert len(want) == 7962, f'{a.ids} holds {len(want)} ids, expected 7962'
    assert not (base_ids & extra_ids), 'the added models overlap the v6 test set'
    assert base_ids | extra_ids == want, 'v6 test + added models != the 7,962 id list'

    # ---- rows for the 80 added models ---------------------------------------------------------
    rows = {c: [] for c in COLS}
    rows['verified'] = []; rows['prompt_exact'] = []
    for r in extra:
        targets = [json.dumps(x, separators=(',', ':')) for x in r['actions']] + ['{"kind":"end"}']
        for t, tgt in enumerate(targets):
            vals = [r['model_id'], t, r['n_steps'], t == r['n_steps'], 'test', r['description'],
                    build_prompt(r['description'], pf, n_done=t, bbox=None), tgt, 0, 0, False,
                    stats.get('dataset', 'cadseq_stepwise_v6') + '/test7962']
            for c, v in zip(COLS, vals):
                rows[c].append(v)
            rows['verified'].append(False); rows['prompt_exact'].append(t == 0)
    add = pd.DataFrame(rows)
    base = base.assign(verified=True, prompt_exact=True)
    allrows = pd.concat([base[add.columns], add], ignore_index=True)
    allrows = allrows.sort_values(['model_id', 'step_idx'], kind='stable').reset_index(drop=True)

    # step-0 prompts must be exactly what cad_prompt.build_prompt gives (the rollout asserts this)
    z = allrows[allrows.step_idx == 0]
    bad = [m for m, d, q in zip(z.model_id, z.description, z.question)
           if build_prompt(d, pf, 0, None) != q]
    assert not bad, f'{len(bad)} step-0 prompts do not match cad_prompt.py, e.g. {bad[:3]}'

    for path, df in ((out_all, allrows), (out_extra, add)):
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression='zstd')

    # ---- stats ------------------------------------------------------------------------------
    unscorable = sorted({r['model_id'] for r in extra if not r['scorable']})
    for path, ids, n_rec in ((out_all, want, len(allrows)), (out_extra, extra_ids, len(add))):
        s = dict(stats)
        s.update({'split': 'test', 'dataset': os.path.basename(path)[:-len('.parquet')],
                  'built_from': {'v6_test': os.path.join(a.v6, 'test.parquet'),
                                 'added': os.path.abspath(a.extra), 'added_sha256': sha256(a.extra)},
                  'counts': {'test': {'models': len(ids), 'records': n_rec}},
                  'selection': 'DeepCAD official test split ∩ Recap-OmniCAD annotations (7,962 of 8,052); '
                               'the 80 models beyond the v6 test set are unverified (verified=False)',
                  'unscorable_models': unscorable,
                  'note': 'evaluation only -- geo_idx/target_idx are 0 for the added models and the stored '
                          'question is exact only at step 0 (prompt_exact); the rollout rebuilds every '
                          'later prompt from its own geometry'})
        json.dump(s, open(path[:-len('.parquet')] + '_stats.json', 'w'), indent=2)

    # ---- reference STEPs ----------------------------------------------------------------------
    os.makedirs(a.refs_out, exist_ok=True)
    n_link = n_copy = 0
    for f in os.listdir(a.refs_src):
        if f.endswith('.step') and f[:-5] in want:
            dst = os.path.join(a.refs_out, f)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(a.refs_src, f), dst)
                except OSError:
                    import shutil; shutil.copy(os.path.join(a.refs_src, f), dst)
            n_link += 1
    import shutil
    for f in sorted(os.listdir(a.refs_extra)):
        if f.endswith('.step'):
            dst = os.path.join(a.refs_out, f)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(a.refs_extra, f), dst)
            n_copy += 1
    n_refs = len([f for f in os.listdir(a.refs_out) if f.endswith('.step')])
    summary = {'models': len(want), 'records': len(allrows), 'added_models': len(extra_ids),
               'references': n_refs, 'from_v6': n_link, 'added_references': n_copy,
               'unscorable_models': unscorable,
               'added_status': {s: sum(1 for r in extra if r['status'] == s) for s in {r['status'] for r in extra}},
               'files': {'all': out_all, 'added_only': out_extra, 'references': a.refs_out}}
    json.dump(summary, open(os.path.join(a.v6, 'test7962_build_summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    ok = (len(want) == 7962 and len(extra_ids) == 80 and n_refs == 7950 and len(unscorable) == 12)
    print('RESULT: OK' if ok else 'RESULT: CHECK THE COUNTS ABOVE')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
