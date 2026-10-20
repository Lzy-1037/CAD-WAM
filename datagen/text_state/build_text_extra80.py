#!/usr/bin/env python
"""Text-state test file for the 80 models that the 7,962-model test set adds to the old 7,882.

  conda activate verl && cd $REMOTE/text_project
  python text_datagen/build_text_extra80.py            # default: the step-count dataset (_v5_count)
  python text_datagen/build_text_extra80.py --dataset dataset/cadseq_stepwise_text_v5   # no count line

Writes <dataset>/test80extra.parquet + test80extra_stats.json (nothing existing is modified), for
  TEST=$PWD/<dataset>/test80extra.parquet PHASE=rollout bash text_cad_eval/run_eval.sh <ckpt> <run>_extra80

The rollout needs only each model's step-0 prompt (description + empty state; every later state comes from
executing the model's own actions), so each model gets exactly one row, step_idx 0, built with this
dataset's own text_prompt.build_prompt and prompt format.  The 80 models are the ones whose ground-truth
program did not reproduce DeepCAD's geometry, so no ground-truth text states exist for later steps; answer
and next_state are left empty (training / oracle only), action holds the first ground-truth action.
Descriptions come from new_cadseq/qwen_datagen/extra80.jsonl (Recap-OmniCAD prompt_exp without <v> tags,
the same rule as every other model).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from text_prompt import EMPTY_STATE_JSON, build_prompt  # noqa: E402

CODE = '$REMOTE/code'
STATE_MARK = 'Current state:\n'
COUNT0 = 'Actions completed so far: 0\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=f'{CODE}/text_cadseq/dataset/cadseq_stepwise_text_v5_count')
    ap.add_argument('--extra', default=f'{CODE}/new_cadseq/qwen_datagen/extra80.jsonl')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = os.path.join(a.dataset, 'test80extra.parquet')
    if os.path.exists(out) and not a.overwrite:
        sys.exit(f'{out} exists -- pass --overwrite to rebuild')
    stats = json.load(open(os.path.join(a.dataset, 'test_stats.json')))
    count = bool((stats.get('prompt_format') or {}).get('count'))
    have = set(pq.read_table(os.path.join(a.dataset, 'test.parquet'), columns=['model_id']).column(0).to_pylist())
    extra = [json.loads(l) for l in open(a.extra)]
    ids = [r['model_id'] for r in extra]
    assert len(ids) == 80 and len(set(ids)) == 80, f'{a.extra}: {len(ids)} records'
    assert not (set(ids) & have), f'{len(set(ids) & have)} of the added models are already in {a.dataset}/test.parquet'

    cols = {c: [] for c in ('model_id', 'step_idx', 'n_steps', 'is_end', 'deepcad_split', 'description', 'question',
                            'answer', 'action', 'state', 'next_state', 'enable_thinking', 'data_source')}
    for r in extra:
        q = build_prompt(r['description'], EMPTY_STATE_JSON, n_done=0 if count else None)
        # exactly what text_cad_eval/rollout.py rebuilds from the step-0 prompt
        k = q.rfind(STATE_MARK)
        head = q[: k - len(COUNT0)] if count else q[: k]
        assert q.endswith(EMPTY_STATE_JSON) and head + (COUNT0 if count else '') + STATE_MARK + EMPTY_STATE_JSON == q
        vals = [r['model_id'], 0, int(r['n_steps']), False, 'test', r['description'], q, '',
                json.dumps(r['actions'][0], separators=(',', ':')), EMPTY_STATE_JSON, '', False,
                f"Snitro/Recap-OmniCAD/{os.path.basename(a.dataset.rstrip('/'))}/test80extra"]
        for c, v in zip(cols, vals):
            cols[c].append(v)
    schema = pa.schema([('model_id', pa.string()), ('step_idx', pa.int32()), ('n_steps', pa.int32()),
                        ('is_end', pa.bool_()), ('deepcad_split', pa.string()), ('description', pa.string()),
                        ('question', pa.string()), ('answer', pa.string()), ('action', pa.string()),
                        ('state', pa.string()), ('next_state', pa.string()),
                        ('enable_thinking', pa.bool_()), ('data_source', pa.string())])
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), out, compression='zstd')
    s = dict(stats)
    s.update({'split': 'test', 'dataset': 'test80extra', 'models': 80, 'records': 80,
              'built_from': os.path.abspath(a.extra),
              'note': 'evaluation only: one step-0 row per model (the rollout rebuilds every later prompt from '
                      'its own geometry); answer / next_state empty, action = first ground-truth action'})
    json.dump(s, open(os.path.splitext(out)[0] + '_stats.json', 'w'), indent=2)
    print(f'80 models, prompt count line {"ON" if count else "off"} -> {out}')
    print('RESULT: OK')


if __name__ == '__main__':
    main()
