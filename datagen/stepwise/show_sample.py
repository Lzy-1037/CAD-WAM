#!/usr/bin/env python
"""Print one record of a cadseq_stepwise dataset (v5/v6) exactly as the SFT trainer feeds it.

  conda activate verl && cd $REMOTE/project
  python qwen_datagen/show_sample.py                                  # random train record (seed printed)
  python qwen_datagen/show_sample.py --seed 7 --split test            # reproducible pick
  python qwen_datagen/show_sample.py --model-id 00012345 --step 2     # a given record
  python qwen_datagen/show_sample.py --whole-model                    # + every step of that model

What the trainer (sft_server/train_sft_cad_state.py) does with a record:
  INPUT   prompt  = chat template applied to `question` (user turn only); its 16 <|image_pad|> tokens
                    are replaced inside the model by geo_embed(latents[geo_idx]); no loss on these tokens
  OUTPUT  action  = `answer` + eos token -> cross-entropy (the only labelled tokens)
          state   = latents[target_idx]  -> state head, MSE, weight state_loss_weight (lambda)
"""
import argparse
import json
import os
import random

import numpy as np

CODE = '$REMOTE/code'


def vec_summary(v):
    v = np.asarray(v, np.float64)
    if not v.any():
        return 'all zeros (the empty state before the first action)'
    head = ', '.join(f'{x:.4f}' for x in v[:6])
    return (f'dim {v.size} | mean {v.mean():.4f} | std {v.std():.4f} | min {v.min():.3f} | max {v.max():.3f}\n'
            f'  first 6: [{head}, ...]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=f'{CODE}/new_cadseq/dataset/cadseq_stepwise_v6')
    ap.add_argument('--split', default='train', choices=('train', 'validation', 'test'))
    ap.add_argument('--seed', type=int, default=None, help='random pick; default: a fresh seed (printed)')
    ap.add_argument('--model-id', default=None)
    ap.add_argument('--step', type=int, default=None, help='with --model-id (default: random step of it)')
    ap.add_argument('--tokenizer', default='/datasets/anon/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--max-length', type=int, default=5632)
    ap.add_argument('--whole-model', action='store_true', help='also list every step of the chosen model')
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    path = os.path.join(a.data, f'{a.split}.parquet')
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(10 ** 6)
    rng = random.Random(seed)
    if a.model_id is None:                        # pick a record uniformly over all records of the split
        keys = pq.read_table(path, columns=['model_id', 'step_idx']).to_pydict()
        row = rng.randrange(len(keys['model_id']))
        mid, step = keys['model_id'][row], keys['step_idx'][row]
        where = f'record {row} of {len(keys["model_id"])} in {a.split}.parquet, seed {seed}'
    else:
        mid, step, where = a.model_id, a.step, f'model {a.model_id} in {a.split}.parquet'
    recs = pq.read_table(path, filters=[('model_id', '=', mid)]).to_pandas().sort_values('step_idx')
    assert len(recs), f'{mid} is not in {path}'
    if step is None:
        step = int(rng.choice(recs.step_idx.tolist()))
    r = recs[recs.step_idx == step].iloc[0]
    lat = np.load(os.path.join(a.data, 'latents.npy'), mmap_mode='r')
    stats = json.load(open(os.path.join(a.data, f'{a.split}_stats.json')))

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompt = tok.apply_chat_template([{'role': 'user', 'content': r.question}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(r.answer + tok.eos_token, add_special_tokens=False).input_ids
    n_geo = sum(t == tok.convert_tokens_to_ids('<|image_pad|>') for t in p_ids)
    z_t, z_next = np.asarray(lat[r.geo_idx]), np.asarray(lat[r.target_idx])

    bar = '=' * 100
    print(bar)
    print(f'{stats["dataset"]} | {where}')
    print(f'model {mid} | step_idx {step} (actions done so far) of n_steps {r.n_steps} | is_end {bool(r.is_end)} '
          f'| encoder epoch {stats.get("encoder_epoch")} md5 {stats["encoder_md5"][:8]}')
    print(bar)
    print(f'INPUT 1/2  prompt text after the chat template ({len(p_ids)} tokens, {n_geo} geo placeholders, '
          f'labels -100 = no loss):')
    print('-' * 100)
    print(prompt)
    print('-' * 100)
    print(f'INPUT 2/2  z_t = latents[geo_idx={r.geo_idx}] -> geo_embed -> replaces the {n_geo} <|image_pad|> tokens')
    print('  ' + vec_summary(z_t))
    print(bar)
    print(f'OUTPUT 1/2  target action text = answer + {tok.eos_token!r} ({len(a_ids)} tokens, cross-entropy):')
    print('-' * 100)
    print(r.answer + tok.eos_token)
    print('-' * 100)
    same = ' (= z_t: the end record keeps the finished state)' if r.target_idx == r.geo_idx else ''
    print(f'OUTPUT 2/2  z_(t+1) = latents[target_idx={r.target_idx}] -> state head, MSE{same}')
    print('  ' + vec_summary(z_next))
    print(f'  mean squared change z_(t+1) - z_t over the 1024 dims: {np.mean((z_next - z_t) ** 2):.4f}')
    n_all = len(p_ids) + len(a_ids)
    print(bar)
    print(f'sequence {n_all} tokens (max_length {a.max_length}: '
          + ('fits' if n_all <= a.max_length else f'answer cut by {n_all - a.max_length} tokens') + ')')
    if a.whole_model:
        print(bar)
        print(f'all {len(recs)} records of model {mid} (step t: action target | geo_idx -> target_idx)')
        for _, x in recs.iterrows():
            ans = x.answer if len(x.answer) <= 160 else x.answer[:157] + '...'
            mark = '  <- shown above' if x.step_idx == step else ''
            print(f'  step {x.step_idx}: {ans} | {x.geo_idx} -> {x.target_idx}{mark}')


if __name__ == '__main__':
    main()
