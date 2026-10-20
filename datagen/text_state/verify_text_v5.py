#!/usr/bin/env python
"""Verify cadseq_stepwise_text_v5 against its inputs and against independent geometry.

A. Splits    each parquet = the models.tsv models of that split, records step 0..n, is_end on the last
B. Records   FULL coverage: action == programs.jsonl (then end), description == descriptions.jsonl,
             question == build_prompt(description, state[, step count if the stats say count]), answer == {"action":..,"state":next_state} and parses,
             chains: state_0 empty, state[t] == next_state[t-1], end record next_state == state
C. Geometry  every state after >= 1 action: bbox and volume in the JSON == solid_truth.npz (the exact values
             recorded when the SetVAE clouds were generated), within the 5-significant-digit rounding
D. Repro     models shared with the old local cadseq_stepwise_text and with the same program: every state
             equals the old one after rounding to 3 significant digits (old_text_fingerprints.json); exact
             byte differences are reported (OpenCASCADE float noise in the 5th digit, measured 0.2% locally)
E. Tokens    prompt (chat template) + answer + eos lengths with the training tokenizer -> MAX_LENGTH / packing
Ends with RESULT: OK (exit 0) or RESULT: MISMATCH (exit 1).

  conda activate verl && python text_datagen/verify_text_v5.py
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_text_v5 import CODE, DEFAULTS, SPLITS, load_models, load_programs  # noqa: E402
from text_prompt import EMPTY_STATE_JSON, END_ACTION, build_answer, build_prompt, state_fingerprints  # noqa: E402

REL_TOL = 6e-5        # 5 significant digits -> relative rounding error <= 5e-5
ZERO_TOL = 1.1e-6     # text_state writes |x| < 1e-6 * bbox diagonal as 0


def truth_layout(models_tsv):
    """(model_id, step k>=1) -> (split, row) in solid_truth.npz: per split, models by id, steps ascending."""
    pos, count = {}, {s: 0 for s in SPLITS}
    for mid, s, n in load_models(models_tsv):
        for k in range(1, n + 1):
            pos[(mid, k)] = (s, count[s]); count[s] += 1
    return pos


def canon_actions(action_strings):
    acts = [json.loads(x) for x in action_strings]
    return hashlib.sha256(json.dumps(acts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULTS["out"])
    ap.add_argument("--models", default=DEFAULTS["models"])
    ap.add_argument("--programs", default=DEFAULTS["programs"])
    ap.add_argument("--descriptions", default=DEFAULTS["descriptions"])
    ap.add_argument("--truth", default=f"{CODE}/SetVAE/datagen/solid_truth.npz")
    ap.add_argument("--fingerprints", default=os.path.join(HERE, "old_text_fingerprints.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--tokenizer", default="/datasets/anon/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--skip-tokens", action="store_true")
    ap.add_argument("--batch", type=int, default=5000)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    bad = []

    def check(ok, msg):
        print(("  ok   " if ok else "  FAIL ") + msg, flush=True)
        if not ok:
            bad.append(msg)

    models = load_models(a.models, a.only)
    nsteps = {m: n for m, _, n in models}
    progs = {m: [json.dumps(x, separators=(",", ":")) for x in v]
             for m, v in load_programs(a.programs, set(nsteps)).items()}
    desc = {r["model_id"]: r["description"] for r in map(json.loads, open(a.descriptions)) if r["model_id"] in nsteps}
    with np.load(a.truth) as z:
        truth = {k: z[k] for k in z.files}
    tpos = truth_layout(a.models)
    fp = json.load(open(a.fingerprints))["models"]
    stats = {s: json.load(open(os.path.join(a.data, f"{s}_stats.json"))) for s in SPLITS}
    count = bool(stats["test"].get("prompt_format", {}).get("count"))      # step-count ablation dataset
    print(f"prompt format: count line {'ON' if count else 'off'}", flush=True)
    tok = None
    if not a.skip_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)

    tot_records, token_report = 0, {}
    geo_n, geo_fail, geo_worst, geo_examples = 0, 0, 0.0, []
    fp_models = fp_models_skipped = fp_states = fp_mismatch = fp_exact_diff = 0
    for s in SPLITS:
        print(f"== {s}", flush=True)
        want = [m for m, sp, _ in models if sp == s]
        pf = pq.ParquetFile(os.path.join(a.data, f"{s}.parquet"))
        seen, n_rec = [], 0
        err = {"structure": 0, "action": 0, "description": 0, "question": 0, "answer": 0, "parse": 0, "chain": 0}
        cur_mid, cur_states, cur_actions, prev_next, expect_t = None, [], [], None, 0
        lens = []

        def close_model(mid, states, actions):
            nonlocal fp_models, fp_models_skipped, fp_states, fp_mismatch, fp_exact_diff
            if mid is None:
                return
            if len(states) != nsteps[mid] + 1:
                err["structure"] += 1
            f = fp.get(mid)
            if f is not None:
                if f["actions"] != canon_actions(actions):
                    fp_models_skipped += 1
                else:
                    fp_models += 1
                    for k, (h_exact, h_coarse) in enumerate(f["states"]):
                        fp_states += 1
                        if k >= len(states):
                            fp_mismatch += 1; continue
                        e, c = state_fingerprints(states[k])
                        fp_exact_diff += e != h_exact
                        fp_mismatch += c != h_coarse

        cols = ["model_id", "step_idx", "n_steps", "is_end", "deepcad_split", "description", "question", "answer",
                "action", "state", "next_state"]
        for batch in pf.iter_batches(batch_size=a.batch, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d["model_id"])):
                mid, t, n, end = d["model_id"][i], d["step_idx"][i], d["n_steps"][i], d["is_end"][i]
                st, nx, act = d["state"][i], d["next_state"][i], d["action"][i]
                if mid != cur_mid:
                    close_model(cur_mid, cur_states, cur_actions)
                    cur_mid, cur_states, cur_actions, prev_next, expect_t = mid, [], [], None, 0
                    seen.append(mid)
                n_rec += 1
                if mid not in nsteps or t != expect_t or n != nsteps[mid] or end != (t == n) or d["deepcad_split"][i] != s:
                    err["structure"] += 1
                    expect_t = t + 1
                    continue
                expect_t += 1
                cur_states.append(st)
                if not end:
                    cur_actions.append(act)
                err["action"] += act != (progs[mid][t] if t < n else END_ACTION)
                err["description"] += d["description"][i] != desc[mid]
                err["question"] += d["question"][i] != build_prompt(desc[mid], st, t if count else None)
                err["answer"] += d["answer"][i] != build_answer(act, nx)
                err["chain"] += (st != EMPTY_STATE_JSON) if t == 0 else (st != prev_next)
                if end:
                    err["chain"] += nx != st
                prev_next = nx
                try:
                    obj = json.loads(d["answer"][i])
                    assert set(obj) == {"action", "state"}
                except Exception:  # noqa: BLE001
                    err["parse"] += 1
                    continue
                if not end:                                           # C: state after t+1 actions
                    js = obj["state"]
                    ts, row = tpos[(mid, t + 1)]
                    tb, tv = truth[f"{ts}_bbox"][row], float(truth[f"{ts}_volume"][row])
                    sb = np.array(js["bbox"]["min"] + js["bbox"]["max"], np.float64)
                    diag = float(np.linalg.norm(tb[3:] - tb[:3]))
                    e_box = np.abs(sb - tb) - (REL_TOL * np.abs(tb) + ZERO_TOL * diag)
                    e_vol = abs(float(js["volume"]) - tv) - REL_TOL * abs(tv)
                    worst = max(float(e_box.max()) / max(diag, 1e-300), e_vol / max(abs(tv), 1e-300))
                    geo_n += 1
                    geo_worst = max(geo_worst, worst)
                    if (e_box > 0).any() or e_vol > 0:
                        geo_fail += 1
                        if len(geo_examples) < 5:
                            geo_examples.append((mid, t + 1, sb.tolist(), tb.tolist(), js["volume"], tv))
            if tok is not None:
                prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False) for q in d["question"]]
                P = tok(prompts, add_special_tokens=False).input_ids
                A = tok([x + tok.eos_token for x in d["answer"]], add_special_tokens=False).input_ids
                lens += [len(p) + len(q) for p, q in zip(P, A)]
            print(f"  ... {n_rec} records", flush=True)
        close_model(cur_mid, cur_states, cur_actions)
        tot_records += n_rec
        check(seen == want, f"{s}: {len(seen)} models == models.tsv {s} ({len(want)}), same order")
        check(n_rec == stats[s]["records"] == sum(nsteps[m] + 1 for m in want),
              f"{s}: {n_rec} records == stats == sum(n_steps + 1)")
        for k, v in err.items():
            check(v == 0, f"{s}: {k} ({v} bad)")
        if tok is not None:
            L = np.array(lens)
            token_report[s] = {"mean": round(float(L.mean())), "p50": int(np.percentile(L, 50)), "p99": int(np.percentile(L, 99)),
                               "max": int(L.max()), **{f">{x}": int((L > x).sum()) for x in (4096, 8192, 10240, 12288, 16384)}}
            print(f"  tokens: {token_report[s]}", flush=True)

    print("C. geometry vs solid_truth.npz")
    check(geo_fail == 0 and geo_n == sum(n for _, _, n in models),
          f"{geo_n} states: bbox + volume within rounding ({geo_fail} outside; worst excess {geo_worst:.2e})")
    for ex in geo_examples:
        print("     e.g.", ex)
    print("D. reproducibility vs the old cadseq_stepwise_text")
    check(fp_states > 0 or a.only is not None, f"{fp_models} shared models with identical programs compared "
                                              f"({fp_models_skipped} skipped: program differs)")
    check(fp_mismatch == 0, f"{fp_states} states equal to the old dataset at 3 significant digits ({fp_mismatch} differ; "
                            f"{fp_exact_diff} differ in the last digit only = OpenCASCADE float noise)")
    if token_report:
        print("E. token lengths (prompt via chat template + answer + eos)")
        mx = max(v["max"] for v in token_report.values())
        mean = sum(v["mean"] * stats[s]["records"] for s, v in token_report.items()) / tot_records
        print(f"  longest record {mx} tokens, mean {mean:.0f}")
        print(f"  suggested: MAX_LENGTH >= {mx}, PACK_LENGTH >= {max(16384, 1 << math.ceil(math.log2(mx)))}, "
              f"~64 records/step = {64 * mean:.0f} tokens")
        json.dump(token_report, open(os.path.join(a.data, "token_stats.json"), "w"), indent=2)
    print("RESULT: OK" if not bad else f"RESULT: MISMATCH ({len(bad)} checks failed)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
