#!/usr/bin/env python
"""Build cadseq_stepwise_text_v5 on the server: the text-state counterpart of cadseq_stepwise_v5.

Same models, split (DeepCAD official), descriptions, actions and record order as the latent v5 dataset;
the SetVAE latent is replaced by the JSON B-rep state of text_state.py, the target is the next action
AND the next state (records exactly as stepwise_dataset/build_text_state.py cmd_build):

  question = INSTRUCTION + CAD description + "Current state:\n" + state_t
  answer   = {"action": <action t>, "state": <state_{t+1}>}   ({"action":{"kind":"end"},"state":state_T} at the end)

Two stages, two envs:
  extract  (cadquery env, e.g. cadquery27; CPU)  execute every program step by step, cache the uncapped
           states in <out>/states/shard_XXXX.jsonl (resumable: finished shards are skipped)
  build    (verl env: pandas + pyarrow)          cap face lists (MAX_FACES), render prompts/answers, write
           <out>/{train,validation,test}.parquet + <split>_stats.json

  conda activate cadquery27 && python text_datagen/build_text_v5.py extract --workers 32
  conda activate verl       && python text_datagen/build_text_v5.py build

Step-count ablation (same records + the latent prompt's "Actions completed so far: k" line), from the states
already extracted for cadseq_stepwise_text_v5:
  python text_datagen/build_text_v5.py build --count-line \
      --states dataset/cadseq_stepwise_text_v5/states --out dataset/cadseq_stepwise_text_v5_count
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from text_prompt import (END_ACTION, MAX_FACES, build_answer, build_prompt, cap_state,  # noqa: E402
                         state_json)

CODE = "$REMOTE/code"
SPLITS = ("train", "validation", "test")
DEFAULTS = dict(
    models=f"{CODE}/SetVAE/datagen/models.tsv",
    programs=f"{CODE}/SetVAE/datagen/programs.jsonl",
    cadseq_root=f"{CODE}/SetVAE/datagen",
    descriptions=f"{CODE}/new_cadseq/qwen_datagen/descriptions.jsonl",
    out=f"{CODE}/text_cadseq/dataset/cadseq_stepwise_text_v5",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_models(models_tsv, only=None):
    """[(model_id, split, n_steps)] sorted by model id (the order of every v5 file)."""
    keep = set(open(only).read().split()) if only else None
    rows = []
    for line in sorted(open(models_tsv).read().splitlines()[1:]):
        if line:
            mid, split, n = line.split("\t")
            if keep is None or mid in keep:
                rows.append((mid, split, int(n)))
    return rows


def load_programs(path, wanted):
    """model_id -> list of action dicts."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["model_id"] in wanted:
                out[r["model_id"]] = r["program"]["actions"]
    return out


# ------------------------------------------------------------------------------------ extract
def process_model(job):
    """Same computation as build_text_state.process_model: state_0 = empty, state_k after k actions."""
    mid, actions = job
    try:
        from cadseq import Action, Executor
        from cadseq.executor import CADState
        from text_state import encode_state
        ex, st = Executor(), CADState()
        states = [encode_state(None)]
        for a in actions:
            st = ex.step(st, Action.from_dict(a))
            states.append(encode_state(st.shape, max_faces=10 ** 9))
        return {"model_id": mid, "status": "valid", "states": states}
    except Exception as e:  # noqa: BLE001
        return {"model_id": mid, "status": "failed", "reason": f"{type(e).__name__}:{str(e)[:200]}"}


def cmd_extract(a):
    import multiprocessing as mp
    sys.path.insert(0, a.cadseq_root)
    import cadquery
    import OCP
    import text_state
    models = load_models(a.models, a.only)
    progs = load_programs(a.programs, {m for m, _, _ in models})
    missing = [m for m, _, _ in models if m not in progs]
    assert not missing, f"{len(missing)} models have no program, e.g. {missing[:3]}"
    sdir = os.path.join(a.out, "states")
    os.makedirs(sdir, exist_ok=True)
    meta = {"models": os.path.abspath(a.models), "models_sha256": sha256(a.models),
            "programs_sha256": sha256(a.programs), "only": a.only, "n_models": len(models),
            "models_per_shard": a.models_per_shard, "text_state_sha256": sha256(text_state.__file__),
            "cadquery": cadquery.__version__, "OCP": OCP.__version__, "max_faces_in_cache": "uncapped"}
    mpath = os.path.join(sdir, "meta.json")
    if os.path.isfile(mpath):
        old = json.load(open(mpath))
        for k in ("models_sha256", "programs_sha256", "only", "models_per_shard", "text_state_sha256"):
            assert old.get(k) == meta[k], f"states/ was built with a different {k}; use a new --out or delete states/"
    json.dump(meta, open(mpath, "w"), indent=2)
    shards = [models[i: i + a.models_per_shard] for i in range(0, len(models), a.models_per_shard)]
    print(f"{len(models)} models in {len(shards)} shards | {a.workers} workers | cadquery {cadquery.__version__} "
          f"OCP {OCP.__version__}", flush=True)
    t0, n_ok, n_bad = time.time(), 0, 0
    with mp.Pool(a.workers, maxtasksperchild=500) as pool:
        for si, shard in enumerate(shards):
            path = os.path.join(sdir, f"shard_{si:04d}.jsonl")
            if os.path.isfile(path):
                continue
            recs = pool.map(process_model, [(m, progs[m]) for m, _, _ in shard], chunksize=8)
            with open(path + ".tmp", "w") as f:
                for r in recs:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            os.replace(path + ".tmp", path)                     # a shard exists only when complete
            bad = [r for r in recs if r["status"] != "valid"]
            n_ok += len(recs) - len(bad); n_bad += len(bad)
            for r in bad[:3]:
                print(f"  FAILED {r['model_id']}: {r['reason']}", flush=True)
            el = time.time() - t0
            print(f"  shard {si + 1}/{len(shards)} | valid {n_ok} failed {n_bad} | {el / 60:.1f} min", flush=True)
    print(f"done: {n_ok} valid, {n_bad} failed this run, {(time.time() - t0) / 60:.1f} min -> {sdir}")
    sys.exit(1 if n_bad else 0)


# ------------------------------------------------------------------------------------ build
def cmd_build(a):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    models = load_models(a.models, a.only)
    split_of = {m: s for m, s, _ in models}
    nsteps = {m: n for m, _, n in models}
    progs = load_programs(a.programs, set(split_of))
    desc = {r["model_id"]: r["description"] for r in map(json.loads, open(a.descriptions))}
    sdir = a.states or os.path.join(a.out, "states")
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    assert meta["models_sha256"] == sha256(a.models) and meta["only"] == a.only, "states/ belongs to another model list"
    n_shards = -(-len(models) // meta["models_per_shard"])
    paths = [os.path.join(sdir, f"shard_{i:04d}.jsonl") for i in range(n_shards)]
    absent = [p for p in paths if not os.path.isfile(p)]
    assert not absent, f"{len(absent)} shard(s) not extracted yet, e.g. {absent[0]} -- rerun extract"

    schema = pa.schema([("model_id", pa.string()), ("step_idx", pa.int32()), ("n_steps", pa.int32()),
                        ("is_end", pa.bool_()), ("deepcad_split", pa.string()), ("description", pa.string()),
                        ("question", pa.string()), ("answer", pa.string()), ("action", pa.string()),
                        ("state", pa.string()), ("next_state", pa.string()),
                        ("enable_thinking", pa.bool_()), ("data_source", pa.string())])
    name = os.path.basename(a.out.rstrip("/"))
    writers = {s: pq.ParquetWriter(os.path.join(a.out, f"{s}.parquet"), schema, compression="zstd") for s in SPLITS}
    bufs = {s: {c: [] for c in schema.names} for s in SPLITS}
    acc = {s: {"models": 0, "records": 0, "faces": [], "truncated": 0, "q": [], "ans": [], "st": []} for s in SPLITS}

    def flush(s, force=False):
        if bufs[s]["model_id"] and (force or len(bufs[s]["model_id"]) >= a.rows_per_group):
            writers[s].write_table(pa.Table.from_pydict(bufs[s], schema=schema))
            bufs[s] = {c: [] for c in schema.names}

    t0, seen, failed = time.time(), 0, []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                mid = r["model_id"]
                if r["status"] != "valid":
                    failed.append((mid, r["reason"])); continue
                s, n = split_of[mid], nsteps[mid]
                raw = r["states"]
                assert len(raw) == n + 1 and len(progs[mid]) == n, f"{mid}: {len(raw)} states / {len(progs[mid])} actions / n_steps {n}"
                states = [state_json(cap_state(x, a.max_faces)) for x in raw]
                acts = [json.dumps(x, separators=(",", ":")) for x in progs[mid]] + [END_ACTION]
                b, st = bufs[s], acc[s]
                for x in raw[1:]:
                    st["faces"].append(x["faces"]); st["truncated"] += x["faces"] > a.max_faces
                for t, act in enumerate(acts):
                    cur, nxt = states[t], states[min(t + 1, n)]
                    q, ans = build_prompt(desc[mid], cur, t if a.count_line else None), build_answer(act, nxt)
                    b["model_id"].append(mid); b["step_idx"].append(t); b["n_steps"].append(n)
                    b["is_end"].append(t == n); b["deepcad_split"].append(s); b["description"].append(desc[mid])
                    b["question"].append(q); b["answer"].append(ans); b["action"].append(act)
                    b["state"].append(cur); b["next_state"].append(nxt)
                    b["enable_thinking"].append(False); b["data_source"].append(f"Snitro/Recap-OmniCAD/{name}")
                    st["q"].append(len(q)); st["ans"].append(len(ans)); st["st"].append(len(nxt))
                st["models"] += 1; st["records"] += n + 1; seen += 1
                flush(s)
        print(f"  {os.path.basename(p)} | {seen} models | {(time.time() - t0) / 60:.1f} min", flush=True)
    for s in SPLITS:
        flush(s, force=True); writers[s].close()
    if failed:
        print(f"WARNING: {len(failed)} models failed to execute and are NOT in the dataset, e.g. {failed[:3]}")

    q = lambda v, p: float(np.percentile(v, p))
    for s in SPLITS:
        st = acc[s]
        stats = {"dataset": name, "split": s, "models": st["models"], "records": st["records"],
                 "models_expected": sum(1 for m in split_of.values() if m == s), "max_faces": a.max_faces,
                 "faces_per_state": {"mean": float(np.mean(st["faces"])), "p50": q(st["faces"], 50),
                                     "p95": q(st["faces"], 95), "p99": q(st["faces"], 99), "max": int(max(st["faces"]))},
                 "states_with_truncated_face_list": int(st["truncated"]),
                 "chars": {k: {"mean": float(np.mean(v)), "p95": q(v, 95), "max": int(max(v))}
                           for k, v in (("prompt", st["q"]), ("answer", st["ans"]), ("state", st["st"]))},
                 "failed_models": len(failed), "failed_examples": failed[:20],
                 "prompt_format": {"count": bool(a.count_line)},
                 "prompt": ("stepwise_dataset/build_text_state.py INSTRUCTION + description + "
                            + ("'Actions completed so far: k' + " if a.count_line else "") + "'Current state:'"),
                 "states_dir": os.path.abspath(sdir),
                 "split_source": "DeepCAD official split (SetVAE/datagen/models.tsv)",
                 "frame": "raw model units, same frame as the actions (no normalisation)",
                 "inputs_sha256": {"models.tsv": sha256(a.models), "programs.jsonl": sha256(a.programs),
                                   "descriptions.jsonl": sha256(a.descriptions),
                                   "text_prompt.py": sha256(os.path.join(HERE, "text_prompt.py"))},
                 "states_meta": meta}
        json.dump(stats, open(os.path.join(a.out, f"{s}_stats.json"), "w"), indent=2)
        print(f"{s}: {st['models']} models, {st['records']} records")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}")
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "build"):
        p = sub.add_parser(name)
        p.add_argument("--models", default=DEFAULTS["models"])
        p.add_argument("--programs", default=DEFAULTS["programs"])
        p.add_argument("--out", default=DEFAULTS["out"])
        p.add_argument("--only", default=None, help="file of model ids to restrict to (tests)")
    e = sub.choices["extract"]
    e.add_argument("--cadseq-root", default=DEFAULTS["cadseq_root"], help="dir holding the cadseq package")
    e.add_argument("--workers", type=int, default=32)
    e.add_argument("--models-per-shard", type=int, default=2000)
    b = sub.choices["build"]
    b.add_argument("--descriptions", default=DEFAULTS["descriptions"])
    b.add_argument("--max-faces", type=int, default=MAX_FACES)
    b.add_argument("--rows-per-group", type=int, default=20000)
    b.add_argument("--count-line", action="store_true",
                   help='add "Actions completed so far: k" (the latent v5 step-count line) to every prompt')
    b.add_argument("--states", default=None, help="states cache to build from (default: <out>/states)")
    a = ap.parse_args()
    cmd_extract(a) if a.cmd == "extract" else cmd_build(a)


if __name__ == "__main__":
    main()
