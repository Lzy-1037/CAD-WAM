#!/usr/bin/env python
"""Phase 1 — closed-loop rollout of the text-state Qwen3-8B (plain SFT) over the test set.

For every test model:
    state_0 = {"solids": 0}
    repeat:
        prompt   = instruction + CAD description + "Current state:\n" + state_t     (same as training)
        answer   = model.generate(prompt)          -> {"action": A, "state": S_pred}
        stop if A is {"kind": "end"} (or max steps / rejected action / unparsable answer)
        execute actions so far + A with cadseq (geometry worker) -> real solid
        state_{t+1} = text_state.encode_state(real solid)                        (fed back)
        S_pred is compared with state_{t+1} for the world-model diagnostics, never fed back
    export the final solid to <out>/pred_step/<model_id>.step

Runs in the GPU env (torch + transformers, or vllm with --engine vllm); all CadQuery/OCC work happens
in geometry_worker.py subprocesses (`--geom-python` selects their interpreter).

  python rollout.py --model <ckpt> --test dataset/cadseq_stepwise_text/test.parquet --out runs/exp1 \\
      --cadseq-root $REMOTE/text_project \\
      [--geom-python /path/to/cadquery/env/bin/python] [--limit 50] [--oracle] [--engine vllm]

--oracle replaces the LLM with the ground-truth actions (pipeline test + upper-bound row).
Models trained with --action-only-loss answer {"action": ...} only; that is accepted (no state diagnostics).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from progress import Progress  # noqa: E402

EMPTY_STATE = '{"solids":0}'
STATE_MARK = "Current state:\n"
COUNT_LINE = "Actions completed so far: {}\n"   # step-count ablation (text_datagen/text_prompt.py, latent v5 wording)


# ----------------------------------------------------------------------------- geometry workers
class WorkerPool:
    def __init__(self, n, python_exe, cadseq_root, timeout=60.0):
        self.args = (python_exe, cadseq_root)
        self.timeout = timeout
        self.procs = [self._spawn() for _ in range(n)]

    def _spawn(self):
        python_exe, cadseq_root = self.args
        env = dict(os.environ, CADSEQ_ROOT=cadseq_root, PYTHONUNBUFFERED="1")
        return subprocess.Popen([python_exe, os.path.join(HERE, "geometry_worker.py")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, env=env, bufsize=1)

    def _restart(self, i):
        try:
            self.procs[i].kill()
        except Exception:
            pass
        self.procs[i] = self._spawn()

    def _readline(self, slot, timeout):
        box = {}

        def _rd():
            try:
                box["line"] = self.procs[slot].stdout.readline()
            except Exception as exc:
                box["err"] = exc

        t = threading.Thread(target=_rd, daemon=True)
        t.start(); t.join(timeout)
        if t.is_alive():
            return None
        if "err" in box:
            raise box["err"]
        return box.get("line", "")

    def run(self, requests):
        out = [None] * len(requests)
        for base in range(0, len(requests), len(self.procs)):
            chunk = list(range(base, min(base + len(self.procs), len(requests))))
            for slot, ri in enumerate(chunk):
                try:
                    self.procs[slot].stdin.write(json.dumps(dict(requests[ri], id=ri)) + "\n")
                    self.procs[slot].stdin.flush()
                except Exception:
                    self._restart(slot)
                    out[ri] = {"ok": False, "error": "worker_write_failed", "n_applied": 0}
            for slot, ri in enumerate(chunk):
                if out[ri] is not None:
                    continue
                try:
                    line = self._readline(slot, self.timeout)
                    if line is None:
                        self._restart(slot)
                        out[ri] = {"ok": False, "error": f"worker_timeout>{self.timeout:g}s", "n_applied": 0}
                    elif not line:
                        self._restart(slot)
                        out[ri] = {"ok": False, "error": "worker_died", "n_applied": 0}
                    else:
                        out[ri] = json.loads(line)
                except Exception as e:
                    self._restart(slot)
                    out[ri] = {"ok": False, "error": f"worker_read:{type(e).__name__}", "n_applied": 0}
        return out

    def close(self):
        for p in self.procs:
            try:
                p.stdin.close(); p.wait(timeout=5)
            except Exception:
                p.kill()


# ----------------------------------------------------------------------------- parsing
def first_json(text):
    """First balanced JSON object in `text` -> (obj, error)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
    i = t.find("{")
    if i < 0:
        return None, "no_json_found"
    depth, in_str, esc = 0, False, False
    for j in range(i, len(t)):
        c = t[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[i:j + 1]), None
                except Exception as e:
                    return None, f"json_decode:{str(e)[:80]}"
    return None, "unterminated_json"


def parse_answer(text):
    """-> (action dict, predicted state dict or None, error)."""
    obj, err = first_json(text)
    if err:
        return None, None, err
    if isinstance(obj, dict) and "action" in obj:
        act, st = obj["action"], obj.get("state")
        if not isinstance(act, dict):
            return None, None, "action_not_object"
        return act, (st if isinstance(st, dict) else None), None
    if isinstance(obj, dict) and "kind" in obj:          # bare action (action-only models)
        return obj, None, None
    return None, None, "no_action_key"


def state_diag(pred, real_str):
    """Compare a predicted state dict with the real state string -> diagnostics dict."""
    real = json.loads(real_str)
    d = {"exact": json.dumps(pred, separators=(",", ":"), ensure_ascii=False) == real_str,
         "solids_match": pred.get("solids") == real.get("solids"),
         "faces_match": pred.get("faces") == real.get("faces"),
         "bbox_err": float("nan"), "volume_rel_err": float("nan")}
    try:
        rb, pb = real["bbox"], pred["bbox"]
        lo, hi = rb["min"], rb["max"]
        diag = math.dist(lo, hi) or 1.0
        d["bbox_err"] = max(abs(x - y) for x, y in zip(lo + hi, pb["min"] + pb["max"])) / diag
        d["volume_rel_err"] = abs(float(pred["volume"]) - float(real["volume"])) / (abs(float(real["volume"])) or 1.0)
    except Exception:
        pass
    return d


# ----------------------------------------------------------------------------- generation engines
class HFEngine:
    def __init__(self, path, device, dtype, max_new_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16 if dtype == "bf16" else torch.float32, device_map=device).eval()
        self.model.config.use_cache = True
        self.device, self.max_new_tokens = device, max_new_tokens

    def generate(self, prompts):
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


class VLLMEngine:
    def __init__(self, path, dtype, max_new_tokens, max_model_len, gpu_mem):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.tok = AutoTokenizer.from_pretrained(path)
        self.llm = LLM(model=path, dtype="bfloat16" if dtype == "bf16" else "float32",
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_mem, seed=0)
        self.sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def generate(self, prompts):
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for p in prompts]
        outs = self.llm.generate(texts, self.sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned checkpoint dir (or 'gt' with --oracle)")
    ap.add_argument("--test", default="dataset/cadseq_stepwise_text/test.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cadseq-root", default="$REMOTE/text_project",
                    help="directory holding cadseq/ and text_state.py (for the geometry workers)")
    ap.add_argument("--geom-python", default=sys.executable, help="interpreter that has cadquery")
    ap.add_argument("--geom-workers", type=int, default=8)
    ap.add_argument("--geom-timeout", type=float, default=60.0)
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch", type=int, default=16, help="models generated in parallel (hf); vllm batches internally")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=6144,
                    help="answer = action + state; longest seen 3901 tokens (old text), 3724 (v5 sample)")
    ap.add_argument("--max-model-len", type=int, default=16384, help="vllm context length")
    ap.add_argument("--gpu-mem", type=float, default=0.85, help="vllm gpu_memory_utilization")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--oracle", action="store_true", help="use ground-truth actions instead of the LLM")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "pred_step"), exist_ok=True)
    df = pd.read_parquet(a.test)
    # the dataset's stats say whether its prompts carry the step-count line; the rollout then rebuilds it every
    # step from the number of actions actually executed (like cad_eval/rollout.py does for the latent model)
    stats_path = os.path.splitext(a.test)[0] + "_stats.json"
    count = bool((json.load(open(stats_path)).get("prompt_format") or {}).get("count")) if os.path.isfile(stats_path) else False
    count0 = COUNT_LINE.format(0)
    models = []
    for mid, g in df.groupby("model_id", sort=False):
        g = g.sort_values("step_idx")
        q0 = g.iloc[0]["question"]
        k = q0.rfind(STATE_MARK)
        assert k > 0 and q0.endswith(EMPTY_STATE), f"{mid}: unexpected step-0 prompt"
        if count:
            assert q0[: k].endswith(count0), f"{mid}: stats say count line, step-0 prompt has none"
            head = q0[: k - len(count0)]
        else:
            head = q0[: k]
        models.append({"model_id": mid, "head": head,
                       "gt_actions": [json.loads(x) for x in g.loc[~g["is_end"], "action"]],
                       "gt_n_steps": int(g.iloc[0]["n_steps"])})
    def prompt_of(m, n_done, state):
        return m["head"] + (COUNT_LINE.format(n_done) if count else "") + STATE_MARK + state

    by_id = {m["model_id"]: m for m in models}
    bad = [(r.model_id, r.step_idx) for r in df.itertuples()
           if prompt_of(by_id[r.model_id], r.step_idx, r.state) != r.question]
    assert not bad, (f"rollout prompts do not reproduce {len(bad)} of {len(df)} dataset prompts of {a.test}, "
                     f"e.g. {bad[:3]} -- check prompt_format in {stats_path}")
    print(f"prompt format: count line {'ON' if count else 'off'} | all {len(df)} dataset prompts reproduced", flush=True)
    if a.limit:
        models = models[: a.limit]
    n_all = len(models)
    tag = ""
    if a.num_shards > 1:
        assert 0 <= a.shard < a.num_shards
        models = models[a.shard::a.num_shards]
        tag = f"_shard{a.shard}"
    print(f"rollout over {len(models)}{f' of {n_all}' if a.num_shards > 1 else ''} test models"
          f"{f' (shard {a.shard}/{a.num_shards})' if a.num_shards > 1 else ''} | oracle={a.oracle} engine={a.engine}", flush=True)

    pool = WorkerPool(a.geom_workers, a.geom_python, a.cadseq_root, timeout=a.geom_timeout)
    r = pool.run([{"cmd": "ping"}])[0]
    assert r.get("ok"), f"geometry worker failed to start ({a.geom_python}, CADSEQ_ROOT={a.cadseq_root}): {r}"
    assert r.get("state_empty") == EMPTY_STATE, f"text_state mismatch: {r}"
    print(f"geometry workers OK ({a.geom_workers} x {a.geom_python})", flush=True)

    engine = None
    if not a.oracle:
        if a.engine == "hf":
            engine = HFEngine(a.model, a.device, a.dtype, a.max_new_tokens)
        else:
            engine = VLLMEngine(a.model, a.dtype, a.max_new_tokens, a.max_model_len, a.gpu_mem)
        print(f"model {a.model} loaded ({a.engine}, {a.dtype})", flush=True)

    for m in models:
        m.update(actions=[], state=EMPTY_STATE, done=False, status="running", error=None,
                 n_gen=0, gen_texts=[], state_diags=[], n_state_missing=0)
    active = list(models)
    t0 = time.time()
    prog = Progress(len(models), f"rollout{tag}", interval=5.0)
    n_actions = 0
    jf = open(os.path.join(a.out, f"rollout{tag}.jsonl"), "w")

    def retire(finished):
        owners = [m for m in finished if m["actions"]]
        reqs = [{"cmd": "exec", "actions": m["actions"],
                 "step_path": os.path.join(os.path.abspath(a.out), "pred_step", f"{m['model_id']}.step")} for m in owners]
        for m, resp in zip(owners, pool.run(reqs) if reqs else []):
            m["export_ok"] = bool(resp.get("ok"))
            m["volume"] = resp.get("volume", 0.0)
            if not resp.get("ok") and m["status"].startswith("ok"):
                m.update(status="export_failed", error=resp.get("error"))
        for m in finished:
            jf.write(json.dumps({
                "model_id": m["model_id"], "status": m["status"], "error": m["error"],
                "n_actions_pred": len(m["actions"]), "n_actions_gt": m["gt_n_steps"],
                "n_generations": m["n_gen"], "has_step": bool(m.get("export_ok")),
                "volume": m.get("volume", 0.0), "actions": m["actions"],
                "state_diags": m["state_diags"], "n_state_missing": m["n_state_missing"],
                "final_state": m["state"], "raw_generations": m["gen_texts"][:3],
            }) + "\n")
        jf.flush()

    batch_size = a.batch if a.engine == "hf" else max(a.batch, 256)
    while active:
        batch = active[:batch_size]
        # ---- 1) next action
        if a.oracle:
            for m in batch:
                t = len(m["actions"])
                m["_cand"] = m["gt_actions"][t] if t < len(m["gt_actions"]) else {"kind": "end"}
                m["_pred_state"], m["_err"] = None, None
        else:
            texts = engine.generate([prompt_of(m, len(m["actions"]), m["state"]) for m in batch])
            for m, txt in zip(batch, texts):
                m["gen_texts"].append(txt); m["n_gen"] += 1
                m["_cand"], m["_pred_state"], m["_err"] = parse_answer(txt)
        # ---- 2) classify / execute
        reqs, owners = [], []
        for m in batch:
            if m["_err"] is not None:
                m.update(done=True, status="parse_error", error=m["_err"]); continue
            if m["_cand"].get("kind") == "end":
                m.update(done=True, status="ok_end" if m["actions"] else "empty_end"); continue
            if len(m["actions"]) >= a.max_steps:
                m.update(done=True, status="max_steps"); continue
            reqs.append({"cmd": "exec", "actions": m["actions"] + [m["_cand"]]}); owners.append(m)
        if reqs:
            for m, resp in zip(owners, pool.run(reqs)):
                if not resp.get("ok"):
                    err = resp.get("error") or ""
                    m.update(done=True, error=err,
                             status="action_timeout" if err.startswith("worker_timeout") else "action_rejected")
                    continue
                m["actions"].append(m["_cand"]); n_actions += 1
                m["state"] = resp["state"]                                  # the REAL state is fed back
                if not a.oracle:
                    if m["_pred_state"] is None:
                        m["n_state_missing"] += 1
                    else:
                        m["state_diags"].append(state_diag(m["_pred_state"], resp["state"]))
        # ---- 3) retire finished
        just_done = [m for m in batch if m["done"]]
        if just_done:
            retire(just_done)
            for m in just_done:
                active.remove(m)
        st = Counter(m["status"] for m in models if m["done"])
        bad = sum(v for k, v in st.items() if not k.startswith("ok"))
        prog.update(len(models) - len(active), f"{len(active)} left, {n_actions} actions, {st.get('ok_end', 0)} ok, {bad} failed")

    prog.close(f"{n_actions} actions executed")
    jf.close(); pool.close()
    st = Counter(m["status"] for m in models)
    n_step = sum(1 for m in models if m.get("export_ok"))
    summary = {"n_models": len(models), "n_with_step": n_step, "invalidity_ratio": 1 - n_step / len(models),
               "status_counts": dict(st), "oracle": a.oracle, "engine": a.engine, "model": a.model,
               "seconds": time.time() - t0, "shard": a.shard, "num_shards": a.num_shards}
    json.dump(summary, open(os.path.join(a.out, f"rollout_summary{tag}.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"STEP files: {n_step}/{len(models)} -> {a.out}/pred_step/")


if __name__ == "__main__":
    main()
