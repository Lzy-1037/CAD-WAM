#!/usr/bin/env python
"""Phase 1 — closed-loop rollout of a trained Qwen3ForCADActionState over the test set.

For every test model:
    z_0 = zeros(1024)                                  (empty CAD state)
    repeat:
        action_text = model.generate(prompt, geo_latent=z_t)     # Action Head
        stop if it parses to {"kind": "end"} (or max steps / a rejected action)
        execute the action with cadseq (geometry worker) -> new partial solid
        cloud = 4096 area-weighted surface points of that solid, normalised to the
                deepcad_10k frame (centred, max half-extent 0.75)
        z_{t+1} = SetVAE encoder(cloud)                 # pc_latent_encoder, deterministic
    export the final solid to <out>/pred_step/<model_id>.step

The predicted next state from the model's State Head is recorded for diagnostics but, by design,
is NOT fed back — the loop is closed through the real executor + encoder (the same protocol the
dataset was built with).

Runs in the GPU env (torch + transformers); all CadQuery/OCC work happens in geometry_worker.py
subprocesses (`--geom-python` selects their interpreter, default: this one).

  python rollout.py --model <ckpt> --test dataset/cadseq_stepwise/test.parquet --out runs/exp1 \\
      [--geom-python /path/to/cadquery/env/bin/python] [--limit 50] [--oracle]

--oracle replaces the LLM with the ground-truth actions: it validates the whole pipeline and gives
the upper-bound row of the report ("perfect action prediction").
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from progress import Progress  # noqa: E402


# ----------------------------------------------------------------------------- geometry workers
class WorkerPool:
    def __init__(self, n, python_exe, cadseq_root, timeout=60.0):
        self.args = (python_exe, cadseq_root, timeout)
        self.timeout = timeout
        self.procs = [self._spawn() for _ in range(n)]

    def _spawn(self):
        python_exe, cadseq_root, _ = self.args
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
        """readline() with a deadline. Returns the line, "" if the worker died, or None on timeout.

        readline() on a pipe cannot be interrupted, so it runs on a daemon thread; on timeout the
        worker is killed, which makes that thread's readline return and the thread exit.
        """
        box = {}

        def _rd():
            try:
                box["line"] = self.procs[slot].stdout.readline()
            except Exception as exc:
                box["err"] = exc

        t = threading.Thread(target=_rd, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None
        if "err" in box:
            raise box["err"]
        return box.get("line", "")

    def run(self, requests):
        """requests: list of dicts. Returns responses in the same order (one per worker round)."""
        out = [None] * len(requests)
        for base in range(0, len(requests), len(self.procs)):
            chunk = list(range(base, min(base + len(self.procs), len(requests))))
            for slot, ri in enumerate(chunk):
                req = dict(requests[ri], id=ri)
                try:
                    self.procs[slot].stdin.write(json.dumps(req) + "\n")
                    self.procs[slot].stdin.flush()
                except Exception:
                    self._restart(slot)
                    out[ri] = {"ok": False, "error": "worker_write_failed", "n_applied": 0}
            for slot, ri in enumerate(chunk):
                if out[ri] is not None:
                    continue
                try:
                    line = self._readline(slot, self.timeout)
                    if line is None:          # OCC is spinning on this action: drop the worker
                        self._restart(slot)
                        out[ri] = {"ok": False, "error": f"worker_timeout>{self.timeout:g}s", "n_applied": 0}
                    elif not line:            # worker died (segfault / exit)
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


def decode_points(b64, n_points):
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(n_points, 3).copy()


def parse_action(text):
    """Extract the first JSON object from the model's answer. Returns (obj, error)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        t = t[t.find("{"):] if "{" in t else t
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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained checkpoint dir (or 'gt' with --oracle)")
    ap.add_argument("--test", default="dataset/cadseq_stepwise_v3/test.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="$REMOTE/code/pc_latent_encoder",
                    help="pc_latent_encoder project dir")
    ap.add_argument("--encoder-ckpt", default=None,
                    help="SetVAE checkpoint; default = the encoder package's own DEFAULT_CKPT. "
                         "MUST be the checkpoint the training latents were built with, or the "
                         "geo_latent distribution at eval time will not match training.")
    ap.add_argument("--cadseq-root", default="$REMOTE/project")
    ap.add_argument("--geom-python", default=sys.executable, help="interpreter that has cadquery")
    ap.add_argument("--geom-workers", type=int, default=8)
    ap.add_argument("--geom-timeout", type=float, default=60.0,
                    help="seconds allowed per geometry request before the worker is killed and the "
                         "model is failed (guards against OCC hanging on a generated action)")
    ap.add_argument("--batch", type=int, default=16, help="models generated in parallel")
    ap.add_argument("--max-steps", type=int, default=20,
                    help="max actions per model before giving up (ground truth never exceeds 10)")
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="per action; the longest ground-truth action is 1705 tokens in the v5 test set "
                         "(299 exceed 512, 35 in v3cb) -- 512 used to fail those models by truncation")
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--geo-frame", choices=["current", "zeros", "predicted"], default="current",
                    help="'current': encode each partial solid in its own bbox frame (causal, default); "
                         "'zeros': never feed geometry back (ablation); "
                         "'predicted': feed the State Head's own z_hat back instead of the encoded solid "
                         "(open-loop / imagination ablation -- actions are still executed for the STEP)")
    ap.add_argument("--allow-bbox-feedback", action="store_true",
                    help="with --geo-frame predicted: permit the bbox prompt line, which still comes from "
                         "the executed solid (otherwise the run refuses to start)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0, help="this process handles models[shard::num_shards]")
    ap.add_argument("--num-shards", type=int, default=1, help="data-parallel rollout across GPUs")
    ap.add_argument("--oracle", action="store_true", help="use ground-truth actions instead of the LLM")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "pred_step"), exist_ok=True)
    df = pd.read_parquet(a.test)
    # the dataset's own stats say which progress lines its prompts carry (cad_prompt.py); the rollout
    # adds exactly the same ones, rebuilt every step from its own actions and geometry
    stats_path = os.path.splitext(a.test)[0] + "_stats.json"
    prompt_format = ((json.load(open(stats_path)).get("prompt_format") or {})
                     if os.path.isfile(stats_path) else {})
    dynamic_prompt = any(prompt_format.values())
    if dynamic_prompt:
        from cad_prompt import build_prompt
        assert "description" in df.columns, f"{a.test}: prompt_format {prompt_format} needs a 'description' column"
    models = []
    for mid, g in df.groupby("model_id", sort=False):
        g = g.sort_values("step_idx")
        models.append({"model_id": mid,
                       # v4 stores question/answer only (no messages column); v2-v3cb have both
                       "prompt": (g.iloc[0]["question"] if "question" in g.columns
                                  else g.iloc[0]["messages"][0]["content"]),
                       "description": g.iloc[0]["description"] if "description" in g.columns else None,
                       "gt_actions": [json.loads(r) for r in g[~g["is_end"]]["answer"]],
                       "gt_n_steps": int(g.iloc[0]["n_steps"])})
    if dynamic_prompt:   # cad_prompt.py must reproduce the dataset's own step-0 prompts exactly
        bad = [m["model_id"] for m in models if build_prompt(m["description"], prompt_format, 0, None) != m["prompt"]]
        assert not bad, (f"cad_eval/cad_prompt.py does not reproduce the step-0 prompts of {a.test} "
                         f"({len(bad)} models, e.g. {bad[:3]}) -- sync it with the dataset builder")
    if a.limit:
        models = models[: a.limit]
    n_all = len(models)
    tag = ""
    if a.num_shards > 1:
        assert 0 <= a.shard < a.num_shards, "--shard must be in [0, --num-shards)"
        models = models[a.shard::a.num_shards]
        tag = f"_shard{a.shard}"
    print(f"rollout over {len(models)}"
          f"{f' of {n_all}' if a.num_shards > 1 else ''} test models"
          f"{f' (shard {a.shard}/{a.num_shards})' if a.num_shards > 1 else ''}"
          f" | oracle={a.oracle} geo_frame={a.geo_frame} prompt_format={prompt_format or 'none'}", flush=True)

    if a.geo_frame == "predicted":
        assert not a.oracle, "--geo-frame predicted needs the model's State Head; it cannot run with --oracle"
        assert not prompt_format.get("bbox") or a.allow_bbox_feedback, (
            "prompt_format has the bbox line, which is read from the EXECUTED solid: that would still feed "
            "real geometry back. Re-run with --allow-bbox-feedback to accept this, or use a dataset whose "
            "prompt has no bbox line.")

    pool = WorkerPool(a.geom_workers, a.geom_python, a.cadseq_root, timeout=a.geom_timeout)
    r = pool.run([{"cmd": "ping"}])[0]
    assert r.get("ok"), f"geometry worker failed to start ({a.geom_python}): {r}"
    print(f"geometry workers OK ({a.geom_workers} x {a.geom_python})", flush=True)

    tok = model = enc = None
    if not a.oracle:
        import torch
        from transformers import AutoTokenizer
        sys.path.insert(0, a.cadseq_root)
        sys.path.insert(0, a.encoder)
        from cadwam import cad_state_classes   # Qwen3 or Qwen2.5 backbone, from config.json
        from encoder import SetVAEEncoder
        tok = AutoTokenizer.from_pretrained(a.model)
        tok.padding_side = "left"                      # required for batched generate
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
        model = cad_state_classes(a.model)[1].from_pretrained(a.model, dtype=dtype, device_map=a.device).eval()
        model.config.use_cache = True
        from encoder import DEFAULT_CKPT
        enc_ckpt = a.encoder_ckpt or os.path.join(a.encoder, DEFAULT_CKPT)
        enc = SetVAEEncoder(enc_ckpt, device=a.device)
        print(f"model   {a.model}", flush=True)
        print(f"encoder {enc_ckpt} (SetVAE epoch {enc.ckpt_epoch}, latent {enc.latent_dim}) "
              f"on {a.device} ({a.dtype})", flush=True)
        torch.manual_seed(a.seed)

    # ---- rollout state per model
    for m in models:
        m.update(actions=[], z=np.zeros(1024, np.float32), done=False, status="running",
                 error=None, n_gen=0, pred_states=[], gen_texts=[], bbox=None)
    active = [m for m in models]
    t0 = time.time()
    rounds = 0
    prog = Progress(len(models), f"rollout{tag}", interval=5.0)
    n_actions = 0
    # records are written the moment a model finishes, so a crash/hang costs only the in-flight
    # models instead of the whole shard
    jf = open(os.path.join(a.out, f"rollout{tag}.jsonl"), "w")

    def retire(finished):
        """Export the STEP of each finished model and append its record."""
        reqs = [{"cmd": "exec", "actions": m["actions"], "n_points": 0,
                 "step_path": os.path.join(os.path.abspath(a.out), "pred_step", f"{m['model_id']}.step")}
                for m in finished if m["actions"]]
        owners = [m for m in finished if m["actions"]]
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
                "pred_next_states": m["pred_states"] if not a.oracle else [],
                "raw_generations": m["gen_texts"][:3],
            }) + "\n")
        jf.flush()

    def prompt_of(m):
        """Prompt for m's NEXT action: fixed for v2/v3, rebuilt from the progress so far otherwise."""
        if not dynamic_prompt:
            return m["prompt"]
        return build_prompt(m["description"], prompt_format, len(m["actions"]), m["bbox"])

    while active:
        batch = active[: a.batch]
        # ---- 1) next action for every active model
        if a.oracle:
            for m in batch:
                t = len(m["actions"])
                m["_cand"] = m["gt_actions"][t] if t < len(m["gt_actions"]) else {"kind": "end"}
                m["_err"] = None
        else:
            import torch
            prompts = [tok.apply_chat_template([{"role": "user", "content": prompt_of(m)}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for m in batch]
            enc_in = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
            z = torch.tensor(np.stack([m["z"] for m in batch]), dtype=torch.float32, device=a.device)
            with torch.no_grad():
                gen = model.generate(input_ids=enc_in.input_ids, attention_mask=enc_in.attention_mask,
                                     geo_latent=z, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                out = model(input_ids=enc_in.input_ids, attention_mask=enc_in.attention_mask, geo_latent=z)
            new = gen[:, enc_in.input_ids.shape[1]:]
            texts = tok.batch_decode(new, skip_special_tokens=True)
            preds = out.pred_next_state.float().cpu().numpy()
            for m, txt, pz in zip(batch, texts, preds):
                m["gen_texts"].append(txt)
                m["pred_states"].append(pz.tolist())
                m["n_gen"] += 1
                m["_cand"], m["_err"] = parse_action(txt)

        # ---- 2) classify / execute
        exec_reqs, exec_owner = [], []
        for m in batch:
            if m["_err"] is not None:
                m.update(done=True, status="parse_error", error=m["_err"]); continue
            if m["_cand"].get("kind") == "end":
                m.update(done=True, status="ok_end" if m["actions"] else "empty_end"); continue
            if len(m["actions"]) >= a.max_steps:
                m.update(done=True, status="max_steps"); continue
            exec_reqs.append({"cmd": "exec", "actions": m["actions"] + [m["_cand"]],
                              "n_points": 0 if a.geo_frame in ("zeros", "predicted") else a.n_points,
                              "frame": "current", "seed": a.seed})
            exec_owner.append(m)
        if exec_reqs:
            for m, resp in zip(exec_owner, pool.run(exec_reqs)):
                if not resp.get("ok"):
                    err = resp.get("error") or ""
                    m.update(done=True, error=err,
                             status="action_timeout" if err.startswith("worker_timeout") else "action_rejected")
                    continue
                m["actions"].append(m["_cand"])
                m["bbox"] = resp.get("bbox")
                if prompt_format.get("bbox") and m["bbox"] is None:
                    raise RuntimeError("geometry worker sent no bbox -- sync cad_eval/geometry_worker.py")
                n_actions += 1
                if a.geo_frame == "zeros":
                    m["z"] = np.zeros(1024, np.float32)
                elif a.geo_frame == "predicted":
                    # the State Head's prediction for this step (recorded just above), not the real solid
                    m["z"] = np.asarray(m["pred_states"][-1], np.float32)
                else:
                    pts = decode_points(resp["points_b64"], a.n_points)
                    m["z"] = enc.encode(pts).astype(np.float32) if enc is not None else np.zeros(1024, np.float32)

        # ---- 3) retire finished models (export + record immediately)
        just_done = [m for m in batch if m["done"]]
        if just_done:
            retire(just_done)
            for m in just_done:
                active.remove(m)
        rounds += 1
        from collections import Counter as _C
        st = _C(m["status"] for m in models if m["done"])
        bad = sum(v for k, v in st.items() if not k.startswith("ok"))
        prog.update(len(models) - len(active),
                    f"{len(active)} left, {n_actions} actions, {st.get('ok_end', 0)} ok, {bad} failed")

    prog.close(f"{n_actions} actions executed")
    jf.close()
    pool.close()
    print(f"rollout finished in {time.time()-t0:.0f}s", flush=True)
    from collections import Counter
    st = Counter(m["status"] for m in models)
    n_step = sum(1 for m in models if m.get("export_ok"))
    summary = {"encoder_ckpt": (enc_ckpt if not a.oracle else None),
               "encoder_epoch": (enc.ckpt_epoch if not a.oracle else None),
               "n_models": len(models), "n_with_step": n_step,
               "invalidity_ratio": 1 - n_step / len(models), "status_counts": dict(st),
               "oracle": a.oracle, "geo_frame": a.geo_frame, "prompt_format": prompt_format, "model": a.model,
               "seconds": time.time() - t0}
    summary["shard"], summary["num_shards"] = a.shard, a.num_shards
    json.dump(summary, open(os.path.join(a.out, f"rollout_summary{tag}.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"STEP files: {n_step}/{len(models)} -> {a.out}/pred_step/")


if __name__ == "__main__":
    main()
