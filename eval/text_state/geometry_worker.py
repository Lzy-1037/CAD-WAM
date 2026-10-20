#!/usr/bin/env python
"""Geometry worker for the text-state rollout — runs in a CadQuery env, JSON lines over stdin/stdout.

Stateless: every request carries the full action list and is executed from an empty document, so a
crashed / killed worker is simply replaced.  Instead of a point cloud (latent pipeline) it returns
the TEXT STATE of the resulting solid (stepwise_dataset/text_state.encode_state), i.e. exactly the
string the model was trained to see as "Current state".

Protocol — one JSON object per line in, one per line out:
  in : {"id":.., "cmd":"exec", "actions":[...], "step_path": optional "/path/out.step"}
  out: {"id":.., "ok":bool, "n_applied":int, "error":str|null, "state":str|null,
        "volume":float, "n_solids":int}
  in : {"cmd":"ping"} -> {"ok":true}

Needs CADSEQ_ROOT on sys.path with both `cadseq/` and `text_state.py`.
"""
import json
import os
import sys

CADSEQ_ROOT = os.environ.get("CADSEQ_ROOT", "$REMOTE/text_project")
sys.path.insert(0, CADSEQ_ROOT)
sys.path.insert(0, os.path.join(CADSEQ_ROOT, "stepwise_dataset"))   # dev layout of the source repo


def handle(req, cache):
    from cadseq import Executor
    from cadseq.actions import Action
    from cadseq.executor import CADState
    from text_state import encode_state, state_json

    if req.get("cmd") == "ping":
        return {"ok": True, "state_empty": state_json(encode_state(None))}

    actions = req["actions"]
    out = {"ok": False, "n_applied": 0, "error": None, "state": None, "volume": 0.0, "n_solids": 0}
    ex = cache.setdefault("ex", Executor())
    state = CADState()
    for i, a in enumerate(actions):
        try:
            state = ex.step(state, Action.from_dict(a))
        except Exception as e:
            out["error"] = f"action{i}:{type(e).__name__}:{str(e)[:160]}"
            return out
        out["n_applied"] = i + 1
    if state.is_empty:
        out["error"] = "empty_model"
        return out
    out["volume"] = float(state.volume)
    out["n_solids"] = int(state.n_solids)
    if req.get("step_path"):
        os.makedirs(os.path.dirname(req["step_path"]), exist_ok=True)
        state.export(req["step_path"])
    out["state"] = state_json(encode_state(state.shape))       # MAX_FACES cap as in the dataset
    out["ok"] = True
    return out


def main():
    cache = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"id": None, "ok": False, "error": f"bad_request:{e}"}), flush=True)
            continue
        try:
            resp = handle(req, cache)
        except Exception as e:
            resp = {"ok": False, "error": f"worker:{type(e).__name__}:{str(e)[:160]}", "n_applied": 0, "state": None}
        resp["id"] = req.get("id")
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
