#!/usr/bin/env python
"""Geometry worker — runs in a CadQuery env, driven over stdin/stdout with JSON lines.

Kept in a separate process from the LLM for three reasons: CadQuery/OCC need not live in the same
conda env as torch; LLM-generated actions can hang or segfault the OCC kernel, and a worker can be
killed and restarted without losing the rollout; per-request timeouts are enforceable.

The worker is STATELESS: every request carries the full action list so far and is executed from an
empty document, so a crashed worker can simply be replaced (no replay bookkeeping).

Protocol — one JSON object per line in, one per line out:
  in : {"id":.., "cmd":"exec", "actions":[...], "n_points":4096, "frame":"current"|"none",
        "half_extent":0.75, "seed":0, "step_path": optional "/path/out.step"}
  out: {"id":.., "ok":bool, "n_applied":int, "error":str|null, "points_b64":str|null,
        "volume":float, "n_solids":int, "center":[3], "scale":float,
        "bbox":[xmin,ymin,zmin,xmax,ymax,zmax] (model units, the same box the frame is built from)}
  in : {"cmd":"ping"} -> {"ok":true}

`points_b64` is base64 of a float32 [n_points, 3] array; with frame="current" the cloud is centred
on the current solid's exact bbox and scaled so its max half-extent is `half_extent` (the deepcad_10k
convention the SetVAE encoder was trained on).
"""
import base64
import hashlib
import json
import os
import sys

import numpy as np

CADSEQ_ROOT = os.environ.get("CADSEQ_ROOT", "$REMOTE/project")
sys.path.insert(0, CADSEQ_ROOT)


def _sample_mesh(verts, tris, n, rng):
    V = np.asarray([(v.x, v.y, v.z) for v in verts], dtype=np.float64)
    F = np.asarray(tris, dtype=np.int64)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    tot = areas.sum()
    if not np.isfinite(tot) or tot <= 0:
        raise ValueError("zero_area_mesh")
    idx = rng.choice(len(F), size=n, p=areas / tot)
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    return ((1 - r1)[:, None] * v0[idx] + (r1 * (1 - r2))[:, None] * v1[idx]
            + (r1 * r2)[:, None] * v2[idx])


def handle(req, cache):
    from cadseq import Executor
    from cadseq.actions import Action
    from cadseq.executor import CADState

    if req.get("cmd") == "ping":
        return {"ok": True}

    actions = req["actions"]
    out = {"ok": False, "n_applied": 0, "error": None, "points_b64": None,
           "volume": 0.0, "n_solids": 0, "center": [0.0, 0.0, 0.0], "scale": 1.0}
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
    bb = state.bbox()
    out["bbox"] = [float(bb.xmin), float(bb.ymin), float(bb.zmin), float(bb.xmax), float(bb.ymax), float(bb.zmax)]
    if req.get("step_path"):
        os.makedirs(os.path.dirname(req["step_path"]), exist_ok=True)
        state.export(req["step_path"])

    n_points = int(req.get("n_points", 0))
    if n_points:
        lo = np.array([bb.xmin, bb.ymin, bb.zmin]); hi = np.array([bb.xmax, bb.ymax, bb.zmax])
        center, half = (lo + hi) / 2., float((hi - lo).max() / 2.)
        if half < 1e-9:
            out["error"] = "degenerate_bbox"
            return out
        diag = float(np.linalg.norm(hi - lo))
        verts, tris = state.shape.tessellate(max(diag, 1e-6) / 1000.)
        h = hashlib.sha256(f"{req.get('seed', 0)}:{json.dumps(actions, sort_keys=True)}".encode()).digest()
        pts = _sample_mesh(verts, tris, n_points, np.random.default_rng(int.from_bytes(h[:8], "little")))
        if req.get("frame", "current") == "current":
            scale = float(req.get("half_extent", 0.75)) / half
            pts = (pts - center) * scale
            out["center"], out["scale"] = center.tolist(), scale
        pts = pts.astype(np.float32)
        if not np.isfinite(pts).all():
            out["error"] = "nonfinite_points"
            return out
        out["points_b64"] = base64.b64encode(pts.tobytes()).decode("ascii")
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
            resp = {"ok": False, "error": f"worker:{type(e).__name__}:{str(e)[:160]}",
                    "n_applied": 0, "points_b64": None}
        resp["id"] = req.get("id")
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
