"""Import DeepCAD / Fusion 360 Gallery JSON (entities + sequence) into a CADSequence.

Each (sketch, profile) referenced by an ExtrudeFeature becomes one Extrude action.  Sketch
curve points are in the sketch's local (x, y) frame.
"""
from __future__ import annotations

import json
import math
import warnings
from typing import Dict, List

import numpy as np

from .actions import BoolOp, Extrude
from .geometry import ANGLE_EPS, Arc, Circle, Line, Loop, Profile, SketchPlane
from .sequence import CADSequence

OPERATIONS = {
    "NewBodyFeatureOperation": BoolOp.NEW,
    "JoinFeatureOperation": BoolOp.JOIN,
    "CutFeatureOperation": BoolOp.CUT,
    "IntersectFeatureOperation": BoolOp.INTERSECT,
}
EXTENT_CODES = {  # DeepCAD's extent_type codes, see Extrude.from_deepcad_extent
    "OneSideFeatureExtentType": 0,
    "SymmetricFeatureExtentType": 1,
    "TwoSidesFeatureExtentType": 2,
}


def _xy(p) -> tuple:
    return (float(p["x"]), float(p["y"]))


def _xyz(p) -> tuple:
    return (float(p["x"]), float(p["y"]), float(p["z"]))


def plane_from_transform(t: Dict) -> SketchPlane:
    return SketchPlane.from_axes(_xyz(t["origin"]), _xyz(t["x_axis"]), _xyz(t["y_axis"]), _xyz(t["z_axis"]))


def _arc_mid(c: Dict, center) -> tuple:
    """DeepCAD's rule (cadlib/curves.py Arc.get_mid_point): rotate reference_vector by
    (start_angle + end_angle) / 2 counter-clockwise in the sketch frame."""
    a = (float(c["start_angle"]) + float(c["end_angle"])) / 2.0
    rx, ry = float(c["reference_vector"]["x"]), float(c["reference_vector"]["y"])
    mx, my = math.cos(a) * rx - math.sin(a) * ry, math.sin(a) * rx + math.cos(a) * ry
    r = float(c["radius"])
    return (center[0] + mx * r, center[1] + my * r)


def _arc_from_json(c: Dict, start, end) -> Arc:
    """Arc travelling from `start` to `end` (already in chain order).

    The radius comes from the JSON and the sweep from chord and radius, so a nearly closed
    arc (start ~ end) keeps its true radius; DeepCAD's mid point only decides the direction
    and whether the short or the long side of the circle is meant."""
    center = _xy(c["center_point"])
    r = float(c["radius"])
    chord = math.dist(start, end)
    if r <= 0 or chord <= 0:
        return Arc.from_center(start, end, center, ccw=float(c.get("normal", {}).get("z", 1.0)) >= 0)
    half = math.asin(max(-1.0, min(1.0, chord / (2.0 * r))))
    short, long_ = 2.0 * half, 2.0 * math.pi - 2.0 * half
    a0 = math.atan2(start[1] - center[1], start[0] - center[0])
    a1 = math.atan2(end[1] - center[1], end[0] - center[0])
    ccw_to_end = (a1 - a0) % (2.0 * math.pi)
    if all(k in c for k in ("start_angle", "end_angle", "reference_vector")):
        m = _arc_mid(c, center)
        ccw_to_mid = (math.atan2(m[1] - center[1], m[0] - center[0]) - a0) % (2.0 * math.pi)
        ccw = ccw_to_mid < ccw_to_end          # the mid point is passed before the end going ccw
    else:
        ccw = float(c.get("normal", {}).get("z", 1.0)) >= 0
    est = ccw_to_end if ccw else 2.0 * math.pi - ccw_to_end
    sweep = short if abs(short - est) <= abs(long_ - est) else long_
    sweep = min(max(sweep, ANGLE_EPS), 2.0 * math.pi - ANGLE_EPS)
    return Arc(end=end, sweep=sweep, ccw=ccw)


def loop_from_json(loop: Dict) -> Loop:
    """Chain the (possibly unordered / reversed) profile curves head-to-tail; drop curves of
    (relatively) zero length."""
    raw = loop["profile_curves"]
    if len(raw) == 1 and raw[0]["type"] == "Circle3D":
        c = raw[0]
        return Loop([Circle(_xy(c["center_point"]), float(c["radius"]))])

    segs = []  # (start, end, curve_json)
    for c in raw:
        if c["type"] == "Circle3D":
            raise ValueError("a Circle3D must be the only curve in its loop")
        segs.append((_xy(c["start_point"]), _xy(c["end_point"]), c))
    pts = np.array([p for s_, e_, _ in segs for p in (s_, e_)])
    size = float(np.max(pts.max(0) - pts.min(0))) or 1.0
    tol = 1e-4 * size  # chaining / zero-length tolerance, relative to the loop size
    segs = [t for t in segs if math.dist(t[0], t[1]) > tol]  # zero-length curves are no-ops
    if len(segs) < 2:
        raise ValueError("loop has fewer than 2 usable curves")

    ordered = [segs.pop(0) + (False,)]  # (start, end, json, flipped)
    while segs:
        cur_end = ordered[-1][1]
        best = None
        for i, (s_, e_, c) in enumerate(segs):
            if math.dist(cur_end, s_) < tol:
                best = (i, False)
                break
            if math.dist(cur_end, e_) < tol:
                best = (i, True)
                break
        if best is None:
            raise ValueError("profile curves do not form a closed chain")
        i, flip = best
        s_, e_, c = segs.pop(i)
        ordered.append((e_, s_, c, True) if flip else (s_, e_, c, False))
    if math.dist(ordered[-1][1], ordered[0][0]) > tol:
        raise ValueError("profile loop is not closed")

    curves = []
    for s_, e_, c, flipped in ordered:
        if c["type"] == "Line3D":
            curves.append(Line(e_))
        elif c["type"] == "Arc3D":
            curves.append(_arc_from_json(c, s_, e_))
        else:
            raise ValueError(f"unsupported curve type {c['type']}")
    return Loop(curves)


def profile_from_json(profile: Dict) -> Profile:
    loops = [loop_from_json(l) for l in profile["loops"]]
    # outer loop first (Fusion marks it with is_outer); nesting is re-derived geometrically anyway
    outer = [l for l, j in zip(loops, profile["loops"]) if j.get("is_outer", False)]
    inner = [l for l, j in zip(loops, profile["loops"]) if not j.get("is_outer", False)]
    return Profile(outer + inner) if outer else Profile(loops)


def _distance(ext: Dict, key: str) -> float:
    d = ext.get(key)
    if not d or "distance" not in d:
        return 0.0
    return float(d["distance"]["value"])


def _taper_deg(ext: Dict) -> float:
    d = ext.get("extent_one", {}).get("taper_angle")
    return math.degrees(float(d["value"])) if d and d.get("value") else 0.0


def from_deepcad_json(data: Dict, split_profiles: bool = True, keep_taper: bool = False) -> CADSequence:
    entities = data["entities"]
    actions: List[Extrude] = []
    for item in data["sequence"]:
        if item["type"] != "ExtrudeFeature":
            continue  # sketches are consumed by the extrude that references them
        ext = entities[item["entity"]]
        if ext.get("start_extent", {}).get("type", "ProfilePlaneStartDefinition") != "ProfilePlaneStartDefinition":
            warnings.warn("start_extent other than ProfilePlaneStartDefinition is treated as the sketch plane")
        op = OPERATIONS[ext["operation"]]
        if not actions:
            op = BoolOp.NEW  # DeepCAD uses the first extrude's tool as the body whatever its operation
        code = EXTENT_CODES[ext["extent_type"]]
        forward, backward = Extrude.from_deepcad_extent(
            code, _distance(ext, "extent_one"), _distance(ext, "extent_two") if code == 2 else 0.0)
        if forward + backward <= 0:
            warnings.warn(f"zero-extent extrude {item['entity']} skipped (a no-op in DeepCAD's reconstruction)")
            continue
        taper = _taper_deg(ext) if keep_taper else 0.0

        refs = ext["profiles"]
        profiles = []
        for ref in refs:
            sk = entities[ref["sketch"]]
            profiles.append((plane_from_transform(sk["transform"]), profile_from_json(sk["profiles"][ref["profile"]])))

        if not split_profiles and len(profiles) > 1:
            # exact Fusion semantics: the tool is the union of all profiles (same sketch plane)
            plane = profiles[0][0]
            loops = [l for _, p in profiles for l in p.loops]
            actions.append(Extrude(plane=plane, profile=Profile(loops), forward=forward, backward=backward,
                                   op=op, taper=taper))
            continue

        for k, (plane, prof) in enumerate(profiles):
            this_op = op
            if k > 0 and op in (BoolOp.NEW, BoolOp.JOIN):
                this_op = BoolOp.JOIN
            elif k > 0 and op == BoolOp.INTERSECT:
                warnings.warn("multi-profile intersect split sequentially; semantics differ from Fusion "
                              "(use split_profiles=False for exact behaviour)")
            actions.append(Extrude(plane=plane, profile=prof, forward=forward, backward=backward,
                                   op=this_op, taper=taper))
    return CADSequence(actions, meta={"source": "deepcad_json"})


def load_deepcad_json(path: str, **kw) -> CADSequence:
    with open(path) as f:
        return from_deepcad_json(json.load(f), **kw)


def to_deepcad_vector(seq: CADSequence, **kw) -> np.ndarray:
    """DeepCAD's (L, 17) command vector: [cmd, x, y, alpha, f, r, theta, phi, gamma,
    px, py, pz, s, e1, e2, b, u] with PAD=-1 in unused slots."""
    from . import tokens
    return tokens.to_deepcad(tokens.encode(seq.actions, **kw))
