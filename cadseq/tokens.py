"""Flat token rows for sequence models: [cmd, 21 args], PAD = -1 where unused.

    cmd  LINE=0 ARC=1 CIRCLE=2 EOS=3 SOL=4 EXT=5 REV=6 FIL=7 CHA=8   (0-5 and the first 16 args = DeepCAD)
    args x y alpha f r | theta phi gamma | px py pz s | e1 e2 b u | ax ay adx ady taper

Sketch rows (SOL, LINE, ARC, CIRCLE) are buffered and committed by the next EXT / REV row.
Sketch coordinates are relative to the loop start point and divided by the profile size s;
(px, py, pz) is that start point in world coordinates.
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence

import numpy as np

from .actions import Action, ActionError, BoolOp, Chamfer, EdgeSelector, Extrude, Fillet, Revolve
from .geometry import Arc, Circle, Line, Loop, Profile, SketchPlane

CMDS = ["LINE", "ARC", "CIRCLE", "EOS", "SOL", "EXT", "REV", "FIL", "CHA"]
LINE, ARC, CIRCLE, EOS, SOL, EXT, REV, FIL, CHA = range(len(CMDS))

ARG_NAMES = ["x", "y", "alpha", "f", "r", "theta", "phi", "gamma", "px", "py", "pz", "s",
             "e1", "e2", "b", "u", "ax", "ay", "adx", "ady", "taper"]
N_ARGS = len(ARG_NAMES)  # 21
N_ARGS_DEEPCAD = 16
COL = {n: i for i, n in enumerate(ARG_NAMES)}
PAD = -1.0
TWO_PI = 2 * math.pi

# Ranges for quantisation (None = categorical).  They assume a normalised sequence (model
# inside [-0.75, 0.75]^3) and leave room for tool bodies larger than the final part.
ARG_RANGES = {
    "x": (-1.0, 1.0), "y": (-1.0, 1.0), "alpha": (0.0, TWO_PI), "f": None, "r": (0.0, 1.0),
    "theta": (0.0, math.pi), "phi": (-math.pi, math.pi), "gamma": (-math.pi, math.pi),
    "px": (-1.5, 1.5), "py": (-1.5, 1.5), "pz": (-1.5, 1.5), "s": (0.0, 3.0),
    "e1": (-2.0, 2.0), "e2": (-2.0, 2.0), "b": None, "u": None,
    "ax": (-2.0, 2.0), "ay": (-2.0, 2.0), "adx": (-1.0, 1.0), "ady": (-1.0, 1.0),
    "taper": (-math.pi / 4, math.pi / 4),
}
N_BINS = 257  # 256 intervals: 0, ±1, π/2, π and 0.75 are exact bins


def _mask(*names):
    m = np.zeros(N_ARGS, dtype=bool)
    m[[COL[n] for n in names]] = True
    return m


PLANE = ("theta", "phi", "gamma", "px", "py", "pz", "s")
CMD_ARGS_MASK = np.stack([
    _mask("x", "y"),                                       # LINE   end point
    _mask("x", "y", "alpha", "f"),                         # ARC    end point, sweep, ccw
    _mask("x", "y", "r"),                                  # CIRCLE center, radius
    _mask(),                                               # EOS
    _mask(),                                               # SOL
    _mask(*PLANE, "e1", "e2", "b", "u", "taper"),          # EXT    e1,e2,u = DeepCAD extent; b = op
    _mask("alpha", *PLANE, "b", "ax", "ay", "adx", "ady"), # REV    alpha = angle; axis point + direction
    _mask("px", "py", "pz", "s", "e1", "u"),               # FIL    probe, within-radius, radius, k
    _mask("px", "py", "pz", "s", "e1", "e2", "u"),         # CHA    probe, within-radius, d1, d2, k
])


def _row(cmd: int, **args) -> np.ndarray:
    r = np.full(1 + N_ARGS, PAD)
    r[0] = cmd
    for k, v in args.items():
        r[1 + COL[k]] = v
    return r


# --------------------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------------------
def _profile_rows(profile: Profile) -> List[np.ndarray]:
    rows = []
    for loop in profile.loops:
        rows.append(_row(SOL))
        for c in loop.curves:
            if isinstance(c, Line):
                rows.append(_row(LINE, x=c.end[0], y=c.end[1]))
            elif isinstance(c, Arc):
                rows.append(_row(ARC, x=c.end[0], y=c.end[1], alpha=c.sweep, f=float(c.ccw)))
            else:
                rows.append(_row(CIRCLE, x=c.center[0], y=c.center[1], r=c.radius))
    return rows


def _selector_args(sel: EdgeSelector) -> dict:
    """probe -> px py pz; k -> u (0 means all); within-radius -> s (0 means nearest)."""
    p = sel.point
    k = 0 if sel.mode == "all" else (1 if sel.mode == "within" else sel.k)
    return dict(px=p[0], py=p[1], pz=p[2], u=k, s=sel.radius if sel.mode == "within" else 0.0)


def _selector_from(g) -> EdgeSelector:
    k, s, point = int(round(g("u"))), g("s"), (g("px"), g("py"), g("pz"))
    if k <= 0:
        return EdgeSelector(point=point, mode="all")
    if s > 0:
        return EdgeSelector(point=point, mode="within", radius=s)
    return EdgeSelector(point=point, mode="nearest", k=k)


def encode(actions: Sequence[Action], normalize_profile: bool = True, canonicalize: bool = True,
           add_eos: bool = True, pad_to: Optional[int] = None) -> np.ndarray:
    rows: List[np.ndarray] = []
    for a in actions:
        if isinstance(a, (Extrude, Revolve)):
            prof = a.profile.canonical() if canonicalize else a.profile
            if normalize_profile:
                sx, sy = prof.start_point
                s = prof.bbox_size
                if s <= 0:
                    raise ActionError("degenerate profile (zero bbox)")
                prof = prof.transformed(translate=(-sx, -sy), scale=1.0 / s)
                origin = a.plane.to_world(sx, sy)
            else:
                sx, sy, s, origin = 0.0, 0.0, 1.0, a.plane.origin
            theta, phi, gamma = a.plane.to_euler()
            plane = dict(theta=theta, phi=phi, gamma=gamma, px=origin[0], py=origin[1], pz=origin[2], s=s)
            rows += _profile_rows(prof)
            if isinstance(a, Extrude):
                u, e1, e2 = a.deepcad_extent()
                rows.append(_row(EXT, e1=e1, e2=e2, b=a.op.index, u=u, taper=math.radians(a.taper), **plane))
            else:
                d = np.subtract(a.axis_end, a.axis_start)
                d = d / np.linalg.norm(d)
                rows.append(_row(REV, alpha=math.radians(a.angle), b=a.op.index, adx=d[0], ady=d[1],
                                 ax=(a.axis_start[0] - sx) / s, ay=(a.axis_start[1] - sy) / s, **plane))
        elif isinstance(a, Fillet):
            rows.append(_row(FIL, e1=a.radius, **_selector_args(a.edges)))
        elif isinstance(a, Chamfer):
            d2 = a.distance if a.distance2 is None else a.distance2
            rows.append(_row(CHA, e1=a.distance, e2=d2, **_selector_args(a.edges)))
        else:
            raise ActionError(f"no token encoding for action kind {a.kind!r}")
    if add_eos:
        rows.append(_row(EOS))
    out = np.stack(rows) if rows else np.zeros((0, 1 + N_ARGS))
    if pad_to is not None:
        if len(out) > pad_to:
            raise ActionError(f"sequence length {len(out)} exceeds pad_to={pad_to}")
        out = np.concatenate([out, np.tile(_row(EOS), (pad_to - len(out), 1))], 0)
    return out


# --------------------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------------------
def decode(rows: np.ndarray, strict: bool = True) -> List[Action]:
    """Rows -> actions.  Sketch rows are buffered until the next EXT / REV commit row.
    strict=False skips malformed fragments instead of raising (use it for model samples)."""
    actions: List[Action] = []
    loops: List[Loop] = []
    cur: Optional[Loop] = None

    def fail(msg):
        if strict:
            raise ActionError(msg)

    for r in np.asarray(rows, dtype=float):
        cmd, a = int(round(r[0])), r[1:]
        g = lambda name: float(a[COL[name]])
        if cmd == EOS:
            break
        if cmd == SOL:
            cur = Loop()
            loops.append(cur)
        elif cmd in (LINE, ARC, CIRCLE):
            if cur is None:
                fail("curve before SOL")
                continue
            if cmd == LINE:
                cur.curves.append(Line((g("x"), g("y"))))
            elif cmd == ARC:
                cur.curves.append(Arc((g("x"), g("y")), g("alpha"), g("f") > 0.5))
            else:
                cur.curves.append(Circle((g("x"), g("y")), g("r")))
        elif cmd in (EXT, REV):
            loops = [l for l in loops if l.curves]
            if not loops:
                fail("EXT/REV without a profile")
                cur = None
                continue
            s = g("s")
            plane = SketchPlane.from_euler((g("px"), g("py"), g("pz")), g("theta"), g("phi"), g("gamma"))
            profile = Profile(loops).transformed(scale=s)
            op = list(BoolOp)[int(round(g("b")))]
            if cmd == EXT:
                fwd, back = Extrude.from_deepcad_extent(int(round(g("u"))), g("e1"), g("e2"))
                taper = 0.0 if g("taper") == PAD else math.degrees(g("taper"))  # PAD: 16-column DeepCAD row
                actions.append(Extrude(plane=plane, profile=profile, forward=fwd, backward=back, op=op, taper=taper))
            else:
                ax, ay = g("ax") * s, g("ay") * s
                actions.append(Revolve(plane=plane, profile=profile, axis_start=(ax, ay),
                                       axis_end=(ax + g("adx") * s, ay + g("ady") * s),
                                       angle=math.degrees(g("alpha")), op=op))
            loops, cur = [], None
        elif cmd == FIL:
            actions.append(Fillet(edges=_selector_from(g), radius=g("e1")))
        elif cmd == CHA:
            d1, d2 = g("e1"), g("e2")
            actions.append(Chamfer(edges=_selector_from(g), distance=d1, distance2=None if abs(d1 - d2) < 1e-12 else d2))
        else:
            fail(f"unknown command id {cmd}")
    if loops and strict:
        raise ActionError("dangling sketch rows without a committing EXT/REV")
    return actions


# --------------------------------------------------------------------------------------
# quantisation (discrete vocabularies)
# --------------------------------------------------------------------------------------
def quantize(rows: np.ndarray, n_bins: int = N_BINS, ranges: Optional[dict] = None,
             on_clip: str = "warn") -> np.ndarray:
    """level = round((v - lo) / (hi - lo) * (n_bins - 1)), clipped to [0, n_bins).
    Categorical columns are kept as integers.  on_clip: 'warn' | 'raise' | 'ignore'."""
    ranges = ranges or ARG_RANGES
    rows = np.asarray(rows, dtype=float)
    cmd = np.rint(rows[:, 0]).astype(int)
    used = CMD_ARGS_MASK[cmd]
    out = np.full(rows.shape, int(PAD), dtype=np.int64)
    out[:, 0] = cmd
    clipped = []
    for j, name in enumerate(ARG_NAMES):
        m = used[:, j]
        if not m.any():
            continue
        v = rows[m, 1 + j]
        if ranges[name] is None:
            q = np.rint(v)
        else:
            lo, hi = ranges[name]
            q = np.rint((v - lo) / (hi - lo) * (n_bins - 1))
            bad = (q < 0) | (q > n_bins - 1)
            clipped += [(int(i), name, float(x)) for i, x in zip(np.flatnonzero(m)[bad], v[bad])]
            q = np.clip(q, 0, n_bins - 1)
        out[m, 1 + j] = q.astype(np.int64)
    if clipped and on_clip != "ignore":
        msg = f"{len(clipped)} value(s) clipped during quantisation: {clipped[:5]}"
        if on_clip == "raise":
            raise ActionError(msg)
        warnings.warn(msg)
    return out


def dequantize(rows: np.ndarray, n_bins: int = N_BINS, ranges: Optional[dict] = None) -> np.ndarray:
    ranges = ranges or ARG_RANGES
    rows = np.asarray(rows)
    cmd = rows[:, 0].astype(int)
    used = CMD_ARGS_MASK[cmd]
    out = np.full(rows.shape, PAD, dtype=float)
    out[:, 0] = cmd
    for j, name in enumerate(ARG_NAMES):
        m = used[:, j]
        if not m.any():
            continue
        v = rows[m, 1 + j].astype(float)
        if ranges[name] is not None:
            lo, hi = ranges[name]
            v = lo + v / (n_bins - 1) * (hi - lo)
        out[m, 1 + j] = v
    return out


# --------------------------------------------------------------------------------------
# DeepCAD's 17-column layout
# --------------------------------------------------------------------------------------
def to_deepcad(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows)
    cmd = rows[:, 0].astype(int)
    if np.any(np.isin(cmd, [REV, FIL, CHA])):
        raise ActionError("sequence uses extension commands; not representable in DeepCAD's layout")
    if np.any(np.abs(rows[cmd == EXT, 1 + COL["taper"]]) > 1e-9):
        raise ActionError("tapered extrusions are not representable in DeepCAD's layout")
    return rows[:, : 1 + N_ARGS_DEEPCAD]


def from_deepcad(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    out = np.concatenate([rows, np.full((len(rows), N_ARGS - N_ARGS_DEEPCAD), PAD)], 1)
    out[out[:, 0].astype(int) == EXT, 1 + COL["taper"]] = 0.0
    return out
