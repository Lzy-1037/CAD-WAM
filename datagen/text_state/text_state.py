"""Text (JSON) state representation of a cadseq CADState — ablation counterpart of the 1024-d
SetVAE latent state used by cadseq_stepwise.

The state is a deterministic B-rep summary of the *current solid only* (no action history), in
the same coordinate frame and units as the cadseq actions, so the model can relate state
numbers to action numbers directly:

    {"solids": 0}                                       # empty model (before the first action)

    {"solids": 1,
     "bbox": {"min": [x, y, z], "max": [x, y, z]},
     "volume": v, "area": a,
     "faces": N, "edges": M, "vertices": K,
     "face_list": [                                     # largest faces first (max MAX_FACES)
        {"type": "plane",    "center": [..], "normal": [..], "area": a},
        {"type": "cylinder", "center": [..], "axis": [..],  "radius": r, "convex": true, "area": a},
        {"type": "sphere",   "center": [..], "radius": r, "convex": .., "area": a},
        {"type": "cone",     "center": [..], "axis": [..], "radius": r, "half_angle": deg, "convex": .., "area": a},
        {"type": "torus",    "center": [..], "axis": [..], "radius": R, "minor_radius": r, "area": a},
        {"type": "bspline"|..., "center": [..], "area": a}]}

* center      = centre of mass of the face (world coordinates)
* normal      = outward unit normal of a planar face
* axis        = unit direction of the cylinder / cone / torus axis, sign-canonicalised
                (first non-zero component positive); "center" of a cylinder is the face centroid
* convex      = the curved face bulges outwards (boss); false = concave (hole / inner wall)
* every number has SIG significant digits; |x| < ZERO_REL * bbox_diag is written as 0
* face_list is sorted by (area desc, center, normal) after rounding -> unique for a geometry;
  faces beyond MAX_FACES are dropped from the list ("faces" keeps the true count)
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp_Face
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt, gp_Vec

SIG = 5            # significant digits per number
ZERO_REL = 1e-6    # |x| < ZERO_REL * bbox diagonal -> 0
MAX_FACES = 40     # longest face_list kept in the JSON
STATE_VERSION = 1


def _sig(v: float, tol: float = 0.0) -> float:
    if not math.isfinite(v) or abs(v) < tol:
        return 0.0
    r = float(f"{v:.{SIG}g}")
    return 0.0 if r == 0 else r


def _vec(v, tol) -> List[float]:
    return [_sig(float(c), tol) for c in (v.x, v.y, v.z)] if hasattr(v, "x") else [_sig(float(c), tol) for c in v]


def _canon_dir(d) -> List[float]:
    """Unit direction with the first non-zero component positive (axis lines have no sign)."""
    v = [float(d.X()), float(d.Y()), float(d.Z())]
    for c in v:
        if abs(c) > 1e-9:
            if c < 0:
                v = [-x for x in v]
            break
    return v


def _face_record(face: cq.Face, tol: float) -> Dict:
    kind = face.geomType().lower()
    c = face.Center()
    rec: Dict = {"type": kind, "center": _vec(c, tol)}
    ad = BRepAdaptor_Surface(face.wrapped)

    # oriented normal + a point on the surface, evaluated at the middle of the face's UV range
    # (the centroid of a full cylinder lies on its axis, where a projection is undefined)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face.wrapped)
    pnt, vn = gp_Pnt(), gp_Vec()
    BRepGProp_Face(face.wrapped).Normal((u0 + u1) / 2, (v0 + v1) / 2, pnt, vn)  # respects face orientation
    normal = (vn.X(), vn.Y(), vn.Z())
    p = (pnt.X(), pnt.Y(), pnt.Z())

    def convex(radial) -> bool:
        return sum(normal[i] * radial[i] for i in range(3)) >= 0

    def radial_to_axis(loc, direction):
        d = [float(x) for x in direction]
        q = [p[i] - loc[i] for i in range(3)]
        t = sum(q[i] * d[i] for i in range(3))
        return [q[i] - t * d[i] for i in range(3)]

    if kind == "plane":
        rec["normal"] = [_sig(x) for x in normal]
    elif kind == "cylinder":
        cyl = ad.Cylinder()
        axis = _canon_dir(cyl.Axis().Direction())
        loc = cyl.Location()
        rec["axis"] = [_sig(x) for x in axis]
        rec["radius"] = _sig(cyl.Radius(), tol)
        rec["convex"] = convex(radial_to_axis((loc.X(), loc.Y(), loc.Z()), axis))
    elif kind == "cone":
        cone = ad.Cone()
        axis = _canon_dir(cone.Axis().Direction())
        loc = cone.Location()
        rec["axis"] = [_sig(x) for x in axis]
        rec["radius"] = _sig(cone.RefRadius(), tol)
        rec["half_angle"] = _sig(math.degrees(cone.SemiAngle()))
        rec["convex"] = convex(radial_to_axis((loc.X(), loc.Y(), loc.Z()), axis))
    elif kind == "sphere":
        sph = ad.Sphere()
        loc = sph.Location()
        rec["center"] = _vec((loc.X(), loc.Y(), loc.Z()), tol)  # the sphere centre is more useful than the centroid
        rec["radius"] = _sig(sph.Radius(), tol)
        rec["convex"] = convex([p[i] - (loc.X(), loc.Y(), loc.Z())[i] for i in range(3)])
    elif kind == "torus":
        tor = ad.Torus()
        rec["axis"] = [_sig(x) for x in _canon_dir(tor.Axis().Direction())]
        rec["radius"] = _sig(tor.MajorRadius(), tol)
        rec["minor_radius"] = _sig(tor.MinorRadius(), tol)
    rec["area"] = _sig(face.Area(), tol * tol)
    return rec


def _sort_key(f: Dict):
    return (-f["area"], tuple(f["center"]), f["type"], tuple(f.get("normal", f.get("axis", ()))))


def encode_state(shape: Optional[cq.Shape], max_faces: int = MAX_FACES) -> Dict:
    """JSON-ready dict describing `shape` (None = empty model)."""
    if shape is None:
        return {"solids": 0}
    bb = shape.BoundingBox()
    diag = math.dist((bb.xmin, bb.ymin, bb.zmin), (bb.xmax, bb.ymax, bb.zmax))
    tol = ZERO_REL * diag
    faces = shape.Faces()
    recs = sorted((_face_record(f, tol) for f in faces), key=_sort_key)
    return {
        "solids": len(shape.Solids()),
        "bbox": {"min": _vec((bb.xmin, bb.ymin, bb.zmin), tol), "max": _vec((bb.xmax, bb.ymax, bb.zmax), tol)},
        "volume": _sig(shape.Volume(), tol ** 3),
        "area": _sig(shape.Area(), tol ** 2),
        "faces": len(faces),
        "edges": len(shape.Edges()),
        "vertices": len(shape.Vertices()),
        "face_list": recs[:max_faces],
    }


def state_json(state: Dict) -> str:
    """Compact, key-order-preserving serialisation (the exact string used in the dataset)."""
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


EMPTY_STATE = {"solids": 0}
