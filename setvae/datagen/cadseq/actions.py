"""Actions: Extrude, Revolve, Fillet, Chamfer.

Each action maps a solid (None before the first action) to a solid, or raises ActionError.
Dataclass fields define the JSON schema; `length()` marks fields scaled by `transformed`.
"""
from __future__ import annotations

import math
import typing
from dataclasses import MISSING, dataclass, field, fields, replace
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Tuple

import cadquery as cq

from .geometry import Profile, SketchPlane, Vec2, Vec3, _v2, _v3


class ActionError(Exception):
    """The action cannot be applied: bad parameters, kernel failure, or the result is not a
    valid solid.  The executor guarantees the state is unchanged when this is raised."""


class BoolOp(str, Enum):
    """How the tool body is combined with the model.  Order = DeepCAD's integer codes."""
    NEW = "new"              # DeepCAD "NewBody": behaves exactly like join
    JOIN = "join"            # model ∪ tool
    CUT = "cut"              # model − tool
    INTERSECT = "intersect"  # model ∩ tool

    @property
    def index(self) -> int:
        return list(BoolOp).index(self)


def length(default):
    """Declare a dataclass field whose value is a length (scaled by `transformed`)."""
    return field(default=default, metadata={"unit": "length"})


# --------------------------------------------------------------------------------------
# generic dataclass <-> JSON and scaling
# --------------------------------------------------------------------------------------
def to_plain(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def from_plain(hint, value):
    """Rebuild a field value from JSON using the field's type annotation."""
    if value is None:
        return None
    if hasattr(hint, "from_dict"):
        return hint.from_dict(value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint(value)
    origin = typing.get_origin(hint)
    if origin is typing.Union:  # Optional[T]
        inner = [a for a in typing.get_args(hint) if a is not type(None)][0]
        return from_plain(inner, value)
    if origin is tuple:
        return tuple(float(x) for x in value)
    if hint in (float, int, bool):
        return hint(value)
    return value


def build(cls, data: Dict, context: str = ""):
    """Instantiate dataclass `cls` from a JSON dict with clear errors for missing fields."""
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = from_plain(hints[f.name], data[f.name])
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ActionError(f"{context or cls.__name__}: missing required field {f.name!r}")
    return cls(**kwargs)


def scaled(value, translate, scale):
    """Scale one field value: nested objects know how to transform themselves; numeric
    lengths and length tuples are multiplied.  Profile coordinates are sketch-local, so a
    world translation does not move them."""
    if isinstance(value, Profile):
        return value.scaled(scale)
    if hasattr(value, "transformed"):
        return value.transformed(translate, scale)
    if isinstance(value, tuple):
        return tuple(x * scale for x in value)
    return None if value is None else value * scale


# --------------------------------------------------------------------------------------
# base class + registry
# --------------------------------------------------------------------------------------
ACTION_REGISTRY: Dict[str, type] = {}


@dataclass
class Action:
    kind: ClassVar[str] = "abstract"

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.kind != "abstract":
            ACTION_REGISTRY[cls.kind] = cls

    # -- contract (override) --------------------------------------------------------------
    def validate(self) -> None:
        """Cheap parameter checks; raise ActionError."""

    def apply(self, body: Optional[cq.Shape]) -> cq.Shape:
        raise NotImplementedError

    # -- derived from the fields ---------------------------------------------------------
    def to_dict(self) -> Dict:
        return {"kind": self.kind, **{f.name: to_plain(getattr(self, f.name)) for f in fields(self)}}

    @staticmethod
    def from_dict(d: Dict) -> "Action":
        kind = d.get("kind")
        if kind not in ACTION_REGISTRY:
            raise ActionError(f"unknown action kind {kind!r}; known: {sorted(ACTION_REGISTRY)}")
        return build(ACTION_REGISTRY[kind], d, context=kind)

    def transformed(self, translate=(0.0, 0.0, 0.0), scale: float = 1.0) -> "Action":
        """Rigid translation followed by uniform scaling of everything the action holds."""
        changes = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if hasattr(v, "transformed") or hasattr(v, "scaled") or f.metadata.get("unit") == "length":
                changes[f.name] = scaled(v, translate, scale)
        return replace(self, **changes)


# --------------------------------------------------------------------------------------
# boolean combination shared by the tool-body actions
# --------------------------------------------------------------------------------------
def combine(body: Optional[cq.Shape], tool: cq.Shape, op: BoolOp, tol: Optional[float] = None) -> cq.Shape:
    """Boolean of the model with a tool body; `tol` is OCC's fuzzy tolerance (retry aid)."""
    if body is None:
        if op in (BoolOp.NEW, BoolOp.JOIN):
            return tool
        raise ActionError(f"{op.value} on an empty model")
    if op in (BoolOp.NEW, BoolOp.JOIN):
        return body.fuse(tool, tol=tol)
    if op == BoolOp.CUT:
        return body.cut(tool, tol=tol)
    return body.intersect(tool, tol=tol)


def fuse_all(solids: List[cq.Shape]) -> cq.Shape:
    out = solids[0]
    for s in solids[1:]:
        out = out.fuse(s)
    return out


TAPER_EPS = 1e-3  # degrees; smaller drafts are treated as straight


# --------------------------------------------------------------------------------------
# Extrude (covers all of DeepCAD)
# --------------------------------------------------------------------------------------
@dataclass
class Extrude(Action):
    """Push `profile` (drawn on `plane`) `forward` along the plane normal and `backward`
    against it, then combine the tool with the model.

        forward=5               one-sided                (DeepCAD one_side,  e1 = 5)
        backward=5              one-sided, the other way (DeepCAD one_side,  e1 = -5)
        forward=3, backward=3   symmetric                (DeepCAD symmetric, e1 = 3)
        forward=10, backward=2  two-sided                (DeepCAD two_side,  e1 = 10, e2 = 2)

    The tool spans [-backward, +forward] along the normal and must have positive length.
    `taper` is a draft angle in degrees, applied from the sketch plane outward.
    """
    kind: ClassVar[str] = "extrude"
    plane: SketchPlane
    profile: Profile
    forward: float = length(0.0)
    backward: float = length(0.0)
    op: BoolOp = BoolOp.JOIN
    taper: float = 0.0

    def __post_init__(self):
        self.op = BoolOp(self.op)

    def validate(self) -> None:
        try:
            self.profile.validate()
        except ValueError as e:
            raise ActionError(f"invalid profile: {e}") from e
        if self.forward + self.backward <= 0:
            raise ActionError("extrude needs forward + backward > 0")
        if not -89.0 < self.taper < 89.0:
            raise ActionError("taper angle out of range")

    def spans(self) -> List[Tuple[float, float]]:
        """Prisms as (offset from the plane, signed length) along the normal."""
        if abs(self.taper) <= TAPER_EPS:
            return [(-self.backward, self.forward + self.backward)]
        return [(0.0, d) for d in (self.forward, -self.backward) if abs(d) > 1e-12]  # draft starts at the plane

    def tool(self) -> cq.Shape:
        n = cq.Vector(*self.plane.normal)
        taper = self.taper if abs(self.taper) > TAPER_EPS else 0.0

        def prism(face, offset, d):
            f = face.translate(n * offset) if offset else face
            return cq.Solid.extrudeLinear(f, n * d, taper)

        prisms = []
        for outer, holes in self.profile.regions(self.plane):
            face = cq.Face.makeFromWires(outer, holes)
            for offset, d in self.spans():
                if holes and not face.isValid():
                    # holes touching the boundary: subtract hole prisms in 3D instead
                    body = prism(cq.Face.makeFromWires(outer), offset, d)
                    prisms.append(body.cut(*[prism(cq.Face.makeFromWires(h), offset, d) for h in holes]))
                else:
                    prisms.append(prism(face, offset, d))
        return fuse_all(prisms)

    def apply(self, body, tol: Optional[float] = None):
        return combine(body, self.tool(), self.op, tol)

    # DeepCAD's (extent_type, extent_one, extent_two) <-> (forward, backward)
    def deepcad_extent(self) -> Tuple[int, float, float]:
        """(code, e1, e2) with code 0 = one_side, 1 = symmetric, 2 = two_side."""
        f, b = self.forward, self.backward
        if b == 0:
            return 0, f, 0.0
        if f == 0:
            return 0, -b, 0.0
        if abs(f - b) < 1e-12:
            return 1, f, 0.0
        return 2, f, b

    @staticmethod
    def from_deepcad_extent(code: int, e1: float, e2: float = 0.0) -> Tuple[float, float]:
        if code == 0:
            return (e1, 0.0) if e1 >= 0 else (0.0, -e1)
        if code == 1:
            return abs(e1), abs(e1)
        # two_side: DeepCAD unions the signed prisms [0, e1] and [0, -e2]
        return max(0.0, e1, -e2), max(0.0, -e1, e2)


# --------------------------------------------------------------------------------------
# Revolve
# --------------------------------------------------------------------------------------
@dataclass
class Revolve(Action):
    """Revolve `profile` about the axis through `axis_start` -> `axis_end` (sketch coordinates)."""
    kind: ClassVar[str] = "revolve"
    plane: SketchPlane
    profile: Profile
    axis_start: Vec2 = length((0.0, 0.0))
    axis_end: Vec2 = length((0.0, 1.0))
    angle: float = 360.0  # degrees
    op: BoolOp = BoolOp.JOIN

    def __post_init__(self):
        self.op = BoolOp(self.op)
        self.axis_start, self.axis_end = _v2(self.axis_start), _v2(self.axis_end)

    def validate(self) -> None:
        try:
            self.profile.validate()
        except ValueError as e:
            raise ActionError(f"invalid profile: {e}") from e
        if math.dist(self.axis_start, self.axis_end) < 1e-9:
            raise ActionError("revolve axis is degenerate")
        if not 0.0 < abs(self.angle) <= 360.0:
            raise ActionError("revolve angle must be in (0, 360]")

    def tool(self) -> cq.Shape:
        a0 = cq.Vector(*self.plane.to_world(*self.axis_start))
        a1 = cq.Vector(*self.plane.to_world(*self.axis_end))
        solids = []
        for outer, holes in self.profile.regions(self.plane):
            if holes and not cq.Face.makeFromWires(outer, holes).isValid():
                body = cq.Solid.revolve(outer, [], self.angle, a0, a1)
                solids.append(body.cut(*[cq.Solid.revolve(h, [], self.angle, a0, a1) for h in holes]))
            else:
                solids.append(cq.Solid.revolve(outer, holes, self.angle, a0, a1))
        return fuse_all(solids)

    def apply(self, body, tol: Optional[float] = None):
        return combine(body, self.tool(), self.op, tol)


# --------------------------------------------------------------------------------------
# Edge selection by geometry, then Fillet / Chamfer
# --------------------------------------------------------------------------------------
@dataclass
class EdgeSelector:
    """Pick edges of the current body by a 3D probe point instead of a topological name.

        mode="nearest"  the k edges closest to `point`
        mode="within"   every edge whose distance to `point` is <= radius
        mode="all"      every edge of the body
    """
    point: Vec3 = (0.0, 0.0, 0.0)
    mode: str = "nearest"
    k: int = 1
    radius: float = length(0.0)

    def __post_init__(self):
        self.point = _v3(self.point)
        if self.mode not in ("nearest", "within", "all"):
            raise ValueError(f"bad selector mode {self.mode!r}")

    @staticmethod
    def candidate_edges(body: cq.Shape) -> List[cq.Edge]:
        """Edges shared by two distinct faces with non-zero length.  Seam edges of
        cylinders and degenerate edges can never be filleted or chamfered."""
        faces_of: Dict[int, set] = {}
        for fi, f in enumerate(body.Faces()):
            for e in f.Edges():
                faces_of.setdefault(e.hashCode(), set()).add(fi)
        return [e for e in body.Edges() if len(faces_of.get(e.hashCode(), ())) >= 2 and e.Length() > 1e-9]

    def select(self, body: cq.Shape) -> List[cq.Edge]:
        edges = self.candidate_edges(body)
        if self.mode == "all":
            return edges
        probe = cq.Vertex.makeVertex(*self.point)
        ranked = sorted(((e.distance(probe), i, e) for i, e in enumerate(edges)), key=lambda t: t[:2])
        if self.mode == "nearest":
            return [e for _, _, e in ranked[: max(1, int(self.k))]]
        return [e for d, _, e in ranked if d <= self.radius + 1e-9]

    def transformed(self, translate=(0.0, 0.0, 0.0), scale: float = 1.0) -> "EdgeSelector":
        p = tuple((c + t) * scale for c, t in zip(self.point, translate))
        return EdgeSelector(point=p, mode=self.mode, k=self.k, radius=self.radius * scale)

    def to_dict(self) -> Dict:
        return {f.name: to_plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Dict) -> "EdgeSelector":
        return build(cls, d, context="edges")


@dataclass
class Fillet(Action):
    kind: ClassVar[str] = "fillet"
    edges: EdgeSelector
    radius: float = length(1.0)

    def validate(self):
        if self.radius <= 0:
            raise ActionError("fillet radius must be positive")

    def apply(self, body):
        if body is None:
            raise ActionError("fillet on an empty model")
        sel = self.edges.select(body)
        if not sel:
            raise ActionError("fillet: no edges selected")
        return body.fillet(self.radius, sel)


@dataclass
class Chamfer(Action):
    kind: ClassVar[str] = "chamfer"
    edges: EdgeSelector
    distance: float = length(1.0)
    distance2: Optional[float] = length(None)

    def validate(self):
        if self.distance <= 0 or (self.distance2 is not None and self.distance2 <= 0):
            raise ActionError("chamfer distances must be positive")

    def apply(self, body):
        if body is None:
            raise ActionError("chamfer on an empty model")
        sel = self.edges.select(body)
        if not sel:
            raise ActionError("chamfer: no edges selected")
        return body.chamfer(self.distance, self.distance2, sel)
