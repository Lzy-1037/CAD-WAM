"""Sketch geometry: SketchPlane, curves (Line, Arc, Circle), Loop, Profile.

Coordinates are local (u, v) on a SketchPlane.  A Loop is a closed chain of curves; each
curve stores only its end point and the loop starts at the end of its last curve.
Profile.loops[0] is the outer boundary; further loops are holes / islands (even-odd).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import cadquery as cq

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

TWO_PI = 2.0 * math.pi
EPS = 1e-9
ANGLE_EPS = 1e-6


# --------------------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------------------
def _v2(p) -> Vec2:
    return (float(p[0]), float(p[1]))


def _v3(p) -> Vec3:
    return (float(p[0]), float(p[1]), float(p[2]))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("zero-length vector cannot be normalised")
    return v / n


def _rotate_about_axis(vec, axis, angle) -> np.ndarray:
    """Rodrigues rotation of `vec` about unit `axis` by `angle` radians."""
    vec = np.asarray(vec, dtype=float)
    k = _unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    return vec * c + np.cross(k, vec) * s + k * np.dot(k, vec) * (1 - c)


# --------------------------------------------------------------------------------------
# Sketch plane
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SketchPlane:
    """A sketch coordinate system: origin + normal (local w) + x_dir (local u).

    The frame is always stored explicitly (no topological references such as "the top
    face of body 2"), which is what makes the representation robust for generative
    models: any plane is valid, nothing can dangle.
    """

    origin: Vec3 = (0.0, 0.0, 0.0)
    normal: Vec3 = (0.0, 0.0, 1.0)
    x_dir: Vec3 = (1.0, 0.0, 0.0)

    def __post_init__(self):
        n = _unit(self.normal)
        x = np.asarray(self.x_dir, dtype=float)
        x = x - np.dot(x, n) * n  # Gram-Schmidt: make x_dir orthogonal to normal
        if np.linalg.norm(x) < 1e-6:
            raise ValueError("x_dir must not be parallel to normal")
        x = _unit(x)
        object.__setattr__(self, "origin", _v3(self.origin))
        object.__setattr__(self, "normal", _v3(n))
        object.__setattr__(self, "x_dir", _v3(x))

    # ---- frame ---------------------------------------------------------------------
    @property
    def y_dir(self) -> Vec3:
        return _v3(np.cross(self.normal, self.x_dir))

    def to_world(self, u: float, v: float, w: float = 0.0) -> np.ndarray:
        return (
            np.asarray(self.origin)
            + u * np.asarray(self.x_dir)
            + v * np.asarray(self.y_dir)
            + w * np.asarray(self.normal)
        )

    def to_local(self, p) -> Vec3:
        d = np.asarray(p, dtype=float) - np.asarray(self.origin)
        return (
            float(np.dot(d, self.x_dir)),
            float(np.dot(d, self.y_dir)),
            float(np.dot(d, self.normal)),
        )

    def to_cq(self) -> cq.Plane:
        return cq.Plane(origin=self.origin, xDir=self.x_dir, normal=self.normal)

    def moved(self, origin: Vec3) -> "SketchPlane":
        return SketchPlane(origin=_v3(origin), normal=self.normal, x_dir=self.x_dir)

    def transformed(self, translate=(0.0, 0.0, 0.0), scale: float = 1.0) -> "SketchPlane":
        """Move the origin by `translate`, then scale it; the orientation is unchanged."""
        return self.moved((np.asarray(self.origin) + np.asarray(translate, dtype=float)) * scale)

    # ---- constructors ----------------------------------------------------------------
    @classmethod
    def named(cls, name: str, origin: Vec3 = (0.0, 0.0, 0.0)) -> "SketchPlane":
        table = {
            "XY": ((0, 0, 1), (1, 0, 0)),
            "YZ": ((1, 0, 0), (0, 1, 0)),
            "ZX": ((0, 1, 0), (0, 0, 1)),
            "XZ": ((0, -1, 0), (1, 0, 0)),
            "YX": ((0, 0, -1), (0, 1, 0)),
            "ZY": ((-1, 0, 0), (0, 0, 1)),
        }
        normal, x_dir = table[name.upper()]
        return cls(origin=origin, normal=normal, x_dir=x_dir)

    @classmethod
    def from_axes(cls, origin, x_axis, y_axis, z_axis) -> "SketchPlane":
        """Fusion 360 / DeepCAD JSON `transform` block (x_axis, y_axis, z_axis)."""
        return cls(origin=_v3(origin), normal=_v3(z_axis), x_dir=_v3(x_axis))

    @classmethod
    def from_euler(cls, origin, theta: float, phi: float, gamma: float) -> "SketchPlane":
        """DeepCAD's (theta, phi, gamma) parameterisation (cadlib.math_utils.
        polar_parameterization_inverse): the normal is the spherical direction
        (theta, phi); x_dir is the reference x axis of that direction rotated about the
        normal by gamma."""
        normal = np.array(
            [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
        )
        ref_x = _rotate_about_axis(
            _rotate_about_axis(np.array([1.0, 0.0, 0.0]), (0, 1, 0), theta), (0, 0, 1), phi
        )
        x_dir = _rotate_about_axis(ref_x, normal, gamma)
        return cls(origin=_v3(origin), normal=_v3(normal), x_dir=_v3(x_dir))

    def to_euler(self) -> Tuple[float, float, float]:
        """Inverse of from_euler: theta in [0, pi], phi in (-pi, pi], gamma in (-pi, pi]."""
        n = np.asarray(self.normal)
        theta = math.acos(max(-1.0, min(1.0, n[2])))
        phi = math.atan2(n[1], n[0]) if theta > ANGLE_EPS and (math.pi - theta) > ANGLE_EPS else 0.0
        ref_x = _rotate_about_axis(
            _rotate_about_axis(np.array([1.0, 0.0, 0.0]), (0, 1, 0), theta), (0, 0, 1), phi
        )
        x = np.asarray(self.x_dir)
        gamma = math.atan2(np.dot(np.cross(ref_x, x), n), np.dot(ref_x, x))
        return float(theta), float(phi), float(gamma)

    # ---- serialisation -----------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {"origin": list(self.origin), "normal": list(self.normal), "x_dir": list(self.x_dir)}

    @classmethod
    def from_dict(cls, d: Dict) -> "SketchPlane":
        return cls(origin=_v3(d["origin"]), normal=_v3(d["normal"]), x_dir=_v3(d["x_dir"]))


# --------------------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Line:
    """Straight segment from the previous end point to `end`."""

    end: Vec2

    def __post_init__(self):
        object.__setattr__(self, "end", _v2(self.end))

    def reversed(self, start: Vec2) -> "Line":
        return Line(end=start)

    def to_dict(self):
        return {"type": "line", "end": list(self.end)}


@dataclass(frozen=True)
class Arc:
    """Circular arc from the previous end point to `end`.

    `sweep` is the swept angle in radians, 0 < sweep < 2*pi.  `ccw` gives the direction of
    travel from start to end as seen from +normal of the sketch plane.  (This is DeepCAD's
    arc parameterisation; the center is derived, never stored.)
    """

    end: Vec2
    sweep: float
    ccw: bool = True

    def __post_init__(self):
        object.__setattr__(self, "end", _v2(self.end))
        object.__setattr__(self, "sweep", float(self.sweep))
        object.__setattr__(self, "ccw", bool(self.ccw))

    def radius(self, start: Vec2) -> float:
        chord = math.dist(start, self.end)
        return chord / (2.0 * math.sin(self.sweep / 2.0))

    def center(self, start: Vec2) -> Vec2:
        sx, sy = start
        ex, ey = self.end
        chord = math.hypot(ex - sx, ey - sy)
        r = self.radius(start)
        mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
        # left-hand perpendicular of the chord direction
        nx, ny = -(ey - sy) / chord, (ex - sx) / chord
        d = r * math.cos(self.sweep / 2.0)  # signed: negative when sweep > pi
        if not self.ccw:
            d = -d
        return (mx + nx * d, my + ny * d)

    def mid(self, start: Vec2) -> Vec2:
        cx, cy = self.center(start)
        r = self.radius(start)
        a0 = math.atan2(start[1] - cy, start[0] - cx)
        am = a0 + (self.sweep / 2.0 if self.ccw else -self.sweep / 2.0)
        return (cx + r * math.cos(am), cy + r * math.sin(am))

    def reversed(self, start: Vec2) -> "Arc":
        return Arc(end=start, sweep=self.sweep, ccw=not self.ccw)

    @classmethod
    def from_center(cls, start: Vec2, end: Vec2, center: Vec2, ccw: bool = True) -> "Arc":
        """Build from start/end/center (the Fusion 360 / DeepCAD JSON way)."""
        a0 = math.atan2(start[1] - center[1], start[0] - center[0])
        a1 = math.atan2(end[1] - center[1], end[0] - center[0])
        sweep = (a1 - a0) % TWO_PI if ccw else (a0 - a1) % TWO_PI
        if sweep < ANGLE_EPS:
            sweep = TWO_PI
        return cls(end=_v2(end), sweep=sweep, ccw=ccw)

    @classmethod
    def from_three_points(cls, start: Vec2, mid: Vec2, end: Vec2) -> "Arc":
        ax, ay = start
        bx, by = mid
        cx, cy = end
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < EPS:
            raise ValueError("collinear points do not define an arc")
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        # direction: sign of cross(mid-start, end-start) tells which side the mid point is
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        ccw = cross > 0  # start->mid->end turning left  => ccw travel around the center
        return cls.from_center(start, end, (ux, uy), ccw=ccw)

    def to_dict(self):
        return {"type": "arc", "end": list(self.end), "sweep": self.sweep, "ccw": self.ccw}


@dataclass(frozen=True)
class Circle:
    """Full circle; it is always the single curve of its loop."""

    center: Vec2
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "center", _v2(self.center))
        object.__setattr__(self, "radius", float(self.radius))

    def to_dict(self):
        return {"type": "circle", "center": list(self.center), "radius": self.radius}


Curve = Union[Line, Arc, Circle]


def curve_from_dict(d: Dict) -> Curve:
    t = d["type"]
    if t == "line":
        return Line(end=d["end"])
    if t == "arc":
        return Arc(end=d["end"], sweep=d["sweep"], ccw=d.get("ccw", True))
    if t == "circle":
        return Circle(center=d["center"], radius=d["radius"])
    raise ValueError(f"unknown curve type {t!r}")


# --------------------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------------------
@dataclass
class Loop:
    """A closed chain of curves (or a single Circle)."""

    curves: List[Curve] = field(default_factory=list)

    # ---- structure -------------------------------------------------------------------
    @property
    def is_circle(self) -> bool:
        return len(self.curves) == 1 and isinstance(self.curves[0], Circle)

    @property
    def start_point(self) -> Vec2:
        if self.is_circle:
            c = self.curves[0]
            return (c.center[0] - c.radius, c.center[1])  # left-most point, for consistency
        return self.curves[-1].end

    def segments(self) -> List[Tuple[Vec2, Curve]]:
        """(start_point, curve) pairs."""
        out = []
        start = self.start_point
        for c in self.curves:
            out.append((start, c))
            if not isinstance(c, Circle):
                start = c.end
        return out

    def vertices(self) -> np.ndarray:
        if self.is_circle:
            return np.asarray([self.start_point])
        return np.asarray([c.end for c in self.curves], dtype=float)

    def sample(self, n_arc: int = 24) -> np.ndarray:
        """Polygonal approximation, shape (N, 2), implicitly closed."""
        if self.is_circle:
            c = self.curves[0]
            t = np.linspace(0, TWO_PI, 4 * n_arc, endpoint=False)
            return np.stack([c.center[0] + c.radius * np.cos(t), c.center[1] + c.radius * np.sin(t)], 1)
        pts = []
        for start, c in self.segments():
            if isinstance(c, Line):
                pts.append(start)
            else:
                cx, cy = c.center(start)
                r = c.radius(start)
                a0 = math.atan2(start[1] - cy, start[0] - cx)
                sgn = 1.0 if c.ccw else -1.0
                k = max(2, int(math.ceil(n_arc * c.sweep / TWO_PI)))
                for i in range(k):
                    a = a0 + sgn * c.sweep * i / k
                    pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return np.asarray(pts, dtype=float)

    def signed_area(self) -> float:
        if self.is_circle:
            return math.pi * self.curves[0].radius ** 2
        p = self.sample()
        x, y = p[:, 0], p[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @property
    def is_ccw(self) -> bool:
        return self.signed_area() > 0

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self.sample()
        return p.min(0), p.max(0)

    # ---- transforms ------------------------------------------------------------------
    def reversed(self) -> "Loop":
        if self.is_circle:
            return Loop(list(self.curves))
        segs = self.segments()
        out = []
        for start, c in reversed(segs):
            out.append(c.reversed(start))
        return Loop(out)

    def rotated(self, k: int) -> "Loop":
        """Start the loop at curve index k (same geometry, different start point)."""
        if self.is_circle:
            return Loop(list(self.curves))
        k %= len(self.curves)
        return Loop(self.curves[k:] + self.curves[:k])

    def canonical(self) -> "Loop":
        """DeepCAD canonical form: counter-clockwise, starting at the left-most (min x,
        then min y) vertex.  Makes the token encoding unique for a given geometry."""
        loop = self if self.is_ccw else self.reversed()
        if loop.is_circle:
            return loop
        verts = loop.vertices()
        starts = np.roll(verts, 1, axis=0)  # start point of curve i is end of curve i-1
        k = int(np.lexsort((np.round(starts[:, 1], 6), np.round(starts[:, 0], 6)))[0])
        return loop.rotated(k)

    def transformed(self, translate=(0.0, 0.0), scale: float = 1.0) -> "Loop":
        """p -> (p + translate) * scale, applied to every point, radius scaled."""
        tx, ty = translate
        out = []
        for c in self.curves:
            if isinstance(c, Line):
                out.append(Line(((c.end[0] + tx) * scale, (c.end[1] + ty) * scale)))
            elif isinstance(c, Arc):
                out.append(Arc(((c.end[0] + tx) * scale, (c.end[1] + ty) * scale), c.sweep, c.ccw))
            else:
                out.append(Circle(((c.center[0] + tx) * scale, (c.center[1] + ty) * scale), c.radius * scale))
        return Loop(out)

    # ---- validation ------------------------------------------------------------------
    def validate(self, min_len: Optional[float] = None) -> None:
        """min_len defaults to 1e-7 of the loop's bounding-box size (unit independent)."""
        if not self.curves:
            raise ValueError("empty loop")
        if min_len is None:
            lo, hi = self.bbox()
            min_len = 1e-7 * max(float(np.max(hi - lo)), 1e-300)
        if any(isinstance(c, Circle) for c in self.curves):
            if len(self.curves) != 1:
                raise ValueError("a Circle must be the only curve of its loop")
            if self.curves[0].radius <= min_len:
                raise ValueError("circle radius must be positive")
            return
        if len(self.curves) < 2:
            raise ValueError("a non-circle loop needs at least 2 curves")
        for start, c in self.segments():
            if math.dist(start, c.end) < min_len:
                raise ValueError(f"zero-length curve at {c}")
            if isinstance(c, Arc) and not (ANGLE_EPS < c.sweep < TWO_PI - ANGLE_EPS):
                raise ValueError(f"arc sweep must be in (0, 2pi), got {c.sweep}")
        if abs(self.signed_area()) < min_len ** 2:
            raise ValueError("loop has (near) zero area")

    # ---- CadQuery --------------------------------------------------------------------
    def to_cq_wire(self, plane: SketchPlane) -> cq.Wire:
        W = lambda p: cq.Vector(*plane.to_world(p[0], p[1]))
        if self.is_circle:
            c = self.curves[0]
            return cq.Wire.makeCircle(c.radius, W(c.center), cq.Vector(*plane.normal))
        edges = []
        for start, c in self.segments():
            if isinstance(c, Line):
                edges.append(cq.Edge.makeLine(W(start), W(c.end)))
            else:
                edges.append(cq.Edge.makeThreePointArc(W(start), W(c.mid(start)), W(c.end)))
        return cq.Wire.assembleEdges(edges)

    # ---- serialisation -----------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {"curves": [c.to_dict() for c in self.curves]}

    @classmethod
    def from_dict(cls, d: Dict) -> "Loop":
        return cls([curve_from_dict(c) for c in d["curves"]])

    # ---- convenience constructors ------------------------------------------------------
    @classmethod
    def polygon(cls, points: Sequence[Vec2]) -> "Loop":
        """Closed polyline through `points` (first point == loop start point)."""
        pts = [_v2(p) for p in points]
        return cls([Line(p) for p in pts[1:]] + [Line(pts[0])])

    @classmethod
    def rect(cls, w: float, h: float, center: Vec2 = (0.0, 0.0)) -> "Loop":
        cx, cy = center
        return cls.polygon([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)])

    @classmethod
    def circle(cls, radius: float, center: Vec2 = (0.0, 0.0)) -> "Loop":
        return cls([Circle(center, radius)])

    @classmethod
    def stadium(cls, length: float, radius: float, center: Vec2 = (0.0, 0.0)) -> "Loop":
        """Slot: two half-circles of `radius` joined by lines, straight part `length`."""
        cx, cy = center
        half = length / 2
        return cls([
            Line((cx + half, cy - radius)),
            Arc((cx + half, cy + radius), math.pi, ccw=True),
            Line((cx - half, cy + radius)),
            Arc((cx - half, cy - radius), math.pi, ccw=True),
        ])


# --------------------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------------------
def _point_in_polygon(pt, poly: np.ndarray) -> bool:
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cond = (ys > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (xj - xs) * (y - ys) / (yj - ys) + xs
    return bool(np.sum(cond & (x < xint)) % 2 == 1)


@dataclass
class Profile:
    """A planar region = one or more closed loops (outer boundaries, holes, islands)."""

    loops: List[Loop] = field(default_factory=list)

    def __post_init__(self):
        if not self.loops:
            raise ValueError("a profile needs at least one loop")

    # ---- geometry --------------------------------------------------------------------
    @property
    def start_point(self) -> Vec2:
        return self.loops[0].start_point

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        mins, maxs = zip(*(l.bbox() for l in self.loops))
        return np.min(mins, 0), np.max(maxs, 0)

    @property
    def bbox_size(self) -> float:
        lo, hi = self.bbox()
        return float(np.max(hi - lo))

    def canonical(self) -> "Profile":
        """Canonical loops, ordered by bbox min corner (DeepCAD Profile.reorder)."""
        loops = [l.canonical() for l in self.loops]
        keys = np.array([np.round(l.bbox()[0], 6) for l in loops])
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        return Profile([loops[i] for i in order])

    def transformed(self, translate=(0.0, 0.0), scale: float = 1.0) -> "Profile":
        """p -> (p + translate) * scale in sketch coordinates."""
        return Profile([l.transformed(translate, scale) for l in self.loops])

    def scaled(self, scale: float) -> "Profile":
        return self.transformed(scale=scale)

    def validate(self) -> None:
        for l in self.loops:
            l.validate()

    def nesting(self) -> List[int]:
        """Parent loop of every loop (-1 = top level), from geometric containment: loop i is
        inside loop j when most of its sample points fall inside j's polygon."""
        polys = [l.sample() for l in self.loops]
        n = len(polys)

        def inside(i, j):
            pts = polys[i][:: max(1, len(polys[i]) // 8)]
            return i != j and sum(_point_in_polygon(p, polys[j]) for p in pts) * 2 > len(pts)

        contains = [[inside(i, j) for j in range(n)] for i in range(n)]
        depth = [sum(row) for row in contains]
        return [max((j for j in range(n) if contains[i][j]), key=lambda j: depth[j], default=-1) for i in range(n)]

    # ---- CadQuery --------------------------------------------------------------------
    def regions(self, plane: SketchPlane) -> List[Tuple[cq.Wire, List[cq.Wire]]]:
        """(outer wire, hole wires) per region on `plane`, by the even-odd nesting rule."""
        wires = [l.to_cq_wire(plane) for l in self.loops]
        if len(wires) == 1:
            return [(wires[0], [])]
        parent = self.nesting()
        depth = []
        for i in range(len(wires)):
            d, p = 0, parent[i]
            while p != -1:
                d, p = d + 1, parent[p]
            depth.append(d)
        return [(w, [wires[j] for j in range(len(wires)) if parent[j] == i])
                for i, w in enumerate(wires) if depth[i] % 2 == 0]

    def to_cq_faces(self, plane: SketchPlane) -> List[cq.Face]:
        """Planar faces on `plane` (one per region)."""
        return [cq.Face.makeFromWires(outer, holes) for outer, holes in self.regions(plane)]

    # ---- serialisation -----------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {"loops": [l.to_dict() for l in self.loops]}

    @classmethod
    def from_dict(cls, d: Dict) -> "Profile":
        return cls([Loop.from_dict(l) for l in d["loops"]])

    # ---- convenience constructors ------------------------------------------------------
    @classmethod
    def rect(cls, w, h, center=(0.0, 0.0), holes: Sequence[Loop] = ()) -> "Profile":
        return cls([Loop.rect(w, h, center), *holes])

    @classmethod
    def circle(cls, radius, center=(0.0, 0.0), holes: Sequence[Loop] = ()) -> "Profile":
        return cls([Loop.circle(radius, center), *holes])

    @classmethod
    def polygon(cls, points, holes: Sequence[Loop] = ()) -> "Profile":
        return cls([Loop.polygon(points), *holes])
