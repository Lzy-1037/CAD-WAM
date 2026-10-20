"""CADSequence: an ordered list of actions with JSON I/O, normalisation and execution."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .actions import Action
from .executor import CADState, ExecConfig, Executor


@dataclass
class CADSequence:
    actions: List[Action] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    # ---- serialisation -----------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {"version": 1, "meta": self.meta, "actions": [a.to_dict() for a in self.actions]}

    @classmethod
    def from_dict(cls, d: Dict) -> "CADSequence":
        return cls([Action.from_dict(a) for a in d["actions"]], meta=dict(d.get("meta", {})))

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """JSON with one point / vector per line, e.g. "end": [40.0, 0.0]."""
        s = re.sub(r"\[\s*((?:-?[\d.eE+-]+|null|true|false)(?:,\s*(?:-?[\d.eE+-]+|null|true|false))*)\s*\]",
                   lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
                   json.dumps(self.to_dict(), indent=indent))
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, path_or_str: str) -> "CADSequence":
        if path_or_str.lstrip().startswith("{"):
            return cls.from_dict(json.loads(path_or_str))
        with open(path_or_str) as f:
            return cls.from_dict(json.load(f))

    # ---- execution -----------------------------------------------------------------------
    def execute(self, config: Optional[ExecConfig] = None, on_error: str = "raise") -> CADState:
        return Executor(config).run(self.actions, on_error=on_error)[0]

    def bbox(self, fit: str = "final") -> Tuple[np.ndarray, np.ndarray]:
        """fit='final': bbox of the executed model.  fit='tools': union of the final model and
        every tool body (so no intermediate profile/extent falls outside the box)."""
        shapes = [self.execute().shape]
        if fit == "tools":
            shapes += [a.tool() for a in self.actions if hasattr(a, "tool")]
        boxes = [s.BoundingBox() for s in shapes]
        lo = np.min([[b.xmin, b.ymin, b.zmin] for b in boxes], 0)
        hi = np.max([[b.xmax, b.ymax, b.zmax] for b in boxes], 0)
        return lo, hi

    # ---- transforms ----------------------------------------------------------------------
    def transformed(self, translate=(0.0, 0.0, 0.0), scale: float = 1.0) -> "CADSequence":
        return CADSequence([a.transformed(translate, scale) for a in self.actions], dict(self.meta))

    def normalized(self, size: float = 1.0, factor: float = 0.75, fit: str = "final") -> Tuple["CADSequence", Dict]:
        """Center the model at the origin and scale it into [-factor*size, factor*size]^3
        (DeepCAD CADSequence.normalize convention).  Returns (sequence, transform).
        fit='tools' fits every intermediate tool body as well (see bbox)."""
        lo, hi = self.bbox(fit)
        center = (lo + hi) / 2.0
        extent = float(np.max(hi - lo)) / 2.0
        scale = size * factor / extent if extent > 0 else 1.0
        seq = self.transformed(translate=-center, scale=scale)
        seq.meta["normalization"] = {"translate": (-center).tolist(), "scale": scale}
        return seq, {"translate": (-center).tolist(), "scale": scale}

    # ---- token view ----------------------------------------------------------------------
    def to_vector(self, **kw) -> np.ndarray:
        from . import tokens
        return tokens.encode(self.actions, **kw)

    @classmethod
    def from_vector(cls, rows: np.ndarray, **kw) -> "CADSequence":
        from . import tokens
        return cls(tokens.decode(rows, **kw))

    def __len__(self):
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)
