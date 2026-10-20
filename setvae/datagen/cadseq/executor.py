"""Executor: applies actions one at a time and accepts a result only if it is a valid solid
(>= 1 Solid, isValid(), volume > min_volume).  A rejected action leaves the state unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, List, Optional, Tuple

import cadquery as cq

from .actions import Action, ActionError, BoolOp, Extrude, Revolve


class InvalidSolidError(ActionError):
    pass


@dataclass
class ExecConfig:
    min_volume: float = 0.0         # absolute floor, model units^3 (0 = only the relative check)
    min_volume_rel: float = 1e-9    # relative floor: volume >= rel * (bbox diagonal)^3, unit independent
    allow_multi_solid: bool = True  # False: reject results that split into several bodies
    auto_new_body: bool = True      # JOIN on an empty state is promoted to NEW
    clean: bool = True              # unify faces after booleans (simpler topology)
    fix: bool = True                # try ShapeFix once if the raw result is invalid
    reject_noop: bool = False       # reject actions that do not change the volume
    noop_tol: float = 1e-9          # relative to the current volume
    check_monotone: bool = True     # a join must not lose volume, a cut / intersect must not gain it
    monotone_tol: float = 1e-6      # relative; violations mean the kernel produced a broken boolean


@dataclass
class CADState:
    """Immutable-by-convention snapshot: the solid + the accepted actions so far."""

    shape: Optional[cq.Shape] = None
    actions: List[Action] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.shape is None

    @property
    def volume(self) -> float:
        return 0.0 if self.shape is None else float(self.shape.Volume())

    @property
    def n_solids(self) -> int:
        return 0 if self.shape is None else len(self.shape.Solids())

    def bbox(self):
        return None if self.shape is None else self.shape.BoundingBox()

    def workplane(self) -> cq.Workplane:
        return cq.Workplane("XY").newObject([self.shape])

    def export(self, path: str) -> None:
        if path.lower().endswith(".stl"):
            self.shape.exportStl(path)
        else:
            self.shape.exportStep(path)


@dataclass
class StepResult:
    index: int
    action: Action
    ok: bool
    error: Optional[str] = None
    volume: float = 0.0
    n_solids: int = 0


class Executor:
    def __init__(self, config: Optional[ExecConfig] = None):
        self.cfg = config or ExecConfig()

    # ---- validity gate -----------------------------------------------------------------
    def check_solid(self, shape: Optional[cq.Shape]) -> cq.Shape:
        """Return a normalised shape (Solid or Compound of Solids) or raise."""
        if shape is None:
            raise InvalidSolidError("result is empty")
        solids = shape.Solids()
        if not solids:
            raise InvalidSolidError("result contains no solid (empty boolean?)")
        if not self.cfg.allow_multi_solid and len(solids) > 1:
            raise InvalidSolidError(f"result split into {len(solids)} bodies")
        for s in solids:
            if not s.isValid():
                raise InvalidSolidError("BRepCheck: invalid solid")
        vol = sum(s.Volume() for s in solids)
        bb = shape.BoundingBox()
        diag = ((bb.xmax - bb.xmin) ** 2 + (bb.ymax - bb.ymin) ** 2 + (bb.zmax - bb.zmin) ** 2) ** 0.5
        if vol < self.cfg.min_volume or vol < self.cfg.min_volume_rel * diag ** 3:
            raise InvalidSolidError(f"result volume {vol:.3g} is degenerate for a model of size {diag:.3g}")
        return solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)

    # ---- one transition -----------------------------------------------------------------
    def step(self, state: CADState, action: Action) -> CADState:
        action = self._coerce(state, action)
        action.validate()
        try:
            raw = action.apply(state.shape)
        except ActionError:
            raise
        except Exception as e:  # OCC kernel errors (StdFail_NotDone, ValueError, ...)
            raise ActionError(f"{action.kind}: kernel failure: {type(e).__name__}: {e}") from e

        shape = self._finish(raw)
        if self.cfg.check_monotone and state.shape is not None and hasattr(action, "op"):
            bad = self._monotone_violation(action.op, state.shape.Volume(), shape.Volume())
            if bad:  # a broken boolean: retry once with a fuzzy tolerance, then reject
                bb = state.shape.BoundingBox()
                diag = ((bb.xmax - bb.xmin) ** 2 + (bb.ymax - bb.ymin) ** 2 + (bb.zmax - bb.zmin) ** 2) ** 0.5
                try:
                    shape = self._finish(action.apply(state.shape, tol=1e-6 * diag))
                    bad = self._monotone_violation(action.op, state.shape.Volume(), shape.Volume())
                except Exception:
                    pass
                if bad:
                    raise InvalidSolidError(f"{action.kind}: {bad}")

        if self.cfg.reject_noop and state.shape is not None:
            if abs(shape.Volume() - state.shape.Volume()) <= self.cfg.noop_tol * abs(state.shape.Volume()):
                raise ActionError(f"{action.kind}: no-op (volume unchanged)")
        return CADState(shape=shape, actions=state.actions + [action])

    def _finish(self, raw: cq.Shape) -> cq.Shape:
        """clean, then gate (with one ShapeFix retry)."""
        if self.cfg.clean:
            try:
                cleaned = raw.clean()
                if cleaned.isValid():
                    raw = cleaned
            except Exception:
                pass
        try:
            return self.check_solid(raw)
        except InvalidSolidError:
            if not self.cfg.fix:
                raise
            return self.check_solid(raw.fix())

    def _monotone_violation(self, op: BoolOp, before: float, after: float) -> str:
        tol = self.cfg.monotone_tol * max(before, after)
        if op in (BoolOp.NEW, BoolOp.JOIN) and after < before - tol:
            return f"join lost volume ({before:.4g} -> {after:.4g}); the kernel dropped part of the model"
        if op in (BoolOp.CUT, BoolOp.INTERSECT) and after > before + tol:
            return f"{op.value} gained volume ({before:.4g} -> {after:.4g}); broken boolean"
        return ""

    def _coerce(self, state: CADState, action: Action) -> Action:
        if state.is_empty and isinstance(action, (Extrude, Revolve)):
            if action.op == BoolOp.JOIN and self.cfg.auto_new_body:
                return replace(action, op=BoolOp.NEW)
            if action.op in (BoolOp.CUT, BoolOp.INTERSECT):
                raise ActionError(f"{action.op.value} on an empty model")
        return action

    # ---- whole sequence -----------------------------------------------------------------
    def run(self, actions: Iterable[Action], state: Optional[CADState] = None,
            on_error: str = "raise") -> Tuple[CADState, List[StepResult]]:
        """on_error: 'raise' | 'skip' (drop failing actions) | 'stop' (keep valid prefix)."""
        state = state or CADState()
        results: List[StepResult] = []
        for i, a in enumerate(actions):
            try:
                state = self.step(state, a)
                results.append(StepResult(i, a, True, None, state.volume, state.n_solids))
            except ActionError as e:
                results.append(StepResult(i, a, False, str(e), state.volume, state.n_solids))
                if on_error == "raise":
                    raise
                if on_error == "stop":
                    break
        return state, results


def execute(actions: Iterable[Action], config: Optional[ExecConfig] = None, on_error="raise") -> CADState:
    return Executor(config).run(actions, on_error=on_error)[0]
