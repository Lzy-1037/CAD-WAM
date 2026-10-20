#!/usr/bin/env python
"""Paper metric suite: ONE protocol for our runs and for every baseline's predicted STEP files.

  python cad_eval/paper_eval.py --pred-dir <dir of <model_id>.step> --ref-dir <reference STEPs> \
      --ids <evaluation id list> --system "<row name>" --out <dir>/paper_metrics.json \
      [--attempted <ids the system attempted>] [--workers 16] [--no-iou] [--sir]
  add --iou-only to recompute just the IoU of an existing JSON + per-model CSV (other fields untouched)

Every definition follows the paper that introduced the metric, so our numbers sit on their scale:

  meshing     Pointer-CAD misc.create_mesh + CADModel.normalize: the solid is fitted into [-1,1]^3
              (bbox centre, uniform scale 2/longest side), then BRepMesh_IncrementalMesh(0.001, relative
              False, angular 0.5, parallel) -> STL -> trimesh (vertices merged).  Same values as
              Text2CAD/CADmium brep2mesh.
  CD          Pointer-CAD measurements/chamfer_distance.py: 8,192 area-uniform surface samples per shape,
              each cloud centred on its mean and divided by (max - min over all coordinates), i.e. about
              [-0.5, 0.5]; squared nearest-neighbour distances, mean per direction, SUM of the two
              directions, x1000.  Median (the headline in Text2CAD, CADmium, Pointer-CAD) and mean.
  HD          no baseline paper defines it: symmetric Hausdorff distance on the same 8,192-point normalised
              clouds as CD (larger of the two maximum nearest-neighbour distances, unsquared, no multiplier).
  NC          (median and mean reported) CADFS compute_normal_consistency on the same 8,192-point clouds (signed dot products of
              face normals of nearest neighbours, mean of the two directions).
  ECD         CADFS edge_chamfer_dist under CADFS's own sampling (100,000 points, centre on mean,
              divide by max |coordinate|, edge radius 0.004, |n.n| < 0.2, x1000): ECD only exists in
              CADFS and its edge radius is tied to that point density.
  IR          Pointer-CAD's strict definition: a prediction is invalid unless its STEP loads into a
              shape with non-zero volume that can be meshed.  Denominator = the attempted ids.
  (watertightness is not reported; trimesh is_watertight is kept per model only as Euler match's precondition)
  Euler match CADmium calculate_CADMium_scores (EECM): only when BOTH meshes are watertight, after
              remove_unreferenced_vertices, 1 if euler_number agrees.
  IoU         Pointer-CAD measurements/intersection_over_union.py: both solids fitted into [-1,1]^3,
              BRepAlgoAPI_Common / BRepAlgoAPI_Fuse volumes.  Their code fuses the UN-normalised shapes
              (a bug); here both booleans use the normalised solids, with a fuzzy value of 1e-5: a correct
              prediction has faces that coincide with the reference to ~1e-7, and plain OCC booleans then
              return an empty common (IoU 0 for 37 of 200 geometrically identical oracle pairs).  Booleans
              still misfire now and then, so each result must satisfy Vu = Vp + Vr - Vi with the exact solid
              volumes (residual <= 1e-3); fuzzy 1e-5, plain, fuzzy 1e-4 are tried in that order and the first
              consistent one is kept; a pair with none is excluded from IoU and counted.

Writes <out> (paper JSON) and <out stem>_per_model.csv (one row per id, for paired tests).
"""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import compute_normal_consistency, normalize_pc  # noqa: E402  (CADFS, vendored)

PROTOCOL = {
    'mesh': 'fit to [-1,1]^3 (bbox centre, 2/longest), BRepMesh linear 0.001 abs, angular 0.5 (Pointer-CAD)',
    'cd_points': 8192, 'cd_normalisation': 'centre on mean, divide by (max-min) over all coords (Pointer-CAD)',
    'cd_formula': 'sum over both directions of mean squared NN distance, x1000',
    'hd': 'symmetric Hausdorff (max of the two max NN distances, unsquared, no multiplier) on the CD clouds',
    'ecd_points': 100000, 'ecd_normalisation': 'centre on mean, divide by max|coord| (CADFS)',
    'ecd_edge_radius': 0.004, 'ecd_dot_threshold': 0.2,
    'iou': 'both solids fitted to [-1,1]^3, Common/Fuse volumes (Pointer-CAD, fuse bug fixed), booleans with '
           'fuzzy value 1e-5: plain booleans return an EMPTY common for near-coincident faces (oracle check: 37 of '
           '200 geometrically identical pairs scored IoU 0); every result checked by inclusion-exclusion against the '
           'exact solid volumes (residual <= 1e-3), methods tried in order fuzzy 1e-5 / plain / fuzzy 1e-4, first '
           'consistent kept, none -> excluded (counts.iou_failed)',
    'invalid': 'no STEP / unreadable / zero volume / not meshable (Pointer-CAD strict IR)',
    'sampling_seed': 'sha256(model_id, role)',
}
CD_POINTS, ECD_POINTS = 8192, 100000
EDGE_R, EDGE_DOT = 0.004, 0.2


# ------------------------------------------------------------------ OCC (pythonocc or cadquery's OCP)
_O = {}


def occ():
    if _O:
        return _O
    try:
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
        from OCC.Core.TopTools import TopTools_ListOfShape
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
        _O.update(bnd_add=brepbndlib.Add, volume_props=brepgprop.VolumeProperties)
    except ImportError:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.Bnd import Bnd_Box
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.StlAPI import StlAPI_Writer
        from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
        _O.update(bnd_add=BRepBndLib.Add_s, volume_props=BRepGProp.VolumeProperties_s)
    _O.update(Common=BRepAlgoAPI_Common, Fuse=BRepAlgoAPI_Fuse, ListOfShape=TopTools_ListOfShape,
              Transform=BRepBuilderAPI_Transform,
              Mesh=BRepMesh_IncrementalMesh, Bnd_Box=Bnd_Box, GProp=GProp_GProps, RetDone=IFSelect_RetDone,
              Reader=STEPControl_Reader, StlWriter=StlAPI_Writer, Pnt=gp_Pnt, Trsf=gp_Trsf, Vec=gp_Vec)
    return _O


def load_step(path):
    o = occ()
    r = o['Reader']()
    if r.ReadFile(str(path)) != o['RetDone']:
        raise ValueError('unreadable')
    r.TransferRoots()
    shape = r.OneShape()
    if shape is None or shape.IsNull():
        raise ValueError('null shape')
    return shape


def volume(shape):
    o = occ()
    props = o['GProp']()
    o['volume_props'](shape, props)
    return abs(props.Mass())


def fit_unit_cube(shape):
    """Pointer-CAD _fit_shape_to_unit_cube: bbox centre to the origin, longest side to 2."""
    o = occ()
    box = o['Bnd_Box']()
    box.SetGap(0.0)
    o['bnd_add'](shape, box, True)
    if box.IsVoid():
        raise ValueError('void bbox')
    x0, y0, z0, x1, y1, z1 = box.Get()
    longest = max(x1 - x0, y1 - y0, z1 - z0)
    if not all(map(math.isfinite, (x0, y0, z0, x1, y1, z1))) or longest < 1e-12:
        raise ValueError('degenerate bbox')
    t = o['Trsf']()
    t.SetTranslation(o['Vec'](-(x0 + x1) / 2, -(y0 + y1) / 2, -(z0 + z1) / 2))
    shape = o['Transform'](shape, t, True).Shape()
    s = o['Trsf']()
    s.SetScale(o['Pnt'](0.0, 0.0, 0.0), 2.0 / longest)
    return o['Transform'](shape, s, True).Shape()


FUZZY = 1e-5          # in the [-1,1]^3 frame of fit_unit_cube


def fuzzy_boolean(kind, a, b, fuzzy=None):
    """BRepAlgoAPI_Common / _Fuse with a fuzzy value, so near-coincident faces are treated as coincident."""
    o = occ()
    alg = o[kind]()
    args, tools = o['ListOfShape'](), o['ListOfShape']()
    args.Append(a)
    tools.Append(b)
    alg.SetArguments(args)
    alg.SetTools(tools)
    alg.SetFuzzyValue(FUZZY if fuzzy is None else fuzzy)
    alg.SetRunParallel(False)
    alg.Build()
    if not alg.IsDone():
        raise RuntimeError(f'{kind} failed')
    return alg.Shape()


def mesh_shape(shape):
    """Pointer-CAD create_mesh on an already normalised solid."""
    import trimesh
    o = occ()
    m = o['Mesh'](shape, 0.001, False, 0.5, True)
    m.Perform()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'm.stl')
        w = o['StlWriter']()
        w.Write(shape, p)
        mesh = trimesh.load(p, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError('empty mesh')
    return mesh


# ------------------------------------------------------------------ point-cloud metrics
def seed_of(mid, role):
    return int.from_bytes(hashlib.sha256(f'{mid}:{role}'.encode()).digest()[:4], 'little')


def sample(mesh, n, seed):
    import trimesh
    try:
        pts, idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    except TypeError:                            # trimesh < 3.21 has no seed argument
        state = np.random.get_state()
        np.random.seed(seed)
        pts, idx = trimesh.sample.sample_surface(mesh, n)
        np.random.set_state(state)
    return np.asarray(pts, np.float64), np.asarray(mesh.face_normals[idx], np.float64)


def pcad_normalize(points):
    """Pointer-CAD chamfer_distance._normalize (centred)."""
    p = points - points.mean(axis=0)
    scale = abs(p.max() - p.min())
    return p / scale if scale > 1e-12 else p


def chamfer(a, b):
    """(CD, HD, tree_a, tree_b): CD = sum of the two mean squared NN distances; HD = symmetric Hausdorff,
    the larger of the two maximum NN distances (unsquared), on the same clouds."""
    ta, tb = cKDTree(a), cKDTree(b)
    d1, _ = tb.query(a)
    d2, _ = ta.query(b)
    return (float(np.mean(np.square(d1)) + np.mean(np.square(d2))), float(max(d1.max(), d2.max())), ta, tb)


def edge_points(points, normals):
    """CADFS edge detection, vectorised: a point is an edge point if a neighbour within EDGE_R has
    |n_i . n_j| < EDGE_DOT.  Identical to the per-point query_ball_point loop (the point itself has dot 1)."""
    pairs = cKDTree(points).query_pairs(EDGE_R, output_type='ndarray')
    flags = np.zeros(len(points), bool)
    if len(pairs):
        dot = np.abs(np.einsum('ij,ij->i', normals[pairs[:, 0]], normals[pairs[:, 1]]))
        sel = pairs[dot < EDGE_DOT]
        flags[sel[:, 0]] = True
        flags[sel[:, 1]] = True
    return points[flags]


def edge_chamfer(gt_pts, gt_nrm, pr_pts, pr_nrm):
    """CADFS edge_chamfer_dist (x1000 applied by the caller), same degenerate handling."""
    ge, pe = edge_points(gt_pts, gt_nrm), edge_points(pr_pts, pr_nrm)
    if len(ge) == 0:
        return 0.0
    if len(pe) == 0:
        pe = np.zeros((1, 3))
    d1, _ = cKDTree(pe).query(ge)
    d2, _ = cKDTree(ge).query(pe)
    return float(np.mean(np.square(d1)) + np.mean(np.square(d2)))


# ------------------------------------------------------------------ self-intersection ratio
def _tri_tri_intersect(A, B, eps):
    """Vectorised separating-axis test for triangle pairs A, B [k,3,3]: True where they cross.
    On a triangle's own normal it projects to a single value d, so there the other triangle must
    strictly straddle d (so faces that merely touch, and coplanar pairs, do not count); on the 9
    edge-cross axes ordinary SAT applies (an axis separates only if the intervals are apart by > eps --
    an edge-cross axis can coincide with a face normal, where a crossing pair has zero-width overlap)."""
    ea = A[:, [1, 2, 0]] - A                     # edges [k,3,3]
    eb = B[:, [1, 2, 0]] - B
    na = np.cross(ea[:, 0], ea[:, 1])
    nb = np.cross(eb[:, 0], eb[:, 1])
    hit = np.ones(len(A), bool)
    for n, own, other in ((na, A, B), (nb, B, A)):
        n = n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
        d = np.einsum('kj,kj->k', own[:, 0], n)
        po = np.einsum('kij,kj->ki', other, n) - d[:, None]
        hit &= (po.min(1) < -eps) & (po.max(1) > eps)
    axes = [np.cross(ea[:, i], eb[:, j]) for i in range(3) for j in range(3)]
    for ax in axes:
        ln = np.linalg.norm(ax, axis=1)
        ok = ln > 1e-12                          # skip degenerate axes (parallel edges)
        ax = np.where(ok[:, None], ax / np.maximum(ln, 1e-300)[:, None], 0.0)
        pa = np.einsum('kij,kj->ki', A, ax)
        pb = np.einsum('kij,kj->ki', B, ax)
        overlap = np.minimum(pa.max(1), pb.max(1)) - np.maximum(pa.min(1), pb.min(1))
        hit &= ~ok | (overlap >= -eps)
    return hit


def self_intersection_ratio(mesh):
    """Fraction of faces that take part in an intersection with a non-adjacent face."""
    V, F = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)
    T = V[F]
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    keep = np.nonzero(area > 1e-14)[0]
    if len(keep) < 2:
        return 0.0
    c = T[keep].mean(1)
    r = np.linalg.norm(T[keep] - c[:, None], axis=2).max(1)
    # candidate pairs by bounding spheres, bucketed by radius so a few large triangles don't blow up
    bucket = np.floor(np.log2(np.maximum(r, 1e-12))).astype(int)
    groups = {b: np.nonzero(bucket == b)[0] for b in np.unique(bucket)}
    trees = {b: cKDTree(c[g]) for b, g in groups.items()}
    rmax = {b: r[g].max() for b, g in groups.items()}
    cand = []
    bs = sorted(groups)
    for ia, a in enumerate(bs):
        for b in bs[ia:]:
            lists = trees[a].query_ball_tree(trees[b], rmax[a] + rmax[b])
            for i, js in enumerate(lists):
                if js:
                    gi = groups[a][i]
                    gj = groups[b][np.asarray(js)]
                    cand.append(np.stack([np.full(len(gj), gi), gj], 1))
    if not cand:
        return 0.0
    P = np.concatenate(cand)
    P = P[P[:, 0] != P[:, 1]]
    P = np.unique(np.sort(P, 1), axis=0)
    P = P[np.linalg.norm(c[P[:, 0]] - c[P[:, 1]], axis=1) <= r[P[:, 0]] + r[P[:, 1]]]
    fi, fj = keep[P[:, 0]], keep[P[:, 1]]
    shared = (F[fi][:, :, None] == F[fj][:, None, :]).any((1, 2))   # adjacent: share a vertex
    fi, fj = fi[~shared], fj[~shared]
    if len(fi) == 0:
        return 0.0
    eps = 1e-9 * float(np.ptp(V, axis=0).max())
    hits = np.zeros(len(fi), bool)
    for s in range(0, len(fi), 200000):
        hits[s:s + 200000] = _tri_tri_intersect(T[fi[s:s + 200000]], T[fj[s:s + 200000]], eps)
    bad = np.unique(np.concatenate([fi[hits], fj[hits]]))
    return float(len(bad) / len(F))


# ------------------------------------------------------------------ one model
def evaluate_model(job):
    mid, pred_path, ref_path, do_iou, do_sir = job
    t0 = time.time()
    row = {'model_id': mid, 'pred_status': None, 'ref_status': None, 'CD': None, 'NC': None, 'ECD': None,
           'HD': None, 'watertight': None, 'SIR': None, 'euler_match': None, 'IoU': None, 'iou_status': None,
           'pred_faces': None, 'note': None}
    pred = pmesh = None
    # ---- prediction validity (independent of the reference)
    try:
        if not pred_path or not os.path.isfile(pred_path):
            raise ValueError('missing')
        pred = load_step(pred_path)
        if volume(pred) <= 0.0:
            raise ValueError('zero volume')
        pred = fit_unit_cube(pred)
        if volume(pred) <= 1e-9:
            raise ValueError('zero volume')
        pmesh = mesh_shape(pred)
        row['pred_status'] = 'ok'
        row['pred_faces'] = int(len(pmesh.faces))
        row['watertight'] = int(bool(pmesh.is_watertight))
    except Exception as e:  # noqa: BLE001
        row['pred_status'] = f'invalid:{str(e)[:60]}'
    if row['pred_status'] == 'ok' and do_sir:
        try:
            row['SIR'] = self_intersection_ratio(pmesh)
        except Exception as e:  # noqa: BLE001
            row['note'] = f'sir_failed:{type(e).__name__}'
    # ---- reference
    ref = rmesh = None
    try:
        if not ref_path or not os.path.isfile(ref_path):
            raise ValueError('no reference file')
        ref = fit_unit_cube(load_step(ref_path))
        rmesh = mesh_shape(ref)
        row['ref_status'] = 'ok'
    except Exception as e:  # noqa: BLE001
        row['ref_status'] = f'unscorable:{str(e)[:60]}'
    if row['pred_status'] != 'ok' or row['ref_status'] != 'ok':
        row['seconds'] = round(time.time() - t0, 2)
        return row
    # ---- CD / NC (Pointer-CAD sampling + normalisation)
    gp, gn = sample(rmesh, CD_POINTS, seed_of(mid, 'ref'))
    pp, pn = sample(pmesh, CD_POINTS, seed_of(mid, 'pred'))
    gp, pp = pcad_normalize(gp), pcad_normalize(pp)
    cd, hd, tg, tp = chamfer(gp, pp)
    row['CD'] = cd * 1000.0
    row['HD'] = hd
    row['NC'] = float(compute_normal_consistency(gp, pp, gn, pn, gt_tree=tg, pred_tree=tp))
    # ---- ECD (CADFS sampling + normalisation)
    gp2, gn2 = sample(rmesh, ECD_POINTS, seed_of(mid, 'ref_ecd'))
    pp2, pn2 = sample(pmesh, ECD_POINTS, seed_of(mid, 'pred_ecd'))
    row['ECD'] = edge_chamfer(normalize_pc(gp2), gn2, normalize_pc(pp2), pn2) * 1000.0
    # ---- Euler characteristic match (CADmium: both watertight)
    if pmesh.is_watertight and rmesh.is_watertight:
        a, b = pmesh.copy(), rmesh.copy()
        a.remove_unreferenced_vertices(); b.remove_unreferenced_vertices()
        row['euler_match'] = int(a.euler_number == b.euler_number)
    # ---- volumetric IoU (Pointer-CAD, both booleans on the normalised solids)
    if do_iou:
        row['IoU'], row['iou_status'] = compute_iou(pred, ref)
    row['seconds'] = round(time.time() - t0, 2)
    return row


IOU_METHODS = (('fuzzy1e-5', 1e-5), ('plain', None), ('fuzzy1e-4', 1e-4))
IOU_RESIDUAL = 1e-3


def compute_iou(pred, ref):
    """(IoU, status) of two solids already fitted to [-1,1]^3.  OCC booleans occasionally return a wrong
    volume (an empty common for near-coincident faces, or a broken fuse), so every result is checked
    against inclusion-exclusion with the exactly computed solid volumes, |Vu - (Vp + Vr - Vi)| <=
    1e-3 * max(Vu, Vp, Vr); methods are tried in order and the first consistent one is kept.  No
    consistent result -> None ('inconsistent': excluded from the IoU statistics and counted)."""
    o = occ()
    try:
        vp, vr = volume(pred), volume(ref)
    except Exception as e:  # noqa: BLE001
        return None, f'failed:{type(e).__name__}'
    for name, fz in IOU_METHODS:
        try:
            if fz is None:
                vi, vu = volume(o['Common'](pred, ref).Shape()), volume(o['Fuse'](pred, ref).Shape())
            else:
                vi, vu = volume(fuzzy_boolean('Common', pred, ref, fz)), volume(fuzzy_boolean('Fuse', pred, ref, fz))
        except Exception:  # noqa: BLE001
            continue
        scale = max(vu, vp, vr, 1e-12)
        if vu > 0 and abs(vu - (vp + vr - vi)) <= IOU_RESIDUAL * scale and vi <= min(vp, vr) * (1 + IOU_RESIDUAL):
            return float(min(1.0, max(0.0, vi / vu))), name
    return None, 'inconsistent'


def iou_only_model(job):
    mid, pred_path, ref_path = job
    try:
        pred = fit_unit_cube(load_step(pred_path))
        ref = fit_unit_cube(load_step(ref_path))
    except Exception as e:  # noqa: BLE001
        return mid, None, f'failed:{type(e).__name__}'
    iou, status = compute_iou(pred, ref)
    return mid, iou, status


def iou_only(a):
    """Recompute IoU for the models an earlier run scored; leave every other number untouched."""
    csv_path = os.path.splitext(a.out)[0] + '_per_model.csv'
    pj = json.load(open(a.out))
    with open(csv_path, newline='') as f:
        rd = csv.DictReader(f)
        fields, rows = rd.fieldnames, list(rd)
    todo = [r for r in rows if r.get('pred_status') == 'ok' and r.get('ref_status') == 'ok']
    jobs = [(r['model_id'], os.path.join(a.pred_dir, f"{r['model_id']}.step"),
             os.path.join(a.ref_dir, f"{r['model_id']}.step")) for r in todo]
    print(f'{a.out}: recomputing IoU for {len(jobs)} scored models, {a.workers} workers', flush=True)
    res, t0 = {}, time.time()
    with mp.Pool(a.workers, maxtasksperchild=20) as pool:
        it = pool.imap_unordered(iou_only_model, jobs)
        for k in range(len(jobs)):
            try:
                mid, iou, status = it.next(timeout=a.timeout)
            except mp.TimeoutError:
                break
            res[mid] = (iou, status)
            if (k + 1) % 1000 == 0:
                print(f'  {k + 1}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min', flush=True)
    for r in todo:
        iou, status = res.get(r['model_id'], (None, 'timeout'))
        r['IoU'] = '' if iou is None else iou
        r['iou_status'] = status
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    vals = [res[r['model_id']][0] for r in todo if r['model_id'] in res]
    iou = stat(vals)
    old = pj['metrics'].get('IoU')
    pj['metrics']['IoU'] = iou
    pj['counts']['iou_failed'] = sum(1 for v in vals if v is None) + (len(todo) - len(res))
    pj['counts'].pop('iou_plain_fallback', None)
    pj['counts']['iou_method'] = {k: sum(1 for _, st in res.values() if st == k)
                                  for k in sorted({st for _, st in res.values()})}
    pj['protocol']['iou'] = PROTOCOL['iou']
    pj['table_row']['IoU'] = rnd(iou['mean'], 4)
    pj['iou_recomputed'] = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'previous': old}
    json.dump(pj, open(a.out, 'w'), indent=2)
    print(f'IoU mean {old and old.get("mean")} -> {iou["mean"]}  (median {iou["median"]}, n {iou["n"]}, '
          f'methods {pj["counts"]["iou_method"]}, excluded {pj["counts"]["iou_failed"]})  '
          f'{(time.time() - t0) / 60:.1f} min -> {a.out}')


# ------------------------------------------------------------------ aggregation
def stat(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], np.float64)
    if len(v) == 0:
        return {'median': None, 'mean': None, 'n': 0}
    return {'median': float(np.median(v)), 'mean': float(v.mean()), 'n': int(len(v))}


def pct(vals):
    v = [x for x in vals if x is not None]
    return (100.0 * float(np.mean(v)) if v else None), len(v)


def rnd(x, k):
    return None if x is None else round(x, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', required=True, help='<model_id>.step files of one system')
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--ids', required=True, help='evaluation id list (e.g. ids_test7962.txt)')
    ap.add_argument('--attempted', default=None,
                    help='ids this system attempted (e.g. CADFS: its 7,278); others are "not attempted", '
                         'excluded from every metric instead of counted as invalid')
    ap.add_argument('--system', required=True, help='row name in the paper table')
    ap.add_argument('--out', required=True, help='paper JSON path')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--timeout', type=float, default=600.0, help='seconds of pool silence before giving up')
    ap.add_argument('--limit', type=int, default=None, help='first N ids only (smoke test)')
    ap.add_argument('--no-iou', action='store_true')
    ap.add_argument('--sir', action='store_true', help='also compute the self-intersection ratio (not reported)')
    ap.add_argument('--iou-only', action='store_true',
                    help='recompute only IoU in an existing --out JSON and its per-model CSV')
    a = ap.parse_args()
    if a.iou_only:
        return iou_only(a)

    ids = sorted(set(open(a.ids).read().split()))
    attempted = set(open(a.attempted).read().split()) if a.attempted else set(ids)
    if a.limit:
        ids = ids[: a.limit]
    todo = [m for m in ids if m in attempted]
    jobs = [(m, os.path.join(a.pred_dir, f'{m}.step'), os.path.join(a.ref_dir, f'{m}.step'),
             not a.no_iou, a.sir) for m in todo]
    print(f'{a.system}: {len(ids)} ids, {len(todo)} attempted, {a.workers} workers', flush=True)
    rows, t0 = {}, time.time()
    with mp.Pool(a.workers, maxtasksperchild=20) as pool:
        it = pool.imap_unordered(evaluate_model, jobs)   # chunksize 1: .next(timeout) needs the iterator
        for k in range(len(jobs)):
            try:
                r = it.next(timeout=a.timeout)
            except mp.TimeoutError:
                break
            rows[r['model_id']] = r
            if (k + 1) % 500 == 0:
                print(f'  {k + 1}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min', flush=True)
    for m in todo:
        if m not in rows:
            rows[m] = {'model_id': m, 'pred_status': 'timeout', 'ref_status': 'timeout'}

    # ---- per-model CSV (all ids, with the not-attempted ones marked)
    fields = ['model_id', 'pred_status', 'ref_status', 'CD', 'HD', 'NC', 'ECD', 'watertight', 'SIR',
              'euler_match', 'IoU', 'iou_status', 'pred_faces', 'note', 'seconds']
    csv_path = os.path.splitext(a.out)[0] + '_per_model.csv'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in ids:
            w.writerow({k: v for k, v in rows.get(m, {'model_id': m, 'pred_status': 'not_attempted'}).items()
                        if k in fields})

    R = [rows[m] for m in todo]
    valid = [r for r in R if r.get('pred_status') == 'ok']
    scored = [r for r in valid if r.get('ref_status') == 'ok']
    ir = 100.0 * (1 - len(valid) / len(R)) if R else None
    em, n_em = pct([r.get('euler_match') for r in scored])
    cd, hd, nc, ecd = (stat([r.get(k) for r in scored]) for k in ('CD', 'HD', 'NC', 'ECD'))
    iou = stat([r.get('IoU') for r in scored])
    out = {
        'system': a.system, 'pred_dir': os.path.abspath(a.pred_dir), 'ref_dir': os.path.abspath(a.ref_dir),
        'ids': os.path.abspath(a.ids), 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'protocol': PROTOCOL,
        'counts': {'eval_ids': len(ids), 'attempted': len(R), 'valid_predictions': len(valid),
                   'scored': len(scored),
                   'unscorable_reference': sum(1 for r in valid if r.get('ref_status') != 'ok'),
                   'iou_failed': sum(1 for r in scored if r.get('IoU') is None and not a.no_iou),
                   'iou_method': {k: sum(1 for r in scored if r.get('iou_status') == k)
                                  for k in sorted({str(r.get('iou_status')) for r in scored})}},
        'metrics': {'IR_pct': ir, 'CD': cd, 'HD': hd, 'NC': nc, 'ECD': ecd,
                    'euler_match_pct': em, 'euler_match_n': n_em, 'IoU': iou},
        'per_model_csv': os.path.abspath(csv_path),
        'seconds': round(time.time() - t0, 1),
    }
    out['table_row'] = {
        'System': a.system, 'n': len(R), 'IR (%)': rnd(ir, 2),
        'CD median': rnd(cd['median'], 3), 'CD mean': rnd(cd['mean'], 2),
        'HD median': rnd(hd['median'], 4), 'HD mean': rnd(hd['mean'], 4),
        'NC median': rnd(nc['median'], 4), 'NC mean': rnd(nc['mean'], 4), 'ECD': rnd(ecd['median'], 3),
        'IoU': rnd(iou['mean'], 4), 'Euler match (%)': rnd(em, 2)}
    json.dump(out, open(a.out, 'w'), indent=2)
    print(json.dumps(out['table_row'], indent=2))
    print(f'-> {a.out}\n-> {csv_path}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
