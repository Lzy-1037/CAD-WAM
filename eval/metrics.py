"""Normal consistency and the point-cloud normalisation used by the edge metric, vendored from
CADFS (`src/metrics/metrics.py`, `src/metrics/utils.py`); see THIRD_PARTY.md.

`paper_eval.py` imports these two functions so that normal consistency and the edge Chamfer
distance stay on the scale of the work that defined them.
"""
import numpy as np
from scipy.spatial import cKDTree


def normalize_pc(points):
    """CADFS normalize_pc: centre on the mean, scale by max |coordinate|."""
    centered = points - np.mean(points, axis=0)
    scale = np.max(np.abs(centered))
    return centered / scale if scale > 1e-8 else centered


def compute_normal_consistency(gt_points, pred_points, gt_normals, pred_normals,
                               gt_tree=None, pred_tree=None):
    """Normal Consistency."""
    if gt_tree is None:
        gt_tree = cKDTree(gt_points)
    _, match_pred_gt = gt_tree.query(pred_points, k=1)
    if pred_tree is None:
        pred_tree = cKDTree(pred_points)
    _, match_gt_pred = pred_tree.query(gt_points, k=1)
    a = np.sum(pred_normals * gt_normals[match_pred_gt], axis=1).mean()
    b = np.sum(gt_normals * pred_normals[match_gt_pred], axis=1).mean()
    return (a + b) / 2
