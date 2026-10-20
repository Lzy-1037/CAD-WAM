"""Prompt format shared by the dataset builder and the closed-loop rollout.

Training and evaluation MUST build prompts with this one function: a single differing character is a
distribution shift the model never saw.  `build_prompt(description)` with no progress lines
reproduces the v2/v3 prompt byte for byte.

Optional progress lines (a dataset records which ones it uses as "prompt_format" in <split>_stats.json,
and the rollout reads that same file, so the two cannot disagree):
  count  "Actions completed so far: k"      -- how many actions have been executed
  bbox   "Current bounding box: size sx x sy x sz, center (cx, cy, cz)"   (model units)
         -- the absolute extent that per-step normalisation strips from the geometric state;
            "Current bounding box: empty" before the first action
Both are known exactly at inference: the rollout counts its own executed actions and the geometry
worker reports the bbox of the solid it just built.
"""
GEO_TOKEN = '<|image_pad|>'
NUM_GEO = 16

INSTRUCTION = (
    'You are constructing a CAD model step by step with cadseq version 1 actions. '
    'You are given the CAD description of the target model and the current geometric state of the '
    'partially built model, encoded as 16 geometric state tokens. '
    'Generate ONLY the next cadseq action as a single JSON object, preserving the described dimensions. '
    'If the model is already complete, return {"kind": "end"}.'
)


def fmt_num(x):
    """4 decimals, trailing zeros stripped, no negative zero: 0.032, 0.11, 0, -0.5."""
    s = f'{float(x):.4f}'.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def bbox_line(bbox):
    """bbox = [xmin, ymin, zmin, xmax, ymax, zmax] in model units, or None for the empty state."""
    if bbox is None:
        return 'Current bounding box: empty'
    lo, hi = bbox[:3], bbox[3:]
    size = ' x '.join(fmt_num(h - l) for l, h in zip(lo, hi))
    center = ', '.join(fmt_num((h + l) / 2) for l, h in zip(lo, hi))
    return f'Current bounding box: size {size}, center ({center})'


def build_prompt(description, prompt_format=None, n_done=0, bbox=None):
    """prompt_format: None / {} for the v2/v3 prompt, or {"count": bool, "bbox": bool}."""
    pf = prompt_format or {}
    extra = ''
    if pf.get('count'):
        extra += f'Actions completed so far: {n_done}\n'
    if pf.get('bbox'):
        extra += bbox_line(bbox) + '\n'
    return (f'{INSTRUCTION}\n\nCAD description:\n{description}\n\n'
            f'{extra}Current geometric state:\n<GEO>{GEO_TOKEN * NUM_GEO}</GEO>')
