"""Command line interpreter:  python -m cadseq program.json [-o model.step]

Executes the program step by step, prints one line per step, exports the final solid."""
import argparse
import sys

import numpy as np

from . import ActionError, CADSequence, ExecConfig, Executor, deepcad_io, tokens


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m cadseq", description=__doc__.split("\n")[0])
    p.add_argument("program", help="cadseq JSON program (or a DeepCAD dataset file with --deepcad)")
    p.add_argument("-o", "--output", help="export the final solid to this .step or .stl file")
    p.add_argument("--deepcad", action="store_true", help="read a DeepCAD / Fusion 360 Gallery JSON instead")
    p.add_argument("--on-error", choices=["stop", "skip", "raise"], default="stop",
                   help="what to do with a rejected action (default: stop, keep the valid prefix)")
    p.add_argument("--single-solid", action="store_true", help="reject actions that split the model into several bodies")
    p.add_argument("--tokens", help="also save the quantised token rows of the normalised program (.npy)")
    args = p.parse_args(argv)

    seq = deepcad_io.load_deepcad_json(args.program) if args.deepcad else CADSequence.from_json(args.program)
    executor = Executor(ExecConfig(allow_multi_solid=not args.single_solid))
    try:
        state, results = executor.run(seq.actions, on_error=args.on_error)
    except ActionError as e:
        print(f"rejected: {e}")
        return 1
    for r in results:
        line = f"step {r.index:2d}  {r.action.kind:8s} {'ok  ' if r.ok else 'FAIL'}  volume {r.volume:12.3f}  solids {r.n_solids}"
        print(line + (f"   {r.error}" if r.error else ""))
    if args.output and not state.is_empty:
        state.export(args.output)
        print("wrote", args.output)
    if args.tokens:
        np.save(args.tokens, tokens.quantize(seq.normalized()[0].to_vector(), on_clip="ignore"))
        print("wrote", args.tokens)
    return 0 if results and all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
