#!/usr/bin/env python
"""Download Qwen3 base models into /datasets/anon/<official name>/ on the H800 server.

Lands real files (not cache symlinks) in directories named exactly like the HF repos, matching the
existing /datasets/anon/Qwen3-8B layout, so they can be used directly as
    --model /datasets/anon/Qwen3-4B

Resumable: re-running skips what is already complete.  After each model it prints the config values
that decide how the CAD wrapper resizes (hidden_size drives GeometricStateEmbedding's output
projection and StatePredictionHead's input) and verifies the folder actually loads.

  python download_qwen3.py                       # 0.6B, 1.7B, 4B  (~13 GB total)
  python download_qwen3.py --models Qwen3-4B     # just one
  python download_qwen3.py --dest /some/other/dir
"""
import argparse
import inspect
import json
import os
import sys
import time

DEFAULT_MODELS = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
# weights + tokenizer + configs; skips README/images/pytorch_model.bin duplicates
PATTERNS = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"]


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f}{u}"
        n /= 1024


def dir_size(p):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs if not f.startswith("."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/datasets/anon")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--org", default="Qwen")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    # `local_dir_use_symlinks` exists on huggingface_hub<1.0 and is gone in 1.x, where local_dir
    # already writes real files; pass it only when the installed version accepts it.
    kw = {}
    if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
        kw["local_dir_use_symlinks"] = False
    if "max_workers" in inspect.signature(snapshot_download).parameters:
        kw["max_workers"] = a.workers

    os.makedirs(a.dest, exist_ok=True)
    print(f"destination {a.dest}  |  models: {', '.join(a.models)}", flush=True)
    ok, failed = [], []
    for name in a.models:
        repo = f"{a.org}/{name}"
        out = os.path.join(a.dest, name)
        print(f"\n=== {repo} -> {out}", flush=True)
        t0 = time.time()
        try:
            snapshot_download(repo_id=repo, local_dir=out, allow_patterns=PATTERNS, **kw)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(name)
            continue
        # ---- verify it is actually usable, and report the dims that matter to the CAD wrapper
        try:
            cfg = json.load(open(os.path.join(out, "config.json")))
            shards = [f for f in os.listdir(out) if f.endswith(".safetensors")]
            assert shards, "no .safetensors found"
            idx = os.path.join(out, "model.safetensors.index.json")
            if os.path.isfile(idx):
                need = {v for v in json.load(open(idx))["weight_map"].values()}
                missing = need - set(shards)
                assert not missing, f"missing shards {sorted(missing)}"
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(out)
            pad = tok.convert_tokens_to_ids("<|image_pad|>")
            assert pad is not None and pad != tok.unk_token_id, "no <|image_pad|> placeholder token"
            print(f"  OK  {human(dir_size(out))} in {time.time()-t0:.0f}s, {len(shards)} shard(s)")
            print(f"      hidden_size={cfg['hidden_size']}  layers={cfg['num_hidden_layers']}  "
                  f"vocab={cfg['vocab_size']}  tie_word_embeddings={cfg.get('tie_word_embeddings')}  "
                  f"<|image_pad|>={pad}")
            ok.append(name)
        except Exception as e:
            print(f"  VERIFY FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(name)

    print(f"\ndone: {len(ok)} ok {ok}" + (f" | {len(failed)} FAILED {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
