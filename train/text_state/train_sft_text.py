#!/usr/bin/env python
"""Plain SFT of a Qwen3 / Qwen2.5 causal LM on cadseq_stepwise_text[_v5] (text-state baseline) — torchrun + FSDP.

Same skeleton as sft_server/train_sft_cad_state.py (FSDP FULL_SHARD, fp32 sharded master weights +
fused AdamW, bf16 compute, gradient checkpointing, warmup + cosine, save_pretrained checkpoints,
metrics.jsonl) with three throughput changes for the 2.5x longer sequences of this dataset:

1. Sequence packing (default when flash_attn is installed).  Records are pre-tokenised once and packed
   into bins of --pack-length tokens; a bin is one row [1, L] with position_ids restarting at every
   record, which makes flash-attention 2 run in varlen mode (no cross-record attention, no padding,
   attention cost per record instead of per bin).  Every rank processes the same number of tokens per
   step, so no rank waits for a straggler.  Batch size is defined in tokens (--tokens-per-step).
2. Loss only where it is supervised.  The lm_head (152k vocab) is applied only to the assistant
   positions (~37% of the tokens), in chunks, with recomputation, so the fp32 logits of a whole
   10k-token sequence are never materialised — the head's memory drops by >10x and its FLOPs by ~3x.
   This is what lets a 16k-token bin fit and the GPUs stay busy.
3. Without flash_attn: falls back to sdpa with padded micro-batches, length-grouped and rank-balanced
   (LengthGroupedDistributedSampler), --micro-batch records per GPU.

Launch (see train_text_sft.sh):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train_sft_text.py \
      --model /datasets/anon/Qwen3-8B \
      --train dataset/cadseq_stepwise_text/train.parquet --val dataset/cadseq_stepwise_text/test.parquet \
      --out /datasets/anon/text_cadseq/run

Prompt = the model's chat template (enable_thinking=False; Qwen2.5 ignores it); only the assistant answer + eos
is supervised.  --action-only-loss: the target becomes {"action": ...} + eos (no next-state prediction).

v5 additions (2026-09-17):
- any Qwen3 / Qwen2.5 checkpoint: FSDP wraps the model's own decoder-layer class
- token cache: rank 0 tokenises a split ONCE into <cache>/{ids.bin, offsets.npy, prompt_len.npy, state_start.npy,
  meta.json} (renamed into place when complete); the other ranks wait for it and every rank memmaps it
  (v5 text is ~1B tokens: no 4x tokenisation time, no 4x RAM).  Reused by later runs with the same inputs.
- validation on a fixed random sample (seed 0) of --eval-samples records; CE reported overall, on the action
  tokens and on the state tokens (the answer tokens from the start of the "state" value on)
- --eval-every / --save-every N optimizer steps (besides the epoch ends)
"""
import argparse
import functools
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a):
    if rank0():
        tqdm.write(" ".join(str(x) for x in a))   # bar-safe


# --------------------------------------------------------------------------------------- data
TOKEN_CACHE_VERSION = 1
STATE_KEY = ',"state":'


def load_texts(path, limit=None, sample_seed=None):
    """-> (questions, answers, actions or None) from question/answer[/action] columns (fallback: messages)."""
    import pyarrow.parquet as pq
    have = set(pq.read_schema(path).names)
    cols = (["question", "answer"] if {"question", "answer"} <= have else ["messages"]) + (["action"] if "action" in have else [])
    df = pd.read_parquet(path, columns=cols)
    if limit and sample_seed is not None and limit < len(df):
        df = df.sample(n=limit, random_state=sample_seed).sort_index()   # records are sorted by model id
    elif limit:
        df = df.iloc[:limit]
    if "question" in df:
        q, ans = df["question"].tolist(), df["answer"].tolist()
    else:
        q, ans = [m[0]["content"] for m in df["messages"]], [m[1]["content"] for m in df["messages"]]
    return q, ans, (df["action"].tolist() if "action" in df else None)


def build_token_cache(final_dir, path, tok, max_length, action_only, truncation, limit, sample_seed, chunk=20000):
    """Tokenise one split into final_dir (atomic rename).  Record = chat-template prompt + answer + eos;
    state_start = index of the first answer token that starts inside the "state" value (a token straddling
    ',"state":{' counts as action); = record length if there is no state."""
    q, ans, acts = load_texts(path, limit, sample_seed)
    if action_only:
        assert acts is not None, f"{path}: --action-only-loss needs an 'action' column"
        ans = ['{"action":' + x + "}" for x in acts]
    tmp = f"{final_dir}.tmp{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    offsets, plen, sstart, n_trunc, t0 = [0], [], [], 0, time.time()
    with open(os.path.join(tmp, "ids.bin"), "wb") as f:
        for s in range(0, len(q), chunk):
            prompts = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) for x in q[s: s + chunk]]
            P = tok(prompts, add_special_tokens=False).input_ids
            A = tok([x + tok.eos_token for x in ans[s: s + chunk]], add_special_tokens=False, return_offsets_mapping=True)
            for p, a_ids, offs, text in zip(P, A.input_ids, A.offset_mapping, ans[s: s + chunk]):
                k = text.find(STATE_KEY)
                if action_only or k < 0:
                    st = len(a_ids)
                else:
                    c = k + len(STATE_KEY)
                    st = next((j for j, (b, _) in enumerate(offs) if b >= c), len(a_ids))
                ids = p + a_ids
                if len(ids) > max_length:
                    if truncation == "error":
                        raise ValueError(f"record of {len(ids)} tokens > --max-length {max_length}; raise it or --truncation right")
                    n_trunc += 1
                    ids = ids[:max_length]
                f.write(np.asarray(ids, np.int32).tobytes())
                offsets.append(offsets[-1] + len(ids)); plen.append(len(p)); sstart.append(min(len(p) + st, len(ids)))
            log(f"  tokenised {min(s + chunk, len(q))}/{len(q)} records of {os.path.basename(path)} ({time.time() - t0:.0f}s)")
    np.save(os.path.join(tmp, "offsets.npy"), np.asarray(offsets, np.int64))
    np.save(os.path.join(tmp, "prompt_len.npy"), np.asarray(plen, np.int32))
    np.save(os.path.join(tmp, "state_start.npy"), np.asarray(sstart, np.int32))
    json.dump({"source": os.path.abspath(path), "records": len(q), "tokens": offsets[-1], "truncated": n_trunc,
               "max_length": max_length, "action_only": action_only, "limit": limit, "sample_seed": sample_seed,
               "tokenizer": tok.name_or_path, "seconds": round(time.time() - t0, 1)},
              open(os.path.join(tmp, "meta.json"), "w"), indent=2)
    try:
        os.rename(tmp, final_dir)
    except OSError:                       # another run finished the same cache first
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


class TokenStore:
    """Memmapped token cache: get(i) -> (input_ids, labels, state_mask) int32/int32/bool arrays."""

    def __init__(self, d):
        self.dir, self.meta = d, json.load(open(os.path.join(d, "meta.json")))
        self.offsets = np.load(os.path.join(d, "offsets.npy"))
        self.plen = np.load(os.path.join(d, "prompt_len.npy"))
        self.sstart = np.load(os.path.join(d, "state_start.npy"))
        self.ids = np.memmap(os.path.join(d, "ids.bin"), dtype=np.int32, mode="r")
        self.lens = np.diff(self.offsets)

    def __len__(self):
        return len(self.lens)

    def get(self, i):
        ids = np.array(self.ids[self.offsets[i]: self.offsets[i + 1]])
        lab = ids.copy()
        lab[: self.plen[i]] = -100
        sm = np.zeros(len(ids), bool)
        sm[self.sstart[i]:] = True
        return ids, lab, sm


def open_token_store(path, tok, max_length, action_only, truncation, limit=None, sample_seed=None, cache_root=None):
    st = os.stat(path)
    key = json.dumps({"v": TOKEN_CACHE_VERSION, "path": os.path.abspath(path), "size": st.st_size, "mtime": st.st_mtime_ns,
                      "tok": tok.name_or_path, "vocab": len(tok), "eos": tok.eos_token, "max_length": max_length,
                      "action_only": action_only, "truncation": truncation, "limit": limit, "seed": sample_seed},
                     sort_keys=True)
    root = cache_root or os.path.join(os.path.dirname(os.path.abspath(path)), "token_cache")
    d = os.path.join(root, f"{os.path.splitext(os.path.basename(path))[0]}_{hashlib.sha1(key.encode()).hexdigest()[:12]}")
    if not os.path.isfile(os.path.join(d, "meta.json")):
        if rank0():
            os.makedirs(root, exist_ok=True)
            log(f"token cache: building {d}")
            build_token_cache(d, path, tok, max_length, action_only, truncation, limit, sample_seed)
        else:                             # no collective while rank 0 tokenises (it can exceed the NCCL timeout)
            while not os.path.isfile(os.path.join(d, "meta.json")):
                time.sleep(5)
    log(f"token cache: {d}")
    return TokenStore(d)


class PackedBins(Dataset):
    """Per epoch: shuffle records, first-fit-decreasing inside chunks -> bins of <= pack_length tokens.
    Item = one bin: input_ids [L], labels [L], position_ids [L] (restart at each record)."""

    def __init__(self, store, pack_length, world, rank, seed=0, shuffle=True):
        self.store, self.L, self.world, self.rank, self.seed, self.shuffle = store, pack_length, world, rank, seed, shuffle
        self.lens = store.lens
        assert self.lens.max() <= pack_length, "a record is longer than --pack-length"
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(len(self.lens)) if self.shuffle else np.arange(len(self.lens))
        bins = []
        for s in range(0, len(order), 2048):                       # pack inside chunks (keeps randomness)
            chunk = order[s: s + 2048]
            chunk = chunk[np.argsort(-self.lens[chunk], kind="stable")]
            open_bins, open_fill = [], []
            for i in chunk:                                        # first fit decreasing
                for b, fill in enumerate(open_fill):
                    if fill + self.lens[i] <= self.L:
                        open_bins[b].append(i); open_fill[b] += self.lens[i]
                        break
                else:
                    open_bins.append([i]); open_fill.append(int(self.lens[i]))
            bins += open_bins
        if self.shuffle:
            bins = [bins[i] for i in rng.permutation(len(bins))]
        n = len(bins) // self.world * self.world                   # equal count per rank
        self.bins = bins[self.rank: n: self.world]
        self.n_bins_total = n
        self.tokens = sum(sum(self.lens[b]) for b in bins[:n])

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, k):
        ids, lab, pos, sm = [], [], [], []
        for i in self.bins[k]:
            a, l, m = self.store.get(i)
            ids.append(a); lab.append(l); sm.append(m); pos.append(np.arange(len(a), dtype=np.int32))
        return {"input_ids": np.concatenate(ids), "labels": np.concatenate(lab), "position_ids": np.concatenate(pos),
                "state_mask": np.concatenate(sm)}


def collate_packed(batch):
    assert len(batch) == 1, "packed mode uses one bin per micro-batch"
    b = batch[0]
    return {k: torch.from_numpy(v.astype(np.int64))[None] for k, v in b.items()}   # [1, L]


class PaddedRecords(Dataset):
    def __init__(self, store):
        self.store = store
        self.lengths = store.lens

    def __len__(self):
        return len(self.store)

    def __getitem__(self, i):
        ids, lab, sm = self.store.get(i)
        return {"input_ids": ids, "labels": lab, "state_mask": sm}


class LengthGroupedDistributedSampler(Sampler):
    """Shuffle, sort inside mega-batches by length, deal so every rank's micro-batch at a step holds
    records of similar length (rank-balanced, little padding).  Same record set as DistributedSampler."""

    def __init__(self, lengths, world, rank, micro_batch, group_size, seed=0):
        self.lengths, self.world, self.rank, self.micro, self.seed = np.asarray(lengths), world, rank, micro_batch, seed
        self.group = max(micro_batch * world, group_size - group_size % (micro_batch * world))
        self.n_per_rank = (len(self.lengths) // (micro_batch * world)) * micro_batch
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_per_rank

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(len(self.lengths))[: self.n_per_rank * self.world]
        chunks = [perm[i: i + self.group] for i in range(0, len(perm), self.group)]
        order = np.concatenate([c[np.argsort(-self.lengths[c], kind="stable")] for c in chunks])
        blocks = order.reshape(-1, self.world, self.micro)
        blocks = blocks[rng.permutation(len(blocks))]
        return iter(blocks[:, self.rank, :].reshape(-1).tolist())


def make_collate_padded(pad_id):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        L += (-L) % 8
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        sm = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.from_numpy(b["input_ids"].astype(np.int64))
            lab[i, :n] = torch.from_numpy(b["labels"].astype(np.int64)); att[i, :n] = 1
            sm[i, :n] = torch.from_numpy(b["state_mask"].astype(np.int64))
        return {"input_ids": ids, "labels": lab, "attention_mask": att, "state_mask": sm}
    return collate


# --------------------------------------------------------------------------------------- model
class SelectiveLossLM(nn.Module):
    """HF causal LM whose lm_head + CE run only on supervised positions, in recomputed chunks.
    forward -> (ce_sum, n_tokens, ce_sum_state, n_state_tokens); the state split is computed in eval mode only.
    Wrapping (instead of calling lm.model directly) keeps the root FSDP
    unit (embeddings, norm, lm_head) on the normal forward path."""

    def __init__(self, lm, chunk=4096):
        super().__init__()
        self.lm, self.chunk = lm, chunk

    def _chunk_ce(self, h, y):
        return F.cross_entropy(self.lm.lm_head(h).float(), y, reduction="sum")

    def forward(self, input_ids, labels, attention_mask=None, position_ids=None, state_mask=None):
        h = self.lm.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                          use_cache=False).last_hidden_state
        y = labels[:, 1:]
        m = y != -100
        hs, ys = h[:, :-1][m], y[m]                                # [N, H], [N]
        ce = h.new_zeros((), dtype=torch.float32)
        ce_state = h.new_zeros((), dtype=torch.float32)
        if self.training or state_mask is None:
            for s in range(0, hs.shape[0], self.chunk):
                hc, yc = hs[s: s + self.chunk], ys[s: s + self.chunk]
                ce = ce + (checkpoint(self._chunk_ce, hc, yc, use_reentrant=False) if self.training else self._chunk_ce(hc, yc))
            return ce, m.sum(), ce_state, m.sum() * 0
        sm = state_mask[:, 1:].bool()[m]                           # target token lies in the state value
        for s in range(0, hs.shape[0], self.chunk):
            v = F.cross_entropy(self.lm.lm_head(hs[s: s + self.chunk]).float(), ys[s: s + self.chunk], reduction="none")
            ce = ce + v.sum()
            ce_state = ce_state + v[sm[s: s + self.chunk]].sum()
        return ce, m.sum(), ce_state, sm.sum()


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--pack", default="auto", choices=["auto", "on", "off"],
                    help="sequence packing (needs flash_attn); auto = on when flash_attn is installed")
    ap.add_argument("--pack-length", type=int, default=16384, help="tokens per packed bin (>= longest record)")
    ap.add_argument("--tokens-per-step", type=int, default=131072,
                    help="packed mode: optimiser step size in tokens (131072 ~= 64 records of this dataset)")
    ap.add_argument("--micro-batch", type=int, default=2, help="padded mode: records per GPU per micro-batch")
    ap.add_argument("--global-batch", type=int, default=64, help="padded mode: records per optimiser step")
    ap.add_argument("--length-group", type=int, default=512, help="padded mode: records per length-sorted group")
    ap.add_argument("--max-length", type=int, default=10240, help="dataset max is 9454 tokens")
    ap.add_argument("--truncation", default="error", choices=["error", "right"])
    ap.add_argument("--attn", default="auto", choices=["auto", "flash_attention_2", "sdpa"])
    ap.add_argument("--loss-chunk", type=int, default=4096, help="supervised positions per lm_head chunk")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--action-only-loss", action="store_true", help='target = {"action": ...} only')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-samples", type=int, default=1024, help="val records per evaluation: fixed random sample (seed 0)")
    ap.add_argument("--eval-every", type=int, default=0, help="also evaluate every N optimizer steps (0: epoch ends only)")
    ap.add_argument("--save-every", type=int, default=0, help="also save <out>/step-N every N optimizer steps")
    ap.add_argument("--token-cache-dir", default=None, help="default: <data dir>/token_cache")
    ap.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--tiny", action="store_true", help="debug: random small model")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N records")
    ap.add_argument("--max-steps", type=int, default=None, help="debug: stop early")
    a = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(a.seed)

    try:
        import flash_attn  # noqa: F401
        has_fa = True
    except ImportError:
        has_fa = False
    attn = a.attn if a.attn != "auto" else ("flash_attention_2" if has_fa else "sdpa")
    packed = (a.pack == "on") or (a.pack == "auto" and attn == "flash_attention_2")
    if packed and attn != "flash_attention_2":
        raise SystemExit("packing needs flash_attention_2 (records would attend across each other under sdpa)")
    log(f"attention={attn}  mode={'packed' if packed else 'padded'}  flash_attn_installed={has_fa}")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- model (fp32 master weights; bf16 compute via FSDP MixedPrecision) -------------------
    if a.tiny:
        cfg = AutoConfig.from_pretrained(a.model)
        cfg.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, tie_word_embeddings=False, layer_types=["full_attention"] * 2))
        lm = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation=attn)
    else:
        lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, low_cpu_mem_usage=True,
                                                  attn_implementation=attn)
    lm.config.use_cache = False
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = SelectiveLossLM(lm, chunk=a.loss_chunk)

    layer_cls = type(lm.model.layers[0])                           # Qwen3DecoderLayer / Qwen2DecoderLayer / ...
    log(f"backbone {lm.config.model_type}: FSDP wraps {layer_cls.__name__}")
    wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
    model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 auto_wrap_policy=wrap, use_orig_params=True,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                buffer_dtype=torch.bfloat16))
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay, fused=True)
        adam_kind = "fused"
    except (RuntimeError, TypeError, ValueError):
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay, foreach=False)
        adam_kind = "foreach_off"
    n_params = torch.tensor([sum(p.numel() for p in model.parameters())], dtype=torch.float64, device=device)
    dist.all_reduce(n_params)
    log(f"model ready: world={world} params={n_params.item()/1e9:.3f}B adam={adam_kind} action_only_loss={a.action_only_loss}")

    # ---- data (tokenised once by rank 0 into a shared cache, memmapped by every rank) -----------
    t0 = time.time()
    train_recs = open_token_store(a.train, tok, a.max_length, a.action_only_loss, a.truncation, a.limit, None, a.token_cache_dir)
    val_recs = open_token_store(a.val, tok, a.max_length, a.action_only_loss, a.truncation, a.eval_samples, 0, a.token_cache_dir)
    lens = train_recs.lens
    log(f"data ready in {time.time()-t0:.0f}s: {len(train_recs)} train / {len(val_recs)} val records | tokens/record mean "
        f"{lens.mean():.0f} max {lens.max()} | supervised fraction {(lens - train_recs.plen).sum() / lens.sum():.2f} "
        f"(state tokens {(train_recs.lens - train_recs.sstart).sum() / (lens - train_recs.plen).sum():.2f} of it) | "
        f"truncated {train_recs.meta['truncated']}")

    if packed:
        train_ds = PackedBins(train_recs, a.pack_length, world, rank, a.seed, shuffle=True)
        val_ds = PackedBins(val_recs, a.pack_length, world, rank, a.seed, shuffle=False)
        accum = max(1, a.tokens_per_step // (a.pack_length * world))
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_packed, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_packed)
        steps_per_epoch = len(train_ds) // accum
        log(f"packed: {train_ds.n_bins_total} bins of <= {a.pack_length} tokens (fill {train_ds.tokens / (train_ds.n_bins_total * a.pack_length):.2f}), "
            f"{len(train_ds)} per rank | accum {accum} -> ~{accum * world * a.pack_length} tokens "
            f"~= {accum * world * a.pack_length / lens.mean():.0f} records per step | {steps_per_epoch} steps/epoch")
    else:
        train_ds, val_ds = PaddedRecords(train_recs), PaddedRecords(val_recs)
        assert a.global_batch % (a.micro_batch * world) == 0, "global batch must divide by micro*world"
        accum = a.global_batch // (a.micro_batch * world)
        if a.length_group > 0:
            sampler = LengthGroupedDistributedSampler(train_ds.lengths, world, rank, a.micro_batch, a.length_group, a.seed)
        else:
            sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=a.seed, drop_last=True)
        train_dl = DataLoader(train_ds, batch_size=a.micro_batch, sampler=sampler, num_workers=2,
                              collate_fn=make_collate_padded(tok.pad_token_id), pin_memory=True, drop_last=True)
        val_order = np.argsort(-val_ds.lengths, kind="stable").tolist()
        val_dl = DataLoader(val_ds, batch_size=a.micro_batch, sampler=val_order, num_workers=1,
                            collate_fn=make_collate_padded(tok.pad_token_id))
        steps_per_epoch = len(train_dl) // accum
        log(f"padded: micro={a.micro_batch} x world={world} x accum={accum} = global {a.global_batch} records | {steps_per_epoch} steps/epoch")
    total_steps = steps_per_epoch * a.epochs
    warmup = max(1, int(total_steps * a.warmup_ratio))

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    log(f"total {total_steps} steps, warmup {warmup}")

    if rank0():
        os.makedirs(a.out, exist_ok=True)
        json.dump({**vars(a), "world": world, "attn": attn, "packed": packed, "accum": accum,
                   "steps_per_epoch": steps_per_epoch, "total_steps": total_steps},
                  open(os.path.join(a.out, "config.json"), "w"), indent=2)
        mfile = open(os.path.join(a.out, "metrics.jsonl"), "a")

    def to_dev(b):
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}

    @torch.no_grad()
    def evaluate():
        """-> (ce over all answer tokens, ce on action tokens, ce on state tokens)."""
        model.eval()
        tot = torch.zeros(4, device=device)
        for b in tqdm(val_dl, desc="  eval", disable=not rank0(), leave=False, dynamic_ncols=True):
            ce, n, ces, ns = model(**to_dev(b))
            tot += torch.stack([ce, n.float(), ces, ns.float()])
        dist.all_reduce(tot)
        model.train()
        t = tot.tolist()
        return t[0] / t[1], (t[0] - t[2]) / max(t[1] - t[3], 1.0), (t[2] / t[3] if t[3] else float("nan"))

    def log_eval(tag, step, epoch):
        ce, ce_a, ce_s = evaluate()
        log(f"== {tag} | val ce={ce:.4f} action_ce={ce_a:.4f} state_ce={ce_s:.4f} | elapsed {(time.time() - t_start) / 60:.1f} min")
        if rank0():
            mfile.write(json.dumps({"step": step, "epoch": epoch, "val_ce": ce, "val_action_ce": ce_a, "val_state_ce": ce_s}) + "\n")
            mfile.flush()

    def save(tag):
        cfgsd = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfgsd):
            sd = model.state_dict()
        if rank0():
            d = os.path.join(a.out, tag)
            sd = {k[len("lm."):]: (v.to(torch.bfloat16) if a.save_dtype == "bf16" and v.is_floating_point() else v)
                  for k, v in sd.items() if k.startswith("lm.")}
            model.module.lm.save_pretrained(d, state_dict=sd, safe_serialization=True)
            tok.save_pretrained(d)
            log(f"saved {d} ({a.save_dtype})")
        dist.barrier()

    # ---- train -------------------------------------------------------------------------------
    model.train()
    step, t0, t_start = 0, time.time(), time.time()
    bar_total = min(total_steps, a.max_steps) if a.max_steps else total_steps
    bar = tqdm(total=bar_total, desc="train", disable=not rank0(), dynamic_ncols=True, unit="step", mininterval=2.0,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    tok_count = torch.zeros((), device=device)
    for epoch in range(a.epochs):
        (train_ds if packed else sampler).set_epoch(epoch)
        run = torch.zeros(1, device=device)
        n_micro = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_dl):
            batch = to_dev(batch)
            ce, n, _, _ = model(**batch)
            loss = ce / n.clamp(min=1)                            # token mean of this micro-batch
            (loss / accum).backward()
            run += loss.detach(); n_micro += 1
            tok_count += batch["input_ids"].numel()
            if (i + 1) % accum:
                continue
            gn = model.clip_grad_norm_(a.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            last_loss = (run / n_micro).item()
            bar.update(1)
            bar.set_postfix_str(f"ep {epoch + 1}/{a.epochs} loss {last_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                                f"gn {float(gn):.2f} mem {torch.cuda.max_memory_allocated() / 2**30:.0f}G", refresh=False)
            if step % a.log_every == 0:
                run.zero_(); n_micro = 0
                dt, elapsed = time.time() - t0, time.time() - t_start
                tps = tok_count.item() * world / dt                       # tokens/s over all GPUs (this rank x world)
                msg = {"step": step, "epoch": epoch, "loss": round(last_loss, 4), "grad_norm": round(float(gn), 3),
                       "lr": sched.get_last_lr()[0], "sec_per_step": round(dt / a.log_every, 2),
                       "tokens_per_sec": round(tps), "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                       "elapsed_min": round(elapsed / 60, 1), "eta_min": round(elapsed / step * (bar_total - step) / 60, 1)}
                t0 = time.time(); tok_count.zero_()
                log(json.dumps(msg))
                if rank0():
                    mfile.write(json.dumps(msg) + "\n"); mfile.flush()
            if a.eval_every and step % a.eval_every == 0 and step % steps_per_epoch:
                log_eval(f"step {step} (epoch {epoch})", step, epoch)
                t0 = time.time(); tok_count.zero_()
            if a.save_every and step % a.save_every == 0 and step % steps_per_epoch:
                save(f"step-{step}")
                t0 = time.time(); tok_count.zero_()
            if a.max_steps and step >= a.max_steps:
                break
        log_eval(f"epoch {epoch} done (step {step})", step, epoch)
        if a.save_every_epoch or epoch == a.epochs - 1:
            save(f"epoch-{epoch}")
        if a.max_steps and step >= a.max_steps:
            break
    bar.close()
    save("final")
    if rank0():
        mfile.close()
    dist.barrier()
    dist.destroy_process_group()
    log("training done")


if __name__ == "__main__":
    main()
