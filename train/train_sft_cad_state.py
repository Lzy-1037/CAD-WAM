#!/usr/bin/env python
"""SFT for the CAD action+state model on Qwen3 or Qwen2.5 (joint action CE + next-state MSE) — torchrun + FSDP.

Self-contained: torch + transformers + pandas/pyarrow only (runs inside the `verl` conda env but does
not touch verl internals).  Data = cadseq_stepwise parquet (messages + geo_latent + target_next_state).

Launch (see train_cad_sft.sh):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train_sft_cad_state.py \
      --model /datasets/anon/Qwen3-8B \
      --train dataset/cadseq_stepwise/train.parquet --val dataset/cadseq_stepwise/test.parquet \
      --out checkpoints/cad_sft

Details
- backbone family (Qwen3ForCADActionState / Qwen2ForCADActionState) and the FSDP wrap layer come from the
  checkpoint's config.json (cadwam.cad_state_classes)
- prompt rendered with the model's own chat template (Qwen3: enable_thinking=False -> assistant answers start
  with an empty <think></think> block; Qwen2.5: default system prompt, no think block); only assistant tokens
  are supervised
- the 16 <|image_pad|> placeholders in the user message survive templating and are fused by the model
- FSDP FULL_SHARD, fp32 sharded master weights + AdamW, bf16 compute (MixedPrecision), grad checkpointing
- two LR groups: backbone/lm_head vs the newly initialised geo_embed/state_head
- loss = action CE + state_loss_weight * MSE(fp32) (computed inside the model); both parts logged
- checkpoints: full state dict via save_pretrained under <out>/epoch-N (loadable with
  <Family>ForCADActionState.from_pretrained), metrics appended to <out>/metrics.jsonl
"""
import argparse
import functools
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadwam import GEO_PLACEHOLDER_TOKEN, cad_state_classes  # noqa: E402


def rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a):
    if rank0():
        print(*a, flush=True)


# --------------------------------------------------------------------------------------- data
class StepwiseSFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=4096, limit=None, sample_seed=None):
        import pyarrow.parquet as pq
        have = set(pq.read_schema(path).names)   # arrow schema: nested columns keep their top-level name
        # v4 keeps the latents in a memmapped latents.npy beside the parquet and stores only row
        # indices: 526k records x 2 x 1024 floats inline would be ~4.3 GB of RAM in EVERY rank.
        text = ["question", "answer"] if {"question", "answer"} <= have else ["messages"]
        lat = ["geo_latent", "target_next_state"] if "geo_latent" in have else ["geo_idx", "target_idx"]
        df = pd.read_parquet(path, columns=text + lat)
        if limit and sample_seed is not None and limit < len(df):
            # records are sorted by model id: the first N would all come from the lowest-id models
            df = df.sample(n=limit, random_state=sample_seed).sort_index()
        elif limit:
            df = df.iloc[:limit]
        self.tok = tokenizer
        self.max_length = max_length
        if text[0] == "question":
            self.prompts = df["question"].tolist()
            self.answers = df["answer"].tolist()
        else:
            self.prompts = [m[0]["content"] for m in df["messages"]]
            self.answers = [m[1]["content"] for m in df["messages"]]
        if lat[0] == "geo_latent":
            self.geo = np.stack(df["geo_latent"].to_numpy()).astype(np.float32)
            self.tgt = np.stack(df["target_next_state"].to_numpy()).astype(np.float32)
            self.lat = None
        else:
            lat_path = os.path.join(os.path.dirname(os.path.abspath(path)), "latents.npy")
            assert os.path.isfile(lat_path), f"{path} stores latent indices but {lat_path} is missing"
            self.lat = np.load(lat_path, mmap_mode="r")
            self.geo = df["geo_idx"].to_numpy()
            self.tgt = df["target_idx"].to_numpy()
        # sanity: template must keep the 16 placeholders
        ids = self.tok(self._render_prompt(self.prompts[0])).input_ids
        n = sum(1 for t in ids if t == self.tok.convert_tokens_to_ids(GEO_PLACEHOLDER_TOKEN))
        assert n == 16, f"expected 16 geo placeholders after templating, got {n}"

    def _render_prompt(self, user_content):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        p_ids = self.tok(self._render_prompt(self.prompts[i]), add_special_tokens=False).input_ids
        a_ids = self.tok(self.answers[i] + self.tok.eos_token, add_special_tokens=False).input_ids
        if len(p_ids) >= self.max_length:   # truncation would cut the 16 geo tokens at the prompt's end
            raise ValueError(f"record {i}: prompt is {len(p_ids)} tokens >= --max-length {self.max_length}; "
                             f"raise MAX_LENGTH (cadseq_stepwise_v5 needs 5632)")
        ids = (p_ids + a_ids)[: self.max_length]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_length]
        if self.lat is None:
            geo, tgt = self.geo[i], self.tgt[i]
        else:
            geo = np.asarray(self.lat[self.geo[i]], dtype=np.float32)
            tgt = np.asarray(self.lat[self.tgt[i]], dtype=np.float32)
        return {"input_ids": ids, "labels": labels, "geo_latent": geo, "target_next_state": tgt}


def make_collate(pad_id):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att,
                "geo_latent": torch.from_numpy(np.stack([b["geo_latent"] for b in batch])),
                "target_next_state": torch.from_numpy(np.stack([b["target_next_state"] for b in batch]))}
    return collate


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=4, help="per-GPU micro batch")
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5, help="backbone + lm_head")
    ap.add_argument("--new-lr", type=float, default=5e-5, help="geo_embed + state_head (newly initialised)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--state-loss-weight", type=float,
                    default=float(os.environ.get("CAD_STATE_LOSS_WEIGHT", 1.0)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-samples", type=int, default=1024,
                    help="val records scored per evaluation: a fixed random sample (seed 0) of --val")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="also evaluate every N optimizer steps (0: only at epoch ends)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save <out>/step-N every N optimizer steps (0: only epoch ends + final)")
    ap.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True,
                    help="save after every epoch (--no-save-every-epoch: only the final model)")
    ap.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp32"],
                    help="checkpoint precision: bf16 ~16.4GB (training compute dtype), fp32 ~32.8GB")
    ap.add_argument("--tiny", action="store_true", help="debug: random small model instead of the 8B (no pretrained load)")
    ap.add_argument("--limit", type=int, default=None, help="debug: use only the first N records")
    ap.add_argument("--max-steps", type=int, default=None, help="debug: stop early")
    a = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)   # device_id: silences the barrier warning
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- model (fp32 master weights; bf16 compute via FSDP MixedPrecision) -------------------
    ConfigCls, ModelCls, DecoderLayerCls = cad_state_classes(a.model)
    log(f"backbone family: {ModelCls.__name__} (FSDP wraps {DecoderLayerCls.__name__})")
    if a.tiny:
        cfg = ConfigCls(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                        vocab_size=len(tok), max_position_embeddings=8192,
                        tie_word_embeddings=False)
        cfg.geo_token_id = tok.convert_tokens_to_ids(GEO_PLACEHOLDER_TOKEN)
        cfg.state_loss_weight = a.state_loss_weight
        model = ModelCls(cfg)
    else:
        cfg = ConfigCls.from_pretrained(a.model)
        cfg.geo_token_id = tok.convert_tokens_to_ids(GEO_PLACEHOLDER_TOKEN)
        cfg.state_loss_weight = a.state_loss_weight
        model = ModelCls.from_pretrained(
            a.model, config=cfg, dtype=torch.float32, low_cpu_mem_usage=True,
            attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={DecoderLayerCls})
    model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 auto_wrap_policy=wrap, use_orig_params=True,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                buffer_dtype=torch.bfloat16))
    new_params, base_params = [], []
    for n, p in model.named_parameters():
        (new_params if (".geo_embed." in n or ".state_head." in n or
                        n.startswith(("geo_embed.", "state_head."))) else base_params).append(p)
    groups = [{"params": base_params, "lr": a.lr}, {"params": new_params, "lr": a.new_lr}]
    try:    # fused Adam: no multi-GB foreach temporaries during step()
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=a.weight_decay, fused=True)
        adam_kind = "fused"
    except (RuntimeError, TypeError, ValueError):
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=a.weight_decay, foreach=False)
        adam_kind = "foreach_off"
    counts = torch.tensor([sum(p.numel() for p in new_params), sum(p.numel() for p in base_params)],
                          dtype=torch.float64, device=device)
    dist.all_reduce(counts)   # use_orig_params shards params across ranks; sum for the true totals
    log(f"model ready: world={world} new-module params={counts[0].item()/1e6:.1f}M "
        f"base={counts[1].item()/1e9:.3f}B state_loss_weight={a.state_loss_weight} adam={adam_kind}")

    # ---- data --------------------------------------------------------------------------------
    train_ds = StepwiseSFTDataset(a.train, tok, a.max_length, limit=a.limit)
    val_ds = StepwiseSFTDataset(a.val, tok, a.max_length, limit=a.eval_samples, sample_seed=0)
    assert a.global_batch % (a.micro_batch * world) == 0, "global batch must divide by micro*world"
    accum = a.global_batch // (a.micro_batch * world)
    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=a.seed, drop_last=True)
    train_dl = DataLoader(train_ds, batch_size=a.micro_batch, sampler=sampler, num_workers=4,
                          collate_fn=make_collate(tok.pad_token_id), pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=a.micro_batch, shuffle=False, num_workers=2,
                        collate_fn=make_collate(tok.pad_token_id))
    steps_per_epoch = len(train_dl) // accum
    total_steps = steps_per_epoch * a.epochs
    warmup = max(1, int(total_steps * a.warmup_ratio))

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    log(f"data: train={len(train_ds)} val={len(val_ds)} | micro={a.micro_batch} x world={world} x accum={accum} "
        f"= global {a.global_batch} | {steps_per_epoch} steps/epoch, total {total_steps}, warmup {warmup}")

    if rank0():
        os.makedirs(a.out, exist_ok=True)
        json.dump({**vars(a), "world": world, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps},
                  open(os.path.join(a.out, "config.json"), "w"), indent=2)
        mfile = open(os.path.join(a.out, "metrics.jsonl"), "a")

    def to_dev(b):
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = torch.zeros(3, device=device)  # ce_sum, mse_sum, n
        for b in val_dl:
            b = to_dev(b)
            out = model(**b)
            n = b["input_ids"].shape[0]
            tot += torch.stack([out.action_loss.detach() * n, out.state_loss.detach() * n,
                                torch.tensor(float(n), device=device)])
        dist.all_reduce(tot)
        model.train()
        return (tot[0] / tot[2]).item(), (tot[1] / tot[2]).item()

    def save(tag):
        cfgsd = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfgsd):
            sd = model.state_dict()
        if rank0():
            d = os.path.join(a.out, tag)
            if a.save_dtype == "bf16":
                sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
            model.module.save_pretrained(d, state_dict=sd, safe_serialization=True)
            tok.save_pretrained(d)
            log(f"saved {d} ({a.save_dtype})")
        dist.barrier()

    # ---- train -------------------------------------------------------------------------------
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        run = torch.zeros(3, device=device)  # loss, ce, mse (micro sums)
        n_micro = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_dl):
            batch = to_dev(batch)
            out = model(**batch)
            (out.loss / accum).backward()
            run += torch.stack([out.loss.detach(), out.action_loss.detach(), out.state_loss.detach()])
            n_micro += 1
            if (i + 1) % accum:
                continue
            gn = model.clip_grad_norm_(a.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                v = (run / n_micro).tolist(); run.zero_(); n_micro = 0
                lr_now = sched.get_last_lr()[0]
                msg = {"step": step, "epoch": epoch, "loss": round(v[0], 4), "action_ce": round(v[1], 4),
                       "state_mse": round(v[2], 5), "grad_norm": round(float(gn), 3), "lr": lr_now,
                       "sec_per_step": round((time.time() - t0) / a.log_every, 2),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                t0 = time.time()
                log(json.dumps(msg))
                if rank0():
                    mfile.write(json.dumps(msg) + "\n"); mfile.flush()
            if a.eval_every and step % a.eval_every == 0 and step % steps_per_epoch:
                ce, mse = evaluate()
                log(f"== step {step} (epoch {epoch}) | val action_ce={ce:.4f} val state_mse={mse:.5f}")
                if rank0():
                    mfile.write(json.dumps({"step": step, "epoch": epoch, "val_action_ce": ce,
                                            "val_state_mse": mse}) + "\n"); mfile.flush()
                t0 = time.time()
            if a.save_every and step % a.save_every == 0 and step % steps_per_epoch:
                save(f"step-{step}")
                t0 = time.time()
            if a.max_steps and step >= a.max_steps:
                break
        ce, mse = evaluate()
        log(f"== epoch {epoch} done (step {step}) | val action_ce={ce:.4f} val state_mse={mse:.5f}")
        if rank0():
            mfile.write(json.dumps({"step": step, "epoch": epoch, "val_action_ce": ce, "val_state_mse": mse}) + "\n"); mfile.flush()
        if a.save_every_epoch or epoch == a.epochs - 1:
            save(f"epoch-{epoch}")
        if a.max_steps and step >= a.max_steps:
            break
    save("final")
    if rank0():
        mfile.close()
    dist.barrier()
    dist.destroy_process_group()
    log("training done")


if __name__ == "__main__":
    main()
