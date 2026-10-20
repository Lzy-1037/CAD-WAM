#!/usr/bin/env python
"""Forward-pass interface validation for Qwen3ForCADActionState with dummy inputs (NO training).

Loads the pretrained Qwen3-8B weights into the modified model, builds dummy text (goal + symbolic state with
16 geometry placeholders) and a dummy geo_latent [B, 1024], runs one forward pass under torch.no_grad() and
asserts every tensor interface listed in the task:

  geo_latent [B,1024] | GeometricStateEmbedding out [B,16,H] | fused input embeds [B,S,H]
  final hidden [B,S,H] | action logits [B,S,vocab] | geometric final hidden [B,16,H]
  mean-pooled state repr [B,H] | predicted next state [B,1024]
  where H = config.hidden_size (4096 for Qwen3-8B, 2048 for Qwen3-1.7B, 2560 for Qwen3-4B)

Usage:  python -m cadwam.test_forward_dummy [--model Qwen/Qwen3-8B] [--device cpu|cuda] [--dtype fp32|bf16]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadwam import (GEO_PLACEHOLDER_TOKEN, GeometricStateEmbedding, Qwen3CADStateConfig,  # noqa: E402
                             Qwen3ForCADActionState, StatePredictionHead, insert_geo_placeholders)
from transformers import AutoTokenizer  # noqa: E402


def check(name, t, shape):
    assert tuple(t.shape) == tuple(shape), f"{name}: expected {tuple(shape)}, got {tuple(t.shape)}"
    assert torch.isfinite(t).all(), f"{name}: non-finite values"
    print(f"  OK  {name:38s} {tuple(t.shape)}  dtype={t.dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    torch.manual_seed(0)

    # ---------------------------------------------------------------- load
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    config = Qwen3CADStateConfig.from_pretrained(a.model)
    config.geo_token_id = tok.convert_tokens_to_ids(GEO_PLACEHOLDER_TOKEN)
    model = Qwen3ForCADActionState.from_pretrained(a.model, config=config, dtype=dtype, device_map=a.device)
    model.eval()
    H, V, N, D = config.hidden_size, config.vocab_size, config.num_geo_tokens, config.latent_dim
    print(f"loaded {a.model} in {time.time() - t0:.0f}s on {a.device} ({a.dtype}); hidden={H} vocab={V} "
          f"layers={config.num_hidden_layers} geo_token_id={config.geo_token_id} ({GEO_PLACEHOLDER_TOKEN!r})")
    # any Qwen3 size works: the new modules are built from config.hidden_size, so H is 4096 for
    # Qwen3-8B, 2048 for Qwen3-1.7B, 2560 for Qwen3-4B, 5120 for 14B/32B
    assert config.model_type in ("qwen3", "cadwam") and H > 0

    # ---------------------------------------------------------------- structure checks
    n_total = sum(p.numel() for p in model.parameters())
    n_geo = sum(p.numel() for p in model.geo_embed.parameters())
    n_state = sum(p.numel() for p in model.state_head.parameters())
    n_backbone = sum(p.numel() for p in model.model.parameters())
    n_lm = model.lm_head.weight.numel()
    print(f"params: total {n_total/1e9:.3f}B = backbone {n_backbone/1e9:.3f}B + lm_head {n_lm/1e6:.1f}M "
          f"+ GeometricStateEmbedding {n_geo/1e6:.2f}M + StatePredictionHead {n_state/1e6:.2f}M")
    assert isinstance(model.geo_embed, GeometricStateEmbedding) and isinstance(model.state_head, StatePredictionHead)
    assert len(model.model.layers) == config.num_hidden_layers
    assert isinstance(model.lm_head, nn.Linear) and model.lm_head.out_features == V
    assert len(model.geo_embed.transformer.layers) == 2
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not bad, f"non-finite parameters after loading/initialisation: {bad[:5]}"
    for n, p in model.named_parameters():        # new modules must be initialised (not torch.empty garbage)
        if n.startswith(("geo_embed.", "state_head.")) and p.dim() > 1:
            assert float(p.std()) < 1.0, f"suspicious init for {n}: std={float(p.std()):.3f}"
    print(f"  OK  backbone {config.num_hidden_layers} layers unchanged, original lm_head kept "
          f"(tied={bool(getattr(config, 'tie_word_embeddings', False))}), new modules present & finite")

    # ---------------------------------------------------------------- dummy inputs
    B = a.batch
    texts = []
    for i in range(B):
        prefix = f"<GOAL>\nCreate a rectangular plate with {2 + i} through holes.\n</GOAL>\n<GEO>"
        suffix = ("</GEO>\n<STATE>\nsketches: 1; bodies: 1; bbox: [0,0,0]-[8,4,1]; faces: 6; edges: 12\n</STATE>\n"
                  "<ACTION>\n" + '{"id": "e1", "op": "extrude"' + ", " * i + "}\n</ACTION>")
        texts.append(insert_geo_placeholders(prefix, suffix, N))
    enc = tok(texts, return_tensors="pt", padding=True)
    input_ids, attention_mask = enc.input_ids.to(a.device), enc.attention_mask.to(a.device)
    S = input_ids.shape[1]
    geo_latent = torch.randn(B, D, device=a.device)
    target_next_state = torch.randn(B, D, device=a.device)
    labels = input_ids.clone(); labels[attention_mask == 0] = -100
    labels[:, : S // 2] = -100                    # only "action" positions contribute (prompt ignored)
    print(f"dummy batch: B={B} S={S} (text tokens + {N} geo placeholders), padding side={tok.padding_side}")
    check("geo_latent", geo_latent, (B, D))
    assert ((input_ids == config.geo_token_id).sum(1) == N).all()

    # ---------------------------------------------------------------- forward (no grad, no training)
    with torch.no_grad():
        # intermediate interfaces, computed exactly as inside forward()
        geo_embeds = model.geo_embed(geo_latent.to(dtype))
        check("GeometricStateEmbedding output", geo_embeds, (B, N, H))
        geo_mask = model.geo_positions_mask(input_ids)
        text_embeds = model.model.embed_tokens(input_ids)
        fused = text_embeds.masked_scatter(geo_mask.unsqueeze(-1).expand_as(text_embeds), geo_embeds)
        check("combined Qwen input embeddings", fused, (B, S, H))
        assert torch.equal(fused[geo_mask].view(B, N, H), geo_embeds), "geo positions must hold the geometric embeddings"
        assert not torch.equal(fused[geo_mask], text_embeds[geo_mask]), "placeholder vocab embeddings must not be used"

        t1 = time.time()
        out = model(input_ids=input_ids, attention_mask=attention_mask, geo_latent=geo_latent,
                    labels=labels, target_next_state=target_next_state)
        print(f"forward pass: {time.time() - t1:.1f}s")

    check("Qwen final hidden states", out.last_hidden_state, (B, S, H))
    check("action logits", out.logits, (B, S, V))
    check("geometric final hidden states", out.geo_hidden_states, (B, N, H))
    check("mean-pooled state representation", out.state_repr, (B, H))
    check("predicted next state", out.pred_next_state, (B, D))
    assert torch.allclose(out.state_repr, out.geo_hidden_states.mean(1)), "state_repr must be the mean over 16 slots"
    assert torch.equal(out.geo_hidden_states, out.last_hidden_state[geo_mask].view(B, N, H)), \
        "geometric hidden states must come from the placeholder positions of the final hidden states"
    assert out.action_loss is not None and out.state_loss is not None and torch.isfinite(out.loss)
    assert torch.allclose(out.loss, out.action_loss + config.state_loss_weight * out.state_loss)
    print(f"  OK  losses: action(CE)={out.action_loss.item():.4f} state(MSE,fp32)={out.state_loss.item():.4f} "
          f"total={out.loss.item():.4f} (state_loss_weight={config.state_loss_weight})")
    # both heads read only the shared backbone output (independence: state branch ignores logits, and vice versa)
    with torch.no_grad():
        pred_direct = model.state_head(out.geo_hidden_states)
    assert torch.allclose(pred_direct, out.pred_next_state), "state head must depend only on geometric hidden states"
    print("ALL INTERFACE CHECKS PASSED")


if __name__ == "__main__":
    main()
