# cadwam — Qwen3-8B with continuous CAD-state input and next-state prediction

Implementation of `../qwen3_8b_cad_state_modification_spec.md`. **Model structure + interface validation only —
no training code, optimizer, scheduler, LoRA, or data pipeline is included.**

```
text goal + symbolic state ──tokenizer──▶ input_ids ──embed_tokens──▶ [B, L, 4096] ─┐
geo_latent [B, 1024] ──GeometricStateEmbedding──▶ [B, 16, 4096] ────────────────────┤ fused at 16 placeholder
                                                                                     ▼ positions (masked_scatter)
                                                          inputs_embeds [B, S, 4096]
                                                                       │
                                                  ORIGINAL Qwen3-8B backbone (36 layers, unchanged)
                                                                       │
                                                      final hidden states [B, S, 4096]
                                                    ┌──────────────────┴──────────────────┐
                                     original lm_head (Action Head)      hidden states at the 16 geo positions
                                                    │                     [B, 16, 4096] → mean pool → [B, 4096]
                                       action logits [B, S, vocab]        → LN → Linear(4096,2048) → GELU → Linear(2048,1024)
                                                                          → pred_next_state [B, 1024]
```

## Files

| file | content |
|---|---|
| `geometric_state_embedding.py` | `GeometricStateEmbedding` (spec §5): LN(1024) → Linear(1024, 16·256) → reshape → + slot embeddings → 2-layer bidirectional `nn.TransformerEncoder` (d 256, 8 heads, FFN 1024, GELU, pre-norm, no mask) → LN(256) → Linear(256, 4096) |
| `state_prediction_head.py` | `StatePredictionHead` (spec §9): mean pooling over 16 slots → LN(4096) → Linear(4096, 2048) → GELU → Linear(2048, 1024), no output activation |
| `modeling_cadwam.py` | `Qwen3CADStateConfig` (Qwen3Config + fixed hyper-parameters, `state_loss_weight`, `geo_token_id`), `Qwen3ForCADActionState` (subclass of `Qwen3ForCausalLM`: original `model` backbone + original `lm_head` + the two new modules), `CADActionStateOutput`, `insert_geo_placeholders` |
| `test_forward_dummy.py` | loads pretrained Qwen3-8B, dummy text + dummy `geo_latent [B,1024]`, one forward pass, asserts all interface shapes |

## Fusion convention (spec §6)

The tokenized sequence carries **16 contiguous placeholder tokens** (`<|image_pad|>`, an existing but unused Qwen3
special token — configurable via `config.geo_token_id` / `model.set_geo_token_id(tokenizer, token)`).  Their vocabulary
embeddings are discarded: `inputs_embeds.masked_scatter(mask, geo_embeds)` writes the 16 continuous geometric
embeddings at those positions, and the backbone is called with `inputs_embeds`.  The same mask recovers the 16
contextualized geometric hidden states from the final hidden states for the state head.

```python
from cadwam import Qwen3CADStateConfig, Qwen3ForCADActionState, insert_geo_placeholders, GEO_PLACEHOLDER_TOKEN
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
cfg = Qwen3CADStateConfig.from_pretrained("Qwen/Qwen3-8B"); cfg.geo_token_id = tok.convert_tokens_to_ids(GEO_PLACEHOLDER_TOKEN)
model = Qwen3ForCADActionState.from_pretrained("Qwen/Qwen3-8B", config=cfg, dtype=torch.bfloat16)   # backbone+lm_head pretrained, new modules initialised
text = insert_geo_placeholders("<GOAL>...</GOAL>\n", "\n<STATE>...</STATE>\n<ACTION>")               # prefix + 16 placeholders + suffix
out = model(**tok(text, return_tensors="pt"), geo_latent=z_t, labels=labels, target_next_state=z_next)
out.logits            # [B, S, vocab]   action logits (original LM head)
out.pred_next_state   # [B, 1024]       one-shot next-state regression
out.loss              # action CE + state_loss_weight * MSE(fp32)   (when targets given)
```

`forward` signature: `input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, geo_latent,
target_next_state, use_cache, logits_to_keep`.  `generate(..., geo_latent=z)` works: the prefill step fuses the
geometry, cached decoding steps (no placeholders in the new token) skip fusion.

## What is *not* changed / added
* No edit to the installed Transformers source; nothing inside the 36 Qwen3 decoder layers, RMSNorm, GQA, RoPE, MLP.
* No new action head, no `<STATE_QUERY>`, no attention pooling, gating, residual/autoregressive state prediction,
  no extra losses, no detach before the state head.
