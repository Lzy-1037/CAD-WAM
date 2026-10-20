"""Qwen3-8B with a continuous CAD-state input branch and an independent next-state head.

Implements qwen3_8b_cad_state_modification_spec.md (§2, §6, §8–§13):

    text (goal + symbolic state) --tokenizer--> input_ids --embed_tokens--> [B, L, 4096]
    geo_latent [B, 1024] --GeometricStateEmbedding--> [B, 16, 4096]
        -> fused at the 16 placeholder positions of the token sequence  -> inputs_embeds [B, S, 4096]
        -> ORIGINAL Qwen3-8B backbone (unchanged)                       -> final hidden states [B, S, 4096]
            -> original LM head  (Action Head)       -> action logits [B, S, vocab]
            -> hidden states at the 16 geometric positions [B, 16, 4096]
                -> StatePredictionHead (mean pool -> LN -> 4096-2048-1024) -> pred_next_state [B, 1024]

The two heads are structurally independent: neither consumes the other's output.  Both may back-propagate into
the shared backbone (no detach).  Loss = CE(action) + state_loss_weight * MSE(state, fp32).

Fusion uses placeholder tokens: the tokenized sequence must contain exactly `num_geo_tokens` (16) contiguous
occurrences of `geo_token_id` per sample; their vocabulary embeddings are discarded and replaced by the
continuous geometric embeddings (spec §6).  `insert_geo_placeholders` / `GEO_PLACEHOLDER_TOKEN` help build such
sequences.  Nothing inside the Qwen3 decoder stack is modified.

Backbones: the design only needs `hidden_size`, `model.embed_tokens`, `lm_head`, a decoder that accepts
`inputs_embeds` and the `<|image_pad|>` token, so the same `CADActionStateMixin` sits on top of Qwen3
(`Qwen3ForCADActionState`) and Qwen2 / Qwen2.5 (`Qwen2ForCADActionState`, used for the backbone-matched comparison
with Pointer-CAD, which fine-tunes Qwen2.5-Instruct).  `cad_state_classes(model_dir)` picks the family from the
checkpoint's config.json, so training and evaluation code never names a family.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.utils import ModelOutput

from .geometric_state_embedding import GeometricStateEmbedding
from .state_prediction_head import StatePredictionHead

# An existing, otherwise unused special token of the Qwen3 tokenizer is used as the positional marker for the
# 16 geometric embeddings (spec §6: placeholders are markers only; their vocabulary embeddings are never used).
# The Qwen2.5 tokenizer has the same token (same id, 151655).
GEO_PLACEHOLDER_TOKEN = "<|image_pad|>"


def _set_cad_state_fields(cfg, latent_dim, num_geo_tokens, geo_token_dim, geo_num_layers, geo_num_heads,
                          state_intermediate_dim, state_loss_weight, geo_token_id):
    cfg.latent_dim = latent_dim
    cfg.num_geo_tokens = num_geo_tokens
    cfg.geo_token_dim = geo_token_dim
    cfg.geo_num_layers = geo_num_layers
    cfg.geo_num_heads = geo_num_heads
    cfg.state_intermediate_dim = state_intermediate_dim
    cfg.state_loss_weight = state_loss_weight
    cfg.geo_token_id = geo_token_id


class Qwen3CADStateConfig(Qwen3Config):
    """Qwen3Config + the fixed hyper-parameters of the two new modules (spec §14)."""
    model_type = "cadwam"

    def __init__(
        self,
        latent_dim: int = 1024,
        num_geo_tokens: int = 16,
        geo_token_dim: int = 256,
        geo_num_layers: int = 2,
        geo_num_heads: int = 8,
        state_intermediate_dim: int = 2048,
        state_loss_weight: float = 1.0,
        geo_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _set_cad_state_fields(self, latent_dim, num_geo_tokens, geo_token_dim, geo_num_layers, geo_num_heads,
                              state_intermediate_dim, state_loss_weight, geo_token_id)


class Qwen2CADStateConfig(Qwen2Config):
    """Qwen2Config (Qwen2 / Qwen2.5) + the same new-module hyper-parameters as Qwen3CADStateConfig."""
    model_type = "qwen2_cad_state"

    def __init__(
        self,
        latent_dim: int = 1024,
        num_geo_tokens: int = 16,
        geo_token_dim: int = 256,
        geo_num_layers: int = 2,
        geo_num_heads: int = 8,
        state_intermediate_dim: int = 2048,
        state_loss_weight: float = 1.0,
        geo_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _set_cad_state_fields(self, latent_dim, num_geo_tokens, geo_token_dim, geo_num_layers, geo_num_heads,
                              state_intermediate_dim, state_loss_weight, geo_token_id)


@dataclass
class CADActionStateOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None            # action logits            [B, S, vocab]
    pred_next_state: Optional[torch.FloatTensor] = None   # predicted next latent    [B, 1024]
    action_loss: Optional[torch.FloatTensor] = None
    state_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    # interface-inspection extras (not part of the architecture)
    last_hidden_state: Optional[torch.FloatTensor] = None  # Qwen final hidden states [B, S, 4096]
    geo_hidden_states: Optional[torch.FloatTensor] = None  # geometric final hidden   [B, 16, 4096]
    state_repr: Optional[torch.FloatTensor] = None         # mean-pooled              [B, 4096]


class CADActionStateMixin:
    """Everything the CAD model adds to a Qwen causal LM; placed BEFORE the `*ForCausalLM` base in the MRO."""

    def _add_cad_state_modules(self, config):
        self.geo_embed = GeometricStateEmbedding(
            latent_dim=config.latent_dim,
            num_geo_tokens=config.num_geo_tokens,
            geo_token_dim=config.geo_token_dim,
            qwen_hidden_dim=config.hidden_size,
            num_layers=config.geo_num_layers,
            num_heads=config.geo_num_heads,
        )
        self.state_head = StatePredictionHead(
            hidden_dim=config.hidden_size,
            intermediate_dim=config.state_intermediate_dim,
            latent_dim=config.latent_dim,
        )

    # ------------------------------------------------------------------ init of the new modules
    def _init_weights(self, module):
        """Backbone modules use the backbone's initialisation; the new modules use plain PyTorch defaults
        (needed because from_pretrained materialises missing weights through this hook)."""
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
                nn.init.zeros_(module.out_proj.bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, GeometricStateEmbedding):
            nn.init.normal_(module.slot_embeddings, std=0.02)
        else:
            super()._init_weights(module)

    # ------------------------------------------------------------------ helpers
    @property
    def geo_token_id(self) -> int:
        if self.config.geo_token_id is None:
            raise ValueError("config.geo_token_id is not set; call set_geo_token_id(tokenizer) first")
        return self.config.geo_token_id

    def set_geo_token_id(self, tokenizer, token: str = GEO_PLACEHOLDER_TOKEN):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(f"placeholder token {token!r} not in tokenizer vocabulary")
        self.config.geo_token_id = int(tid)
        return self.config.geo_token_id

    def geo_positions_mask(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """[B, S] bool mask of the 16 contiguous placeholder positions (validated)."""
        mask = input_ids == self.geo_token_id
        n = self.config.num_geo_tokens
        counts = mask.sum(dim=1)
        if not bool((counts == n).all()):
            raise ValueError(f"every sample must contain exactly {n} geo placeholder tokens, got {counts.tolist()}")
        first = mask.float().argmax(dim=1)
        ar = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        contiguous = (ar >= first[:, None]) & (ar < first[:, None] + n)
        if not bool((contiguous == mask).all()):
            raise ValueError("the geo placeholder tokens must form one contiguous block per sample")
        return mask

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        geo_latent: Optional[torch.FloatTensor] = None,
        target_next_state: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int = 0,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CADActionStateOutput:
        # 1) text embeddings from Qwen's embedding table
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids (with geo placeholders) or inputs_embeds is required")
            inputs_embeds = self.model.embed_tokens(input_ids)                     # [B, S, 4096]

        # 2-3) geometric embeddings fused at the placeholder positions
        geo_mask = None
        if geo_latent is not None and input_ids is not None:
            has_placeholders = bool((input_ids == self.geo_token_id).any())
            if has_placeholders:
                geo_mask = self.geo_positions_mask(input_ids)                      # [B, S]
                geo_embeds = self.geo_embed(geo_latent.to(inputs_embeds.dtype))    # [B, 16, 4096]
                inputs_embeds = inputs_embeds.masked_scatter(
                    geo_mask.unsqueeze(-1).expand_as(inputs_embeds), geo_embeds.to(inputs_embeds.dtype))
            elif past_key_values is None or past_key_values.get_seq_length() == 0:
                raise ValueError("geo_latent given but input_ids contain no geo placeholder tokens")
            # else: incremental decoding step (KV cache holds the prefix incl. geometry) -> nothing to fuse

        # 4) original backbone via inputs_embeds
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state                                 # [B, S, 4096]

        # 5) Action Head = original LM head
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])                  # [B, S', vocab]

        # 6-7) State Prediction Head on the 16 contextualised geometric hidden states (no detach)
        pred_next_state = geo_hidden = state_repr = None
        if geo_mask is not None:
            B = hidden_states.shape[0]
            geo_hidden = hidden_states[geo_mask].view(B, self.config.num_geo_tokens, -1)   # [B, 16, 4096]
            state_repr = self.state_head.pool(geo_hidden)                                   # [B, 4096]
            pred_next_state = self.state_head(geo_hidden)                                   # [B, 1024]

        # 8) losses
        loss = action_loss = state_loss = None
        if labels is not None:
            action_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            loss = action_loss
        if target_next_state is not None and pred_next_state is not None:
            state_loss = F.mse_loss(pred_next_state.float(), target_next_state.float())
            loss = state_loss * self.config.state_loss_weight if loss is None else loss + self.config.state_loss_weight * state_loss

        return CADActionStateOutput(
            loss=loss,
            logits=logits,
            pred_next_state=pred_next_state,
            action_loss=action_loss,
            state_loss=state_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_state=hidden_states,
            geo_hidden_states=geo_hidden,
            state_repr=state_repr,
        )

    # keep geo_latent flowing through generate() (prefill fuses; cached steps skip fusion)
    def prepare_inputs_for_generation(self, *args, geo_latent=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if geo_latent is not None:
            model_inputs["geo_latent"] = geo_latent
        return model_inputs


class Qwen3ForCADActionState(CADActionStateMixin, Qwen3ForCausalLM):
    """ModifiedQwen3ForCausalLM (spec §13): original backbone + original LM head + GeometricStateEmbedding
    + StatePredictionHead.  Loading a stock Qwen3-8B checkpoint initialises only the two new modules."""
    config_class = Qwen3CADStateConfig

    def __init__(self, config: Qwen3CADStateConfig):
        super().__init__(config)            # self.model (backbone) + self.lm_head (Action Head), unchanged
        self._add_cad_state_modules(config)
        self.post_init()


class Qwen2ForCADActionState(CADActionStateMixin, Qwen2ForCausalLM):
    """The same CAD model on a Qwen2 / Qwen2.5 backbone.  Loading a stock Qwen2.5 checkpoint initialises only
    the two new modules."""
    config_class = Qwen2CADStateConfig

    def __init__(self, config: Qwen2CADStateConfig):
        super().__init__(config)
        self._add_cad_state_modules(config)
        self.post_init()


_FAMILIES = {
    "qwen3": (Qwen3CADStateConfig, Qwen3ForCADActionState, Qwen3DecoderLayer),
    "cadwam": (Qwen3CADStateConfig, Qwen3ForCADActionState, Qwen3DecoderLayer),
    "qwen2": (Qwen2CADStateConfig, Qwen2ForCADActionState, Qwen2DecoderLayer),
    "qwen2_cad_state": (Qwen2CADStateConfig, Qwen2ForCADActionState, Qwen2DecoderLayer),
}


def cad_state_classes(model_dir: str):
    """(ConfigClass, ModelClass, DecoderLayerClass) for a stock Qwen2/2.5/3 checkpoint or a trained CAD one,
    chosen from the `model_type` in its config.json (a local directory or a Hugging Face repo id)."""
    path = os.path.join(model_dir, "config.json")
    if os.path.isfile(path):
        model_type = json.load(open(path)).get("model_type")
    else:
        from transformers import PretrainedConfig
        model_type = PretrainedConfig.get_config_dict(model_dir)[0].get("model_type")
    if model_type not in _FAMILIES:
        raise ValueError(f"{model_dir}: model_type {model_type!r} is not supported "
                         f"(expected one of {sorted(_FAMILIES)})")
    return _FAMILIES[model_type]


def insert_geo_placeholders(prefix_text: str, suffix_text: str, num_geo_tokens: int = 16,
                            token: str = GEO_PLACEHOLDER_TOKEN) -> str:
    """Build `prefix + 16 placeholders + suffix` (spec §6: H_input = [H_prefix; H_geo; H_suffix])."""
    return prefix_text + token * num_geo_tokens + suffix_text
