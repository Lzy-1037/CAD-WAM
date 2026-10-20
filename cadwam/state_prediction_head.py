"""StatePredictionHead — spec §9 (qwen3_8b_cad_state_modification_spec.md).

    geometric final hidden states [B, 16, 4096]
        -> mean pooling over the 16 geometric positions -> [B, 4096]
        -> LayerNorm(4096) -> Linear(4096, 2048) -> GELU -> Linear(2048, 1024)   (no output activation)
        -> predicted next latent  [B, 1024]
"""
import torch
import torch.nn as nn


class StatePredictionHead(nn.Module):
    def __init__(self, hidden_dim: int = 4096, intermediate_dim: int = 2048, latent_dim: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, latent_dim),
        )

    def pool(self, geo_hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, 16, 4096] -> [B, 4096]
        return geo_hidden_states.mean(dim=1)

    def forward(self, geo_hidden_states: torch.Tensor) -> torch.Tensor:
        # geo_hidden_states: [B, 16, 4096]
        state_repr = self.pool(geo_hidden_states)      # [B, 4096]
        state_repr = self.norm(state_repr)
        pred_next_state = self.mlp(state_repr)         # [B, 1024], no activation / normalisation
        return pred_next_state
