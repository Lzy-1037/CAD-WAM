"""GeometricStateEmbedding — spec §5 (qwen3_8b_cad_state_modification_spec.md).

Maps one continuous CAD latent z_t in R^1024 to 16 Qwen-compatible geometric embeddings in R^{16 x 4096}:

    [B, 1024] -> LayerNorm(1024) -> Linear(1024, 16*256) -> reshape [B, 16, 256]
              -> + learnable slot embeddings [1, 16, 256]
              -> 2-layer bidirectional TransformerEncoder (d=256, 8 heads, FFN 1024, GELU, pre-norm, no causal mask)
              -> LayerNorm(256) -> Linear(256, 4096) -> [B, 16, 4096]
"""
import torch
import torch.nn as nn


class GeometricStateEmbedding(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        num_geo_tokens: int = 16,
        geo_token_dim: int = 256,
        qwen_hidden_dim: int = 4096,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_geo_tokens = num_geo_tokens
        self.geo_token_dim = geo_token_dim

        self.latent_norm = nn.LayerNorm(latent_dim)
        self.expand = nn.Linear(latent_dim, num_geo_tokens * geo_token_dim)
        self.slot_embeddings = nn.Parameter(torch.randn(1, num_geo_tokens, geo_token_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=geo_token_dim,
            nhead=num_heads,
            dim_feedforward=geo_token_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # bidirectional: no mask is ever passed to the encoder (all 16 slots attend to each other)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.output_norm = nn.LayerNorm(geo_token_dim)
        self.output_proj = nn.Linear(geo_token_dim, qwen_hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim]
        B = z.shape[0]
        z = self.latent_norm(z)
        x = self.expand(z)                                        # [B, 16*256]
        x = x.view(B, self.num_geo_tokens, self.geo_token_dim)    # [B, 16, 256]
        x = x + self.slot_embeddings
        x = self.transformer(x)                                   # [B, 16, 256]  (bidirectional)
        x = self.output_norm(x)
        x = self.output_proj(x)                                   # [B, 16, 4096]
        return x
