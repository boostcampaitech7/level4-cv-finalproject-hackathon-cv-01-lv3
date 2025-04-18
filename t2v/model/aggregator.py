import torch
import torch.nn as nn

class SpaceTimeBridgeEncoder(nn.Module):
    def __init__(self, transformer_dim, max_T=32, max_P=1024):
        super().__init__()
        self.max_T = max_T
        self.max_P = max_P
        self.transformer_dim = transformer_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.transformer_dim))  # (1, 1, D)

        # Spatial, Temporal Positional Embedding
        self.temporal_pos = nn.Parameter(torch.randn(1, self.max_T, 1, self.transformer_dim))  # (1, T, 1, D)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.max_P, self.transformer_dim))  # (1, 1, P, D)

        # Combined Spatial-Temporal Attention Transformer
        self.spatial_temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8, dim_feedforward=7168),
            num_layers=1
        )

    def forward(self, vit_embeds):
        # vit_embeds: (B, T, P, D)
        B, T, P, D = vit_embeds.shape

        if D != self.transformer_dim:
            raise ValueError(f"Embedding dim D ({D}) must match transformer_dim ({self.transformer_dim})")

        # Positional Encoding
        vit_embeds = vit_embeds + self.temporal_pos[:, :T, :, :] + self.spatial_pos[:, :, :P, :]  # (B, T, P, D)

        # Flatten spatial and temporal
        vit_embeds = vit_embeds.view(B, T * P, D)  # (B, T*P, D)

        # Expand cls_token for each batch
        cls_tokens = self.cls_token.expand(B, 1, D)  # (B, 1, D)

        # Concatenate cls_token to the beginning of the sequence
        vit_embeds = torch.cat([cls_tokens, vit_embeds], dim=1)  # (B, 1 + T*P, D)

        # Transformer expects (sequence_length, batch_size, embedding_dim)
        vit_embeds = vit_embeds.transpose(0, 1)  # (1 + T*P, B, D)

        transformer_out = self.spatial_temporal_transformer(vit_embeds)  # (1 + T*P, B, D)
        transformer_out = transformer_out.transpose(0, 1)  # (B, 1 + T*P, D)

        # Extract cls_token output
        final_embed = transformer_out[:, 0, :]  # (B, D)

        return final_embed