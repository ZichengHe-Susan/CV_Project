# video-captioning/models/fusion.py
import torch
import torch.nn as nn


class SimpleFusion(nn.Module):
    """
    Converts a *single* video/image embedding into a sequence of
    `context_tokens` GPT‑2‑sized (768‑D) prefix tokens.

    Parameters
    ----------
    in_dim : int
        Dimension of the flattened input vector
        (e.g. 15360 for C3D, 768 for a CLIP image embedding,
        or 768*F if you concatenate F CLIP frame embeddings).
    context_tokens : int, default 4
        How many prefix tokens to generate for GPT‑2.
    """

    def __init__(self, in_dim: int, context_tokens: int = 4):
        super().__init__()
        self.context_tokens = context_tokens

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, context_tokens * 768),
            nn.GELU(),
            nn.Linear(context_tokens * 768, context_tokens * 768),
        )

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape can be:
            (B, 15360)             – C3D
            (B, 1, 768) or (B, 768) – single CLIP image
            (B, F, 768)            – CLIP frame sequence

        Returns
        -------
        (B, context_tokens, 768)
        """
        # unify to (B, D)
        if x.dim() == 3:                    # (B, F, 768)  or (B, 1, 768)
            x = x.mean(1)                   # simple average pool
        elif x.dim() != 2:                  # unexpected rank
            x = x.flatten(1)

        out = self.mlp(x)                   # (B, ctx*768)
        B = out.size(0)
        return out.view(B, self.context_tokens, 768)
