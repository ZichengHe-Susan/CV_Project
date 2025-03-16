# video-captioning/models/fusion.py

import torch
import torch.nn as nn
from config import Config

class SimpleFusion(nn.Module):
    """
    Takes (B, F, 768) from the CLIP encoder or (B, 1, 15360) from the C3D encoder 
    and converts it into (B, T_ctx, 768) as a prefix for GPT-2. 
    """
    def __init__(self, context_tokens=Config.CONTEXT_TOKENS):
        super().__init__()
        self.context_tokens = context_tokens
        # For example, a small MLP that goes from 15360 -> 768 * context_tokens
        self.mlp = nn.Sequential(
            nn.Linear(15360, 768 * context_tokens),
            nn.Tanh()
        )

    def forward(self, video_emb):
        """
        If using C3D, video_emb shape might be (B, 15360).
        If using CLIP (w. separate MLP), maybe (B, F, 768). 
        We'll handle the C3D shape in this example.
        """
        # Suppose we're working with C3D: (B, 15360) -> expand to (B, context_tokens, 768)
        if len(video_emb.shape) == 2:
            out = self.mlp(video_emb)  # (B, 768 * context_tokens)
            B, dim = out.shape
            out = out.view(B, self.context_tokens, 768)
            return out
        
        # If using CLIP, it's already (B, F, 768). Possibly we flatten or pool frames:
        elif len(video_emb.shape) == 3:
            # Let F = number of frames. We want T_ctx = 20 tokens total.
            B, F, E = video_emb.shape
            # Flatten
            out = video_emb.view(B, F*E)
            # project to context_tokens * 768
            out = self.mlp(out).view(B, self.context_tokens, 768)
            return out
        else:
            raise ValueError("Unexpected shape for video_emb.")
