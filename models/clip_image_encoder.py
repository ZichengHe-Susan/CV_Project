# video-captioning/models/clip_image_encoder.py
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class CLIPImageEncoder(nn.Module):
    """
    Encodes a single RGB image with CLIP's vision tower and returns a
    (B, 1, 768) tensor ready for the fusion module.

    Args
    ----
    model_name   : str   Hugging‑Face id of the vision backbone
                        (default: "openai/clip-vit-base-patch32").
    freeze_clip  : bool  If True, CLIP weights are frozen (fine‑tune = False).
    use_pool     : bool  If True, use the model's pooled output; otherwise
                         take the CLS token from last_hidden_state.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_clip: bool = True,
        use_pool: bool = True,
    ):
        super().__init__()
        self.processor  = CLIPImageProcessor.from_pretrained(model_name)
        self.vision     = CLIPVisionModel.from_pretrained(model_name)
        self.use_pool   = use_pool
        self.hidden_dim = self.vision.config.hidden_size  # 768 for ViT‑B/32

        if freeze_clip:
            for p in self.vision.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs : torch.Tensor of shape (B, 3, H, W) in [0,1] RGB
        Returns CLIP‑normalized tensor of shape (B, 3, 224, 224).
        """
        # The HF processor wants PIL or numpy, so we replicate its
        # normalization here to stay on‑device and avoid Python overhead.
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=imgs.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=imgs.device).view(1, 3, 1, 1)

        imgs = nn.functional.interpolate(imgs, size=224, mode="bilinear",
                                         align_corners=False)
        return (imgs - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) tensor in [0,1] RGB
        Returns: (B, 1, hidden_dim)  — a single token per image.
        """
        x = self._preprocess(x)                 # (B, 3, 224, 224)
        outputs = self.vision(pixel_values=x)

        if self.use_pool and outputs.pooler_output is not None:
            img_emb = outputs.pooler_output     # (B, hidden_dim)
        else:
            img_emb = outputs.last_hidden_state[:, 0, :]  # CLS token

        return img_emb.unsqueeze(1)             # (B, 1, hidden_dim)
