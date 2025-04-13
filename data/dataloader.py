# video-captioning/data/dataloader.py
import os
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import json
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# import PIL
# from PIL import Image


class VideoCaptionDataset(Dataset):
    """
    A dataset that returns (video_features, caption_text).
    'video_features' is a preprocessed .npy file
    'caption_text' is a single string or list of tokens
    """
    def __init__(self, feature_dir, annotations, tokenizer, max_length=30):
        """
        feature_dir: path with preprocessed .npy files
        annotations: list of tuples (video_id, caption_str)
        tokenizer: GPT2 tokenizer
        max_length: maximum length of tokens
        """
        super().__init__()
        self.feature_dir = feature_dir
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.root = Path(feature_dir)

        self.img_tf = transforms.Compose([
            transforms.Resize(224),          # shortest side = 224
            transforms.CenterCrop(224),      # 224 × 224 exactly
            transforms.ToTensor(),           # 0‑1 float32
        ])

    def __len__(self):
        return len(self.annotations)
    
    def _load_feature_or_image(self, vid: str) -> torch.Tensor:
        """
        Tries, in order:
          <vid>.npy, <vid>.mp4.npy, <vid>.avi.npy  (legacy video features)
          <vid>.<jpg|png|jpeg|bmp>                 (raw image)
        """
        # ---------- 1.  pre‑extracted .npy ----------
        for suffix in [".npy", ".mp4.npy", ".avi.npy"]:
            path = self.root / f"{vid}{suffix}"
            if path.exists():
                arr = np.load(path)
                return torch.from_numpy(arr).float()

        # ---------- 2.  raw image ----------
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = self.root / f"{vid}{ext}"
            if p.exists():
                return self.img_tf(Image.open(p).convert("RGB"))     

        raise FileNotFoundError(f"No feature/image found for id '{vid}'")


    def __getitem__(self, idx):
        vid, caption = self.annotations[idx]

        feats = self._load_feature_or_image(vid)    # tensor

        tokens = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids      = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return feats, input_ids, attention_mask

def collate_fn(batch):
    """
    Collate function to combine a list of (video_features, input_ids, attention_mask)
    into a batch. You may want more complex padding logic here.
    """
    # Each video_features might have different shapes for C3D vs CLIP
    video_feats, in_ids, att_msks = zip(*batch)
    
    # For features, just stack them
    video_feats = torch.stack(video_feats, dim=0) 
    
    # Pad input_ids and attention masks
    # For simplicity, let's do naive padding to the max length in this batch
    max_len = max(x.size(0) for x in in_ids)
    
    padded_in_ids = []
    padded_att_msks = []
    
    for i in range(len(in_ids)):
        seq_len = in_ids[i].size(0)
        pad_len = max_len - seq_len
        padded_in_ids.append(
            torch.cat([in_ids[i], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_att_msks.append(
            torch.cat([att_msks[i], torch.zeros(pad_len, dtype=torch.long)])
        )
    
    padded_in_ids = torch.stack(padded_in_ids, dim=0)
    padded_att_msks = torch.stack(padded_att_msks, dim=0)
    
    return video_feats, padded_in_ids, padded_att_msks

def get_dataloader(feature_dir, annotations, tokenizer, batch_size=8, shuffle=True):
    dataset = VideoCaptionDataset(feature_dir, annotations, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return loader
