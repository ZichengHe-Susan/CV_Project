# video-captioning/data/dataloader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_id, caption_str = self.annotations[idx]
        
        # Load video features from .npy
        feat_path = os.path.join(self.feature_dir, video_id + ".mp4.npy")
        if not os.path.exists(feat_path):
            # Try .avi or you can handle differently
            feat_path = os.path.join(self.feature_dir, video_id + ".avi.npy")
        
        video_features = np.load(feat_path)  # shape depends on your pre-processing
        video_features = torch.from_numpy(video_features).float()
        
        # Tokenize the caption
        tokens = self.tokenizer(
            caption_str,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)      # shape: (seq_len)
        attention_mask = tokens["attention_mask"].squeeze(0)  # shape: (seq_len)

        return video_features, input_ids, attention_mask

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
