# video-captioning/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from data.dataloader import get_dataloader
from transformers import GPT2Tokenizer
from models.c3d_model import C3DEncoder
from models.clip_model import CLIPEncoder
from models.fusion import SimpleFusion
from models.gpt2_decoder import GPT2Decoder
from models.clip_image_encoder import CLIPImageEncoder
from data.coco.train.train_annotations_coco import TRAIN_ANNOTATIONS_COCO


# Example dummy annotations
# Each item: (video_id_without_extension, "caption text")
TRAIN_ANNOTATIONS = TRAIN_ANNOTATIONS_COCO
USE_C3D   = False      # 3‑D ConvNet over 16‑frame clips
USE_CLIP  = True       # single‑image CLIP ViT
USE_FCLIP = False    



def train_one_epoch(loader, c3d_encoder, image_encoder, clip_encoder,
                    fusion_model, gpt2_decoder, optimizer, device):
    if c3d_encoder  is not None: c3d_encoder.train()
    if image_encoder is not None and not Config.FREEZE_CLIP: image_encoder.train()
    if clip_encoder is not None and not Config.FREEZE_CLIP:  clip_encoder.train()
    fusion_model.train()
    gpt2_decoder.train()
    
    total_loss = 0
    
    for imgs, in_ids, att_msks in tqdm(loader, desc="Training"):
        imgs, in_ids, att_msks = imgs.to(device), in_ids.to(device), att_msks.to(device)

        # Forward depending on the approach
        if c3d_encoder is not None:                    # (B, 3, 16, 112, 112)
            enc_out = c3d_encoder(imgs)                # (B, 15360)
            context_embeds = fusion_model(enc_out)     # (B, T_ctx, 768)

        elif image_encoder is not None:                # (B, 3, H, W)
            enc_out = image_encoder(imgs)              # (B, 1, 768)
            context_embeds = fusion_model(enc_out)     # (B, T_ctx, 768)

        else:                                          # legacy per‑frame CLIP
            enc_out = clip_encoder(imgs)               # (B, F, 768)
            context_embeds = fusion_model(enc_out)
        
        # We also need token embeddings for the text portion
        # GPT2 can build them from input_ids, or we can manually embed
        # Easiest is to do: embedding = gpt2_decoder.gpt2.transformer.wte(input_ids)
        B, T = in_ids.shape
        token_embeds = gpt2_decoder.gpt2.transformer.wte(in_ids)  # (B, T, 768)
        
        # Combine context_embeds + token_embeds along time dimension
        inputs_embeds = torch.cat([context_embeds, token_embeds], dim=1)  # (B, T_ctx + T, 768)
        
        # Build an attention mask that allows attending to everything up to current position
        # For simplicity, let's say we allow all context and text tokens
        attention_mask = torch.cat([
            torch.ones((B, context_embeds.size(1)), device=device),
            att_msks
        ], dim=1)
        
        # Our 'labels' for cross-entropy are the text tokens, shifted
        # but in GPT2 usage, we can just feed them with an offset or set 
        # the context positions to -100 so they're ignored
        labels = in_ids.clone()
        
        # We need to shift the labels to align with the text portion ONLY
        # One trick is to pad with -100 for the context positions so they don't affect the loss
        labels_pad = torch.full((B, context_embeds.size(1)), -100, device=device, dtype=torch.long)
        labels = torch.cat([labels_pad, labels], dim=1)
        
        outputs = gpt2_decoder(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask
        )
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    torch.manual_seed(Config.SEED)
    
    device = torch.device(Config.DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    
    # Create your DataLoader
    train_loader = None
    if( USE_CLIP):
        train_loader= get_dataloader(
            feature_dir=Config.IMAGE_DIR,  # or PROCESSED_CLIP_FEATS
            annotations=TRAIN_ANNOTATIONS,
            tokenizer=tokenizer,
            batch_size=Config.BATCH_SIZE,
            shuffle=True
        )
    elif USE_C3D:
        train_loader = get_dataloader(
            feature_dir=Config.C3D_DIR,
            annotations=TRAIN_ANNOTATIONS,
            tokenizer=tokenizer,
            batch_size=Config.BATCH_SIZE,
            shuffle=True
        )
    # Choose your approach
    if USE_C3D:
        c3d_encoder  = C3DEncoder().to(device)
        image_encoder = clip_encoder = None
    elif USE_CLIP:
        image_encoder = CLIPImageEncoder(freeze_clip=Config.FREEZE_CLIP).to(device)
        c3d_encoder  = clip_encoder = None
    else:                     # per‑frame CLIP
        clip_encoder  = CLIPEncoder(freeze_clip=Config.FREEZE_CLIP).to(device)
        c3d_encoder  = image_encoder = None
    in_dim = 768 if USE_CLIP else 15360
    fusion_model = SimpleFusion(
    in_dim=in_dim,
    context_tokens=Config.CONTEXT_TOKENS
).to(device)
    gpt2_decoder = GPT2Decoder().to(device)
    
    # Collect all trainable parameters
    params_to_optimize = list(fusion_model.parameters()) + list(gpt2_decoder.parameters())
    if c3d_encoder is not None:
        params_to_optimize += list(c3d_encoder.parameters())
    if clip_encoder is not None and not Config.FREEZE_CLIP:
        params_to_optimize += list(clip_encoder.parameters())
    if image_encoder is not None and not Config.FREEZE_CLIP:
        params_to_optimize += list(image_encoder.parameters())
    
    
    optimizer = optim.Adam(params_to_optimize, lr=Config.LEARNING_RATE)
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        loss_val = train_one_epoch(
            train_loader,
            c3d_encoder,
            image_encoder,   # ←‑‑ correct object
            clip_encoder,    # ←‑‑ will be None when USE_CLIP = True
            fusion_model,
            gpt2_decoder,
            optimizer,
            device,
        )

        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Loss: {loss_val:.4f}")
        
        # Save checkpoint
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # inside the loop, right before torch.save
        ckpt = {
            "epoch": epoch + 1,
            "fusion_model": fusion_model.state_dict(),
            "gpt2_decoder": gpt2_decoder.state_dict(),
        }
        if image_encoder is not None and not Config.FREEZE_CLIP:
            ckpt["image_encoder"] = image_encoder.state_dict()

        # (same for c3d_encoder or clip_encoder if you ever train them)

        torch.save(ckpt,
                os.path.join(Config.OUTPUT_DIR, f"model_epoch_{epoch+1}.pt"),
                _use_new_zipfile_serialization=False)   # avoids 4‑GB zip bug

if __name__ == "__main__":
    main()
