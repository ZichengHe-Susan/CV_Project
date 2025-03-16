# video-captioning/inference.py

import os
import torch
from config import Config
from data.preprocess import extract_frames_c3d, extract_frames_clip
from PIL import Image
from models.c3d_model import C3DEncoder
from models.clip_model import CLIPEncoder
from models.fusion import SimpleFusion
from models.gpt2_decoder import GPT2Decoder
from utils.beam_search import beam_search_decoding
from transformers import GPT2Tokenizer

def load_model_checkpoint(ckpt_path, use_c3d=True):
    device = torch.device(Config.DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    c3d_encoder = None
    clip_encoder = None
    if use_c3d:
        c3d_encoder = C3DEncoder().to(device)
        c3d_encoder.load_state_dict(checkpoint["c3d_encoder"])
    else:
        clip_encoder = CLIPEncoder(freeze_clip=Config.FREEZE_CLIP).to(device)
        clip_encoder.load_state_dict(checkpoint["clip_encoder"])
    
    fusion_model = SimpleFusion(context_tokens=Config.CONTEXT_TOKENS).to(device)
    fusion_model.load_state_dict(checkpoint["fusion_model"])
    
    gpt2_decoder = GPT2Decoder().to(device)
    gpt2_decoder.load_state_dict(checkpoint["gpt2_decoder"])
    
    return c3d_encoder, clip_encoder, fusion_model, gpt2_decoder

def run_inference_on_video(video_path, ckpt_path, use_c3d=True, beam_width=5):
    device = torch.device(Config.DEVICE)
    c3d_encoder, clip_encoder, fusion_model, gpt2_decoder = load_model_checkpoint(ckpt_path, use_c3d)
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    
    if use_c3d:
        frames = extract_frames_c3d(video_path, Config.NUM_FRAMES_C3D, 112)  # np array
        frames_tensor = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1, 16, 3, 112, 112)
        with torch.no_grad():
            enc_out = c3d_encoder(frames_tensor.permute(0,2,1,3,4))
            context_embeds = fusion_model(enc_out)
        
        caption_str = beam_search_decoding(
            gpt2_decoder,
            context_embeds,
            beam_width=beam_width,
            max_steps=Config.MAX_SEQ_LEN,
            tokenizer=tokenizer
        )
    else:
        # if using CLIP
        frames_pil = extract_frames_clip(video_path, Config.NUM_FRAMES_CLIP, 224)
        # If you have a direct approach, do clip_encoder.model.encode_image(...) 
        # but if storing features offline, do that first
        # Here we assume we do not store them, but pass through the clip_encoder
        # We need the raw CLIP embeddings though. We'll do a workaround:
        
        # Actually, for quick demonstration let's say we can't do that on the fly 
        # unless we also have the original clip model + preprocess. 
        # We'll do a simplified approach if pre-extracted. 
        # If truly on the fly, see preprocess_clip_features logic.
        
        raise NotImplementedError("On-the-fly CLIP inference not fully shown here.")
    
    return caption_str

if __name__ == "__main__":
    video_path_test = "test_video.mp4"
    ckpt_path = os.path.join(Config.OUTPUT_DIR, "model_epoch_10.pt")
    caption = run_inference_on_video(video_path_test, ckpt_path, use_c3d=True, beam_width=5)
    print("Generated Caption:", caption)
