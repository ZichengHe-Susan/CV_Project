# video-captioning/evaluate.py

import os
import torch
from tqdm import tqdm
from config import Config
from data.dataloader import get_dataloader
from transformers import GPT2Tokenizer
from models.c3d_model import C3DEncoder
from models.clip_model import CLIPEncoder
from models.fusion import SimpleFusion
from models.gpt2_decoder import GPT2Decoder
from utils.metrics import calculate_bleu
from utils.beam_search import beam_search_decoding
from data.coco.val.val_annotations_coco import VAL_ANNOTATIONS_COCO
# Example dummy annotations (video_id, ref_caption)
VAL_ANNOTATIONS = VAL_ANNOTATIONS_COCO
USE_C3D  = False        
USE_CLIP = True

def generate_caption_c3d(c3d_encoder, fusion_model, gpt2_decoder, video_feats, tokenizer, device, beam_width=1):
    # video_feats shape: (1, 16, 3, 112, 112)
    video_feats = video_feats.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        enc_out = c3d_encoder(video_feats)  # (1, 15360)
        context_embeds = fusion_model(enc_out)  # (1, T_ctx, 768)
    
    if beam_width > 1:
        # custom beam search
        caption_str = beam_search_decoding(
            gpt2_decoder,
            context_embeds,
            beam_width=beam_width,
            max_steps=Config.MAX_SEQ_LEN,
            tokenizer=tokenizer
        )
    else:
        # use GPT-2 generate with greedy
        gen_ids = gpt2_decoder.generate(
            inputs_embeds=context_embeds,
            attention_mask=torch.ones((context_embeds.size(0), context_embeds.size(1)), device=device),
            max_length=Config.MAX_SEQ_LEN,
            num_beams=1
        )
        caption_str = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return caption_str

def generate_caption_clip(clip_encoder, fusion_model, gpt2_decoder, video_feats, tokenizer, device, beam_width=1):
    # video_feats shape: (1, F, 512)
    with torch.no_grad():
        enc_out = clip_encoder(video_feats)        # (1, F, 768)
        context_embeds = fusion_model(enc_out)     # (1, T_ctx, 768)
    
    if beam_width > 1:
        caption_str = beam_search_decoding(
            gpt2_decoder,
            context_embeds,
            beam_width=beam_width,
            max_steps=Config.MAX_SEQ_LEN,
            tokenizer=tokenizer
        )
    else:
        gen_ids = gpt2_decoder.generate(
            context_embeds,
            max_length=Config.MAX_SEQ_LEN,
            num_beams=1
        )
        caption_str = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return caption_str

def main():
    device = torch.device(Config.DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    
    # Load your best checkpoint
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "model_epoch_10.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model instances
    c3d_encoder = None
    clip_encoder = None
    image_encoder = None
    # set to True or False depending on which you used

    
    if USE_C3D:
        c3d_encoder = C3DEncoder().to(device)
        c3d_encoder.load_state_dict(checkpoint["c3d_encoder"])
        in_dim = 15360
        image_encoder = None
    else:                       # single‑image CLIP
        from models.clip_image_encoder import CLIPImageEncoder
        image_encoder = CLIPImageEncoder(freeze_clip=Config.FREEZE_CLIP).to(device)
        if "image_encoder" in checkpoint:          # only saved if trainable
            image_encoder.load_state_dict(checkpoint["image_encoder"])
        in_dim = 768
        c3d_encoder = None

    fusion_model = SimpleFusion(in_dim=in_dim,
                                context_tokens=Config.CONTEXT_TOKENS).to(device)
    fusion_model.load_state_dict(checkpoint["fusion_model"])
    
    
    gpt2_decoder = GPT2Decoder().to(device)
    gpt2_decoder.load_state_dict(checkpoint["gpt2_decoder"])
    
    # Prepare dataset/loader for evaluation
    val_loader = get_dataloader(
        feature_dir=Config.IMAGE_DIR_TEST,      # same folder as training
        annotations=VAL_ANNOTATIONS,       # replace with real val list
        tokenizer=tokenizer,
        batch_size=1,
        shuffle=False,
    )
    
    references = []
    hypotheses = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        video_feats, in_ids, att_msks = batch
        video_feats = video_feats.to(device)
        
        # just take the reference text from in_ids for BLEU
        ref_str = tokenizer.decode(in_ids[0], skip_special_tokens=True)
        references.append(ref_str.split())
        
        if c3d_encoder:
            hyp_str = generate_caption_c3d(
                c3d_encoder, 
                fusion_model, 
                gpt2_decoder, 
                video_feats, 
                tokenizer,
                device,
                beam_width=5  # or 1 for greedy
            )
        else:
            img_feats = video_feats.unsqueeze(1)        # (B, 1, 3, 224, 224)
            enc_out   = image_encoder(img_feats.squeeze(1))  # (1, 1, 768)

            context   = fusion_model(enc_out)           # (1, T_ctx, 768)
            gen_ids = gpt2_decoder.generate(
                inputs_embeds=context,
                attention_mask=torch.ones((context.size(0), context.size(1)), device=device),
                max_length=Config.MAX_SEQ_LEN,
                num_beams=5
            )
            hyp_str   = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        
        hypotheses.append(hyp_str.split())
    
    # Compute average BLEU
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu = calculate_bleu(ref, hyp)
        bleu_scores.append(bleu)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Validation BLEU-4: {avg_bleu:.4f}")

if __name__ == "__main__":
    main()
