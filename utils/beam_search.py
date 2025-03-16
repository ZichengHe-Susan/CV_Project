# video-captioning/utils/beam_search.py

import torch
import torch.nn.functional as F

def beam_search_decoding(
    model, 
    prefix_embeds, 
    beam_width=5, 
    max_steps=30, 
    tokenizer=None
):
    """
    Custom beam search on top of GPT2. 
    prefix_embeds: (1, context_len, 768)
    Returns the best sequence of tokens as a string.
    """
    device = prefix_embeds.device
    
    # Each element in beams is (log_prob, token_ids, past_key_values)
    # Start with a single beam that has only prefix embeddings
    beams = [(0.0, [], None)]
    
    for step in range(max_steps):
        new_beams = []
        for log_prob, token_ids, pkv in beams:
            # If we already ended with <EOS>, keep as is
            if len(token_ids) > 0 and token_ids[-1] == tokenizer.eos_token_id:
                new_beams.append((log_prob, token_ids, pkv))
                continue
            
            # Prepare input_ids for GPT2
            current_input = torch.tensor([token_ids[-1]], device=device).unsqueeze(0) if len(token_ids) > 0 else None
            
            # If step=0, we feed prefix_embeds as input_embeds
            if step == 0:
                outputs = model.gpt2(inputs_embeds=prefix_embeds, use_cache=True)
                logits = outputs.logits[:, -1, :]  # (1, vocab_size)
                pkv_next = outputs.past_key_values
            else:
                outputs = model.gpt2(input_ids=current_input, past_key_values=pkv, use_cache=True)
                logits = outputs.logits[:, -1, :]
                pkv_next = outputs.past_key_values
            
            # Get top beam_width next tokens
            probs = F.log_softmax(logits, dim=-1)
            topk = torch.topk(probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                next_token = topk.indices[0, i].item()
                next_prob = topk.values[0, i].item()
                new_beams.append((log_prob + next_prob, token_ids + [next_token], pkv_next))
        
        # Sort new beams by total log_prob, keep top beam_width
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
    
    # Return the beam with the highest log-prob
    best_seq = beams[0][1]
    text = tokenizer.decode(best_seq, skip_special_tokens=True)
    return text
