# video-captioning/models/gpt2_decoder.py

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from config import Config

class GPT2Decoder(nn.Module):
    """
    A wrapper around GPT2 that allows for inserting 'visual context' tokens at the front.
    """
    def __init__(self, model_name=Config.GPT2_MODEL_NAME):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Resize token embeddings if needed (e.g., if we add special tokens)
        # self.gpt2.resize_token_embeddings(len(self.tokenizer))
        
        # Freeze or not freeze GPT2 if desired
        # (In many experiments, you only fine-tune GPT2; in some, you freeze it.)
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False

    def forward(self, inputs_embeds, labels=None, attention_mask=None):
        """
        inputs_embeds: shape (B, T, 768), already includes context + text tokens
        labels: for language modeling, shape (B, T)
        attention_mask: shape (B, T)
        
        GPT2LMHeadModel can accept inputs_embeds directly if you pass None as input_ids.
        """
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        # outputs has (loss, logits, past_key_values, ...)
        return outputs

    def generate(
        self,
        inputs_embeds,
        attention_mask=None,
        max_length=Config.MAX_SEQ_LEN,
        num_beams=1
    ):
        """
        Use GPT-2's generate method with the provided inputs_embeds as prefix.
        """
        return self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )
