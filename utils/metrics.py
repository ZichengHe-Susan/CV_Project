# video-captioning/utils/metrics.py

import nltk
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, hypothesis):
    """
    reference: list of tokens (or list-of-lists if multiple references)
    hypothesis: list of tokens
    Returns BLEU-4 score (float)
    """
    if isinstance(reference[0], str):
        # single reference
        reference = [reference]
    return sentence_bleu(reference, hypothesis)

def calculate_bleu_corpus(references, hypotheses):
    """
    references: list of lists of tokens (or list-of-lists-of-lists if multiple references)
    hypotheses: list of lists of tokens
    A simple corpus BLEU aggregator
    """
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu_scores.append(calculate_bleu(ref, hyp))
    return sum(bleu_scores) / len(bleu_scores)
