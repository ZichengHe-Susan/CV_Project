# video-captioning/utils/metrics.py

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, hypothesis):
    """
    reference: list of tokens (or list-of-lists if multiple references)
    hypothesis: list of tokens
    Returns BLEU-4 score (float)
    """
    smoothie = SmoothingFunction()
    return sentence_bleu(
        [reference],
        hypothesis,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie.method5
    )

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
