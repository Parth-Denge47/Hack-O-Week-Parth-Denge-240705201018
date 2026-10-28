"""Weeks 17-18 (f): Hugging Face Transformers basics - pretrained models via pipeline() and via the explicit
tokenizer + model API.

Needs internet on the first run to download the model weights (cached afterwards).

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner  # noqa: E402

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SAMPLES = [
    "The new deep learning lab was fantastic, I learned so much!",
    "The training kept crashing and the documentation was useless.",
    "The results are okay, nothing special.",
]


def pipeline_sentiment(texts):
    """High-level API: tokenizer, model and post-processing in a single call."""
    clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    return clf(texts)


def manual_sentiment(texts):
    """Low-level API: tokenise -> forward pass -> softmax over the logits."""
    tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).eval()
    batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        probs = model(**batch).logits.softmax(-1)
    return [{"label": model.config.id2label[int(p.argmax())], "score": float(p.max())} for p in probs], batch


def main():
    banner("Weeks 17-18f - Hugging Face Transformers")
    print(f"Model: {SENTIMENT_MODEL}\n")
    print("pipeline() API:")
    for t, r in zip(SAMPLES, pipeline_sentiment(SAMPLES)):
        print(f"  {r['label']:8s} {r['score']:.3f}  {t}")
    manual, batch = manual_sentiment(SAMPLES)
    print("\nTokenizer + model API (same predictions):")
    for t, r in zip(SAMPLES, manual):
        print(f"  {r['label']:8s} {r['score']:.3f}  {t}")
    tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    print("\nTokens for sample 1:", tok.convert_ids_to_tokens(batch["input_ids"][0])[:12], "...")


if __name__ == "__main__":
    main()
