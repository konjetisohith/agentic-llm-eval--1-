"""Context pruning: keep the k sentences most similar (TF-IDF cosine) to the question, in original order."""
from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", (text or "").strip()) if s]


def parse_config(cfg: str) -> int | None:
    """'none' -> None (no pruning); 'topk:2' -> 2."""
    if cfg in ("none", "", None):
        return None
    kind, _, val = cfg.partition(":")
    if kind != "topk" or not val.isdigit() or int(val) < 1:
        raise ValueError(f"bad prune config '{cfg}' (use 'none' or 'topk:N')")
    return int(val)


def prune_context(context: str, question: str, k: int | None) -> str:
    if k is None:
        return context
    sents = split_sentences(context)
    if len(sents) <= k:
        return context
    try:
        vec = TfidfVectorizer(stop_words="english").fit(sents + [question])
        scores = cosine_similarity(vec.transform([question]), vec.transform(sents))[0]
        idx = sorted(sorted(range(len(sents)), key=lambda i: (-scores[i], i))[:k])
    except ValueError:  # empty vocabulary
        idx = list(range(k))
    return " ".join(sents[i] for i in idx)
