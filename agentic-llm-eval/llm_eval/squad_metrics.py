"""SQuAD-style answer normalization, exact match and token F1 (same algorithm as the official script)."""
import re, string
from collections import Counter

_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _tokens(s: str):
    return normalize_answer(s).split() if s else []


def exact_match(gold: str, pred: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(pred))


def f1_score(gold: str, pred: str) -> float:
    g, p = _tokens(gold), _tokens(pred)
    if not g or not p:
        return float(g == p)
    common = sum((Counter(g) & Counter(p)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def best_over_golds(fn, golds, pred) -> float:
    return max((fn(g, pred) for g in golds), default=0.0)
