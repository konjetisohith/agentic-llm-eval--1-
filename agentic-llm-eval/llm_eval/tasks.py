"""Scenario loading, prompts, and scoring.

Scoring rules (these are the definitions the report uses; see README):
  * completed : the agent ended with status 'finished' and a final answer.
  * success   : the answer is CORRECT. Missing/unparseable output is a failure, never dropped.
  * sst2      : correct iff parsed label == gold label.
  * squad     : answerable -> exact match; unanswerable -> must abstain (answer 'UNANSWERABLE').
  * tools     : answer correct. Separately: tool compliance (used when required / not used when unnecessary).
"""
from __future__ import annotations
import json, re
from pathlib import Path

from .squad_metrics import best_over_golds, exact_match, f1_score, normalize_answer
from .agent import AgentTrace

ABSTAIN_PHRASES = ("unanswerable", "no answer", "cannot be answered", "can't be answered", "not mentioned",
                   "not enough information", "i don't know", "i do not know", "not provided", "not stated")


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_scenarios(data_dir, tasks=("sst2", "squad", "tools"), limit: int | None = None) -> list[dict]:
    out = []
    for t in tasks:
        p = Path(data_dir) / f"{t}.jsonl"
        if p.exists():
            rows = load_jsonl(p)
            for r in rows:
                r["task"] = t
            out += rows[:limit] if limit else rows
    return out


# ---------------- prompts ----------------
def build_prompt(sc: dict, context: str | None = None) -> str:
    t = sc["task"]
    if t == "sst2":
        return ('Classify the sentiment of this sentence as "positive" or "negative".\n'
                f'Sentence: {sc["sentence"]}\nFinish with answer "positive" or "negative".')
    if t == "squad":
        return ("Answer the question using ONLY the context. Reply with the shortest exact span copied from the context. "
                'If the context does not contain the answer, finish with answer "UNANSWERABLE".\n'
                f'Context: {context if context is not None else sc["context"]}\nQuestion: {sc["question"]}')
    if t == "tools":
        return f'{sc["question"]}\nGive only the final answer (no units, no explanation) in the finish action.'
    raise ValueError(f"unknown task {t}")


# ---------------- scoring ----------------
def parse_sentiment(answer: str | None):
    a = (answer or "").lower()
    pos, neg = "positive" in a, "negative" in a
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return None


def is_abstain(answer: str | None) -> bool:
    a = (answer or "").strip().lower()
    return a == "" or any(p in a for p in ABSTAIN_PHRASES)


def parse_number(answer: str | None):
    nums = re.findall(r"-?\d+(?:\.\d+)?|-?\.\d+", (answer or "").replace(",", ""))
    return float(nums[-1]) if nums else None


def score(sc: dict, tr: AgentTrace, context_used: str | None = None) -> dict:
    completed = tr.status == "finished" and tr.final_answer is not None
    ans = tr.final_answer if completed else None
    s = {"completed": completed}
    t = sc["task"]
    if t == "sst2":
        pred = parse_sentiment(ans)
        s.update(pred=pred, parse_ok=pred is not None, success=pred == sc["label"])
    elif t == "squad":
        abstained = completed and is_abstain(ans)
        pred_text = "" if (not completed or abstained) else ans
        if sc["is_impossible"]:
            em = f1 = float(abstained)
            s.update(success=abstained)
        else:
            em = best_over_golds(exact_match, sc["answers"], pred_text)
            f1 = best_over_golds(f1_score, sc["answers"], pred_text)
            s.update(success=em == 1.0)
            # grounded: is the answer text found inside the ORIGINAL context? (extractive groundedness only)
            s["grounded"] = (normalize_answer(pred_text) in normalize_answer(sc["context"])) if pred_text else None
            ctx = normalize_answer(context_used if context_used is not None else sc["context"])
            s["answer_retained"] = any(normalize_answer(g) in ctx for g in sc["answers"])
        s.update(abstained=abstained, em=em, f1=f1)
    elif t == "tools":
        used = len(tr.tool_calls) > 0
        if sc.get("kind") == "numeric":
            p = parse_number(ans)
            g = float(sc["gold"])
            correct = p is not None and abs(p - g) <= max(0.01, 1e-6 * abs(g))
        else:
            correct = completed and normalize_answer(ans) == normalize_answer(str(sc["gold"]))
        compliant = used if sc["tool_policy"] == "required" else not used
        s.update(success=bool(correct), tool_used=used, tool_compliant=compliant,
                 tool_errors=sum(1 for c in tr.tool_calls if not c["ok"]))
    return s
