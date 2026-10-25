"""Non-LLM baselines. A reported LLM number only means something relative to these."""
from __future__ import annotations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .agent import AgentTrace

_V = SentimentIntensityAnalyzer()
NAME = "baseline-rules"


def predict(sc: dict) -> AgentTrace:
    t = sc["task"]
    if t == "sst2":
        ans = "positive" if _V.polarity_scores(sc["sentence"])["compound"] >= 0 else "negative"
    elif t == "squad":
        ans = "UNANSWERABLE"   # always-abstain: scores 100% on unanswerable, 0% on answerable
    else:
        ans = "0"
    return AgentTrace(status="finished", final_answer=ans, turns=0)
