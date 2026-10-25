"""Run models over scenarios and write one raw JSONL record per scenario.

Metrics are NEVER computed here: raw records are the source of truth, and report.py derives metrics from
them. A scoring fix therefore never requires re-running (slow) model inference.
"""
from __future__ import annotations
import json, re
from pathlib import Path

from . import baselines
from .agent import Agent
from .ollama_client import OllamaClient, OllamaError
from .pruning import parse_config, prune_context
from .tasks import build_prompt, score


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["scenario_id"] for l in path.read_text().splitlines() if l.strip()}


def make_record(model, prune, sc, tr, context_used=None, context_full_chars=None) -> dict:
    s = score(sc, tr, context_used)
    rec = {"model": model, "prune": prune, "task": sc["task"], "scenario_id": sc["id"], "status": tr.status,
           "final_answer": tr.final_answer, "latency_s": round(tr.latency_s, 4), "load_s": round(tr.load_s, 4),
           "eval_s": round(tr.eval_s, 4), "tokens_per_s": round(tr.tokens_per_s, 2),
           "prompt_tokens": tr.prompt_tokens, "completion_tokens": tr.completion_tokens, "turns": tr.turns,
           "repairs": tr.repairs, "tool_calls": tr.tool_calls, "truncated": tr.truncated, "error": tr.error,
           "steps": tr.steps, **s}
    if sc["task"] == "squad":
        rec["context_chars_full"] = len(sc["context"])
        rec["context_chars_used"] = len(context_used if context_used is not None else sc["context"])
        rec["is_impossible"] = bool(sc["is_impossible"])
    if sc["task"] == "tools":
        rec["tool_policy"] = sc["tool_policy"]
    return rec


def run_model(model, scenarios, prune_configs, out_dir, client: OllamaClient, options, fmt="json",
              agent_cfg=None, resume=True, log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    agent_cfg = agent_cfg or {}
    if model not in client.list_models():
        raise OllamaError(f"model '{model}' is not pulled. Run: ollama pull {model}")
    client.chat(model, [{"role": "user", "content": 'Reply with {"ok": true}'}], options, fmt)  # warm-up: load model
    agent = Agent(client, model, options, fmt, **agent_cfg)
    for ci, cfg in enumerate(prune_configs):
        k = parse_config(cfg)
        path = out / f"{slug(model)}__{slug(cfg)}.jsonl"
        done = _done_ids(path) if resume else set()
        if not resume and path.exists():
            path.unlink()
        todo = [sc for sc in scenarios if sc["id"] not in done and (sc["task"] == "squad" or ci == 0)]
        log(f"[{model} | prune={cfg}] {len(todo)} scenarios to run ({len(done)} already done)")
        with path.open("a") as f:
            for i, sc in enumerate(todo, 1):
                ctx = prune_context(sc["context"], sc["question"], k) if sc["task"] == "squad" else None
                tr = agent.run(build_prompt(sc, ctx))
                f.write(json.dumps(make_record(model, cfg if sc["task"] == "squad" else "none", sc, tr, ctx)) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    log(f"  {i}/{len(todo)}")


def run_baselines(scenarios, out_dir, resume=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{slug(baselines.NAME)}__none.jsonl"
    if path.exists():
        path.unlink()
    with path.open("w") as f:
        for sc in scenarios:
            f.write(json.dumps(make_record(baselines.NAME, "none", sc, baselines.predict(sc))) + "\n")
