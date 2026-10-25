"""ReAct-style agent loop with bounded turns, bounded output repair, and tool-error recovery.

Each turn the model must return one JSON action. Invalid output gets a repair prompt (bounded);
tool errors are returned as observations so the model can recover; every loop ends in an explicit
status: finished | parse_failure | turn_limit | backend_error.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field

from .ollama_client import OllamaError
from .tools import TOOLS, ToolError

SYSTEM_PROMPT = """You are a careful assistant that works step by step. Each turn, reply with exactly one JSON object:
{"thought": "<brief reasoning>", "action": "<calculator|finish>", "args": {...}}
Actions:
- calculator: args {"expression": "<arithmetic expression>"}. Returns the exact result. Use it ONLY when exact arithmetic is needed; never for tasks you can answer directly.
- finish: args {"answer": "<final answer>"}. Ends the task.
Never invent tool results. After an Observation, continue or finish."""

REPAIR = ('Your last reply was not a valid action. Reply with exactly one JSON object like '
          '{"thought": "...", "action": "calculator" or "finish", "args": {...}}.')


@dataclass
class AgentTrace:
    status: str = "init"
    final_answer: str | None = None
    steps: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    repairs: int = 0
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    load_s: float = 0.0
    eval_s: float = 0.0
    error: str = ""
    truncated: bool = False

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.eval_s if self.eval_s > 0 else 0.0


def parse_action(text: str):
    obj = None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except ValueError:
                obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("action"), str):
        return None
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    if obj["action"] == "finish" and "answer" not in args and "answer" in obj:
        args = {"answer": obj["answer"]}
    return {"thought": str(obj.get("thought", "")), "action": obj["action"], "args": args}


class Agent:
    def __init__(self, client, model: str, options: dict, fmt="json", max_turns: int = 6, max_repairs: int = 2):
        self.client, self.model, self.options, self.fmt = client, model, options, fmt
        self.max_turns, self.max_repairs = max_turns, max_repairs

    def run(self, task_prompt: str) -> AgentTrace:
        tr = AgentTrace()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task_prompt}]
        num_ctx = self.options.get("num_ctx", 4096)
        while tr.turns < self.max_turns:
            tr.turns += 1
            try:
                res = self.client.chat(self.model, msgs, self.options, self.fmt)
            except OllamaError as e:
                tr.status, tr.error = "backend_error", str(e)
                return tr
            tr.prompt_tokens += res.prompt_tokens; tr.completion_tokens += res.completion_tokens
            tr.latency_s += res.wall_s; tr.load_s += res.load_s; tr.eval_s += res.eval_s
            est_tokens = sum(len(m["content"]) for m in msgs) / 3.5
            if res.prompt_tokens >= num_ctx - 16 or est_tokens >= num_ctx:
                tr.truncated = True  # input may have been silently truncated by Ollama
            act = parse_action(res.content)
            if act is None or (act["action"] == "finish" and "answer" not in act["args"]):
                tr.repairs += 1
                tr.steps.append({"raw": res.content[:300], "invalid": True})
                if tr.repairs > self.max_repairs:
                    tr.status = "parse_failure"
                    return tr
                msgs += [{"role": "assistant", "content": res.content}, {"role": "user", "content": REPAIR}]
                continue
            step = {"thought": act["thought"], "action": act["action"], "args": act["args"]}
            if act["action"] == "finish":
                tr.final_answer = str(act["args"]["answer"]).strip()
                tr.status = "finished"
                tr.steps.append(step)
                return tr
            if act["action"] in TOOLS:
                expr = act["args"].get("expression", "")
                try:
                    result, ok = TOOLS[act["action"]](expr), True
                except ToolError as e:
                    result, ok = f"ERROR: {e}", False
            else:
                result, ok = f"ERROR: unknown action '{act['action']}'", False
            tr.tool_calls.append({"name": act["action"], "args": act["args"], "ok": ok, "result": result})
            step["observation"] = result
            tr.steps.append(step)
            msgs += [{"role": "assistant", "content": res.content}, {"role": "user", "content": f"Observation: {result}"}]
        tr.status = "turn_limit"
        return tr
