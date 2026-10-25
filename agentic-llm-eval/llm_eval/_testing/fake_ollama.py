"""A FAKE Ollama server for tests and offline smoke runs. It is NOT a language model.

Behaviour is scripted from the fixtures (it can see the gold answers), which lets us verify the harness
plumbing: agent loop, repair, tool use, scoring, resume, reports. Results produced against it say
nothing about real model quality and must never be reported as results.

  fake-good  : oracle; uses the calculator only when the scenario requires it
  fake-bad   : always 'positive', fabricates a wrong span on QA, never uses tools
  fake-flaky : like fake-good but its first reply per scenario is non-JSON (exercises repair)
"""
from __future__ import annotations
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ..tasks import load_scenarios

MODELS = ("fake-good", "fake-bad", "fake-flaky")


def _find(scenarios, prompt):
    for sc in scenarios:
        t = sc["task"]
        if (t == "sst2" and sc["sentence"] in prompt) or (t == "squad" and f"Question: {sc['question']}" in prompt) \
                or (t == "tools" and prompt.startswith(sc["question"])):
            return sc
    return None


def _act(action, **args):
    return json.dumps({"thought": "scripted", "action": action, "args": args})


def respond(model, scenarios, messages) -> str:
    sc = _find(scenarios, messages[1]["content"]) if len(messages) > 1 else None
    if sc is None:
        return _act("finish", answer="ok")
    repaired = any("not a valid action" in m["content"] for m in messages)
    if model == "fake-flaky" and not repaired:
        return "Sure! Here is my answer."
    obs = messages[-1]["content"] if messages[-1]["content"].startswith("Observation:") else None
    t = sc["task"]
    if model == "fake-bad":
        return _act("finish", answer={"sst2": "positive", "squad": "Paris", "tools": "0"}[t])
    if t == "sst2":
        return _act("finish", answer=sc["label"])
    if t == "squad":
        return _act("finish", answer="UNANSWERABLE" if sc["is_impossible"] else sc["answers"][0])
    if sc["tool_policy"] == "required":
        if obs is None:
            return _act("calculator", expression=sc["reference_expression"])
        return _act("finish", answer=obs.split(":", 1)[1].strip())
    return _act("finish", answer=str(sc["gold"]))


class FakeOllama:
    def __init__(self, scenarios, port=0, models=MODELS):
        outer = self

        class H(BaseHTTPRequestHandler):
            def _send(self, obj, code=200):
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                if self.path == "/api/version":
                    self._send({"version": "fake-0"})
                elif self.path == "/api/tags":
                    self._send({"models": [{"name": m} for m in outer.models]})
                else:
                    self._send({}, 404)

            def do_POST(self):
                p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if p["model"] not in outer.models:
                    return self._send({"error": "model not found"}, 404)
                content = respond(p["model"], outer.scenarios, p["messages"])
                self._send({"message": {"role": "assistant", "content": content},
                            "prompt_eval_count": sum(len(m["content"]) for m in p["messages"]) // 4,
                            "eval_count": max(1, len(content) // 4), "load_duration": 0, "eval_duration": int(4e7)})

            def log_message(self, *a):
                pass
        self.scenarios, self.models = scenarios, models
        self.srv = HTTPServer(("127.0.0.1", port), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"

    def start(self):
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self.url

    def stop(self):
        self.srv.shutdown(); self.srv.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/fixtures"); ap.add_argument("--port", type=int, default=11500)
    a = ap.parse_args()
    s = FakeOllama(load_scenarios(a.data), a.port)
    print("fake ollama on", s.url, flush=True)
    s.srv.serve_forever()
