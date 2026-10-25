"""Minimal Ollama HTTP client (stdlib only): timeouts, bounded retries, timing breakdown."""
from __future__ import annotations
import json, os, time, urllib.error, urllib.request
from dataclasses import dataclass


class OllamaError(Exception):
    pass


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0     # client-measured end-to-end time
    load_s: float = 0.0     # model load time reported by Ollama (excluded from fair latency via warm-up)
    eval_s: float = 0.0     # generation time reported by Ollama

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.eval_s if self.eval_s > 0 else 0.0


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: int = 180, max_retries: int = 2):
        h = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.host = h if h.startswith("http") else "http://" + h
        self.timeout, self.max_retries = timeout, max_retries
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy local inference

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.host + path, data, {"Content-Type": "application/json"},
                                     method="POST" if data is not None else "GET")
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener.open(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code in (400, 404):  # not retryable (bad request / model missing)
                    raise OllamaError(f"HTTP {e.code} from Ollama: {body}") from e
                last = f"HTTP {e.code}: {body}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = str(e)
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 4))
        raise OllamaError(f"Ollama request to {path} failed after {self.max_retries + 1} attempts: {last}")

    def version(self) -> str:
        return self._request("/api/version").get("version", "unknown")

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._request("/api/tags").get("models", [])]

    def chat(self, model: str, messages: list[dict], options: dict | None = None, fmt: str | dict | None = "json") -> ChatResult:
        payload = {"model": model, "messages": messages, "stream": False, "options": options or {}}
        if fmt:
            payload["format"] = fmt
        t = time.perf_counter()
        d = self._request("/api/chat", payload)
        wall = time.perf_counter() - t
        ns = 1e9
        return ChatResult(d.get("message", {}).get("content", ""), d.get("prompt_eval_count", 0),
                          d.get("eval_count", 0), wall, d.get("load_duration", 0) / ns, d.get("eval_duration", 0) / ns)
