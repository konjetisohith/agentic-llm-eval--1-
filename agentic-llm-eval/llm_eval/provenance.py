"""Run provenance so every number can be traced to code + config + data."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD") or "no-git", "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "no-git", "dirty": None}


def collect(config: dict, data_dir, ollama_version: str = "unknown") -> dict:
    data_hashes = {}
    for p in sorted(Path(data_dir).glob("*.jsonl")):
        data_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return {"created_utc": datetime.now(timezone.utc).isoformat(), **git_info(),
            "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "ollama_version": ollama_version, "config": config,
            "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
            "data_hashes": data_hashes}
