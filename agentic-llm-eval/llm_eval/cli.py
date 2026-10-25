"""Command line: prepare | check | run | report."""
from __future__ import annotations
import argparse, json
from pathlib import Path

from . import baselines, data_prep, provenance
from .ollama_client import OllamaClient, OllamaError
from .report import write_report
from .runner import run_baselines, run_model
from .tasks import load_scenarios

import os
CFG = Path(os.environ.get("LLM_EVAL_CONFIG", str(Path(__file__).resolve().parent.parent / "configs/default.json")))


def main(argv=None):
    cfg = json.loads(CFG.read_text())
    ap = argparse.ArgumentParser(prog="llm_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="download SST-2 + SQuAD 2.0 subsets and build the tool set")
    p.add_argument("--out", default="data"); p.add_argument("--seed", type=int, default=cfg["data"]["seed"])
    p.add_argument("--source", choices=["github", "hf"], default="github", help="github (default, verified by hash) or Hugging Face")
    for k in ("n_sst2", "n_squad", "n_tools"):
        p.add_argument(f"--{k.replace('_','-')}", type=int, default=cfg["data"][k], dest=k)
    sub.add_parser("check", help="verify Ollama is reachable and models are pulled")
    r = sub.add_parser("run", help="run models over the benchmark")
    r.add_argument("--models", nargs="+", default=cfg["models"]); r.add_argument("--prune", nargs="+", default=cfg["prune_configs"])
    r.add_argument("--data", default="data"); r.add_argument("--out", default="results/runs")
    r.add_argument("--tasks", nargs="+", default=["sst2", "squad", "tools"])
    r.add_argument("--limit", type=int, default=None, help="max scenarios per task (use 10-20 for a first try)")
    r.add_argument("--no-resume", action="store_true"); r.add_argument("--no-baselines", action="store_true")
    rp = sub.add_parser("report", help="build report.md/json from raw run records")
    rp.add_argument("--runs", default="results/runs"); rp.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    if a.cmd == "prepare":
        m = data_prep.prepare_all(a.out, a.n_sst2, a.n_squad, cfg["data"]["squad_impossible_fraction"], a.n_tools, a.seed, a.source)
        print(json.dumps(m, indent=2))
    elif a.cmd == "check":
        c = OllamaClient()
        try:
            print("Ollama", c.version(), "at", c.host); have = c.list_models()
        except OllamaError as e:
            raise SystemExit(f"Cannot reach Ollama: {e}\nStart it with `ollama serve`.")
        for m in cfg["models"]:
            print(("OK       " if m in have else "MISSING  ") + m + ("" if m in have else f"   -> ollama pull {m}"))
    elif a.cmd == "run":
        c = OllamaClient()
        scenarios = load_scenarios(a.data, tuple(a.tasks), a.limit)
        if not scenarios:
            raise SystemExit(f"No scenarios in {a.data}/. Run `python -m llm_eval.cli prepare` (or use --data data/fixtures).")
        meta = provenance.collect({**cfg, "models": a.models, "prune_configs": a.prune, "limit": a.limit}, a.data, c.version())
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"run_{meta['created_utc'][:19].replace(':','-')}.meta.json").write_text(json.dumps(meta, indent=2))
        if not a.no_baselines:
            run_baselines(scenarios, a.out)
        for m in a.models:
            run_model(m, scenarios, a.prune, a.out, c, cfg["options"], cfg["format"], cfg["agent"], resume=not a.no_resume)
        print(write_report(a.out, Path(a.out).parent))
    elif a.cmd == "report":
        print(write_report(a.runs, a.out))


if __name__ == "__main__":
    main()
