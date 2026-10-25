"""Derive all metrics from raw run records (single source of truth for the CLI report and the dashboard)."""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff, wilson


def load_records(runs_dir) -> list[dict]:
    recs = []
    for p in sorted(Path(runs_dir).glob("*.jsonl")):
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return recs


def load_meta(runs_dir) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*.meta.json"))]


def _rate(vals):
    vals = list(vals)
    n, k = len(vals), sum(bool(v) for v in vals)
    lo, hi = wilson(k, n)
    return {"rate": k / n if n else math.nan, "lo": lo, "hi": hi, "k": k, "n": n}


def _mean(vals):
    vals = [float(v) for v in vals]
    if not vals:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    lo, hi = bootstrap_ci(vals)
    return {"mean": float(np.mean(vals)), "lo": lo, "hi": hi, "n": len(vals)}


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["prune"], r["task"])].append(r)
    rows = []
    for (model, prune, task), rs in sorted(groups.items()):
        lat = [r["latency_s"] for r in rs]
        tps = [r["tokens_per_s"] for r in rs if r["tokens_per_s"] > 0]
        row = {"model": model, "prune": prune, "task": task, "n": len(rs),
               "success": _rate(r["success"] for r in rs),          # denominator = ALL scenarios; failures never dropped
               "completed": _rate(r["completed"] for r in rs),
               "latency_p50": float(np.percentile(lat, 50)), "latency_p95": float(np.percentile(lat, 95)),
               "tokens_per_s_median": float(np.median(tps)) if tps else None,
               "repair_rate": float(np.mean([r["repairs"] > 0 for r in rs])),
               "backend_errors": sum(r["status"] == "backend_error" for r in rs),
               "truncated": sum(bool(r["truncated"]) for r in rs),
               "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in rs]))}
        if task == "sst2":
            parsed = [r for r in rs if r["parse_ok"]]
            row["parse_fail_rate"] = 1 - len(parsed) / len(rs)
            row["acc_over_parsed_only"] = _rate(r["success"] for r in parsed)  # shown ONLY to expose the inflation
        elif task == "squad":
            ans = [r for r in rs if not r["is_impossible"]]
            imp = [r for r in rs if r["is_impossible"]]
            row["has_ans_em"] = _mean(r["em"] for r in ans)
            row["has_ans_f1"] = _mean(r["f1"] for r in ans)
            row["no_ans_acc"] = _rate(r["abstained"] for r in imp)                # abstention recall
            row["fabrication_rate"] = _rate(not r["abstained"] for r in imp)      # answered when it should abstain
            row["false_abstain_rate"] = _rate(r["abstained"] for r in ans)        # abstained on an answerable question
            abst = [r for r in rs if r["abstained"]]
            row["abstain_precision"] = _rate(r["is_impossible"] for r in abst) if abst else None
            nonabs = [r for r in ans if not r["abstained"]]
            row["ungrounded_rate"] = _rate(r["grounded"] is False for r in nonabs) if nonabs else None
            row["answer_retention"] = _rate(r["answer_retained"] for r in ans) if ans else None
            full = sum(r["context_chars_full"] for r in rs); used = sum(r["context_chars_used"] for r in rs)
            row["context_reduction"] = 1 - used / full if full else 0.0
        elif task == "tools":
            req = [r for r in rs if r["tool_policy"] == "required"]
            nn = [r for r in rs if r["tool_policy"] == "not_needed"]
            row["acc_required"] = _rate(r["success"] for r in req) if req else None
            row["acc_not_needed"] = _rate(r["success"] for r in nn) if nn else None
            row["tool_use_when_required"] = _rate(r["tool_used"] for r in req) if req else None
            row["unnecessary_call_rate"] = _rate(r["tool_used"] for r in nn) if nn else None
            calls = sum(len(r["tool_calls"]) for r in rs)
            row["tool_error_rate"] = sum(r["tool_errors"] for r in rs) / calls if calls else None
        rows.append(row)
    return rows


def compare(records: list[dict], task: str, a: tuple, b: tuple) -> dict:
    """Paired comparison of system a vs b ((model, prune) each) on scenarios both completed running."""
    def idx(key):
        return {r["scenario_id"]: r for r in records if (r["model"], r["prune"]) == key and r["task"] == task}
    A, B = idx(a), idx(b)
    ids = sorted(set(A) & set(B))
    out = {"task": task, "a": a, "b": b, "n_pairs": len(ids)}
    if not ids:
        return out
    sa, sb = [bool(A[i]["success"]) for i in ids], [bool(B[i]["success"]) for i in ids]
    out.update(success_a=float(np.mean(sa)), success_b=float(np.mean(sb)), **mcnemar_exact(sa, sb))
    if task == "squad":
        out["f1_diff"] = paired_bootstrap_diff([A[i]["f1"] for i in ids], [B[i]["f1"] for i in ids])
    return out


def pruning_comparisons(records):
    models = sorted({r["model"] for r in records if r["task"] == "squad"})
    out = []
    for m in models:
        cfgs = sorted({r["prune"] for r in records if r["task"] == "squad" and r["model"] == m})
        if "none" in cfgs:
            out += [compare(records, "squad", (m, "none"), (m, c)) for c in cfgs if c != "none"]
    return out


def fmt_rate(d) -> str:
    return "-" if not d or d["n"] == 0 else f"{d['rate']*100:.1f}% [{d['lo']*100:.0f}-{d['hi']*100:.0f}] (n={d['n']})"


def fmt_mean(d) -> str:
    return "-" if not d or d["n"] == 0 else f"{d['mean']*100:.1f}% [{d['lo']*100:.0f}-{d['hi']*100:.0f}] (n={d['n']})"


def to_markdown(records: list[dict]) -> str:
    rows = summarize(records)
    md = ["# LLM evaluation report", "",
          "Rates show `value [95% interval] (n)`. Wilson intervals for rates, bootstrap for means. "
          "Denominators include ALL scenarios (missing/unparseable outputs count as failures).", ""]
    def table(task, cols, title):
        rs = [r for r in rows if r["task"] == task]
        if not rs:
            return
        md.extend([f"## {title}", "", "| model | prune | " + " | ".join(c[0] for c in cols) + " |",
                   "|---|---|" + "---|" * len(cols)])
        for r in rs:
            md.append(f"| {r['model']} | {r['prune']} | " + " | ".join(c[1](r) for c in cols) + " |")
        md.append("")
    lat = lambda r: f"{r['latency_p50']:.2f} / {r['latency_p95']:.2f}"
    tps = lambda r: "-" if r["tokens_per_s_median"] is None else f"{r['tokens_per_s_median']:.1f}"
    table("sst2", [("accuracy (all)", lambda r: fmt_rate(r["success"])), ("parse-fail", lambda r: f"{r['parse_fail_rate']*100:.1f}%"),
                   ("acc on parsed only", lambda r: fmt_rate(r["acc_over_parsed_only"])), ("lat p50/p95 s", lat), ("tok/s", tps)],
          "Sentiment (GLUE SST-2 validation subset)")
    table("squad", [("answerable EM", lambda r: fmt_mean(r["has_ans_em"])), ("answerable F1", lambda r: fmt_mean(r["has_ans_f1"])),
                    ("abstains correctly", lambda r: fmt_rate(r["no_ans_acc"])), ("fabrication", lambda r: fmt_rate(r["fabrication_rate"])),
                    ("false abstain", lambda r: fmt_rate(r["false_abstain_rate"])), ("answer retained", lambda r: fmt_rate(r["answer_retention"])),
                    ("ctx reduction", lambda r: f"{r['context_reduction']*100:.0f}%"), ("lat p50/p95 s", lat)],
          "Question answering (SQuAD 2.0 validation subset)")
    table("tools", [("accuracy (all)", lambda r: fmt_rate(r["success"])), ("acc when tool required", lambda r: fmt_rate(r["acc_required"])),
                    ("tool used when required", lambda r: fmt_rate(r["tool_use_when_required"])),
                    ("unnecessary calls", lambda r: fmt_rate(r["unnecessary_call_rate"])), ("lat p50/p95 s", lat)],
          "Tool use (synthetic labelled set)")
    comps = pruning_comparisons(records)
    if comps:
        md += ["## Context pruning (paired, same scenarios)", "", "| model | config | n | success none | success pruned | only none right | only pruned right | McNemar p | F1 diff [95%] |", "|---|---|---|---|---|---|---|---|---|"]
        for c in comps:
            if c["n_pairs"]:
                d = c["f1_diff"]
                md.append(f"| {c['a'][0]} | {c['b'][1]} | {c['n_pairs']} | {c['success_a']*100:.1f}% | {c['success_b']*100:.1f}% | "
                          f"{c['only_a']} | {c['only_b']} | {c['p_value']:.3f} | {d['mean_diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] |")
        md.append("")
    md += ["## Notes", "- `fabrication` = answered an unanswerable question (the primary hallucination measure here).",
           "- `answer retained` = gold answer still present in the (pruned) context; low values mean pruning removed the evidence.",
           "- Latency excludes model load (a warm-up call precedes each model). p95 over few scenarios is noisy.", ""]
    return "\n".join(md)


def write_report(runs_dir, out_dir):
    recs = load_records(runs_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    md = to_markdown(recs)
    (Path(out_dir) / "report.md").write_text(md)
    (Path(out_dir) / "report.json").write_text(json.dumps({"summary": summarize(recs), "pruning": pruning_comparisons(recs)}, indent=2, default=str))
    return md
