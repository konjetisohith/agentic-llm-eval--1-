"""Streamlit dashboard: compare model runs and context-pruning configurations.
Run:  streamlit run app/dashboard.py
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import streamlit as st

from llm_eval.report import compare, fmt_mean, fmt_rate, load_meta, load_records, pruning_comparisons, summarize

st.set_page_config(page_title="Local LLM Evaluation", layout="wide")
st.title("Local LLM evaluation: agentic tasks on Ollama")
runs_dir = st.sidebar.text_input("Runs directory", os.environ.get("LLM_EVAL_RUNS", "results/full-benchmark"))
recs = load_records(runs_dir) if Path(runs_dir).exists() else []
if not recs:
    st.info("No run records found. Run `python -m llm_eval.cli run --limit 10` first (see README).")
    st.stop()

models = sorted({r["model"] for r in recs})
sel = st.sidebar.multiselect("Models", models, default=models)
recs = [r for r in recs if r["model"] in sel]
rows = summarize(recs)
st.sidebar.caption("Rates: value [95% interval] (n). Failures always count in the denominator.")
if any(r["model"].startswith("fake-") for r in recs):
    st.warning("These records come from the FAKE Ollama test server, not real models. Do not report them.")

t1, t2, t3, t4, t5 = st.tabs(["Leaderboard", "Context pruning", "Latency", "Failure explorer", "Provenance"])

with t1:
    for task, title, cols in [
        ("sst2", "Sentiment (SST-2)", [("accuracy (all)", lambda r: fmt_rate(r["success"])), ("parse-fail %", lambda r: f"{r['parse_fail_rate']*100:.1f}"),
                                        ("acc on parsed only", lambda r: fmt_rate(r["acc_over_parsed_only"]))]),
        ("squad", "Question answering (SQuAD 2.0)", [("answerable EM", lambda r: fmt_mean(r["has_ans_em"])), ("answerable F1", lambda r: fmt_mean(r["has_ans_f1"])),
                                                      ("abstains correctly", lambda r: fmt_rate(r["no_ans_acc"])), ("fabrication", lambda r: fmt_rate(r["fabrication_rate"])),
                                                      ("false abstain", lambda r: fmt_rate(r["false_abstain_rate"]))]),
        ("tools", "Tool use (synthetic, labelled)", [("accuracy (all)", lambda r: fmt_rate(r["success"])), ("tool used when required", lambda r: fmt_rate(r["tool_use_when_required"])),
                                                      ("unnecessary calls", lambda r: fmt_rate(r["unnecessary_call_rate"]))])]:
        rs = [r for r in rows if r["task"] == task]
        if rs:
            st.subheader(title)
            st.dataframe(pd.DataFrame([{"model": r["model"], "prune": r["prune"], **{c[0]: c[1](r) for c in cols}} for r in rs]), hide_index=True)
    st.caption("`baseline-rules` is a non-LLM baseline (VADER sentiment, always-abstain). A model only means something relative to it.")

with t2:
    comps = [c for c in pruning_comparisons(recs) if c["n_pairs"]]
    if not comps:
        st.write("Run with `--prune none topk:2` to compare pruning configurations.")
    for c in comps:
        d = c["f1_diff"]
        st.markdown(f"**{c['a'][0]}**: `none` vs `{c['b'][1]}` on {c['n_pairs']} paired scenarios. Success {c['success_a']*100:.1f}% vs {c['success_b']*100:.1f}% "
                    f"(McNemar p={c['p_value']:.3f}); F1 diff {d['mean_diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}].")
    sq = [r for r in rows if r["task"] == "squad"]
    if sq:
        st.dataframe(pd.DataFrame([{"model": r["model"], "prune": r["prune"], "context reduction": f"{r['context_reduction']*100:.0f}%",
                                    "gold answer still in context": fmt_rate(r["answer_retention"]), "mean prompt tokens": round(r["mean_prompt_tokens"])} for r in sq]),
                     hide_index=True)
        st.caption("If 'gold answer still in context' is low, pruning removed the evidence, so lower accuracy is expected rather than a model failure.")

with t3:
    df = pd.DataFrame([{"model": f"{r['model']} / {r['task']} / {r['prune']}", "p50 s": r["latency_p50"], "p95 s": r["latency_p95"],
                        "tokens/s": r["tokens_per_s_median"]} for r in rows if r["model"] != "baseline-rules"])
    if len(df):
        st.bar_chart(df.set_index("model")[["p50 s", "p95 s"]])
        st.dataframe(df, hide_index=True)
    st.caption("Latency excludes model load (warm-up call first). Compare models on the same hardware only.")

with t4:
    c1, c2 = st.columns(2)
    fm = c1.selectbox("Model", sel); ft = c2.selectbox("Task", ["sst2", "squad", "tools"])
    bad = [r for r in recs if r["model"] == fm and r["task"] == ft and not r["success"]]
    st.write(f"{len(bad)} failures")
    for r in bad[:50]:
        with st.expander(f"{r['scenario_id']} | status={r['status']} | answer={r['final_answer']!r}"):
            st.json({k: r[k] for k in ("status", "final_answer", "repairs", "turns", "tool_calls", "steps") if k in r})

with t5:
    meta = load_meta(runs_dir)
    st.json(meta[-1] if meta else {"note": "no .meta.json found"})
