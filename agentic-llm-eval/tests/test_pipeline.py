import json
from pathlib import Path
import pytest

from llm_eval import baselines
from llm_eval.data_prep import build_tool_set, sample_sst2, sample_squad
from llm_eval.ollama_client import OllamaClient, OllamaError
from llm_eval.report import compare, load_records, summarize, to_markdown, write_report
from llm_eval.runner import run_baselines, run_model
from llm_eval.tasks import load_scenarios
from llm_eval.tools import calculator
from llm_eval._testing.fake_ollama import FakeOllama

FIX = Path(__file__).resolve().parent.parent / "data/fixtures"
OPTS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 300}


@pytest.fixture(scope="module")
def server():
    s = FakeOllama(load_scenarios(FIX)); url = s.start()
    yield url
    s.stop()


@pytest.fixture(scope="module")
def runs(server, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    sc = load_scenarios(FIX); c = OllamaClient(server, timeout=10, max_retries=0)
    run_baselines(sc, out)
    for m in ("fake-good", "fake-bad", "fake-flaky"):
        run_model(m, sc, ["none", "topk:1"], out, c, OPTS, log=lambda *a: None)
    return out


def row(rows, model, task, prune="none"):
    return next(r for r in rows if r["model"] == model and r["task"] == task and r["prune"] == prune)


def test_client_roundtrip_and_errors(server):
    c = OllamaClient(server, timeout=5, max_retries=0)
    assert c.version() == "fake-0" and "fake-good" in c.list_models()
    r = c.chat("fake-good", [{"role": "user", "content": "hi"}], OPTS)
    assert json.loads(r.content)["action"] == "finish" and r.completion_tokens > 0 and r.tokens_per_s > 0
    with pytest.raises(OllamaError):
        c.chat("not-pulled", [{"role": "user", "content": "x"}], OPTS)
    with pytest.raises(OllamaError):
        OllamaClient("http://127.0.0.1:9", timeout=1, max_retries=1).version()


def test_missing_model_gives_actionable_error(server, tmp_path):
    with pytest.raises(OllamaError, match="ollama pull"):
        run_model("llama3.2:3b", load_scenarios(FIX), ["none"], tmp_path, OllamaClient(server, max_retries=0), OPTS)


def test_oracle_vs_bad_model_separated_by_metrics(runs):
    rows = summarize(load_records(runs))
    assert row(rows, "fake-good", "sst2")["success"]["rate"] == 1.0
    assert row(rows, "fake-bad", "sst2")["success"]["rate"] == 0.5          # always 'positive' on a balanced set
    g, b = row(rows, "fake-good", "squad"), row(rows, "fake-bad", "squad")
    assert g["has_ans_em"]["mean"] == 1.0 and g["no_ans_acc"]["rate"] == 1.0 and g["fabrication_rate"]["rate"] == 0.0
    assert b["has_ans_em"]["mean"] == 0.0 and b["fabrication_rate"]["rate"] == 1.0 and b["ungrounded_rate"]["rate"] == 1.0
    gt, bt = row(rows, "fake-good", "tools"), row(rows, "fake-bad", "tools")
    assert gt["success"]["rate"] == 1.0 and gt["tool_use_when_required"]["rate"] == 1.0 and gt["unnecessary_call_rate"]["rate"] == 0.0
    assert bt["tool_use_when_required"]["rate"] == 0.0 and bt["acc_required"]["rate"] == 0.0


def test_flaky_model_triggers_repairs_but_still_succeeds(runs):
    r = row(summarize(load_records(runs)), "fake-flaky", "sst2")
    assert r["repair_rate"] == 1.0 and r["success"]["rate"] == 1.0 and r["completed"]["rate"] == 1.0


def test_baselines_are_meaningful_and_rule_based(runs):
    rows = summarize(load_records(runs))
    sq = row(rows, "baseline-rules", "squad")
    assert sq["no_ans_acc"]["rate"] == 1.0 and sq["has_ans_em"]["mean"] == 0.0 and sq["false_abstain_rate"]["rate"] == 1.0
    assert row(rows, "baseline-rules", "sst2")["success"]["rate"] >= 0.75   # VADER on easy sentences


def test_pruning_runs_only_squad_and_reports_retention(runs):
    rows = summarize(load_records(runs))
    assert not [r for r in rows if r["prune"] == "topk:1" and r["task"] != "squad"]
    pr = row(rows, "fake-good", "squad", "topk:1")
    assert pr["context_reduction"] > 0.3 and pr["answer_retention"]["rate"] <= 1.0
    c = compare(load_records(runs), "squad", ("fake-good", "none"), ("fake-good", "topk:1"))
    assert c["n_pairs"] == 10 and 0 <= c["p_value"] <= 1


def test_resume_does_not_duplicate_records(server, tmp_path):
    sc = load_scenarios(FIX); c = OllamaClient(server, timeout=5, max_retries=0)
    for _ in range(2):
        run_model("fake-good", sc, ["none"], tmp_path, c, OPTS, log=lambda *a: None)
    ids = [r["scenario_id"] for r in load_records(tmp_path)]
    assert len(ids) == len(set(ids)) == len(sc)


def test_report_regression_missing_predictions_count_as_wrong():
    """Audit finding: 21 unparsed predictions must stay in the denominator (60/100, not 60/79)."""
    recs = []
    for i in range(100):
        parsed = i < 79
        recs.append({"model": "m", "prune": "none", "task": "sst2", "scenario_id": f"s{i}", "status": "finished" if parsed else "parse_failure",
                     "completed": parsed, "success": parsed and i < 60, "parse_ok": parsed, "latency_s": 1.0, "tokens_per_s": 10.0,
                     "repairs": 0, "backend_errors": 0, "truncated": False, "prompt_tokens": 50})
    r = summarize(recs)[0]
    assert r["success"]["rate"] == pytest.approx(0.60) and r["success"]["n"] == 100
    assert r["acc_over_parsed_only"]["rate"] == pytest.approx(60 / 79)
    assert r["parse_fail_rate"] == pytest.approx(0.21) and r["completed"]["rate"] == pytest.approx(0.79)


def test_report_files_and_markdown(runs, tmp_path):
    md = write_report(runs, tmp_path)
    assert "fabrication" in md and "McNemar" in md and (tmp_path / "report.json").exists()


def test_data_prep_is_seeded_balanced_and_correct():
    sst = [{"sentence": f"s{i}", "label": i % 2} for i in range(200)]
    a, b = sample_sst2(sst, 20, 7), sample_sst2(sst, 20, 7)
    assert a == b and sum(r["label"] == "positive" for r in a) == 10 and sample_sst2(sst, 20, 8) != a
    sq = [{"id": str(i), "context": "c", "question": "q", "answers": {"text": [] if i % 2 else ["a", "a", "b"]}} for i in range(100)]
    s = sample_squad(sq, 20, 1, 0.5)
    assert sum(r["is_impossible"] for r in s) == 10 and all(r["answers"] == [] or r["answers"] == ["a", "b"] for r in s)
    ts = build_tool_set(3, 20)
    assert ts == build_tool_set(3, 20) and sum(t["tool_policy"] == "required" for t in ts) == 10
    for t in ts:
        if t["tool_policy"] == "required":
            assert calculator(t["reference_expression"]) == t["gold"]


def test_dashboard_renders_without_exceptions(runs, monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("LLM_EVAL_RUNS", str(runs))
    at = st_testing.AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app/dashboard.py"), default_timeout=60).run()
    assert not at.exception and len(at.tabs) == 5 and len(at.dataframe) >= 3


# ---------- real-dataset loaders (official file formats, tiny inline samples) ----------
def test_flatten_official_squad_v2_format_and_integrity():
    from llm_eval.data_prep import flatten_squad_v2, DataIntegrityError
    obj = {"version": "v2.0", "data": [{"title": "T", "paragraphs": [{"context": "Paris is in France.", "qas": [
        {"id": "a", "question": "Where is Paris?", "answers": [{"text": "France", "answer_start": 12}, {"text": "France", "answer_start": 12}], "is_impossible": False},
        {"id": "b", "question": "Who founded Paris?", "answers": [], "plausible_answers": [{"text": "x", "answer_start": 0}], "is_impossible": True}]}]}]}
    rows = flatten_squad_v2(obj)
    assert [r["id"] for r in rows] == ["a", "b"] and rows[0]["answers"]["text"] == ["France", "France"] and rows[1]["answers"]["text"] == []
    s = sample_squad(rows, 2, 0, 0.5)
    assert {r["is_impossible"] for r in s} == {True, False} and next(r for r in s if r["id"] == "a")["answers"] == ["France"]
    obj["data"][0]["paragraphs"][0]["qas"][1]["is_impossible"] = False
    with pytest.raises(DataIntegrityError):
        flatten_squad_v2(obj)


def test_parse_sst2_label_first_and_sampling():
    from llm_eval.data_prep import parse_sst2_label_first
    rows = parse_sst2_label_first("1\ta great movie .\n0\ta dull movie .\nlabel\tsentence\n\n")
    assert rows == [{"sentence": "a great movie .", "label": 1}, {"sentence": "a dull movie .", "label": 0}]
    assert {r["label"] for r in sample_sst2(rows, 2, 0)} == {"positive", "negative"}


def test_fetch_rejects_modified_file(tmp_path, monkeypatch):
    from llm_eval import data_prep
    (tmp_path / "dev.tsv").write_text("tampered")
    monkeypatch.setitem(data_prep.SOURCES["sst2_dev"], "url", "https://example.invalid/dev.tsv")
    with pytest.raises(data_prep.DataIntegrityError, match="sha256"):
        data_prep.fetch("sst2_dev", tmp_path)
