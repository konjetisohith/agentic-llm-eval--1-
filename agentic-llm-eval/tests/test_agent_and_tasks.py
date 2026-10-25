import json
from llm_eval.agent import Agent, parse_action
from llm_eval.ollama_client import ChatResult, OllamaError
from llm_eval.tasks import score, parse_sentiment, is_abstain, parse_number, build_prompt
from llm_eval.agent import AgentTrace


class Scripted:
    """Returns pre-set replies in order (or raises)."""
    def __init__(self, replies): self.replies, self.calls = list(replies), []
    def chat(self, model, messages, options=None, fmt="json"):
        self.calls.append([m["content"] for m in messages])
        r = self.replies.pop(0)
        if isinstance(r, Exception): raise r
        return ChatResult(r, 50, 10, 0.5, 0.0, 0.2)


A = lambda a, **kw: json.dumps({"thought": "t", "action": a, "args": kw})
mk = lambda c, **k: Agent(c, "m", {"num_ctx": 4096}, **k)


def test_direct_finish():
    tr = mk(Scripted([A("finish", answer="positive")])).run("q")
    assert tr.status == "finished" and tr.final_answer == "positive" and tr.tool_calls == []
    assert tr.prompt_tokens == 50 and tr.latency_s == 0.5


def test_tool_then_finish_with_observation_fed_back():
    c = Scripted([A("calculator", expression="12*12"), A("finish", answer="144")])
    tr = mk(c).run("q")
    assert tr.status == "finished" and tr.tool_calls[0]["result"] == "144" and tr.tool_calls[0]["ok"]
    assert "Observation: 144" in c.calls[1][-1]


def test_tool_error_is_recoverable():
    c = Scripted([A("calculator", expression="1/0"), A("calculator", expression="1+1"), A("finish", answer="2")])
    tr = mk(c).run("q")
    assert tr.status == "finished" and [x["ok"] for x in tr.tool_calls] == [False, True]


def test_malformed_output_repaired_then_succeeds():
    tr = mk(Scripted(["not json", A("finish", answer="x")])).run("q")
    assert tr.status == "finished" and tr.repairs == 1


def test_repair_budget_exhausted():
    tr = mk(Scripted(["nope"] * 5), max_repairs=2).run("q")
    assert tr.status == "parse_failure" and tr.repairs == 3


def test_unknown_tool_gets_error_observation():
    tr = mk(Scripted([A("web_search", q="x"), A("finish", answer="ok")])).run("q")
    assert tr.status == "finished" and tr.tool_calls[0]["ok"] is False


def test_turn_limit_and_backend_error():
    tr = mk(Scripted([A("calculator", expression="1+1")] * 9), max_turns=3).run("q")
    assert tr.status == "turn_limit" and tr.turns == 3 and tr.final_answer is None
    tr2 = mk(Scripted([OllamaError("boom")])).run("q")
    assert tr2.status == "backend_error" and "boom" in tr2.error


def test_parse_action_variants():
    assert parse_action('Sure: {"action":"finish","answer":"x"}')["args"] == {"answer": "x"}
    assert parse_action("{}") is None and parse_action("hello") is None


# ---------- scoring ----------
def tr(answer, status="finished", calls=()):
    return AgentTrace(status=status, final_answer=answer, tool_calls=list(calls))


def test_sst2_unparseable_counts_as_failure_in_denominator():
    sc = {"task": "sst2", "label": "positive"}
    assert score(sc, tr("positive"))["success"] is True
    bad = score(sc, tr("It is quite good, honestly"))
    assert bad["success"] is False and bad["parse_ok"] is False
    assert score(sc, tr(None, status="parse_failure"))["success"] is False
    assert parse_sentiment("positive and negative") is None


def test_squad_answerable_and_unanswerable_scoring():
    ans = {"task": "squad", "context": "Tobias Hale built it in 1872.", "question": "q", "answers": ["Tobias Hale"], "is_impossible": False}
    s = score(ans, tr("tobias hale"))
    assert s["success"] and s["em"] == 1.0 and s["grounded"] is True and not s["abstained"]
    s = score(ans, tr("Marcus Webb"))
    assert not s["success"] and s["grounded"] is False            # fabricated, not in context
    s = score(ans, tr("UNANSWERABLE"))
    assert s["abstained"] and not s["success"] and s["f1"] == 0.0  # false abstention
    imp = dict(ans, answers=[], is_impossible=True)
    assert score(imp, tr("UNANSWERABLE"))["success"] is True
    assert score(imp, tr("1872"))["success"] is False              # answered an unanswerable question
    assert score(imp, tr(None, "turn_limit"))["success"] is False


def test_squad_answer_retention_uses_pruned_context():
    sc = {"task": "squad", "context": "A. Tobias Hale built it.", "question": "q", "answers": ["Tobias Hale"], "is_impossible": False}
    assert score(sc, tr("x"), context_used="A.")["answer_retained"] is False
    assert score(sc, tr("x"), context_used="Tobias Hale built it.")["answer_retained"] is True


def test_tools_scoring_and_compliance():
    req = {"task": "tools", "kind": "numeric", "gold": "2010", "tool_policy": "required"}
    call = [{"name": "calculator", "args": {}, "ok": True, "result": "2010"}]
    s = score(req, tr("2,010", calls=call)); assert s["success"] and s["tool_compliant"]
    s = score(req, tr("2010")); assert s["success"] and not s["tool_compliant"]       # right answer, skipped required tool
    nn = {"task": "tools", "kind": "text", "gold": "PLANET", "tool_policy": "not_needed"}
    assert score(nn, tr("planet"))["success"]
    assert not score(nn, tr("planet", calls=call))["tool_compliant"]                  # unnecessary call
    assert parse_number("$1,234.50 total") == 1234.5 and is_abstain("I don't know") and not is_abstain("Paris")
    assert "UNANSWERABLE" in build_prompt({"task": "squad", "context": "c", "question": "q"})
