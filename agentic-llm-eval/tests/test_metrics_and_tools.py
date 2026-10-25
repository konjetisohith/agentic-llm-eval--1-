import math, pytest
from llm_eval.squad_metrics import normalize_answer, exact_match, f1_score
from llm_eval.tools import calculator, ToolError
from llm_eval.pruning import prune_context, split_sentences, parse_config
from llm_eval.stats import wilson, mcnemar_exact, bootstrap_ci, paired_bootstrap_diff


def test_normalize_removes_articles_punct_case():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


def test_exact_match_and_f1_known_values():
    assert exact_match("the Eiffel Tower", "eiffel tower") == 1.0
    assert exact_match("Eiffel Tower", "Eiffel") == 0.0
    # gold 'new york city' vs pred 'new york': P=1, R=2/3 -> F1=0.8
    assert f1_score("new york city", "new york") == pytest.approx(0.8)
    assert f1_score("cat", "dog") == 0.0
    assert f1_score("", "") == 1.0 and f1_score("cat", "") == 0.0


def test_calculator_correct_and_safe():
    assert calculator("3 * (4 + 5)") == "27"
    assert calculator("1,000 * 2") == "2000"
    assert calculator("2 ^ 10") == "1024"
    assert calculator("7 / 2") == "3.5"
    for bad in ["__import__('os').system('ls')", "open('x')", "1/0", "9**9**9", "abs(-1)", "", "a + 1", "x" * 300]:
        with pytest.raises(ToolError):
            calculator(bad)


def test_pruning_keeps_relevant_sentence_in_order():
    ctx = ("The harbor opens at dawn. Tobias Hale built the lighthouse in 1872. Gulls nest on the cliffs. "
           "Visitors arrive by ferry. The cafe sells tea.")
    out = prune_context(ctx, "Who built the lighthouse?", 2)
    assert "Tobias Hale built the lighthouse" in out and len(split_sentences(out)) == 2
    assert out.index("harbor") < out.index("Tobias") if "harbor" in out else True
    assert prune_context(ctx, "x", None) == ctx and prune_context("One sentence.", "q", 3) == "One sentence."
    assert parse_config("none") is None and parse_config("topk:2") == 2
    with pytest.raises(ValueError):
        parse_config("topk:zero")


def test_wilson_matches_hand_calculation():
    lo, hi = wilson(2, 15)
    assert (lo, hi) == (pytest.approx(0.037, abs=0.005), pytest.approx(0.379, abs=0.005))
    assert math.isnan(wilson(0, 0)[0])


def test_mcnemar_and_bootstrap():
    a = [True] * 20 + [False] * 5
    b = [False] * 20 + [False] * 5
    r = mcnemar_exact(a, b)
    assert r["only_a"] == 20 and r["only_b"] == 0 and r["p_value"] < 0.001
    same = mcnemar_exact([True, False], [True, False])
    assert same["p_value"] == 1.0
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert lo < 0.5 < hi
    d = paired_bootstrap_diff([1, 1, 1, 0], [0, 0, 1, 0])
    assert d["mean_diff"] == pytest.approx(0.5)
