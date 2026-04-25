from bedrock_rag.eval import (
    GoldenItem,
    evaluate_retrieval,
    parse_judge_response,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5


def test_recall_at_k_no_hit():
    assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0


def test_recall_at_k_outside_window():
    # 'a' is at rank 5 but k=3 -- should not count
    assert recall_at_k(["x", "y", "z", "w", "a"], ["a"], k=3) == 0.0


def test_recall_at_k_empty_expected():
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == 1 / 3


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_reciprocal_rank_uses_first_match():
    # 'a' at rank 2, 'b' at rank 3 -- should use rank 2 since 'a' comes first
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


def test_evaluate_retrieval_aggregates():
    golden = [
        GoldenItem(query="q1", expected_doc_ids=["a"]),
        GoldenItem(query="q2", expected_doc_ids=["b"]),
    ]
    # Mock retriever: returns expected for q1, nothing for q2
    def fake_run(query: str) -> list[str]:
        return {"q1": ["a", "x"], "q2": ["x", "y"]}[query]

    metrics = evaluate_retrieval(golden, fake_run, k=2)
    # q1: recall=1.0, rr=1.0
    # q2: recall=0.0, rr=0.0
    assert metrics.recall_at_k == 0.5
    assert metrics.mrr == 0.5
    assert metrics.n_queries == 2


def test_evaluate_retrieval_empty_set():
    metrics = evaluate_retrieval([], lambda q: [], k=10)
    assert metrics.n_queries == 0
    assert metrics.recall_at_k == 0.0


def test_parse_judge_score_rationale():
    text = "SCORE: 0.85\nRATIONALE: The answer is well-grounded in the provided context."
    j = parse_judge_response(text)
    assert j.score == 0.85
    assert "well-grounded" in j.rationale


def test_parse_judge_handles_extra_whitespace():
    text = "  SCORE: 0.5  \n  RATIONALE: meh.  "
    j = parse_judge_response(text)
    assert j.score == 0.5
    assert j.rationale == "meh."


def test_parse_judge_invalid_score_falls_back_to_zero():
    text = "SCORE: not-a-number\nRATIONALE: noise"
    j = parse_judge_response(text)
    assert j.score == 0.0


def test_parse_judge_missing_score():
    text = "RATIONALE: only rationale here"
    j = parse_judge_response(text)
    assert j.score == 0.0
    assert "only rationale" in j.rationale
