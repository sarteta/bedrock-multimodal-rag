"""Retrieval and generation evaluators.

Retrieval metrics (Recall@k, MRR) are deterministic and cheap. Run
them after every retrieval change.

Generation metrics use an LLM-as-judge. More expensive and noisier;
run on a smaller golden set, weekly or before releases.
"""
from dataclasses import dataclass


@dataclass
class GoldenItem:
    query: str
    expected_doc_ids: list[str]


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    n_queries: int


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Fraction of expected docs that appear in top-k retrieved."""
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(expected_ids))
    return hits / len(expected_ids)


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """1/rank of the first expected doc in retrieved list. 0 if none found."""
    expected_set = set(expected_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    golden_set: list[GoldenItem],
    run_query: callable,
    k: int = 10,
) -> RetrievalMetrics:
    """run_query: callable(query) -> list of doc_ids in retrieval order."""
    if not golden_set:
        return RetrievalMetrics(recall_at_k=0.0, mrr=0.0, n_queries=0)

    recalls: list[float] = []
    rrs: list[float] = []
    for item in golden_set:
        retrieved = run_query(item.query)
        recalls.append(recall_at_k(retrieved, item.expected_doc_ids, k))
        rrs.append(reciprocal_rank(retrieved, item.expected_doc_ids))

    return RetrievalMetrics(
        recall_at_k=sum(recalls) / len(recalls),
        mrr=sum(rrs) / len(rrs),
        n_queries=len(golden_set),
    )


@dataclass
class FaithfulnessJudgement:
    score: float  # 0.0 to 1.0
    rationale: str


def parse_judge_response(text: str) -> FaithfulnessJudgement:
    """Parse 'SCORE: 0.7\\nRATIONALE: ...' format from the judge LLM."""
    score = 0.0
    rationale = ""
    for raw in text.strip().split("\n"):
        line = raw.strip()
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 0.0
        elif line.startswith("RATIONALE:"):
            rationale = line.replace("RATIONALE:", "").strip()
    return FaithfulnessJudgement(score=score, rationale=rationale)
