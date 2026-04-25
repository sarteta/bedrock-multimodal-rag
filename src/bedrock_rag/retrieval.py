"""Hybrid retrieval: BM25 keyword + dense semantic, fused via reciprocal
rank fusion. Optional cross-encoder rerank pass.

Two strategies:
- pure_semantic: cosine similarity over dense embeddings. Cheap, fast.
- hybrid: BM25 + dense, RRF-fused, optionally reranked. ~2x cost but
  better Recall@10 on most eval sets.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    source: str  # "semantic" | "bm25" | "fused"


class VectorStore(Protocol):
    def search(self, query_embedding: list[float], k: int) -> list[RetrievedDoc]: ...


class BM25Index(Protocol):
    def search(self, query_text: str, k: int) -> list[RetrievedDoc]: ...


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]: ...


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedDoc]],
    k: int = 60,
) -> list[RetrievedDoc]:
    """Combine multiple ranked lists into one using RRF.

    The constant k=60 is the value from the original Cormack et al
    paper. It biases toward documents that appear in multiple rankings
    even at moderate ranks, which is what we want — a doc that's #5 in
    semantic AND #5 in BM25 should beat one that's #1 in semantic but
    not in BM25 at all.
    """
    scores: dict[str, float] = {}
    docs_by_id: dict[str, RetrievedDoc] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0) + 1 / (k + rank + 1)
            if doc.doc_id not in docs_by_id:
                docs_by_id[doc.doc_id] = doc

    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [
        RetrievedDoc(doc_id=i, text=docs_by_id[i].text, score=scores[i], source="fused")
        for i in sorted_ids
    ]


def pure_semantic(
    query_embedding: list[float],
    vector_store: VectorStore,
    k: int = 10,
) -> list[RetrievedDoc]:
    """Cheapest path. Cosine over dense embeddings."""
    return vector_store.search(query_embedding, k)


def hybrid(
    query_text: str,
    query_embedding: list[float],
    vector_store: VectorStore,
    bm25_index: BM25Index,
    k: int = 10,
    reranker: Reranker | None = None,
) -> list[RetrievedDoc]:
    """BM25 + dense, fused via RRF, optionally reranked."""
    semantic_results = vector_store.search(query_embedding, k * 2)
    bm25_results = bm25_index.search(query_text, k * 2)

    fused = reciprocal_rank_fusion([semantic_results, bm25_results])[:k]

    if reranker is not None:
        return reranker.rerank(query_text, fused)
    return fused
