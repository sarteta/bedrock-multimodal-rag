"""BM25 + dense retrieval with reciprocal rank fusion and optional
cross-encoder rerank. Two strategies exposed: pure_semantic and hybrid.
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

    k=60 follows the Cormack et al paper. The effect is that a doc
    appearing at moderate rank in multiple rankings beats one ranked
    high in only a single ranking.
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
