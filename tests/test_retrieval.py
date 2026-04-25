from bedrock_rag.retrieval import (
    RetrievedDoc,
    pure_semantic,
    hybrid,
    reciprocal_rank_fusion,
)


class FakeVectorStore:
    def __init__(self, results: list[RetrievedDoc]):
        self._results = results
        self.calls: list[tuple[list[float], int]] = []

    def search(self, query_embedding, k):
        self.calls.append((query_embedding, k))
        return self._results[:k]


class FakeBM25:
    def __init__(self, results: list[RetrievedDoc]):
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query_text, k):
        self.calls.append((query_text, k))
        return self._results[:k]


class CapturingReranker:
    def __init__(self):
        self.calls: list[tuple[str, list[RetrievedDoc]]] = []

    def rerank(self, query, docs):
        self.calls.append((query, docs))
        # Reverse order to make it observable
        return list(reversed(docs))


def _doc(doc_id: str, score: float = 0.5, source: str = "semantic") -> RetrievedDoc:
    return RetrievedDoc(doc_id=doc_id, text=f"text-{doc_id}", score=score, source=source)


def test_rrf_single_ranking_preserves_order():
    ranking = [_doc("a"), _doc("b"), _doc("c")]
    fused = reciprocal_rank_fusion([ranking])
    assert [d.doc_id for d in fused] == ["a", "b", "c"]


def test_rrf_promotes_doc_in_both_rankings():
    sem = [_doc("a"), _doc("b"), _doc("c")]
    bm = [_doc("c"), _doc("d"), _doc("a")]
    fused = reciprocal_rank_fusion([sem, bm])
    # 'a' is rank 1 in semantic, rank 3 in bm
    # 'c' is rank 3 in semantic, rank 1 in bm
    # Both should rank above 'b' and 'd' (each only in one list)
    top_two = [d.doc_id for d in fused[:2]]
    assert "a" in top_two
    assert "c" in top_two


def test_rrf_lower_k_promotes_high_ranks_more():
    sem = [_doc("a"), _doc("b"), _doc("c")]
    fused_low_k = reciprocal_rank_fusion([sem], k=1)
    fused_high_k = reciprocal_rank_fusion([sem], k=200)
    # With low k, gap between a and b is larger
    gap_low = fused_low_k[0].score - fused_low_k[1].score
    gap_high = fused_high_k[0].score - fused_high_k[1].score
    assert gap_low > gap_high


def test_pure_semantic_returns_vector_store_results():
    store = FakeVectorStore([_doc("a"), _doc("b"), _doc("c")])
    out = pure_semantic([0.1] * 1024, store, k=2)
    assert [d.doc_id for d in out] == ["a", "b"]
    assert store.calls[0] == ([0.1] * 1024, 2)


def test_hybrid_calls_both_stores():
    sem_store = FakeVectorStore([_doc("a"), _doc("b")])
    bm_store = FakeBM25([_doc("c"), _doc("d")])
    out = hybrid("query text", [0.1] * 1024, sem_store, bm_store, k=2)
    assert sem_store.calls
    assert bm_store.calls
    assert sem_store.calls[0][1] == 4  # k * 2 — over-retrieve before fusion


def test_hybrid_with_reranker():
    sem_store = FakeVectorStore([_doc("a"), _doc("b")])
    bm_store = FakeBM25([_doc("c"), _doc("d")])
    reranker = CapturingReranker()
    out = hybrid("q", [0.1] * 1024, sem_store, bm_store, k=2, reranker=reranker)
    # Reranker called with the fused top-k
    assert len(reranker.calls) == 1
    # Output is the reranked order (reversed by our fake reranker)
    fused_ids = [d.doc_id for d in reranker.calls[0][1]]
    out_ids = [d.doc_id for d in out]
    assert out_ids == list(reversed(fused_ids))


def test_hybrid_without_reranker_returns_fused_directly():
    sem_store = FakeVectorStore([_doc("a"), _doc("b")])
    bm_store = FakeBM25([_doc("a"), _doc("c")])
    out = hybrid("q", [0.1] * 1024, sem_store, bm_store, k=2)
    # 'a' should be top because it's in both
    assert out[0].doc_id == "a"
    assert out[0].source == "fused"


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
