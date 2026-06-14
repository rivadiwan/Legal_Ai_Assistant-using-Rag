
from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    return _reranker

def rerank_results(query, results, top_k=5):
    if not results:
        return []

    pairs = [
        [query, result["chunk_text"]]
        for result in results
    ]

    scores = get_reranker().predict(pairs)

    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [item[0] for item in ranked[:top_k]]
