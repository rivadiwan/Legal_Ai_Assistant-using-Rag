from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_results(
    query,
    results,
    top_k=5
):

    pairs = []

    for result in results:

        if "chunk" in result:

            text = result["chunk"]["chunk_text"]

        else:

            text = result["chunk_text"]

        pairs.append(
            [query, text]
        )

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        item[0]
        for item in ranked[:top_k]
    ]