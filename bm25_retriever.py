
from rank_bm25 import BM25Okapi

def build_bm25(chunks):
    corpus = [chunk["chunk_text"].lower().split() for chunk in chunks]
    return BM25Okapi(corpus)

def bm25_search(query, bm25, chunks, top_k=5):
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for idx, score in ranked[:top_k]:
        if score <= 0:
            continue

        results.append({
            "score": float(score),
            "case_name": chunks[idx]["case_name"],
            "year": chunks[idx]["year"],
            "chunk_text": chunks[idx]["chunk_text"]
        })

    return results
