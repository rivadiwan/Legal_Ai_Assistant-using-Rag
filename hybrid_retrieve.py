
from retrieve import retrieve_chunks
from bm25_retriever import bm25_search

def normalize_result(result):
    if "chunk" in result:
        chunk = result["chunk"]
        return {
            "case_name": chunk["case_name"],
            "year": chunk["year"],
            "chunk_text": chunk["chunk_text"],
            "score": result.get("score")
        }
    return result

def hybrid_search(query, model, index, chunks, bm25, top_k=10):
    faiss_results = retrieve_chunks(
        query, model, index, chunks, top_k=top_k
    )

    if (
        len(faiss_results) > 0 and
        faiss_results[0].get("score") == "CASE_MATCH"
    ):
        return faiss_results

    bm25_results = bm25_search(
        query, bm25, chunks, top_k=top_k
    )

    merged = []
    seen = set()

    for result in bm25_results + [normalize_result(r) for r in faiss_results]:
        text = result["chunk_text"]
        if text not in seen:
            merged.append(result)
            seen.add(text)

    return merged[:top_k]