from retrieve import retrieve_chunks
from bm25_retriever import bm25_search


def hybrid_search(
    query,
    model,
    index,
    chunks,
    bm25,
    top_k=10
):

    # First check case match
    faiss_results = retrieve_chunks(
        query,
        model,
        index,
        chunks,
        top_k=top_k
    )

    # If retrieve_chunks already found a case
    if (
        len(faiss_results) > 0 and
        faiss_results[0]["score"] == "CASE_MATCH"
    ):
        return faiss_results

    # Otherwise normal hybrid retrieval
    bm25_results = bm25_search(
        query,
        bm25,
        chunks,
        top_k=top_k
    )

    merged = []
    seen = set()

    for result in bm25_results:

        text = result["chunk_text"]

        if text not in seen:
            merged.append(result)
            seen.add(text)

    for result in faiss_results:

        if "chunk" in result:

            chunk = result["chunk"]

            text = chunk["chunk_text"]

            if text not in seen:

                merged.append({
                    "case_name": chunk["case_name"],
                    "year": chunk["year"],
                    "chunk_text": chunk["chunk_text"]
                })

                seen.add(text)

        else:

            text = result["chunk_text"]

            if text not in seen:

                merged.append(result)

                seen.add(text)

    return merged[:top_k]