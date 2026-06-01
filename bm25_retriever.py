import json
from rank_bm25 import BM25Okapi


def load_chunks():

    with open(
        "output/chunks.json",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    return chunks


def build_bm25(chunks):

    corpus = []

    for chunk in chunks:

        tokens = chunk["chunk_text"].lower().split()

        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)

    return bm25


def bm25_search(
    query,
    bm25,
    chunks,
    top_k=5
):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked[:top_k]:

        results.append({

            "score": score,

            "case_name":
                chunks[idx]["case_name"],

            "year":
                chunks[idx]["year"],

            "chunk_text":
                chunks[idx]["chunk_text"]

        })

    return results