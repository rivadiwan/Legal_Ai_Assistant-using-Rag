from retrieve import *
from bm25_retriever import *
from reranker import *

model = load_model()
index = load_faiss_index()
chunks = load_chunks()

bm25 = build_bm25(chunks)

query = input("Query: ")

bm25_results = bm25_search(
    query,
    bm25,
    chunks,
    top_k=10
)

faiss_results = retrieve_chunks(
    query,
    model,
    index,
    chunks,
    top_k=10
)

all_results = bm25_results + faiss_results

reranked = rerank_results(
    query,
    all_results,
    top_k=5
)

print("\n===== RERANKED =====\n")

for result in reranked:

    if "chunk" in result:

        print(
            result["chunk"]["case_name"]
        )

    else:

        print(
            result["case_name"]
        )