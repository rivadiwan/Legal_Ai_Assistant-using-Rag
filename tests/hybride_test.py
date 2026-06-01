from retrieve import *
from bm25_retriever import *

model = load_model()
index = load_faiss_index()
chunks = load_chunks()

bm25 = build_bm25(chunks)

query = input("Query: ")

# BM25
bm25_results = bm25_search(
    query,
    bm25,
    chunks,
    top_k=5
)

print("\n===== BM25 =====")

for result in bm25_results:

    print(
        result["case_name"]
    )

# FAISS
faiss_results = retrieve_chunks(
    query,
    model,
    index,
    chunks,
    top_k=5
)

print("\n===== FAISS =====")

for result in faiss_results:

    if "case_name" in result:

        print(
            result["case_name"]
        )

    else:

        print(
            result["chunk"]["case_name"]
        )