from retrieve import *

from bm25_retriever import *

from hybrid_retrieve import *

print("Loading model...")
model = load_model()

print("Loading index...")
index = load_faiss_index()

print("Loading chunks...")
chunks = load_chunks()

print("Building BM25...")
bm25 = build_bm25(chunks)

while True:

    query = input("\nQuery: ")

    if query == "exit":
        break

    results = hybrid_search(
        query,
        model,
        index,
        chunks,
        bm25,
        top_k=5
    )

    print("\nRESULTS\n")

    for i, result in enumerate(results, 1):

        print("=" * 60)

        print(
            f"{i}. {result['case_name']}"
        )

        print(
            result["chunk_text"][:400]
        )