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

queries = [

    "section 302 ipc",

    "murder",

    "rape",

    "death sentence",

    "what was verdict in abdul nassar case"

]

for query in queries:

    print("\n" + "=" * 80)

    print("QUERY:", query)

    print("=" * 80)

    results = hybrid_search(
        query,
        model,
        index,
        bm25,
        chunks,
        top_k=3
    )

    for i, result in enumerate(results, 1):

        print(f"\nRESULT {i}")

        print("-" * 50)

        print(
            "Case:",
            result["case_name"]
        )

        print(
            result["chunk_text"][:500]
        )