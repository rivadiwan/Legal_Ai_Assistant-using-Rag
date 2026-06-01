from bm25_retriever import *

chunks = load_chunks()

bm25 = build_bm25(chunks)

queries = [

    "section 302 ipc",

    "murder",

    "rape",

    "death sentence"

]

for query in queries:

    print("\n" + "=" * 80)

    print("QUERY:", query)

    print("=" * 80)

    results = bm25_search(
        query,
        bm25,
        chunks
    )

    for i, result in enumerate(results, 1):

        print(f"\nRESULT {i}")

        print("-" * 50)

        print("Case:",
              result["case_name"])

        print(
            result["chunk_text"][:500]
        )