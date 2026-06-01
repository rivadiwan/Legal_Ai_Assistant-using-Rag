from retrieve import *

print("Loading model...")
model = load_model()

print("Loading index...")
index = load_faiss_index()

print("Loading chunks...")
chunks = load_chunks()

queries = [

    "abdul nassar",

    "explain abdul nassar",
    "section 302 ipc",
    "murder case",  

]

for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = retrieve_chunks(
        query,
        model,
        index,
        chunks
    )

    for i, result in enumerate(results, 1):

        print(f"\nRESULT {i}")
        print("-" * 50)

        if result["score"] == "CASE_MATCH":

            print(
                result["chunk_text"][:800]
            )

        else:

            print(
                "FAISS Score:",
                result["score"]
            )

            print(
                result["chunk"]["chunk_text"][:800]
            )