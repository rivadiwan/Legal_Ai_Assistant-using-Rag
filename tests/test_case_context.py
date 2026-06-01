from retrieve import load_chunks
from case_context import get_case_chunks

chunks = load_chunks()

case_name = (
    "Abdul Nassar vs The State Of Kerala"
)

results = get_case_chunks(
    case_name,
    chunks
)

print(
    f"Found {len(results)} chunks\n"
)

for chunk in results:

    print(
        chunk["chunk_id"],
        chunk["chunk_text"][:300]
    )

    print("\n---\n")