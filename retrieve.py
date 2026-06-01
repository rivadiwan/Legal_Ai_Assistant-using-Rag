import faiss
import json
from rapidfuzz import fuzz
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================
# LOAD EMBEDDING MODEL
# =========================

def load_model():

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    return model


# =========================
# LOAD FAISS INDEX
# =========================

def load_faiss_index():

    index = faiss.read_index(
        "output/legal_index.faiss"
    )

    return index


# =========================
# LOAD CHUNKS
# =========================

def load_chunks():

    with open(
        "output/chunks.json",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    return chunks


# =========================
# CASE NAME SEARCH
# =========================
""""
def find_case_chunks(
    query,
    chunks
):

    query = query.lower().strip()

    matching_chunks = []

    for chunk in chunks:

        case_name = chunk.get(
            "case_name",
            ""
        ).lower()

        if query in case_name:

            matching_chunks.append(chunk)

    return matching_chunks
"""
# =========================
# FUZZY CASE SEARCH
# =========================

def find_case_match(query, chunks):

    query = query.lower()

    best_score = 0
    best_case = None

    case_names = {}

    # Collect unique case names
    for chunk in chunks:

        case_name = chunk["case_name"]

        if case_name not in case_names:
            case_names[case_name] = True

    # Compare query with every case name
    for case_name in case_names:

        score = fuzz.partial_ratio(
            query,
            case_name.lower()
        )

        if score > best_score:
            best_score = score
            best_case = case_name
    print(
        f"Best Match: {best_case} "
        f"Score: {best_score}"
    )

    # Threshold
    if best_score >= 50:
        return best_case

    return None

# =========================
# RETRIEVE CHUNKS
# =========================

def retrieve_chunks(
    query,
    model,
    index,
    chunks,
    top_k=5
):
    case_match = find_case_match(
    query,
    chunks
    )
    if case_match:
        print("\nCase Match Found:", case_match)

        results = []

        for chunk in chunks:

            if chunk["case_name"] == case_match:

                results.append({

                "score": "CASE_MATCH",

                "case_name":
                    chunk["case_name"],

                "year":
                    chunk["year"],

                "chunk_text":
                    chunk["chunk_text"]
                 })

        return results


    # -------------------------
    # STEP 2:
    # FAISS SEARCH
    # -------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx, distance in zip(
        indices[0],
        distances[0]
    ):

        results.append(
            {
                "score": float(distance),
                "chunk": chunks[idx]
            }
        )

    return results