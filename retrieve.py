
import faiss
import json
import os
from rapidfuzz import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

def load_faiss_index():
    path = "output/legal_index.faiss"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run create_embeddings.py first."
        )
    return faiss.read_index(path)

def load_chunks():
    path = "output/chunks.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the preprocessing pipeline first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_case_match(query, chunks):
    query = query.lower().strip()

    best_score = 0
    best_case = None

    case_names = {chunk["case_name"] for chunk in chunks}

    for case_name in case_names:
        score = fuzz.partial_ratio(query, case_name.lower())
        if score > best_score:
            best_score = score
            best_case = case_name

    print(f"Best Match: {best_case} Score: {best_score}")

    if best_score >= 75:
        return best_case

    return None

def retrieve_chunks(query, model, index, chunks, top_k=5):
    case_match = find_case_match(query, chunks)

    if case_match:
        return [
            {
                "score": "CASE_MATCH",
                "case_name": chunk["case_name"],
                "year": chunk["year"],
                "chunk_text": chunk["chunk_text"]
            }
            for chunk in chunks
            if chunk["case_name"] == case_match
        ]

    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, top_k)

    return [
        {
            "score": float(distance),
            "chunk": chunks[idx]
        }
        for idx, distance in zip(indices[0], distances[0])
        if idx >= 0
    ]
