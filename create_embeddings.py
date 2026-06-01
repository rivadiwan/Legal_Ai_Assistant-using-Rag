import json
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# =========================
# LOAD CHUNKS
# =========================

def load_chunks(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


# =========================
# CREATE EMBEDDINGS
# =========================

def create_embeddings(chunks):

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    texts = [
        chunk["chunk_text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings, model


# =========================
# CREATE FAISS INDEX
# =========================

def build_faiss_index(embeddings):

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# =========================
# SAVE FILES
# =========================

def save_index(index, path):

    faiss.write_index(index, path)


def save_metadata(chunks, path):

    with open(path, "wb") as f:
        pickle.dump(chunks, f)


# =========================
# MAIN
# =========================

def main():

    print("Loading chunks...")

    chunks = load_chunks(
        "output/chunks.json"
    )

    print(f"Total Chunks: {len(chunks)}")

    print("\nCreating embeddings...")

    embeddings, model = create_embeddings(chunks)

    print("\nBuilding FAISS index...")

    index = build_faiss_index(embeddings)

    print("\nSaving FAISS index...")

    save_index(
        index,
        "output/legal_index.faiss"
    )

    print("Saving metadata...")

    save_metadata(
        chunks,
        "output/chunks_metadata.pkl"
    )

    print("\nDONE!")
    print("Embeddings + FAISS index created successfully!")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    main()