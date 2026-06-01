import pandas as pd
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

def load_dataset():

    df = pd.read_csv("output/cleaned_legal_dataset.csv")

    print(f"Documents Loaded: {len(df)}")

    return df

# ==========================================
# CREATE CHUNKS
# ==========================================

def create_chunks(df):

    text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        " "
    ],
    chunk_size=1000,
    chunk_overlap=200
    )

    all_chunks = []

    chunk_counter = 1

    for _, row in df.iterrows():

        text = str(row["cleaned_text"])

        chunks = text_splitter.split_text(text)

        for chunk in chunks:

            chunk_data = {
                "chunk_id": f"CHUNK_{chunk_counter}",
                "doc_id": row["doc_id"],
                "case_name": row["case_name"],
                "year": row["year"],
                "chunk_text": chunk
            }

            all_chunks.append(chunk_data)

            chunk_counter += 1

    return all_chunks

# ==========================================
# SAVE CHUNKS
# ==========================================

def save_chunks(chunks):

    with open("output/chunks.json", "w", encoding="utf-8") as f:

        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print("\nChunks saved successfully!")

    print(f"Total Chunks: {len(chunks)}")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    df = load_dataset()

    chunks = create_chunks(df)

    save_chunks(chunks)

    print("\nChunking Completed!")