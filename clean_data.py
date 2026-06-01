import pandas as pd
import re
import os

# =========================
# CONFIGURATION
# =========================

INPUT_FILE = "output/legal_dataset.csv"
OUTPUT_FILE = "output/cleaned_legal_dataset.csv"

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(INPUT_FILE)

print("Original Rows:", len(df))

# =========================
# CLEANING FUNCTION
# =========================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Remove Indian Kanoon links
    text = re.sub(r'Indian Kanoon - http\S+', ' ', text)

    # Remove page numbers
    text = re.sub(r'\bPage \d+\b', ' ', text)

    # Remove standalone numbers lines
    text = re.sub(r'\n\s*\d+\s*\n', ' ', text)

    # Remove tabs/newlines
    text = re.sub(r'[\n\t\r]+', ' ', text)

    # Remove weird unicode spaces
    text = text.replace('\xa0', ' ')

    # Remove repeated dots
    text = re.sub(r'\.{2,}', '.', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove repeated case title pattern
    # Example:
    # Abdul Nassar vs The State Of Kerala on January...
    text = re.sub(
        r'([A-Za-z0-9 .,&()\-]+?)\s+vs\s+([A-Za-z0-9 .,&()\-]+?)\s+on\s+[A-Za-z0-9 ,]+',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Final strip
    text = text.strip()

    return text

# =========================
# APPLY CLEANING
# =========================

df["cleaned_text"] = df["text"].apply(clean_text)

# =========================
# REMOVE EMPTY ROWS
# =========================

df = df[df["cleaned_text"].str.len() > 100]

# =========================
# KEEP IMPORTANT COLUMNS
# =========================

final_df = df[
    [
        "doc_id",
        "year",
        "case_name",
        "judgment_date",
        "file_name",
        "cleaned_text"
    ]
]

# =========================
# SAVE CLEANED DATASET
# =========================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# =========================
# SUMMARY
# =========================

print("\n=========================")
print("CLEANING COMPLETED")
print("=========================")

print("\nRemaining Rows:", len(final_df))

print("\nColumns:")
print(final_df.columns)

print("\nSample Cleaned Text:\n")
print(final_df["cleaned_text"].iloc[0][:1000])