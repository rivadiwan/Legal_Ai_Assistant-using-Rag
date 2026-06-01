import fitz
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
import os

# =========================
# CONFIGURATION
# =========================

DATASET_PATH = "dataset"
OUTPUT_FILE = "output/legal_dataset.csv"

# Create output folder if missing
os.makedirs("output", exist_ok=True)

# Store all extracted data
all_data = []

# Counter for document IDs
doc_counter = 1

# =========================
# CLEANING FUNCTION
# =========================

def clean_text(text):

    # Remove Indian Kanoon footer links
    text = re.sub(r'Indian Kanoon - http\S+', ' ', text)

    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', ' ', text)

    # Remove repeated dots
    text = re.sub(r'\.{2,}', ' ', text)

    # Remove extra spaces/tabs/newlines
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =========================
# PROCESS YEAR FOLDERS
# =========================

for year_folder in Path(DATASET_PATH).iterdir():

    # Skip non-folders
    if not year_folder.is_dir():
        continue

    year = year_folder.name

    print(f"\nProcessing Year: {year}")

    pdf_files = list(year_folder.glob("*.pdf"))

    # =========================
    # PROCESS PDF FILES
    # =========================

    for pdf_file in tqdm(pdf_files):

        try:

            # Open PDF
            doc = fitz.open(pdf_file)

            full_text = ""

            # =========================
            # EXTRACT TEXT PAGE BY PAGE
            # =========================

            for page in doc:

                text = page.get_text()

                text = clean_text(text)

                # Skip empty pages
                if len(text.strip()) < 20:
                    continue

                full_text += text + " "

            # Skip empty PDFs
            if len(full_text.strip()) < 100:
                print(f"Skipped empty PDF: {pdf_file.name}")
                continue

            # Final cleanup
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            # =========================
            # EXTRACT METADATA
            # =========================

            case_name = "Unknown"
            judgment_date = "Unknown"

            first_1000_chars = full_text[:1000]

            # Example:
            # Abdul Nassar vs The State Of Kerala on 7 January, 2025

            match = re.search(
                r"([A-Za-z0-9 .,&()\-]+?)\s+vs\s+([A-Za-z0-9 .,&()\-]+?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})",
                first_1000_chars,
                re.IGNORECASE
            )

            if match:

                petitioner = match.group(1).strip()
                respondent = match.group(2).strip()

                case_name = f"{petitioner} vs {respondent}"
                judgment_date = match.group(3).strip()

            # =========================
            # CREATE DOCUMENT ID
            # =========================

            doc_id = f"SC_{year}_{doc_counter:04d}"

            # =========================
            # STORE DATA
            # =========================

            all_data.append({
                "doc_id": doc_id,
                "year": year,
                "case_name": case_name,
                "judgment_date": judgment_date,
                "file_name": pdf_file.name,
                "text": full_text
            })

            doc_counter += 1

        except Exception as e:
            print(f"\nError processing {pdf_file.name}: {e}")

# =========================
# CREATE DATAFRAME
# =========================

df = pd.DataFrame(all_data)

# =========================
# SAVE CSV
# =========================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# =========================
# SUMMARY
# =========================

print("\n=========================")
print("EXTRACTION COMPLETED")
print("=========================")

print(f"\nTotal documents processed: {len(df)}")

print("\nColumns:")
print(df.columns)

print("\nSample Data:")
print(df.head())