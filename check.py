import pandas as pd




df = pd.read_csv("output/cleaned_legal_dataset.csv")

# Check empty texts
print(len(df))
empty_rows = df[df['text'].str.len() < 50]

print("Empty rows:", len(empty_rows))

print("\nSample Cleaned Text:\n")
print(df["cleaned_text"].iloc[0])