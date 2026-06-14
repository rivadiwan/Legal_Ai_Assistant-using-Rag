import json
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve import (
    load_model,
    load_faiss_index,
    load_chunks
)

from bm25_retriever import build_bm25
from hybrid_retrieve import hybrid_search
from reranker import rerank_results

USE_RERANKER = True # Set to True to use the reranker, False to skip it
# -----------------------------
# Load evaluation dataset
# -----------------------------
with open("evaluate/evaluation_queries.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

print(f"Loaded {len(queries)} evaluation queries")


# -----------------------------
# Load retrieval system
# -----------------------------
print("Loading model...")
model = load_model()

print("Loading FAISS...")
index = load_faiss_index()

print("Loading chunks...")
chunks = load_chunks()

print("Building BM25...")
bm25 = build_bm25(chunks)


# -----------------------------
# Metrics
# -----------------------------
total_queries = 0

recall1_hits = 0
recall5_hits = 0
recall10_hits = 0

mrr_total = 0.0

failed_queries = []
all_results = []


# -----------------------------
# Evaluation Loop
# -----------------------------
for item in queries:

    query = item["query"]
    expected_case = item["expected_case"]

    total_queries += 1

    try:
        
        results = hybrid_search(
            query=query,
            model=model,index=index,
            chunks=chunks,
            bm25=bm25,
            top_k=10
        )

        if USE_RERANKER and results:
            results = rerank_results(
                query=query,
                results=results,
                top_k=10
            )
            #debug for self after reranking
            if total_queries <= 3:
                print("\nAFTER RERANK:")
                print(results[0])

        retrieved_cases = []

        for r in results:

            case_name = r.get("case_name")

            if case_name:
                retrieved_cases.append(case_name)

        # remove duplicates while preserving order
        retrieved_cases = list(dict.fromkeys(retrieved_cases))

        # -------------------------
        # Recall@1
        # -------------------------
        if len(retrieved_cases) > 0:

            if retrieved_cases[0] == expected_case:
                recall1_hits += 1

        # -------------------------
        # Recall@5
        # -------------------------
        if expected_case in retrieved_cases[:5]:
            recall5_hits += 1

        # -------------------------
        # Recall@10
        # -------------------------
        if expected_case in retrieved_cases[:10]:
            recall10_hits += 1

        # -------------------------
        # MRR
        # -------------------------
        rank = None

        for i, case in enumerate(retrieved_cases, start=1):

            if case == expected_case:
                rank = i
                break

        if rank:
            mrr_total += 1 / rank

        # -------------------------
        # Store detailed results
        # -------------------------
        all_results.append({
            "query_id": item["query_id"],
            "query": query,
            "expected_case": expected_case,
            "retrieved_cases": retrieved_cases,
            "rank": rank
        })

        # -------------------------
        # Failed query
        # -------------------------
        if rank is None:

            failed_queries.append({
                "query_id": item["query_id"],
                "query": query,
                "expected_case": expected_case,
                "retrieved_cases": retrieved_cases
            })

    except Exception as e:

        failed_queries.append({
            "query_id": item["query_id"],
            "query": query,
            "expected_case": expected_case,
            "error": str(e)
        })


# -----------------------------
# Final Metrics
# -----------------------------
recall_at_1 = recall1_hits / total_queries
recall_at_5 = recall5_hits / total_queries
recall_at_10 = recall10_hits / total_queries

mrr = mrr_total / total_queries


metrics = {
    "total_queries": total_queries,
    "recall@1": round(recall_at_1, 4),
    "recall@5": round(recall_at_5, 4),
    "recall@10": round(recall_at_10, 4),
    "mrr": round(mrr, 4),
    "failed_queries": len(failed_queries)
}


# -----------------------------
# Save outputs
# -----------------------------
output_name = (
    "evaluation_results_reranker.json"
    if USE_RERANKER
    else "evaluation_results_hybrid.json"
)

with open(output_name, "w", encoding="utf-8") as f:
    json.dump(
        {
            "metrics": metrics,
            "details": all_results
        },
        f,
        indent=4
    )
failed_name = (
    "failed_queries_reranker.json"
    if USE_RERANKER
    else "failed_queries_hybrid.json"
)
with open(failed_name, "w", encoding="utf-8") as f:
    json.dump(
        failed_queries,
        f,
        indent=4
)

# -----------------------------
# Console Output
# -----------------------------
print("\n" + "="*50)
print("RETRIEVAL EVALUATION RESULTS")
print("="*50)

print(f"Total Queries : {total_queries}")
print(f"Recall@1      : {recall_at_1:.4f}")
print(f"Recall@5      : {recall_at_5:.4f}")
print(f"Recall@10     : {recall_at_10:.4f}")
print(f"MRR           : {mrr:.4f}")

print(f"\nFailed Queries: {len(failed_queries)}")

print("\nSaved:")
print(output_name)
print(failed_name)