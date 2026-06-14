"""Run retrieval evaluation over case_based_queries.json using the hybrid retriever.

Usage:
    python run_evaluation.py --k 5

Outputs:
    - prints aggregated metrics and writes evaluation_report.json
"""

import json
import argparse
import math
import os
import statistics
import sys

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

QUERY_FILE = os.path.join(HERE, "case_based_queries.json")


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_retriever_resources():
    from retrieve import load_model, load_faiss_index, load_chunks
    from bm25_retriever import build_bm25

    model = load_model()
    index = load_faiss_index()
    chunks = load_chunks()
    bm25 = build_bm25(chunks)
    return model, index, chunks, bm25


def retrieve(query, model, index, chunks, bm25, top_k):
    from hybrid_retrieve import hybrid_search
    return hybrid_search(query, model, index, chunks, bm25, top_k=top_k)


def extract_case_names(raw_results, k):
    cases = []
    seen = set()
    for r in raw_results:
        cn = r.get("case_name", "")
        if cn and cn not in seen:
            cases.append(cn)
            seen.add(cn)
        if len(cases) >= k:
            break
    return cases


def is_case_match(raw_results):
    return len(raw_results) > 0 and raw_results[0].get("score") == "CASE_MATCH"


def precision_at_k(retrieved_cases, gold):
    if not retrieved_cases:
        return 0.0
    return 1.0 / len(retrieved_cases) if gold in retrieved_cases else 0.0


def recall_at_k(retrieved_cases, gold):
    return 1.0 if gold in retrieved_cases else 0.0


def reciprocal_rank(retrieved_cases, gold):
    try:
        rank = retrieved_cases.index(gold) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(retrieved_cases, gold):
    for i, case in enumerate(retrieved_cases):
        if case == gold:
            return 1.0 / math.log2(i + 2)
    return 0.0


def evaluate(queries, k, model, index, chunks, bm25, debug=False):
    results = []
    case_match_count = 0

    for q in queries:
        qid = q["query_id"]
        gold = q["case_name"]
        query_text = q["query"]

        raw_results = retrieve(query_text, model, index, chunks, bm25, top_k=k)
        retrieved_cases = extract_case_names(raw_results, k)
        matched = is_case_match(raw_results)
        if matched:
            case_match_count += 1

        p = precision_at_k(retrieved_cases, gold)
        r = recall_at_k(retrieved_cases, gold)
        rr = reciprocal_rank(retrieved_cases, gold)
        ndcg = ndcg_at_k(retrieved_cases, gold)

        if debug:
            match_label = "[CASE_MATCH]" if matched else "[VECTOR+BM25]"
            print(f"  {qid} {match_label} gold={gold[:50]}...  found={r > 0}  rank_reciprocal={rr:.3f}")

        results.append({
            "query_id": qid,
            "query": query_text,
            "gold": gold,
            "retrieved": retrieved_cases,
            "case_match": matched,
            "precision@k": p,
            "recall@k": r,
            "reciprocal_rank": rr,
            "ndcg@k": ndcg,
        })

    precision_vals = [r["precision@k"] for r in results]
    recall_vals = [r["recall@k"] for r in results]
    rr_vals = [r["reciprocal_rank"] for r in results]
    ndcg_vals = [r["ndcg@k"] for r in results]

    agg = {
        "precision@k_mean": statistics.mean(precision_vals) if precision_vals else 0.0,
        "recall@k_mean": statistics.mean(recall_vals) if recall_vals else 0.0,
        "hits@k": statistics.mean([1.0 if v > 0 else 0.0 for v in recall_vals]) if recall_vals else 0.0,
        "mrr": statistics.mean(rr_vals) if rr_vals else 0.0,
        "ndcg@k_mean": statistics.mean(ndcg_vals) if ndcg_vals else 0.0,
        "queries_evaluated": len(results),
        "case_match_queries": case_match_count,
        "vector_search_queries": len(results) - case_match_count,
    }

    return results, agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="evaluation_report.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    queries = load_queries(QUERY_FILE)
    print(f"Loaded {len(queries)} queries from {QUERY_FILE}")

    print("\nLoading retriever resources (model, index, chunks, BM25)...")
    model, index, chunks, bm25 = load_retriever_resources()
    print(f"Resources loaded: model={type(model).__name__}, chunks={len(chunks)}")

    print(f"\nRunning evaluation with k={args.k}...")
    results, agg = evaluate(
        queries, k=args.k, model=model, index=index,
        chunks=chunks, bm25=bm25, debug=args.debug
    )

    report = {"aggregate": agg, "per_query": results}
    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nEvaluation summary:")
    for key, val in agg.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()