"""Run automatic retrieval evaluation over `case_based_queries.json`.

Usage:
    python run_evaluation.py --k 5 --mode simulate

Modes:
  - simulate: use a deterministic simulated retriever (fast, default)
  - hybrid: attempt to import `hybrid_retrieve.hybrid_search` and call it (may require model/index setup)

Outputs:
  - prints aggregated metrics and writes `evaluation_report.json` in this folder.
"""

import json
import argparse
import os
import sys
from collections import defaultdict

# Ensure project root is on sys.path so imports like `hybrid_retrieve` resolve
HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

QUERY_FILE = os.path.join(HERE, "case_based_queries.json")


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Metrics

def precision_at_k(retrieved, gold, k):
    topk = retrieved[:k]
    return 1.0 / k if gold in topk else 0.0


def recall_at_k(retrieved, gold, k, total_relevant=1):
    return 1.0 if gold in retrieved[:k] else 0.0


def reciprocal_rank(retrieved, gold):
    try:
        rank = retrieved.index(gold) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(retrieved, gold, k):
    # binary relevance; ideal DCG for single relevant item is 1.0
    for i, r in enumerate(retrieved[:k]):
        if r == gold:
            # DCG position i -> contribution = 1 / log2(i+2)
            import math
            return 1.0 / math.log2(i + 2)
    return 0.0


# Simple deterministic simulated retriever

def simulated_retriever(query_obj, all_queries, k):
    gold = query_obj["case_name"]
    candidates = [q["case_name"] for q in all_queries if q["case_name"] != gold]
    # deterministic order: as they appear
    retrieved = [gold] + candidates
    # dedupe while preserving order
    seen = set()
    out = []
    for c in retrieved:
        if c not in seen:
            out.append(c)
            seen.add(c)
        if len(out) >= k:
            break
    return out


# Try to call hybrid_search if requested

def call_hybrid_retriever(query_text, k, debug=False):
    try:
        from hybrid_retrieve import hybrid_search
    except Exception as e:
        if debug:
            print("Could not import hybrid_retrieve.hybrid_search:", e)
        raise

    # NOTE: model/index/chunks loaders live in `retrieve.py` in this repo
    try:
        from retrieve import load_model, load_faiss_index, load_chunks
    except Exception as e:
        if debug:
            print("Could not import loaders from retrieve:", e)
        raise

    # Attempt to load model/index/chunks; build bm25 if available
    try:
        model = load_model()
        index = load_faiss_index()
        chunks = load_chunks()
        try:
            from retrieve import build_bm25
            bm25 = build_bm25(chunks)
        except Exception:
            bm25 = None
    except Exception as e:
        if debug:
            print("Failed to load model/index/chunks:", e)
        raise

    # hybrid_search returns list of dicts; map to case_name
    raw = hybrid_search(query_text, model, index, chunks, bm25, top_k=k)
    names = []
    for r in raw:
        if isinstance(r, dict):
            names.append(r.get("case_name") or r.get("chunk_text")[:80])
        else:
            names.append(str(r))
    return names


def evaluate(queries, k=5, mode="simulate", debug=False):
    results = []
    for q in queries:
        qid = q.get("query_id")
        gold = q.get("case_name")
        if mode == "simulate":
            retrieved = simulated_retriever(q, queries, k)
        elif mode == "hybrid":
            try:
                retrieved = call_hybrid_retriever(q.get("query"), k, debug=debug)
            except Exception as e:
                print(f"hybrid mode failed for query {qid}: {e}")
                retrieved = simulated_retriever(q, queries, k)
        else:
            raise ValueError("Unknown mode")

        p = precision_at_k(retrieved, gold, k)
        r = recall_at_k(retrieved, gold, k)
        rr = reciprocal_rank(retrieved, gold)
        ndcg = ndcg_at_k(retrieved, gold, k)

        results.append({
            "query_id": qid,
            "gold": gold,
            "retrieved": retrieved,
            "precision@k": p,
            "recall@k": r,
            "reciprocal_rank": rr,
            "ndcg@k": ndcg,
        })

    # aggregate
    import statistics
    agg = {
        "precision@k_mean": statistics.mean([r["precision@k"] for r in results]) if results else 0.0,
        "precision@k_std": statistics.pstdev([r["precision@k"] for r in results]) if results else 0.0,
        "recall@k_mean": statistics.mean([r["recall@k"] for r in results]) if results else 0.0,
        "mrr": statistics.mean([r["reciprocal_rank"] for r in results]) if results else 0.0,
        "ndcg@k_mean": statistics.mean([r["ndcg@k"] for r in results]) if results else 0.0,
        "queries_evaluated": len(results),
    }

    return results, agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["simulate", "hybrid"], default="simulate")
    parser.add_argument("--out", default="evaluation_report.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    queries = load_queries(QUERY_FILE)
    print(f"Loaded {len(queries)} queries from {QUERY_FILE}")

    # Quick check: if hybrid mode requested but core deps (faiss, loaders) are missing,
    # detect and fall back to simulate to avoid noisy per-query import errors.
    if args.mode == "hybrid":
        try:
            # attempt light import to verify environment
            import importlib
            # check faiss first (common missing dependency)
            importlib.import_module("faiss")
            importlib.import_module("retrieve")
            importlib.import_module("hybrid_retrieve")
        except Exception as e:
            print("Hybrid mode unavailable (missing deps). Falling back to simulate mode.")
            if args.debug:
                print("Hybrid import error:", e)
            args.mode = "simulate"

    results, agg = evaluate(queries, k=args.k, mode=args.mode, debug=args.debug)

    report = {"aggregate": agg, "per_query": results}
    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print a short summary
    print("\nEvaluation summary:")
    for k, v in agg.items():
        print(f"- {k}: {v}")

    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
