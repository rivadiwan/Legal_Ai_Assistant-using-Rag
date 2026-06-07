"""
Legal RAG Evaluation Pipeline
==============================
Evaluates a legal RAG system across three dimensions:

1. RETRIEVAL METRICS
   - Recall@5, Recall@10, MRR, Hit Rate

2. GENERATION METRICS (via RAGAS)
   - Faithfulness, Context Precision, Context Recall, Answer Relevancy

3. CUSTOM LEGAL METRICS
   - Case Accuracy, Section Accuracy, Verdict Accuracy

Usage:
    python evaluation_final/evaluation_code.py

Output:
    - evaluation_final/results.csv          (per-query results)
    - evaluation_final/metrics_summary.json (aggregated metrics)
    - evaluation_final/checkpoint.json      (resume from interruption)
"""

import json
import os
import re
import sys
import time
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

os.chdir(PROJECT_ROOT)

from groq import Groq
from retrieve import load_model, load_faiss_index, load_chunks
from hybrid_retrieve import hybrid_search
from reranker import rerank_results
from bm25_retriever import build_bm25
from ask_llm import ask_groq, build_context
from case_context import get_case_chunks

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation_final")
QUERIES_FILE = os.path.join(EVAL_DIR, "case_based_queries.json")
RESULTS_CSV = os.path.join(EVAL_DIR, "results.csv")
METRICS_JSON = os.path.join(EVAL_DIR, "metrics_summary.json")
CHECKPOINT_FILE = os.path.join(EVAL_DIR, "checkpoint.json")

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5

# ---------------------------------------------------------------
# SECTION EXTRACTION (regex-based, no LLM call)
# ---------------------------------------------------------------

SECTION_PATTERN = re.compile(
    r"(?:Section|S\.|s\.|Sec\.)\s+(\d+[A-Za-z]*(?:\s*\([^)]*\))?)",
    re.IGNORECASE,
)

SECTION_STOP_WORDS = {
    "and", "or", "of", "the", "for", "was", "were", "has", "had",
    "not", "but", "with", "from", "that", "this", "which", "shall",
    "under", "also", "section", "sec", "s", "is", "it", "be", "by",
    "to", "in", "on", "at", "an", "a", "if", "as", "so", "no", "per",
    "can", "may", "any", "all", "each", "every", "other", "such",
    "said", "been", "being", "have", "do", "does", "did", "will",
    "would", "could", "should", "are", "am", "its", "his", "her",
    "he", "she", "they", "we", "you", "i", "my", "our", "your",
    "their", "there", "here", "where", "when", "what", "who", "how",
    "then", "than", "now", "just", "only", "very", "too", "much",
    "many", "more", "most", "some", "over", "into", "out", "up",
    "down", "about", "before", "after", "above", "below", "between",
    "during", "through", "without", "within", "against", "towards",
    "toward", "upon", "since", "until", "while", "although", "though",
    "because", "therefore", "however", "further", "furthermore",
    "according", "pursuant", "IPC", "CrPC", "Code", "Act",
}

KNOWN_ACTS = {"IPC", "CrPC", "CPC", "IEA", "NI Act", "POCSO", "POCSO Act",
              "NDPS", "NDPS Act", "UAPA", "PMLA", "TADA", "MCA", "SCST",
              "Arbitration Act", "Constitution"}  # uppercase for matching

VERDICT_KEYWORDS = [
    "allowed", "dismissed", "acquitted", "convicted",
    "upheld", "set aside", "quashed", "remanded",
    "granted", "denied", "rejected", "sustained",
    "overturned", "affirmed", "disposed of",
    "appeal succeeds", "appeal fails",
    "sentence upheld", "sentence reduced", "sentence enhanced",
    "bail granted", "bail rejected",
    "compensation awarded", "compensation denied",
    "liable", "not liable",
    "guilty", "not guilty",
]


def extract_sections(text):
    sections = []
    for match in SECTION_PATTERN.finditer(text):
        num = match.group(1)
        match_end = match.end()
        tail = text[match_end:match_end + 60]

        act = ""
        act_match = re.match(r"\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", tail)
        if act_match:
            words = act_match.group(1).split()
            filtered = [w for w in words if w.lower() not in SECTION_STOP_WORDS]
            if filtered:
                act = " ".join(filtered[:2])

        full = f"Section {num}"
        if act:
            full += f" {act}"
        sections.append(full.strip())

    return list(dict.fromkeys(sections))


def extract_verdict(text):
    text_lower = text.lower()
    found = []
    for kw in VERDICT_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return "; ".join(found) if found else ""


def case_name_in_answer(case_name, answer):
    case_lower = case_name.lower()
    answer_lower = answer.lower()
    parts = case_lower.split(" vs ")
    if len(parts) == 2:
        if parts[0].strip() in answer_lower or parts[1].strip() in answer_lower:
            return True
    return case_lower in answer_lower


# ---------------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------------

def compute_retrieval_metrics(eval_results):
    recall_5_scores = []
    recall_10_scores = []
    mrr_scores = []
    hit_rate_scores = []

    for r in eval_results:
        relevant_ids = set(r.get("relevant_chunk_ids", []))
        retrieved_ids = [c.get("chunk_id", "") for c in r.get("retrieved_chunks", [])]

        if not relevant_ids:
            continue

        top5 = set(retrieved_ids[:5])
        recall_5 = len(top5 & relevant_ids) / len(relevant_ids)
        recall_5_scores.append(recall_5)

        top10 = set(retrieved_ids[:10])
        recall_10 = len(top10 & relevant_ids) / len(relevant_ids)
        recall_10_scores.append(recall_10)

        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        hit = 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:10]) else 0.0
        hit_rate_scores.append(hit)

    return {
        "Recall@5": round(float(np.mean(recall_5_scores)), 4) if recall_5_scores else 0.0,
        "Recall@10": round(float(np.mean(recall_10_scores)), 4) if recall_10_scores else 0.0,
        "MRR": round(float(np.mean(mrr_scores)), 4) if mrr_scores else 0.0,
        "Hit Rate": round(float(np.mean(hit_rate_scores)), 4) if hit_rate_scores else 0.0,
    }


def compute_custom_legal_metrics(eval_results):
    case_scores = []
    section_scores = []
    verdict_scores = []

    for r in eval_results:
        gt = r.get("ground_truth_entities", {})
        pred = r.get("predicted_entities", {})

        case_scores.append(1.0 if r.get("case_name_match", False) else 0.0)

        gt_secs = set(s.strip().upper() for s in (gt.get("sections", []) or []) if s.strip())
        pred_secs = set(s.strip().upper() for s in (pred.get("sections", []) or []) if s.strip())
        section_scores.append(
            len(gt_secs & pred_secs) / len(gt_secs) if gt_secs else 1.0
        )

        gt_verdict = (gt.get("verdict") or "").strip().lower()
        pred_verdict = (pred.get("verdict") or "").strip().lower()
        verdict_scores.append(
            1.0 if gt_verdict and pred_verdict and gt_verdict in pred_verdict else 0.0
        )

    return {
        "Case Accuracy": round(float(np.mean(case_scores)), 4) if case_scores else 0.0,
        "Section Accuracy": round(float(np.mean(section_scores)), 4) if section_scores else 0.0,
        "Verdict Accuracy": round(float(np.mean(verdict_scores)), 4) if verdict_scores else 0.0,
    }


def compute_ragas_metrics(eval_results):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            context_precision,
            context_recall,
            answer_relevancy,
        )

        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for r in eval_results:
            dataset_dict["question"].append(r["query"])
            dataset_dict["answer"].append(r.get("generated_answer", "") or "")
            dataset_dict["contexts"].append([
                c.get("chunk_text", "") for c in r.get("retrieved_chunks", [])
            ])
            dataset_dict["ground_truth"].append(r.get("ground_truth_answer", "") or "")

        dataset = Dataset.from_dict(dataset_dict)
        scores = evaluate(
            dataset,
            metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        )

        return {
            "Faithfulness": round(float(scores.get("faithfulness", 0.0)), 4),
            "Context Precision": round(float(scores.get("context_precision", 0.0)), 4),
            "Context Recall": round(float(scores.get("context_recall", 0.0)), 4),
            "Answer Relevancy": round(float(scores.get("answer_relevancy", 0.0)), 4),
        }
    except ImportError:
        print("[WARNING] RAGAS not installed. Run: pip install ragas datasets")
        return {}
    except Exception as e:
        print(f"[WARNING] RAGAS computation failed: {e}")
        return {}


# ---------------------------------------------------------------
# EVALUATION PIPELINE
# ---------------------------------------------------------------

class LegalRAGEvaluator:
    def __init__(self):
        print("\n" + "=" * 60)
        print("INITIALIZING LEGAL RAG EVALUATOR")
        print("=" * 60)

        print("\n[1/5] Loading Groq client...")
        self.client = Groq(api_key=os.getenv("API_KEY"))

        print("[2/5] Loading embedding model...")
        self.model = load_model()

        print("[3/5] Loading FAISS index...")
        self.index = load_faiss_index()

        print("[4/5] Loading chunks...")
        self.chunks = load_chunks()

        print("[5/5] Building BM25 index...")
        self.bm25 = build_bm25(self.chunks)

        self.chunk_id_map = {c["chunk_id"]: c for c in self.chunks}
        self.case_chunk_cache = {}
        self.gt_entities_cache = {}

        print("\nEvaluator ready.\n")

    def get_relevant_chunk_ids(self, case_name):
        if case_name in self.case_chunk_cache:
            return self.case_chunk_cache[case_name]
        case_chunks = get_case_chunks(case_name, self.chunks)
        chunk_ids = [c["chunk_id"] for c in case_chunks]
        self.case_chunk_cache[case_name] = chunk_ids
        return chunk_ids

    def get_ground_truth_entities(self, case_name):
        if case_name in self.gt_entities_cache:
            return self.gt_entities_cache[case_name]
        case_chunks = get_case_chunks(case_name, self.chunks)
        full_text = " ".join(c["chunk_text"] for c in case_chunks)
        entities = {
            "case_name": case_name,
            "sections": extract_sections(full_text),
            "verdict": extract_verdict(full_text),
        }
        self.gt_entities_cache[case_name] = entities
        return entities

    def retrieve(self, query):
        results = hybrid_search(
            query, self.model, self.index, self.chunks, self.bm25,
            top_k=RETRIEVAL_TOP_K,
        )

        if not (len(results) > 0 and results[0].get("score") == "CASE_MATCH"):
            results = rerank_results(query, results, top_k=RERANK_TOP_K)

        return results

    def generate_answer(self, query, retrieved_results):
        context = build_context(retrieved_results)
        return ask_groq(query, context)

    def generate_ground_truth_answer(self, query, case_name):
        case_chunks = get_case_chunks(case_name, self.chunks)
        context = build_context(case_chunks)
        return ask_groq(query, context)

    def run_single_query(self, query_data):
        query_id = query_data["query_id"]
        query = query_data["query"]
        case_name = query_data["case_name"]

        print(f"  [{query_id}] {query[:90]}...")

        relevant_chunk_ids = self.get_relevant_chunk_ids(case_name)
        retrieved_chunks = self.retrieve(query)
        generated_answer = self.generate_answer(query, retrieved_chunks)
        ground_truth_answer = self.generate_ground_truth_answer(query, case_name)
        ground_truth_entities = self.get_ground_truth_entities(case_name)

        predicted_entities = {
            "case_name": case_name if case_name_in_answer(case_name, generated_answer) else "",
            "sections": extract_sections(generated_answer),
            "verdict": extract_verdict(generated_answer),
        }

        return {
            "query_id": query_id,
            "query": query,
            "case_name": case_name,
            "query_type": query_data.get("query_type", ""),
            "relevant_chunk_ids": relevant_chunk_ids,
            "retrieved_chunks": retrieved_chunks,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "ground_truth_entities": ground_truth_entities,
            "predicted_entities": predicted_entities,
            "case_name_match": case_name_in_answer(case_name, generated_answer),
        }

    def evaluate(self):
        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            queries = json.load(f)
        print(f"\nLoaded {len(queries)} queries for evaluation.\n")

        completed_ids = set()
        eval_results = []
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                eval_results = checkpoint.get("results", [])
                completed_ids = set(r["query_id"] for r in eval_results)
            print(f"Resuming from checkpoint: {len(completed_ids)} queries already processed.\n")

        print("=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)

        for i, qd in enumerate(queries):
            if qd["query_id"] in completed_ids:
                continue

            try:
                result = self.run_single_query(qd)
                eval_results.append(result)

                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump({"results": eval_results}, f, indent=2, ensure_ascii=False)

                time.sleep(1)
            except Exception as e:
                print(f"\n  [ERROR] {qd['query_id']}: {e}")
                continue

        print("\n" + "=" * 60)
        print("COMPUTING METRICS")
        print("=" * 60)

        retrieval_metrics = compute_retrieval_metrics(eval_results)
        ragas_metrics = compute_ragas_metrics(eval_results)
        legal_metrics = compute_custom_legal_metrics(eval_results)

        all_metrics = {**retrieval_metrics, **ragas_metrics, **legal_metrics}

        print("\n" + "-" * 40)
        print("RETRIEVAL METRICS")
        print("-" * 40)
        for k, v in retrieval_metrics.items():
            print(f"  {k:20s}: {v}")

        print("\n" + "-" * 40)
        print("GENERATION METRICS (RAGAS)")
        print("-" * 40)
        if ragas_metrics:
            for k, v in ragas_metrics.items():
                print(f"  {k:20s}: {v}")
        else:
            print("  (not computed)")

        print("\n" + "-" * 40)
        print("CUSTOM LEGAL METRICS")
        print("-" * 40)
        for k, v in legal_metrics.items():
            print(f"  {k:20s}: {v}")

        self.save_results(eval_results, all_metrics)
        return eval_results, all_metrics

    def save_results(self, eval_results, metrics):
        rows = []
        for r in eval_results:
            rows.append({
                "query_id": r["query_id"],
                "query": r["query"],
                "case_name": r["case_name"],
                "query_type": r.get("query_type", ""),
                "num_relevant_chunks": len(r["relevant_chunk_ids"]),
                "num_retrieved_chunks": len(r["retrieved_chunks"]),
                "retrieved_case_names": ", ".join(
                    dict.fromkeys(
                        c.get("case_name", "") for c in r["retrieved_chunks"]
                    )
                ),
                "case_name_match": r.get("case_name_match", False),
                "predicted_sections": ", ".join(
                    r.get("predicted_entities", {}).get("sections", [])
                ),
                "predicted_verdict": r.get("predicted_entities", {}).get("verdict", ""),
                "generated_answer": (r.get("generated_answer") or "")[:500],
                "ground_truth_answer": (r.get("ground_truth_answer") or "")[:500],
            })

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
        print(f"\nDetailed results saved to: {RESULTS_CSV}")

        with open(METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics summary saved to: {METRICS_JSON}")


# ---------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------

if __name__ == "__main__":
    evaluator = LegalRAGEvaluator()
    evaluator.evaluate()