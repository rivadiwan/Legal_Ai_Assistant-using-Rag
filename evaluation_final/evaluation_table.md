# Evaluation Matrix for Legal-RAG Project

This document describes the recommended evaluation metrics, protocol, and reporting table for the Legal-RAG retrieval + generation system.

## Metrics

- Precision@k
  - Definition: proportion of retrieved items in the top-k that are relevant.
  - Formula: Precision@k = (1/k) * (# relevant in top-k)

- Recall@k
  - Definition: proportion of all relevant items that appear in top-k. With single-ground-truth case this is 1 if ground-truth is in top-k, else 0.
  - Formula: Recall@k = (# relevant in top-k) / (# relevant total)

- F1@k
  - Harmonic mean of Precision@k and Recall@k when both > 0.

- Mean Reciprocal Rank (MRR)
  - Definition: average of reciprocal rank (1 / rank of first relevant) across queries.

- nDCG@k (Normalized Discounted Cumulative Gain)
  - Definition: a graded relevance metric that rewards relevant items earlier in the ranking.

- Mean Average Precision (MAP)
  - Definition: mean of average precision across queries (useful when multiple relevant items per query exist).

- Case Match Hit Rate
  - Definition: fraction of queries where the top-1 retrieved case equals the ground-truth case.

- Rerank Gain
  - Definition: improvement in a chosen metric (e.g., nDCG@5 or Precision@5) after applying the reranker.
  - Report as delta and relative percentage.

- Hallucination Rate (human-eval)
  - Definition: fraction of generated answers that contain statements not grounded in retrieved context (requires human labeling).

- Answer Accuracy (human-eval)
  - Definition: human-judged correctness of the final answer (binary or graded).

- Latency
  - End-to-end time for retrieval + reranking + context build (median and 95th percentile).

## Evaluation Protocol (Recommended)

1. Dataset & Queries
   - Use `case_based_queries.json` in this folder as the query set (or add your curated queries).
   - For each query the `case_name` field is treated as the ground-truth case.

2. Retrieval Outputs
   - For automatic evaluation provide per-query top-K ranked lists of retrieved case identifiers or case names.
   - If the system returns chunk-level results, map chunks to their `case_name` for case-level metrics.

3. Metric Computation
   - Compute Precision@k, nDCG@k, and MRR for each query and report mean ± std.
   - Compute Case Match Hit Rate (Top-1 accuracy) separately.
   - Measure Rerank Gain by running the pipeline with and without reranker and computing the metric deltas.

4. Human Evaluation (for final answers)
   - Sample N queries (N ≥ 100 recommended for stable estimates) and have 2 annotators label: correctness, citation accuracy, hallucination presence, and helpfulness (1–5).
   - Report inter-annotator agreement (Cohen's kappa) and resolve disagreements by adjudication.

## Reporting Table (example)

| Metric | Top-K | Value (mean) | Std Dev | Notes |
|---|---:|---:|---:|---|
| Precision@5 | 5 | 0.72 | 0.12 | -- |
| Recall@10 | 10 | 0.85 | 0.10 | -- |
| MRR | 5 | 0.63 | 0.18 | -- |
| nDCG@5 | 5 | 0.68 | 0.15 | -- |
| Case Match Hit Rate | 1 | 0.56 | - | Top-1 case correct |
| Rerank Gain (nDCG@5) | 5 | +0.07 | - | Absolute improvement over baseline |
| Answer Accuracy (human) | - | 0.79 | - | Proportion correct |
| Hallucination Rate (human) | - | 0.12 | - | Fraction with unsupported facts |
| Latency (ms) | - | 450 | 120 | median (95th pct: 1200ms) |

## Deliverables
- `case_based_queries.json` — queries (already present).
- `run_evaluation.py` — evaluator script (in same folder) that computes the automatic metrics and saves a CSV/JSON report.

## Notes
- For legal tasks, prioritize human evaluation for final answers; retrieval metrics approximate effectiveness but do not capture legal correctness.
- Use consistent tokenization and mapping from chunks to case identifiers when aggregating metrics.


