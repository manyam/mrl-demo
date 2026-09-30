"""Evaluation and benchmarking module for MRL bi-encoder.

Benchmarks retrieval performance across Matryoshka dimensions (NDCG@10, MRR@10,
Recall@1, Recall@5, Recall@10, vector storage size, and search latency).
Generates Markdown report and exports reports/benchmark_mrl.csv.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from src.models.mrl_model import get_default_nested_dims

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mrl.evaluate")


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSON lines file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_retrieval_metrics(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    query_to_positive_idx: list[set[int]],
    k_list: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Compute NDCG@10, MRR@10, Recall@1, Recall@5, Recall@10 using matrix multiplication."""
    num_queries = query_embeddings.size(0)
    device = query_embeddings.device

    recall_counts = {k: 0.0 for k in k_list}
    mrr_sum = 0.0
    ndcg_sum = 0.0

    # Batch queries to avoid huge similarity matrices in memory
    batch_size = 256
    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)
        q_batch = query_embeddings[start_idx:end_idx]

        # Cosine similarity matrix: [batch_size, num_docs]
        sim_matrix = torch.matmul(q_batch, corpus_embeddings.T)

        # Get top-10 ranked document indices
        topk_scores, topk_indices = torch.topk(sim_matrix, k=10, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        for i, q_idx in enumerate(range(start_idx, end_idx)):
            pos_indices = query_to_positive_idx[q_idx]
            ranked_docs = topk_indices[i]

            # Check positions
            hit_rank = None
            for rank_0, doc_id in enumerate(ranked_docs):
                if doc_id in pos_indices:
                    hit_rank = rank_0 + 1  # 1-indexed
                    break

            if hit_rank is not None:
                # MRR@10
                mrr_sum += 1.0 / hit_rank
                # NDCG@10 (assuming single binary positive: IDCG = 1.0)
                ndcg_sum += 1.0 / math.log2(hit_rank + 1)
                # Recall@k
                for k in k_list:
                    if hit_rank <= k:
                        recall_counts[k] += 1.0

    return {
        "ndcg@10": ndcg_sum / num_queries,
        "mrr@10": mrr_sum / num_queries,
        "recall@1": recall_counts[1] / num_queries,
        "recall@5": recall_counts[5] / num_queries,
        "recall@10": recall_counts[10] / num_queries,
    }


def benchmark_search_latency(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    num_eval_queries: int = 1000,
) -> float:
    """Measure average cosine similarity search latency in milliseconds over 1,000 queries."""
    device = query_embeddings.device
    num_queries = query_embeddings.size(0)

    # Sample or cycle queries to reach num_eval_queries
    if num_queries < num_eval_queries:
        indices = torch.randint(0, num_queries, (num_eval_queries,), device=device)
        eval_queries = query_embeddings[indices]
    else:
        eval_queries = query_embeddings[:num_eval_queries]

    # Warmup
    for _ in range(5):
        _ = torch.matmul(eval_queries[:10], corpus_embeddings.T)
    if device.type == "mps":
        torch.mps.synchronize()

    # Benchmark individual query latency
    start_time = time.perf_counter()
    for i in range(num_eval_queries):
        q = eval_queries[i : i + 1]
        sims = torch.matmul(q, corpus_embeddings.T)
        _ = torch.topk(sims, k=10, dim=1)
    if device.type == "mps":
        torch.mps.synchronize()

    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency_ms = total_time_ms / num_eval_queries
    return avg_latency_ms


def main():
    parser = argparse.ArgumentParser(description="Evaluate MRL embedding dimensions on held-out test split.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing processed test data")
    parser.add_argument("--output_csv", type=str, default="reports/benchmark_mrl.csv", help="Path to export CSV report")
    parser.add_argument("--max_test_queries", type=int, default=1000, help="Max test queries to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_queries_file = data_dir / "test_queries.jsonl"
    corpus_file = data_dir / "corpus.jsonl"

    if not test_queries_file.exists() or not corpus_file.exists():
        raise FileNotFoundError(f"Processed test data not found in {data_dir}. Run preprocess.py first.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Loading evaluation model from {args.model_path} onto {device}...")
    model = SentenceTransformer(args.model_path, device=str(device))
    hidden_size = model.get_sentence_embedding_dimension()
    nested_dims = get_default_nested_dims(hidden_size)

    logger.info(f"Base model hidden dimension: {hidden_size}. Matryoshka dimensions to evaluate: {nested_dims}")

    # Load test queries and corpus
    test_records = load_jsonl(test_queries_file)
    if args.max_test_queries and len(test_records) > args.max_test_queries:
        test_records = test_records[: args.max_test_queries]

    corpus_records = load_jsonl(corpus_file)

    logger.info(f"Evaluating {len(test_records)} queries against {len(corpus_records)} candidate corpus businesses...")

    # Map corpus doc_id to array index
    doc_id_to_idx = {doc["doc_id"]: idx for idx, doc in enumerate(corpus_records)}
    query_to_positive_idx = []
    valid_test_records = []

    for q in test_records:
        gmap_id = q["gmap_id"]
        if gmap_id in doc_id_to_idx:
            query_to_positive_idx.append({doc_id_to_idx[gmap_id]})
            valid_test_records.append(q)

    query_texts = [q["query"] for q in valid_test_records]
    corpus_texts = [d["text"] for d in corpus_records]

    # Pre-compute full untruncated embeddings once
    logger.info("Generating full embeddings for corpus...")
    corpus_embeds_raw = model.encode(
        corpus_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )

    # Save cached corpus embeddings for CLI demo fast startup
    cache_path = data_dir / "corpus_embeddings.pt"
    torch.save(corpus_embeds_raw.cpu(), cache_path)
    logger.info(f"Cached full corpus embeddings to {cache_path}")

    logger.info("Generating full embeddings for test queries...")
    query_embeds_raw = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )

    results = []
    full_dim_ndcg = None

    for dim in nested_dims:
        logger.info(f"Benchmarking dimension {dim}...")
        # Slice and L2-normalize
        c_dim = F.normalize(corpus_embeds_raw[:, :dim], p=2, dim=1)
        q_dim = F.normalize(query_embeds_raw[:, :dim], p=2, dim=1)

        metrics = compute_retrieval_metrics(
            query_embeddings=q_dim,
            corpus_embeddings=c_dim,
            query_to_positive_idx=query_to_positive_idx,
        )

        avg_latency_ms = benchmark_search_latency(
            query_embeddings=q_dim,
            corpus_embeddings=c_dim,
            num_eval_queries=min(1000, len(valid_test_records)),
        )

        # Storage size in MB per 10k entities: 10000 * dim * 4 bytes / 10^6 bytes = 0.04 * dim MB
        size_10k_mb = (10000 * dim * 4) / 1_000_000

        record = {
            "Dimension": dim,
            "NDCG@10": round(metrics["ndcg@10"], 4),
            "MRR@10": round(metrics["mrr@10"], 4),
            "Recall@1": round(metrics["recall@1"], 4),
            "Recall@5": round(metrics["recall@5"], 4),
            "Recall@10": round(metrics["recall@10"], 4),
            "Size/10k Vectors (MB)": round(size_10k_mb, 2),
            "Avg Latency (ms)": round(avg_latency_ms, 3),
        }
        results.append(record)

        if full_dim_ndcg is None:
            full_dim_ndcg = metrics["ndcg@10"]

    # Export CSV report
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Exported benchmark metrics to {output_csv}")

    # Print Markdown table required by spec
    # Expected table format:
    # | Dimension | NDCG@10 | MRR@10 | Recall@10 | Size/10k Vectors (MB) | Avg Latency (ms) |
    summary_cols = ["Dimension", "NDCG@10", "MRR@10", "Recall@10", "Size/10k Vectors (MB)", "Avg Latency (ms)"]
    summary_df = df[summary_cols]
    try:
        md_table = summary_df.to_markdown(index=False)
    except Exception:
        headers = list(summary_df.columns)
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in summary_df.values]
        md_table = "\n".join([header_row, sep_row] + data_rows)
    print("\n" + "=" * 80)
    print("MRL RETRIEVAL BENCHMARK REPORT")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")

    # MRL Verification Criterion 3:
    # Test vectors truncated to 64 dimensions must demonstrate non-random retrieval
    # (NDCG@10 retaining at least 80% of the full-dimension score).
    dim_64_result = next((r for r in results if r["Dimension"] == 64), None)
    if dim_64_result and full_dim_ndcg and full_dim_ndcg > 0:
        retention = (dim_64_result["NDCG@10"] / full_dim_ndcg) * 100.0
        print(f"MRL Retention Check (64-dim vs Full-dim): {retention:.1f}%")
        if retention >= 80.0:
            print("[PASS] Acceptance Criterion 3: 64-dim NDCG@10 retained >= 80% of full-dimension score.")
        else:
            print(f"[WARNING] 64-dim NDCG@10 retention is {retention:.1f}% (< 80%).")
    elif dim_64_result:
        print("[INFO] Full-dim NDCG is zero or baseline. 64-dim NDCG@10:", dim_64_result["NDCG@10"])


if __name__ == "__main__":
    main()
